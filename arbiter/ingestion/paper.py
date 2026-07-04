"""Main-paper ingestion for RCT PDFs."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

from arbiter.config import EnvSettings
from arbiter.ingestion.docling_adapter import (
    convert_pdf,
    docling_markdown_by_page,
    docling_page_boxes,
)
from arbiter.models import DocumentSection, PageBox, ParsingQuality, SectionMap

DOMAIN_TAGS = ("D1", "D2", "D3", "D4", "D5")
ALL_DOMAIN_TAGS = list(DOMAIN_TAGS)
NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "D1": (
        "random",
        "randomisation",
        "randomization",
        "allocation",
        "concealment",
        "baseline",
        "sequence",
    ),
    "D2": (
        "blinding",
        "masking",
        "open-label",
        "deviation",
        "adherence",
        "compliance",
        "intention-to-treat",
        "itt",
        "per-protocol",
    ),
    "D3": (
        "missing",
        "lost to follow-up",
        "dropout",
        "withdrawal",
        "imputation",
        "censoring",
        "analysed",
        "analyzed",
    ),
    "D4": (
        "outcome",
        "endpoint",
        "measure",
        "assessment",
        "assessor",
        "adjudication",
        "central review",
    ),
    "D5": (
        "pre-specified",
        "prespecified",
        "pre-registered",
        "preregistered",
        "protocol",
        "statistical analysis plan",
        "registry",
        "clinicaltrials.gov",
    ),
    "METHODS": (
        "method",
        "methods",
        "statistical analysis",
        "participants",
        "interventions",
        "procedures",
    ),
    "RESULTS": (
        "result",
        "results",
        "participant flow",
        "baseline characteristics",
        "efficacy",
        "safety",
    ),
}

CANONICAL_SECTION_LABELS = {
    "ABSTRACT",
    "BACKGROUND",
    "INTRODUCTION",
    "METHOD",
    "METHODS",
    "MATERIALS AND METHODS",
    "PATIENTS AND METHODS",
    "PARTICIPANTS AND METHODS",
    "STATISTICAL ANALYSIS",
    "RESULT",
    "RESULTS",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "REFERENCES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "SUPPLEMENTARY MATERIAL",
}

TOP_LEVEL_SECTION_LABELS = CANONICAL_SECTION_LABELS - {
    "STATISTICAL ANALYSIS",
}


@dataclass(frozen=True)
class _SectionStart:
    label: str
    page: int
    offset: int


def ingest_paper(path: Path) -> tuple[SectionMap, str]:
    """Parse a main RCT paper into labelled sections plus raw text."""

    source_path = str(path)
    try:
        document = convert_pdf(path, EnvSettings())
        page_texts, page_starts = docling_markdown_by_page(document)
        page_boxes = docling_page_boxes(document)
    except Exception:
        return _degraded_section_map(source_path, ""), ""

    full_text = "\n".join(page_texts).strip()
    raw_stream = full_text
    if not full_text:
        return _degraded_section_map(source_path, raw_stream), raw_stream

    headers = _section_starts(page_texts, page_starts, page_boxes)
    nct_match = NCT_PATTERN.search(raw_stream)
    section_map = SectionMap(
        source_path=source_path,
        full_text=full_text,
        sections=_build_sections(full_text, page_starts, headers),
        page_boxes=page_boxes,
        parsing_quality=ParsingQuality.STANDARD,
        nct_number=nct_match.group(0).upper() if nct_match else None,
    )
    return section_map, raw_stream


def _section_starts(
    page_texts: list[str],
    page_starts: list[int],
    page_boxes: list[PageBox],
) -> list[_SectionStart]:
    headers: list[_SectionStart] = []
    for page_index, page_text in enumerate(page_texts):
        page_start = page_starts[page_index]
        offset = 0
        for line in page_text.splitlines(keepends=True):
            line_text = line.rstrip()
            label = normalize_heading(line_text.lstrip("#").strip())
            if label in CANONICAL_SECTION_LABELS:
                headers.append(_SectionStart(label=label, page=page_index, offset=page_start + offset))
            offset += len(line)
    for box in page_boxes:
        if box.boxclass != "section-header":
            continue
        label = normalize_heading(box.text)
        if label in CANONICAL_SECTION_LABELS and 0 <= box.page < len(page_starts):
            if any(header.label == label and header.page == box.page for header in headers):
                continue
            page_text = page_texts[box.page]
            found_at = page_text.find(box.text)
            headers.append(_SectionStart(label=label, page=box.page, offset=page_starts[box.page] + max(found_at, 0)))
    return _dedupe_headers(headers)


def _build_sections(
    full_text: str,
    page_starts: list[int],
    headers: list[_SectionStart],
) -> list[DocumentSection]:
    usable_headers = [header for header in headers if 0 <= header.offset < len(full_text)]
    if not usable_headers:
        return [
            DocumentSection(
                label="FULL_TEXT",
                pages=list(range(len(page_starts))),
                char_start=0,
                char_end=len(full_text),
                text=full_text,
                domain_tags=_domain_tags("FULL_TEXT", full_text),
            )
        ]

    sections: list[DocumentSection] = []
    for index, header in enumerate(usable_headers):
        start = header.offset
        end = _section_end_offset(header, index, usable_headers, len(full_text))
        text = full_text[start:end].strip()
        sections.append(
            DocumentSection(
                label=header.label,
                pages=_pages_for_range(start, end, page_starts),
                char_start=start,
                char_end=end,
                text=text,
                domain_tags=_domain_tags(header.label, text),
            )
        )
    return sections or [
        DocumentSection(
            label="FULL_TEXT",
            pages=list(range(len(page_starts))),
            char_start=0,
            char_end=len(full_text),
            text=full_text,
            domain_tags=_domain_tags("FULL_TEXT", full_text),
        )
    ]


def _section_end_offset(
    header: _SectionStart,
    index: int,
    headers: list[_SectionStart],
    full_text_length: int,
) -> int:
    if _is_top_level_anchor(header.label):
        for next_header in headers[index + 1 :]:
            if _is_top_level_anchor(next_header.label):
                return next_header.offset
        return full_text_length
    return headers[index + 1].offset if index + 1 < len(headers) else full_text_length


def _is_top_level_anchor(label: str) -> bool:
    return label in TOP_LEVEL_SECTION_LABELS


def _pages_for_range(start: int, end: int, page_starts: list[int]) -> list[int]:
    pages = [idx for idx, page_start in enumerate(page_starts) if start <= page_start < end]
    start_page = max((idx for idx, page_start in enumerate(page_starts) if page_start <= start), default=0)
    return sorted(set([start_page, *pages]))


def _domain_tags(label: str, text: str) -> list[str]:
    scan_chars = EnvSettings().domain_tag_scan_chars
    haystack = f"{label}\n{text[:scan_chars]}".lower()
    return [domain for domain in DOMAIN_TAGS if any(keyword in haystack for keyword in SECTION_KEYWORDS[domain])]


def _degraded_section_map(source_path: str, raw_stream: str) -> SectionMap:
    nct_match = NCT_PATTERN.search(raw_stream)
    return SectionMap(
        source_path=source_path,
        full_text=raw_stream,
        sections=[
            DocumentSection(
                label="FULL_TEXT",
                pages=[0] if raw_stream else [],
                char_start=0,
                char_end=len(raw_stream),
                text=raw_stream,
                domain_tags=ALL_DOMAIN_TAGS.copy(),
            )
        ],
        page_boxes=[],
        parsing_quality=ParsingQuality.DEGRADED,
        nct_number=nct_match.group(0).upper() if nct_match else None,
    )


def _dedupe_headers(headers: list[_SectionStart]) -> list[_SectionStart]:
    deduped: list[_SectionStart] = []
    for header in sorted(headers, key=lambda item: item.offset):
        if deduped and header.offset == deduped[-1].offset:
            continue
        deduped.append(header)
    return deduped


def normalize_heading(text: str) -> str:
    """Return the PRD's uppercase, punctuation-trimmed section label."""

    return text.strip().strip(string.whitespace + string.punctuation).upper()
