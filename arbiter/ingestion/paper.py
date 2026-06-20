"""Main-paper ingestion for RCT PDFs."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pymupdf

from arbiter.config import EnvSettings
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


@dataclass(frozen=True)
class _Line:
    text: str
    bbox: tuple[float, float, float, float]
    page: int
    max_size: float
    is_bold: bool


@dataclass(frozen=True)
class _SectionStart:
    label: str
    page: int
    offset: int


def ingest_paper(path: Path) -> tuple[SectionMap, str]:
    """Parse a main RCT paper into labelled sections plus raw text."""

    source_path = str(path)
    try:
        with pymupdf.open(path) as doc:
            raw_page_texts = [doc.load_page(page_index).get_text() for page_index in range(len(doc))]
            raw_stream = "\n".join(raw_page_texts)
            page_texts, page_starts, page_boxes, headers = _parse_layout(doc)
    except Exception:
        raw_stream = _read_raw_stream_best_effort(path)
        return _degraded_section_map(source_path, raw_stream), raw_stream

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        return _degraded_section_map(source_path, raw_stream), raw_stream

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


def _parse_layout(
    doc: pymupdf.Document,
) -> tuple[list[str], list[int], list[PageBox], list[_SectionStart]]:
    page_texts: list[str] = []
    page_starts: list[int] = []
    page_boxes: list[PageBox] = []
    headers: list[_SectionStart] = []
    running_offset = 0

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_starts.append(running_offset)
        lines = _extract_lines(page, page_index)
        median_size = _median([line.max_size for line in lines]) if lines else 0.0
        page_text = page.get_text()
        page_texts.append(page_text)

        for line in lines:
            label = normalize_heading(line.text)
            is_header = _is_section_header(line, label, median_size)
            boxclass = "section-header" if is_header else "text"
            page_boxes.append(
                PageBox(
                    boxclass=boxclass,
                    text=line.text,
                    bbox=line.bbox,
                    page=page_index,
                )
            )
            if is_header:
                line_offset = page_text.find(line.text)
                offset = running_offset + max(line_offset, 0)
                headers.append(_SectionStart(label=label, page=page_index, offset=offset))

        running_offset += len(page_text)
        if page_index < len(doc) - 1:
            running_offset += 1

    return page_texts, page_starts, page_boxes, _dedupe_headers(headers)


def _extract_lines(page: pymupdf.Page, page_index: int) -> list[_Line]:
    raw = page.get_text("dict")
    lines: list[_Line] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            span_texts: list[str] = []
            sizes: list[float] = []
            is_bold = False
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    span_texts.append(text)
                size = span.get("size")
                if isinstance(size, int | float):
                    sizes.append(float(size))
                font = str(span.get("font", "")).lower()
                is_bold = is_bold or "bold" in font
            text = " ".join("".join(span_texts).split())
            if not text:
                continue
            bbox_values = tuple(float(value) for value in line.get("bbox", (0, 0, 0, 0)))
            bbox = cast(tuple[float, float, float, float], bbox_values)
            lines.append(
                _Line(
                    text=text,
                    bbox=bbox,
                    page=page_index,
                    max_size=max(sizes) if sizes else 0.0,
                    is_bold=is_bold,
                )
            )
    return lines


def _is_section_header(line: _Line, label: str, median_size: float) -> bool:
    if not label or len(label) > 80:
        return False
    word_count = len(label.split())
    if word_count > 8:
        return False
    if label in CANONICAL_SECTION_LABELS:
        return True
    has_heading_style = line.is_bold or (median_size > 0 and line.max_size >= median_size + 1.5)
    starts_numbered = bool(re.match(r"^\d+(\.\d+)*\s+[A-Z]", line.text.strip()))
    has_title_case_shape = line.text[:1].isupper() and line.text.count(".") == 0
    return has_heading_style and (starts_numbered or has_title_case_shape)


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
        end = usable_headers[index + 1].offset if index + 1 < len(usable_headers) else len(full_text)
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


def _pages_for_range(start: int, end: int, page_starts: list[int]) -> list[int]:
    pages = [idx for idx, page_start in enumerate(page_starts) if start <= page_start < end]
    start_page = max((idx for idx, page_start in enumerate(page_starts) if page_start <= start), default=0)
    return sorted(set([start_page, *pages]))


def _domain_tags(label: str, text: str) -> list[str]:
    scan_chars = EnvSettings().domain_tag_scan_chars
    haystack = f"{label}\n{text[:scan_chars]}".lower()
    tags = [domain for domain in DOMAIN_TAGS if any(keyword in haystack for keyword in SECTION_KEYWORDS[domain])]
    return tags


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


def _read_raw_stream_best_effort(path: Path) -> str:
    try:
        with pymupdf.open(path) as doc:
            return "\n".join(doc.load_page(page_index).get_text() for page_index in range(len(doc)))
    except Exception:
        return ""


def _dedupe_headers(headers: list[_SectionStart]) -> list[_SectionStart]:
    deduped: list[_SectionStart] = []
    for header in sorted(headers, key=lambda item: item.offset):
        if deduped and header.offset == deduped[-1].offset:
            continue
        deduped.append(header)
    return deduped


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def normalize_heading(text: str) -> str:
    """Return the PRD's uppercase, punctuation-trimmed section label."""

    return text.strip().strip(string.whitespace + string.punctuation).upper()
