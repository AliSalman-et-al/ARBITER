"""Supplement document segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arbiter.config import EnvSettings
from arbiter.ingestion.paper import ALL_DOMAIN_TAGS, SECTION_KEYWORDS, normalize_heading
from arbiter.models import DocType, PageBox, SupplementSegment


@dataclass(frozen=True)
class ParsedSupplementWindow:
    full_text: str
    page_starts: list[int]
    page_boxes: list[PageBox]
    page_offset: int = 0


@dataclass(frozen=True)
class DocumentTypeDetection:
    doc_type: DocType
    lexicon: tuple[str, ...]


DOC_TYPE_LEXICONS: dict[DocType, tuple[str, ...]] = {
    DocType.SAP: (
        "statistical analysis plan",
        "analysis population",
        "interim analysis",
        "multiplicity",
        "estimand",
        "sample size",
    ),
    DocType.PROTOCOL: (
        "study protocol",
        "trial protocol",
        "randomisation",
        "randomization",
        "eligibility",
        "intervention",
    ),
    DocType.APPENDIX: (
        "supplementary appendix",
        "appendix",
        "supplementary material",
        "supplemental appendix",
        "web appendix",
    ),
}


def detect_document_type(
    page_boxes: list[PageBox],
    *,
    settings: EnvSettings | None = None,
) -> DocumentTypeDetection:
    settings = settings or EnvSettings()
    header_text = "\n".join(
        box.text
        for box in page_boxes
        if box.boxclass == "section-header" and box.page < settings.doctype_scan_pages
    ).lower()
    if not header_text.strip():
        return DocumentTypeDetection(DocType.UNKNOWN, DOC_TYPE_LEXICONS[DocType.SAP])

    scores = {
        doc_type: sum(header_text.count(term) for term in lexicon)
        for doc_type, lexicon in DOC_TYPE_LEXICONS.items()
    }
    best_score = max(scores.values())
    if best_score <= 0:
        return DocumentTypeDetection(DocType.UNKNOWN, DOC_TYPE_LEXICONS[DocType.SAP])
    winners = [doc_type for doc_type, score in scores.items() if score == best_score]
    doc_type = DocType.SAP if DocType.SAP in winners else winners[0]
    return DocumentTypeDetection(doc_type, DOC_TYPE_LEXICONS[doc_type])


def segment_document(
    source_file: Path,
    windows: list[ParsedSupplementWindow],
    *,
    doc_type: DocType,
    settings: EnvSettings | None = None,
) -> list[SupplementSegment]:
    settings = settings or EnvSettings()
    segments: list[SupplementSegment] = []
    for window_index, window in enumerate(windows):
        segments.extend(_segment_window(source_file, window, doc_type, window_index, settings))

    if len(segments) < settings.min_segments:
        full_text = "\n".join(window.full_text for window in windows).strip()
        pages = sorted({box.page for window in windows for box in window.page_boxes})
        if not pages:
            pages = [
                page
                for window in windows
                for page in range(window.page_offset, window.page_offset + len(window.page_starts))
            ]
        return [
            SupplementSegment(
                segment_id=f"{source_file.name}__FULL_DOCUMENT__0",
                source_file=str(source_file),
                doc_type=doc_type,
                heading="FULL_DOCUMENT",
                pages=pages,
                raw_text=full_text or " ",
                annotation="No risk-of-bias relevant content.",
                domain_tags=ALL_DOMAIN_TAGS.copy(),
                char_count=len(full_text),
            )
        ]

    return segments


def _segment_window(
    source_file: Path,
    window: ParsedSupplementWindow,
    doc_type: DocType,
    window_index: int,
    settings: EnvSettings,
) -> list[SupplementSegment]:
    full_text = window.full_text
    headers = _header_offsets(window)
    if not headers:
        text = full_text.strip()
        return [
            SupplementSegment(
                segment_id=f"{source_file.name}__WINDOW_{window_index}__0",
                source_file=str(source_file),
                doc_type=doc_type,
                heading=f"WINDOW_{window_index}",
                pages=_pages_for_range(0, len(full_text), window.page_starts, window.page_offset),
                raw_text=text or " ",
                annotation="No risk-of-bias relevant content.",
                domain_tags=ALL_DOMAIN_TAGS.copy(),
                char_count=len(text),
            )
        ]

    segments: list[SupplementSegment] = []
    for idx, (heading, start) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(full_text)
        raw_text = full_text[start:end].strip()
        if not raw_text:
            continue
        segment_index = len(segments)
        segments.append(
            SupplementSegment(
                segment_id=f"{source_file.name}__{_slug(heading)}__{window_index}_{segment_index}",
                source_file=str(source_file),
                doc_type=doc_type,
                heading=heading,
                pages=_pages_for_range(start, end, window.page_starts, window.page_offset),
                raw_text=raw_text,
                annotation="No risk-of-bias relevant content.",
                domain_tags=_domain_tags(heading, raw_text, settings),
                char_count=len(raw_text),
            )
        )
    return segments


def _header_offsets(window: ParsedSupplementWindow) -> list[tuple[str, int]]:
    headers: list[tuple[str, int]] = []
    for box in window.page_boxes:
        if box.boxclass != "section-header":
            continue
        local_page = box.page - window.page_offset
        page_start = window.page_starts[local_page] if 0 <= local_page < len(window.page_starts) else 0
        relative_page_text = window.full_text[page_start:]
        found_at = relative_page_text.find(box.text)
        offset = page_start + max(found_at, 0)
        headers.append((normalize_heading(box.text), offset))
    return sorted(set(headers), key=lambda item: item[1])


def _domain_tags(heading: str, text: str, settings: EnvSettings) -> list[str]:
    haystack = f"{heading}\n{text[: settings.domain_tag_scan_chars]}".lower()
    return [
        domain
        for domain, keywords in SECTION_KEYWORDS.items()
        if domain.startswith("D") and any(keyword in haystack for keyword in keywords)
    ]


def _pages_for_range(start: int, end: int, page_starts: list[int], page_offset: int) -> list[int]:
    pages = [page_offset + idx for idx, page_start in enumerate(page_starts) if start <= page_start < end]
    start_page = max((idx for idx, page_start in enumerate(page_starts) if page_start <= start), default=0)
    return sorted(set([page_offset + start_page, *pages]))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "SEGMENT"
