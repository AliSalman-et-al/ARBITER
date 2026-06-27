"""Supplementary-material ingestion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from arbiter.config import EnvSettings
from arbiter.ingestion.paper import (
    _extract_lines,
    _normalize_markdown_text,
)
from arbiter.llm.base import LLMClient
from arbiter.models import (
    AnnotationStatus,
    NO_RISK_OF_BIAS_ANNOTATION,
    PageBox,
    SupplementSegment,
)
from arbiter.retrieval.annotator import (
    annotate_segment,
    choose_segments_for_annotation,
    document_preamble,
)
from arbiter.retrieval.segmenter import (
    ParsedSupplementWindow,
    detect_document_type,
    segment_document,
)
from arbiter.retrieval.supplement_index import SupplementIndex

STRUCTURED_CONTENT_PATTERN = re.compile(
    r"\b(figure|fig\.?|diagram|consort|table|missing|lost to follow-up|withdrew|withdrawal|randomi[sz]ed)\b",
    re.IGNORECASE,
)
TABLE_CONTENT_PATTERN = re.compile(r"\b(table|col\d+|missing|total)\b|[|]\s*---", re.IGNORECASE)
SPARSE_MARKDOWN_RATIO = 0.6


async def ingest_supplements(
    paths: list[Path], aux_client: LLMClient
) -> SupplementIndex:
    """Parse, annotate, and index supplementary PDFs.

    Directories are expanded non-recursively to ``*.pdf`` files.
    """

    settings = EnvSettings()
    supplement_paths = _expand_supplement_paths(paths)
    segments: list[SupplementSegment] = []
    for path in supplement_paths:
        document_segments = await _ingest_one_supplement(path, aux_client, settings)
        segments.extend(document_segments)
    return SupplementIndex(segments, settings=settings)


def _expand_supplement_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pdf")))
        else:
            expanded.append(path)
    return expanded


async def _ingest_one_supplement(
    path: Path,
    aux_client: LLMClient,
    settings: EnvSettings,
) -> list[SupplementSegment]:
    windows = _parse_pdf_windows(path, settings)
    page_boxes = [box for window in windows for box in window.page_boxes]
    doc_type = detect_document_type(page_boxes, settings=settings).doc_type
    segments = segment_document(path, windows, doc_type=doc_type, settings=settings)
    if not segments:
        return []

    full_text = "\n".join(window.full_text for window in windows)
    preamble = document_preamble(full_text, settings=settings)
    selected_ids = choose_segments_for_annotation(segments, settings=settings)
    annotated: list[SupplementSegment] = []
    for segment in segments:
        if segment.segment_id not in selected_ids:
            annotated.append(segment)
            continue
        try:
            annotation = await annotate_segment(
                segment,
                document_preamble=preamble,
                aux_client=aux_client,
                settings=settings,
            )
        except Exception as exc:
            annotated.append(
                segment.model_copy(
                    update={
                        "annotation": NO_RISK_OF_BIAS_ANNOTATION,
                        "annotation_status": AnnotationStatus.FAILED,
                        "annotation_error": str(exc),
                    }
                )
            )
            continue

        status = (
            AnnotationStatus.SUCCEEDED_EMPTY
            if annotation == NO_RISK_OF_BIAS_ANNOTATION
            else AnnotationStatus.SUCCEEDED_SUBSTANTIVE
        )
        annotated.append(
            segment.model_copy(
                update={
                    "annotation": annotation,
                    "annotation_status": status,
                    "annotation_error": None,
                }
            )
        )
    return annotated


def _parse_pdf_windows(
    path: Path, settings: EnvSettings
) -> list[ParsedSupplementWindow]:
    try:
        doc = pymupdf.open(path)
    except Exception:
        return [
            ParsedSupplementWindow(
                full_text="",
                page_starts=[],
                page_boxes=[],
                page_offset=0,
            )
        ]

    windows: list[ParsedSupplementWindow] = []
    try:
        markdown_chunks = _extract_supplement_markdown_chunks(path, len(doc))
        window_size = max(1, settings.supplement_parse_window)
        for start in range(0, len(doc), window_size):
            end = min(start + window_size, len(doc))
            windows.append(_parse_pdf_window(doc, start, end, markdown_chunks=markdown_chunks))
    finally:
        doc.close()
    return windows


def _extract_supplement_markdown_chunks(path: Path, page_count: int) -> list[dict] | None:
    try:
        pymupdf4llm.use_layout(False)
        headers = pymupdf4llm.IdentifyHeaders(str(path), max_levels=3)
        chunks = pymupdf4llm.to_markdown(
            str(path),
            hdr_info=headers,
            page_chunks=True,
            page_separators=False,
            margins=(0, 54, 0, 54),
            table_strategy="lines_strict",
            ignore_images=True,
        )
    except Exception:
        return None
    if not isinstance(chunks, list) or len(chunks) != page_count:
        return None
    return chunks


def _parse_pdf_window(
    doc: Any,
    start: int,
    end: int,
    *,
    markdown_chunks: list[dict] | None = None,
) -> ParsedSupplementWindow:
    page_texts: list[str] = []
    page_starts: list[int] = []
    page_boxes: list[PageBox] = []
    running_offset = 0

    for page_index in range(start, end):
        page_starts.append(running_offset)
        try:
            page = doc.load_page(page_index)
            page_text = _page_text(page, page_index, markdown_chunks)
            page_texts.append(page_text)
            headings = _markdown_heading_lines(page_text)
            for heading in headings:
                page_boxes.append(
                    PageBox(
                        boxclass="section-header",
                        text=heading,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        page=page_index,
                    )
                )
            for line in _extract_lines(page, page_index):
                page_boxes.append(
                    PageBox(
                        boxclass="text",
                        text=line.text,
                        bbox=line.bbox,
                        page=page_index,
                    )
                )
        except Exception:
            page_text = ""
            page_texts.append(page_text)
            page_boxes.append(
                PageBox(
                    boxclass="degraded-page",
                    text="",
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    page=page_index,
                )
            )
        running_offset += len(page_text)
        if page_index < end - 1:
            running_offset += 1

    return ParsedSupplementWindow(
        full_text="\n".join(page_texts),
        page_starts=page_starts,
        page_boxes=page_boxes,
        page_offset=start,
    )


def _page_text(page: Any, page_index: int, markdown_chunks: list[dict] | None) -> str:
    raw_text = _normalize_plain_text(page.get_text())
    if markdown_chunks is None:
        return raw_text

    markdown_text = _normalize_markdown_text(markdown_chunks[page_index]["text"])
    if _looks_table_like(markdown_text, raw_text):
        table_text = _extract_table_markdown(page)
        if table_text and table_text not in markdown_text:
            markdown_text = _join_text_parts(markdown_text, table_text)

    if _needs_spatial_text_fallback(markdown_text, raw_text):
        fallback = f"Spatial text fallback:\n{raw_text}"
        return _join_text_parts(markdown_text, fallback)
    return markdown_text


def _normalize_plain_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    normalized_lines = [line for line in lines if line]
    return "\n".join(normalized_lines).strip()


def _extract_table_markdown(page: Any) -> str:
    try:
        tables = page.find_tables(strategy="lines_strict")
    except TypeError:
        try:
            tables = page.find_tables()
        except Exception:
            return ""
    except Exception:
        return ""

    markdown_tables: list[str] = []
    for table in getattr(tables, "tables", []):
        try:
            table_markdown = table.to_markdown().strip()
        except Exception:
            try:
                rows = table.extract()
            except Exception:
                continue
            table_markdown = _rows_to_markdown(rows)
        if table_markdown:
            markdown_tables.append(table_markdown)
    return "\n\n".join(markdown_tables)


def _looks_table_like(markdown_text: str, raw_text: str) -> bool:
    if "|" in markdown_text and "---" in markdown_text:
        return False
    return bool(TABLE_CONTENT_PATTERN.search(f"{markdown_text}\n{raw_text}"))


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    clean_rows = [
        ["" if cell is None else " ".join(str(cell).split()) for cell in row]
        for row in rows
    ]
    clean_rows = [row for row in clean_rows if any(cell for cell in row)]
    if not clean_rows:
        return ""
    width = max(len(row) for row in clean_rows)
    padded_rows = [row + [""] * (width - len(row)) for row in clean_rows]
    header, *body = padded_rows
    separator = ["---"] * width
    return "\n".join(_markdown_row(row) for row in [header, separator, *body])


def _markdown_row(row: list[str]) -> str:
    return "|" + "|".join(cell.replace("|", "\\|") for cell in row) + "|"


def _needs_spatial_text_fallback(markdown_text: str, raw_text: str) -> bool:
    if not raw_text:
        return False
    if not STRUCTURED_CONTENT_PATTERN.search(raw_text):
        return False
    if not markdown_text:
        return True
    markdown_words = set(markdown_text.lower().split())
    missing_structured_terms = [
        term
        for term in ("randomized", "randomised", "follow-up", "withdrew", "withdrawal")
        if term in raw_text.lower() and term not in markdown_text.lower()
    ]
    if missing_structured_terms:
        return True
    raw_word_count = len(raw_text.split())
    markdown_word_count = len(markdown_words)
    return markdown_word_count < raw_word_count * SPARSE_MARKDOWN_RATIO


def _join_text_parts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _markdown_heading_lines(page_text: str) -> list[str]:
    headings: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip().strip("*_").strip()
        heading = " ".join(heading.replace("**", " ").replace("__", " ").split())
        if heading:
            headings.append(heading)
    return headings
