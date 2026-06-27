"""Supplementary-material ingestion."""

from __future__ import annotations

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
    if markdown_chunks is None:
        return page.get_text()
    return _normalize_markdown_text(markdown_chunks[page_index]["text"])


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
