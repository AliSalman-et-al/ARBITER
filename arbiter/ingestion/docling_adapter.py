"""Docling integration helpers for PDF ingestion and chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, cast

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from arbiter.config import EnvSettings
from arbiter.models import PageBox


SOFT_HYPHEN_PATTERN = re.compile(r"\xad\s*\n\s*")
MARKDOWN_HEADING_PREFIX = re.compile(r"^\s{0,3}#{1,6}\s+")
FURNITURE_LABELS = {"page_header", "page_footer"}


def build_docling_converter(settings: EnvSettings | None = None) -> DocumentConverter:
    """Build the tuned Docling PDF converter used by ARBITER ingestion."""

    settings = settings or EnvSettings()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = settings.docling_do_ocr
    pipeline_options.do_table_structure = True
    table_options = cast(Any, pipeline_options.table_structure_options)
    table_options.mode = TableFormerMode.FAST
    table_options.do_cell_matching = True
    if settings.docling_artifacts_path is not None:
        pipeline_options.artifacts_path = settings.docling_artifacts_path

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def build_hybrid_chunker(settings: EnvSettings | None = None) -> HybridChunker:
    """Build Docling's structure-preserving HybridChunker.

    When a dense embedding model is configured, reuse that model's tokenizer so
    chunk size tracks the retrieval backend's token budget. Without a configured
    model, Docling's default tokenizer keeps unit tests and offline runs light.
    """

    settings = settings or EnvSettings()
    if settings.dense_embedding_model:
        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )

            tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name=settings.dense_embedding_model,
                max_tokens=settings.docling_chunk_max_tokens,
            )
            return HybridChunker(tokenizer=tokenizer, merge_peers=True)
        except Exception:
            return HybridChunker(merge_peers=True)
    return HybridChunker(merge_peers=True)


def convert_pdf(path: Path, settings: EnvSettings | None = None) -> Any:
    """Convert a PDF into a DoclingDocument."""

    converter = build_docling_converter(settings)
    return converter.convert(path, raises_on_error=True).document


def load_docling_chunks(path: Path, settings: EnvSettings | None = None) -> list[Document]:
    """Load Docling HybridChunker chunks through langchain-docling."""

    settings = settings or EnvSettings()
    loader = DoclingLoader(
        file_path=str(path),
        converter=build_docling_converter(settings),
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_hybrid_chunker(settings),
    )
    return list(loader.load())


def docling_markdown_by_page(document: Any) -> tuple[list[str], list[int]]:
    page_texts: list[str] = []
    page_starts: list[int] = []
    running_offset = 0
    for one_based_page in range(1, _num_pages(document) + 1):
        page_starts.append(running_offset)
        page_text = normalize_docling_text(
            document.export_to_markdown(
                page_no=one_based_page,
                image_placeholder="",
                page_break_placeholder=None,
            )
        )
        page_texts.append(page_text)
        running_offset += len(page_text)
        if one_based_page < _num_pages(document):
            running_offset += 1
    return page_texts, page_starts


def docling_page_boxes(document: Any) -> list[PageBox]:
    boxes: list[PageBox] = []
    for item, _level in document.iterate_items():
        text = _item_text(item)
        if not text:
            continue
        label = _label_value(getattr(item, "label", "text"))
        if label in FURNITURE_LABELS:
            continue
        prov = _first_prov(item)
        page = max(0, int(getattr(prov, "page_no", 1)) - 1) if prov is not None else 0
        boxes.append(
            PageBox(
                boxclass="section-header" if label == "section_header" else label or "text",
                text=text,
                bbox=_bbox_tuple(getattr(prov, "bbox", None)),
                page=page,
            )
        )
    return boxes


def normalize_docling_text(text: str) -> str:
    text = SOFT_HYPHEN_PATTERN.sub("", text).replace("\xad", "")
    lines = [
        line.rstrip()
        for line in text.replace("\r", "\n").splitlines()
        if not _is_markdown_furniture_line(line)
    ]
    return "\n".join(lines).strip()


def markdown_heading_lines(page_text: str) -> list[str]:
    headings: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = MARKDOWN_HEADING_PREFIX.sub("", stripped)
        heading = heading.strip().strip("*_").strip()
        if heading:
            headings.append(heading)
    return headings


def chunk_headings(metadata: dict[str, Any]) -> list[str]:
    dl_meta = _dl_meta(metadata)
    headings = dl_meta.get("headings")
    if isinstance(headings, list):
        return [str(heading).strip() for heading in headings if str(heading).strip()]
    return []


def chunk_doc_item_labels(metadata: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for item in _doc_items(metadata):
        label = item.get("label") if isinstance(item, dict) else None
        value = _label_value(label)
        if value:
            labels.append(value)
    return sorted(set(labels))


def chunk_pages(metadata: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for item in _doc_items(metadata):
        if not isinstance(item, dict):
            continue
        for prov in _provenance_items(item.get("prov")):
            page_no = prov.get("page_no")
            if isinstance(page_no, int):
                pages.add(max(0, page_no - 1))
    return sorted(pages)


def _num_pages(document: Any) -> int:
    num_pages = getattr(document, "num_pages", None)
    if callable(num_pages):
        return int(num_pages())
    if isinstance(num_pages, int):
        return num_pages
    pages = getattr(document, "pages", None)
    if isinstance(pages, dict | list | tuple):
        return len(pages)
    return 1


def _dl_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("dl_meta", {})
    return value if isinstance(value, dict) else {}


def _doc_items(metadata: dict[str, Any]) -> Iterable[Any]:
    items = _dl_meta(metadata).get("doc_items", [])
    return items if isinstance(items, list) else []


def _provenance_items(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def _first_prov(item: Any) -> Any | None:
    prov = getattr(item, "prov", None)
    return prov[0] if prov else None


def _item_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return normalize_docling_text(text)
    caption = getattr(item, "caption", None)
    if isinstance(caption, str) and caption.strip():
        return normalize_docling_text(caption)
    return ""


def _bbox_tuple(bbox: Any) -> tuple[float, float, float, float]:
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    values = []
    for name in ("l", "t", "r", "b"):
        value = getattr(bbox, name, None)
        if value is None:
            break
        values.append(float(value))
    if len(values) == 4:
        return (values[0], values[1], values[2], values[3])
    if isinstance(bbox, dict):
        keys = ("l", "t", "r", "b")
        if all(key in bbox for key in keys):
            return (
                float(bbox["l"]),
                float(bbox["t"]),
                float(bbox["r"]),
                float(bbox["b"]),
            )
    return (0.0, 0.0, 0.0, 0.0)


def _label_value(label: Any) -> str:
    value = getattr(label, "value", label)
    return str(value or "").strip().lower()


def _is_markdown_furniture_line(line: str) -> bool:
    normalized = line.strip().strip("_*").strip().lower()
    return (
        normalized.startswith("copyright ")
        or normalized.startswith("(c) ")
        or normalized.startswith("© ")
        or normalized.startswith("downloaded from ")
        or "new england journal of medicine" in normalized
    )
