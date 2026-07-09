"""Docling integration helpers for PDF ingestion and chunking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, TypeAlias, cast

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from langchain_core.documents import Document

from arbiter.config import EnvSettings
from arbiter.models import PageBox


SOFT_HYPHEN_PATTERN = re.compile(r"\xad\s*\n\s*")
MARKDOWN_HEADING_PREFIX = re.compile(r"^\s{0,3}#{1,6}\s+")
FURNITURE_LABELS = {"page_header", "page_footer"}
DoclingBackendClass: TypeAlias = type[AbstractDocumentBackend]
DOCLING_PDF_BACKENDS: dict[str, DoclingBackendClass] = {
    "docling-parse-v1": DoclingParseDocumentBackend,
    "docling-parse-v2": DoclingParseV2DocumentBackend,
    "docling-parse-v4": DoclingParseV4DocumentBackend,
    "pypdfium2": PyPdfiumDocumentBackend,
}


class _MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialize Docling table chunks as compact Markdown for quoting and embedding."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(
                compact_tables=True,
                image_placeholder="",
                page_break_placeholder=None,
            ),
        )


def build_docling_converter(
    settings: EnvSettings | None = None, *, do_table_structure: bool = True
) -> DocumentConverter:
    """Build the tuned Docling PDF converter used by ARBITER ingestion."""

    settings = settings or EnvSettings()
    os.environ["OMP_NUM_THREADS"] = str(settings.docling_num_threads)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=settings.docling_num_threads,
        device=AcceleratorDevice.AUTO,
    )
    pipeline_options.do_ocr = settings.docling_do_ocr
    pipeline_options.do_table_structure = do_table_structure
    if do_table_structure:
        table_options = cast(Any, pipeline_options.table_structure_options)
        table_options.mode = TableFormerMode.FAST
        table_options.do_cell_matching = True
    if settings.docling_artifacts_path is not None:
        pipeline_options.artifacts_path = settings.docling_artifacts_path

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=_docling_pdf_backend(settings.docling_backend),
            )
        }
    )
    setattr(converter, "_arbiter_do_table_structure", do_table_structure)
    return converter


def _docling_pdf_backend(name: str) -> DoclingBackendClass:
    normalized = name.strip().lower()
    try:
        return DOCLING_PDF_BACKENDS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DOCLING_PDF_BACKENDS))
        raise ValueError(
            f"Unsupported ARBITER_DOCLING_BACKEND {name!r}. "
            f"Supported values: {supported}."
        ) from exc


def build_hybrid_chunker(settings: EnvSettings | None = None) -> HybridChunker:
    """Build Docling's structure-preserving HybridChunker.

    When a dense embedding model is configured, reuse that model's tokenizer so
    chunk size tracks the retrieval backend's token budget. Without a configured
    model, use a local tiktoken tokenizer so tests and offline runs do not reach
    Hugging Face for Docling's default tokenizer.
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
            return HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
                repeat_table_header=True,
                serializer_provider=_MarkdownTableSerializerProvider(),
            )
        except Exception:
            return HybridChunker(
                tokenizer=_local_tokenizer(settings),
                merge_peers=True,
                repeat_table_header=True,
                serializer_provider=_MarkdownTableSerializerProvider(),
            )
    return HybridChunker(
        tokenizer=_local_tokenizer(settings),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=_MarkdownTableSerializerProvider(),
    )


def convert_pdf(
    path: Path,
    settings: EnvSettings | None = None,
    *,
    converter: DocumentConverter | None = None,
    force_refresh_cache: bool = False,
    do_table_structure: bool = True,
) -> Any:
    """Convert a PDF into a DoclingDocument."""

    settings = settings or EnvSettings()
    effective_do_table_structure = _effective_table_structure(
        converter, do_table_structure
    )
    if not force_refresh_cache:
        cached = _read_cached_document(
            path, settings, do_table_structure=effective_do_table_structure
        )
        if cached is not None:
            return cached

    converter = converter or build_docling_converter(
        settings, do_table_structure=do_table_structure
    )
    document = converter.convert(path, raises_on_error=True).document
    if not force_refresh_cache:
        _write_cached_document(
            path, settings, document, do_table_structure=effective_do_table_structure
        )
    return document


def load_docling_chunks(
    path: Path,
    settings: EnvSettings | None = None,
    *,
    do_table_structure: bool = True,
    converter: DocumentConverter | None = None,
    force_refresh_cache: bool = False,
) -> list[Document]:
    """Load Docling HybridChunker chunks through langchain-docling."""

    settings = settings or EnvSettings()
    document = convert_pdf(
        path,
        settings,
        converter=converter,
        force_refresh_cache=force_refresh_cache,
        do_table_structure=do_table_structure,
    )
    chunker = build_hybrid_chunker(settings)
    return [
        Document(
            page_content=chunker.contextualize(chunk=chunk),
            metadata={"source": str(path), "dl_meta": chunk.meta.export_json_dict()},
        )
        for chunk in chunker.chunk(document)
    ]


def _local_tokenizer(settings: EnvSettings) -> OpenAITokenizer:
    import tiktoken

    return OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("o200k_base"),
        max_tokens=settings.docling_chunk_max_tokens,
    )


def _effective_table_structure(
    converter: DocumentConverter | None, requested: bool
) -> bool:
    value = getattr(converter, "_arbiter_do_table_structure", requested)
    return bool(value)


def _read_cached_document(
    path: Path, settings: EnvSettings, *, do_table_structure: bool
) -> Any | None:
    if not settings.docling_parse_cache_enabled:
        return None
    cache_path = _docling_parse_cache_file(
        path, settings, do_table_structure=do_table_structure
    )
    if cache_path is None or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != 1:
            return None
        return _document_from_cache_payload(payload["document"])
    except Exception:
        return None


def _write_cached_document(
    path: Path, settings: EnvSettings, document: Any, *, do_table_structure: bool
) -> None:
    if not settings.docling_parse_cache_enabled:
        return
    cache_path = _docling_parse_cache_file(
        path, settings, do_table_structure=do_table_structure
    )
    if cache_path is None:
        return
    export_to_dict = getattr(document, "export_to_dict", None)
    if not callable(export_to_dict):
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": 1,
            "source_sha256": _file_sha256(path),
            "converter_fingerprint": _docling_converter_fingerprint(
                settings, do_table_structure=do_table_structure
            ),
            "document": export_to_dict(),
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(cache_path)
    except Exception:
        try:
            if "temp_name" in locals():
                Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass


def _document_from_cache_payload(document_payload: Any) -> DoclingDocument:
    return DoclingDocument.model_validate(document_payload)


def _docling_parse_cache_file(
    path: Path, settings: EnvSettings, *, do_table_structure: bool
) -> Path | None:
    try:
        source_hash = _file_sha256(path)
    except OSError:
        return None
    fingerprint = _docling_converter_fingerprint(
        settings, do_table_structure=do_table_structure
    )
    return settings.docling_parse_cache_path / f"{source_hash}-{fingerprint}.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _docling_converter_fingerprint(
    settings: EnvSettings, *, do_table_structure: bool
) -> str:
    payload = {
        "backend": settings.docling_backend,
        "do_ocr": settings.docling_do_ocr,
        "do_table_structure": do_table_structure,
        "num_threads": settings.docling_num_threads,
        "artifacts_path": str(settings.docling_artifacts_path)
        if settings.docling_artifacts_path is not None
        else None,
        "chunk_max_tokens": settings.docling_chunk_max_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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
                boxclass="section-header"
                if label == "section_header"
                else label or "text",
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
