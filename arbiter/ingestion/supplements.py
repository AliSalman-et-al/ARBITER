"""Supplementary-material ingestion founded on Docling HybridChunker chunks."""

from __future__ import annotations

import re
from pathlib import Path

from arbiter.config import EnvSettings
from arbiter.ingestion.docling_adapter import (
    chunk_doc_item_labels,
    chunk_headings,
    chunk_pages,
    load_docling_chunks,
)
from arbiter.ingestion.paper import ALL_DOMAIN_TAGS
from arbiter.llm.base import LLMClient
from arbiter.models import DocType, SupplementSegment
from arbiter.retrieval.supplement_index import (
    DenseEmbeddingBackend,
    SupplementIndex,
    sentence_transformer_backend,
)


DOC_TYPE_LEXICONS: dict[DocType, tuple[str, ...]] = {
    DocType.DISCLOSURE: (
        "conflict of interest",
        "conflicts of interest",
        "disclose",
        "disclosure",
        "disclosures",
        "disclosure statement",
        "financial disclosure",
        "author disclosure",
        "competing interests",
        "declaration of interests",
        "consulting fees",
        "institutional grants",
    ),
    DocType.ADMINISTRATIVE: (
        "copyright",
        "licence",
        "license",
        "creative commons",
        "reuse permissions",
        "publisher",
        "administrative",
        "regulatory",
        "monitoring",
        "hipaa",
        "audit",
    ),
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
        "protocol",
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


async def ingest_supplements(
    paths: list[Path], aux_client: LLMClient
) -> SupplementIndex:
    """Parse and index supplementary PDFs.

    Directories are expanded non-recursively to ``*.pdf`` files. The auxiliary
    client is accepted for compatibility but no LLM annotation is performed.
    """

    _ = aux_client
    settings = EnvSettings()
    dense_backend = _dense_backend(settings)
    segments: list[SupplementSegment] = []
    for path in _expand_supplement_paths(paths):
        segments.extend(_ingest_one_supplement(path, settings))
    return SupplementIndex(segments, settings=settings, dense_backend=dense_backend)


def _expand_supplement_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pdf")))
        else:
            expanded.append(path)
    return expanded


def _ingest_one_supplement(
    path: Path, settings: EnvSettings
) -> list[SupplementSegment]:
    try:
        chunks = load_docling_chunks(path, settings)
    except Exception:
        return []
    if not chunks:
        return []

    document_text = "\n".join(chunk.page_content for chunk in chunks)
    doc_type = detect_document_type(path, document_text)
    segments: list[SupplementSegment] = []
    for index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue
        headings = chunk_headings(chunk.metadata)
        labels = chunk_doc_item_labels(chunk.metadata)
        pages = chunk_pages(chunk.metadata)
        heading = " / ".join(headings) if headings else _fallback_heading(labels)
        segments.append(
            SupplementSegment(
                segment_id=f"{path.name}__docling_chunk_{index}",
                source_file=str(path),
                doc_type=doc_type,
                heading=heading,
                pages=pages,
                raw_text=text,
                domain_tags=ALL_DOMAIN_TAGS.copy(),
                doc_item_labels=labels,
                metadata={
                    "docling": chunk.metadata.get("dl_meta", {}),
                    "embedding_text": text,
                },
                char_count=len(text),
            )
        )
    return segments


def detect_document_type(source_file: Path, text: str) -> DocType:
    evidence = f"{re.sub(r'[-_.]+', ' ', source_file.stem)}\n{text}".lower()
    if not evidence.strip():
        return DocType.UNKNOWN
    scores = {
        doc_type: sum(evidence.count(term) for term in lexicon)
        for doc_type, lexicon in DOC_TYPE_LEXICONS.items()
    }
    best_score = max(scores.values())
    if best_score <= 0:
        return DocType.UNKNOWN
    winners = [doc_type for doc_type, score in scores.items() if score == best_score]
    return _break_doc_type_tie(winners)


def _break_doc_type_tie(winners: list[DocType]) -> DocType:
    for doc_type in (
        DocType.DISCLOSURE,
        DocType.ADMINISTRATIVE,
        DocType.SAP,
        DocType.PROTOCOL,
        DocType.APPENDIX,
    ):
        if doc_type in winners:
            return doc_type
    return winners[0]


def _fallback_heading(labels: list[str]) -> str:
    if "table" in labels:
        return "TABLE"
    if "section_header" in labels:
        return "SECTION"
    return "DOC_CHUNK"


def _dense_backend(settings: EnvSettings) -> DenseEmbeddingBackend | None:
    if settings.dense_embedding_model is None:
        return None
    try:
        return sentence_transformer_backend(
            settings.dense_embedding_model,
            settings.dense_embedding_cache_path,
        )
    except Exception:
        return None
