# ADR 0006: Docling-founded ingestion and retrieval

## Status

Accepted

## Context

ARBITER previously parsed main papers and supplements into multiple divergent text representations: plain PyMuPDF text for quote verification, pymupdf4llm Markdown for model context, and layout dictionaries for page localization. Supplement retrieval then rebuilt structure with local heading heuristics, semantic domain tags, low-yield suppression, hand-written BM25/RRF fusion, and sentence subranking.

That architecture made quote grounding and page localization depend on reconciling representations after parsing had already discarded useful structure. It also kept table evidence, section breadcrumbs, and page provenance as heuristics rather than source metadata.

## Decision

Ingestion uses Docling as the canonical PDF representation. Main-paper section maps, raw quote-verification text, and page boxes are derived from the same Docling document. Supplement ingestion uses langchain-docling with Docling HybridChunker, so each `SupplementSegment` is created from a Docling chunk whose text is already contextualized with heading breadcrumbs and whose metadata preserves document-item labels and provenance.

Supplement retrieval remains local and CPU-friendly, but it is now founded on Docling chunks and metadata:

- Sparse scoring uses the maintained `bm25s` package instead of local BM25 code.
- Dense scoring and reranking are preserved, with the `top_score` confidence contract unchanged.
- Low-yield document types are a metadata filter during selection.
- Domain and table intent use chunk headings and `doc_item_labels`, including table boosts for D3/D5 evidence.
- The old segmenter, semantic domain tagger, fabricated headings, and RRF boost path are retired.

OCR is disabled by default for born-digital trial PDFs, table structure is enabled with TableFormer FAST mode, and a Docling artifacts path can be configured for prefetched/offline model assets.

## Consequences

The text shown to the model, verified as a quote source, and localized to pages now comes from one structured document path. Supplement tables and section breadcrumbs survive into retrieval as first-class metadata. Unit tests mock the Docling boundary to keep the suite fast; full PDF conversion and performance gates remain evaluation-suite responsibilities because Docling model initialization can require local artifacts.

This removes several legacy heuristics and narrows future retrieval work to a Docling chunk contract rather than parser-specific repair code.
