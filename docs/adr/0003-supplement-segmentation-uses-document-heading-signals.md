# ADR 0003: Supplement segmentation uses document heading signals

## Status

Accepted

## Context

Supplement ingestion previously treated bold, large, capitalized, or isolated lines as section headers on a per-page basis. Protocol appendices and disclosure forms often contain bold form labels, page routing fields, dates, and repeated running headers. Promoting those lines to section headers caused over-fragmentation and exposed non-semantic headings to retrieval and signaling-question prompts.

The alternative was to keep the local heuristic and add a large denylist. That would be brittle because each journal, trial group, and form template can introduce different page furniture.

## Decision

Supplement PDF parsing uses document-wide `pymupdf4llm.IdentifyHeaders` Markdown heading detection when available. Segment creation then applies generic guardrails:

- Tiny form-like headings and parser window placeholders are merged into adjacent content.
- Repeated all-caps headings across a document are treated as running furniture.
- Disclosure and administrative supplements collapse to a single low-yield segment.
- Documents with too many candidate segments fall back to coarse neutral document parts instead of indexing fabricated headings.

## Consequences

Retrieval sees fewer, larger, and more coherent supplement segments. When a PDF has no trustworthy section hierarchy, ARBITER preserves retrievable chunks with neutral part headings rather than pretending that page furniture is a section title.

The trade-off is that some genuine short sections may be merged into neighboring content. This is preferable to allowing form labels or repeated page headers to dominate the candidate pool, and the thresholds remain configurable.
