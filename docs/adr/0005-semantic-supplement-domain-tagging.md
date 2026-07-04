# ADR 0005: Semantic Supplement Domain Tagging

## Status

Superseded by [ADR 0006](0006-docling-founded-ingestion-and-retrieval.md)

## Context

Supplement segments previously received RoB 2 domain tags from keyword lexicons and keyword counts. After retrieval changed to use those tags only as a soft ranking boost, keyword tags could no longer exclude evidence, but they still failed to prioritize paraphrased domain-relevant supplement passages.

ARBITER already has an optional dense embedding backend for supplement retrieval with asymmetric document/query encoding and a persistent content-hash cache.

## Decision

Supplement domain tagging uses semantic similarity between each segment and one prototype description per RoB 2 domain. Segment text is encoded as a document, domain prototypes are encoded as queries, and the highest-scoring domains become the segment's soft tags.

When the dense backend is unavailable or semantic scoring fails, supplement segments receive all RoB 2 domain tags as a neutral fail-open signal. Retrieval must continue to rank all candidates and must not turn domain tags back into a hard filter.

Low-yield document-type detection remains keyword-based because it answers a different, document-level suppression question and has adequate precision for disclosure and administrative materials.

## Consequences

Paraphrased supplement evidence can receive a useful domain boost without relying on exact lexicon terms. Runs without dense embeddings preserve recall and avoid reintroducing brittle keyword tagging, at the cost of losing domain-specific boost quality for that run.
