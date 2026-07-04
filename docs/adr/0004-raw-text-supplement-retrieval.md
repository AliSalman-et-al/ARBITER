# ADR 0004: Raw-text supplement retrieval

## Status

Superseded by [ADR 0006](0006-docling-founded-ingestion-and-retrieval.md)

## Context

ARBITER previously enriched selected supplement segments with an auxiliary LLM annotation before indexing. A CHAARTED / Overall Survival QA run showed the annotation layer consumed meaningful latency and LLM budget while not changing the answers that mattered; the observed over-calls came from main-paper reasoning rather than supplement retrieval. The annotation call also carried the same structured-output fragility as other native-schema auxiliary calls.

Supplementary material is still valuable evidence. The useful boundary is segmentation plus hybrid retrieval over the source text, with low-yield disclosure and administrative documents suppressed from reviewer-facing evidence when they are the only apparent match.

## Decision

Remove the LLM supplement-annotation layer. Supplement ingestion parses, classifies, segments, domain-tags, and indexes each supplement segment's raw text directly. Retrieval continues to use sparse and optional dense signals with reciprocal-rank fusion. Context assembly and QA trace artifacts render the same raw segment text that quote verification can later match.

Low-yield supplement suppression remains part of retrieval selection. Disclosure and administrative segments may remain in the index for observability, but they are not selected as evidence when no high-yield relevant segment is available.

## Consequences

Supplement ingestion no longer makes auxiliary LLM calls for segment annotation, so supplement-heavy runs spend less latency and budget before assessment begins. The pipeline loses a vocabulary-bridging enrichment layer, so future retrieval-recall work should improve the raw-text path directly rather than reintroducing pre-enrichment.

The domain model no longer includes supplement annotation budgets, annotation status, candidate-first enrichment, or supplement annotation ablations. Any evaluation comparing annotated and raw retrieval is historical context, not a live gate.
