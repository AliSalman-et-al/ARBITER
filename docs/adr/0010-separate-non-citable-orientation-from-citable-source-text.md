# Separate non-citable orientation from citable source text

ARBITER separates derived trial orientation from citable source text in signaling-question prompts. Derived trial metadata may orient the worker, but it is not source evidence and must not be copied into the quote field; citable source text remains the main paper, supplements, and ClinicalTrials.gov record.

ClinicalTrials.gov stays citable and cacheable as part of the shared source prefix. If a substantive answer quotes non-citable orientation, finalization should treat that as a distinct non-citable orientation quote, attempt a constrained orientation quote repair against the full citable source surface, and show the repaired verified quote in reviewer-facing output while preserving the original non-citable quote in QA trace.

This trades a simpler mixed prompt prefix for a stricter provenance boundary. The boundary prevents ARBITER's own summaries from becoming apparent evidence while preserving useful derived context and reliable registry fields such as `Masking: NONE`.
