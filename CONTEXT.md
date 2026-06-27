# Context

## Assessment Runtime

An assessment runtime is the per-run bundle of non-serializable handles used by assessment graph nodes, including LLM clients, the supplement index, and optional tracing. It is supplied through LangGraph runtime context, not stored in graph state.

## Trial Context

A trial context is the once-per-trial ingestion bundle consumed by assessment orchestration. It contains serializable trial data plus runtime handles, letting eligibility and assessment reuse the same ingestion output without re-parsing or rebuilding the shared prefix.

## Batch Manifest

A batch manifest is the reviewer-authored work list for unattended ARBITER runs. Each entry names one main paper plus optional supplements, NCT number, outcome list, and trial label; enumerated outcomes define the resume key set before ingestion.

## Two-Tier Assessment Graph

A two-tier assessment graph is ARBITER's split orchestration model: the trial tier judges D1 once per trial, and the outcome tier reuses that D1 while judging D2-D5 separately for each assessed outcome.

## Confidence Flag

A confidence flag is reliability metadata on a signaling-question answer. It helps reviewers spot weak retrieval or quote-verification cases and can route an assessment to human review, but it is not a RoB 2 answer code. Confidence flags do not directly change deterministic domain or overall judgments; quote verification itself can finalize an unsupported substantive answer as `NI` before judgment.

## ClinicalTrials.gov Record

A ClinicalTrials.gov record is the verbatim v2 registry JSON for a single NCT-numbered study. It is structured source evidence for downstream context assembly and metadata checks, not a normalized ARBITER model.

## Supplement Annotation

A supplement annotation is an auxiliary LLM-authored enrichment attached to a supplementary-material segment. Its purpose is to improve final signaling-question answer accuracy by making risk-of-bias evidence easier to retrieve; it is not valuable merely because it summarizes a segment.

## Supplement Annotation Budget

A supplement annotation budget is the per-document limit on auxiliary annotation work for supplementary material. It should preserve recall for plausible risk-of-bias evidence while preventing exhaustive annotation of low-yield or administrative documents.

## Low-Yield Supplement

A low-yield supplement is a supplementary document whose detected purpose makes risk-of-bias evidence unlikely, such as conflict-of-interest disclosures, copyright notices, licences, or administrative forms. It remains searchable as raw text but should not normally receive auxiliary LLM annotation.

## Supplement Annotation Status

A supplement annotation status records whether auxiliary annotation was not run, succeeded with substantive content, succeeded with no relevant content, or failed. It keeps true absence of risk-of-bias evidence distinct from missing enrichment.

## Supplement Segment

A supplement segment is a retrieval unit cut from supplementary material. It should represent a real document section when reliable section structure is available; otherwise it may be a neutral coarse document part rather than a fabricated heading from page furniture or form fields.

## Candidate-First Enrichment

Candidate-first enrichment is the supplement-processing strategy where cheap raw-text retrieval identifies likely useful segments before auxiliary LLM annotation is spent. It contrasts with blanket pre-enrichment of all domain-tagged supplement segments.

## Supplement Annotation Ablation

A supplement annotation ablation is an evaluation comparison that measures whether auxiliary annotation improves final signaling-question answers, quote support, or `NI` rates relative to raw hybrid retrieval. Annotation yield alone is not sufficient evidence that annotation is worth keeping.

## Outcome Comparison

An outcome comparison is the deterministic pre-D5 match between an assessed outcome and the registered ClinicalTrials.gov outcome set. It is evidence for D5 context assembly, not a risk-of-bias judgment.

## Signaling-Question Raw Answer

A signaling-question raw answer is the validated LLM output for one signaling question before deterministic post-processing. It can contain only substantive answer codes or `NI`; structural `NA` is outside the raw answer and belongs to branching.

## Signaling-Question Answer

A signaling-question answer is the finalized answer record consumed by deterministic RoB 2 branching and judgment logic. It combines the answer code with verified quote evidence, deterministic page location, and advisory confidence metadata.

## Signaling-Question Worker

A signaling-question worker is the assessment node that processes exactly one signaling question. It does not decide question ordering or domain judgments; those remain deterministic graph and algorithm responsibilities.

## Study Design

A study design is ARBITER's classification of the trial structure. Only an individually randomised parallel-group RCT is inside the v0.1 RoB 2-IRPG assessment scope; other designs are metadata for deterministic eligibility handling.

## Skip Record

A skip record is the audit artifact for an input trial that is outside ARBITER's v0.1 assessment scope. It records why no RoB 2 assessment was produced and uses sentinel trial-level output keys instead of fabricating nullable domain judgments.

## Reviewer-Facing Report

A reviewer-facing report is the Markdown audit artifact rendered from an existing assessment. It presents deterministic judgments and advisory signaling-question evidence for human inspection, but it does not introduce new extraction, new LLM reasoning, or a review gate.

## QA Trace Bundle

A QA trace bundle is the read-only per-run observability artifact for live pipeline testing. It is written incrementally while a run executes and exposes pipeline inputs, outputs, intermediate artifacts, and full raw LLM prompt and response bodies without becoming part of the deterministic assessment record.
