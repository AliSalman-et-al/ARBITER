# Context

## Assessment Runtime

An assessment runtime is the per-run bundle of non-serializable handles used by assessment graph nodes, including LLM clients, the supplement index, and optional tracing. It is supplied through LangGraph runtime context, not stored in graph state.

## Trial Context

A trial context is the once-per-trial ingestion bundle consumed by assessment orchestration. It contains serializable trial data plus runtime handles, letting eligibility and assessment reuse the same ingestion output without re-parsing or rebuilding the shared prefix.

## Non-Citable Trial Orientation

Non-citable trial orientation is derived trial-level context shown to a signaling-question worker to frame interpretation, but it is not source evidence and must not be copied as quote text. It can guide which citable source evidence is relevant, while quote verification remains limited to citable source text.

## Citable Source Text

Citable source text is verbatim evidence from the main paper, supplements, or ClinicalTrials.gov record that a signaling-question answer may copy into its quote field. It excludes derived ARBITER summaries, labels, and reviewer-facing orientation even when those summaries were derived from reliable sources.

## Non-Citable Orientation Quote

A non-citable orientation quote is model-supplied quote text that matches non-citable trial orientation rather than citable source text. It is a prompt-boundary failure distinct from an ordinary unsupported quote because the quoted wording came from ARBITER's derived context.

## Orientation Quote Repair

Orientation quote repair is a constrained second signaling-question LLM call that replaces a non-citable orientation quote with verbatim citable source text without changing the answer code. A successful repair is shown as the answer's quote, while the original non-citable quote remains QA trace evidence.

## Batch Manifest

A batch manifest is the reviewer-authored work list for unattended ARBITER runs. Each entry names one main paper plus optional supplements, NCT number, outcome list, and trial label; enumerated outcomes define the resume key set before ingestion.

## Two-Tier Assessment Graph

A two-tier assessment graph is ARBITER's split orchestration model: the trial tier judges D1 once per trial, and the outcome tier reuses that D1 while judging D2-D5 separately for each assessed outcome.

## Confidence Flag

A confidence flag is reliability metadata on a signaling-question answer. It helps reviewers spot weak retrieval or quote-verification cases and can route an assessment to human review, but it is not a RoB 2 answer code. Confidence flags do not directly change deterministic domain or overall judgments.

## Failure-Fallback Answer

A failure-fallback answer is a signaling-question answer emitted because the signaling-question worker could not obtain a valid model response after retries and repair attempts. It is distinct from a genuine evidence-based `NI` answer and contributes to assessment-level reliability gating.

## ClinicalTrials.gov Record

A ClinicalTrials.gov record is the verbatim v2 registry JSON for a single NCT-numbered study. It is structured source evidence for downstream context assembly and metadata checks, not a normalized ARBITER model.

## Low-Yield Supplement

A low-yield supplement is a supplementary document whose detected purpose makes risk-of-bias evidence unlikely, such as conflict-of-interest disclosures, copyright notices, licences, or administrative forms. Its raw text remains in the supplement index, but retrieval suppresses it when it is the only apparent match so it does not surface as reviewer-facing evidence.

## Supplement Segment

A supplement segment is a retrieval unit cut from supplementary material. It should represent a real document section when reliable section structure is available; otherwise it may be a neutral coarse document part rather than a fabricated heading from page furniture or form fields.

## Semantic Domain Tag

A semantic domain tag is a soft retrieval-ranking signal assigned to a supplement segment by comparing the segment text with RoB 2 domain prototype embeddings. It helps prioritize likely domain-relevant supplement evidence, but it is not a candidate filter and should fail open to neutral broad tags when semantic scoring is unavailable.

## Outcome Comparison

An outcome comparison is the deterministic pre-D5 match between an assessed outcome and the registered ClinicalTrials.gov outcome set. It is evidence for D5 context assembly, not a risk-of-bias judgment.

## Outcome Measurement Profile

An outcome measurement profile is the Domain 4 reasoning frame for how much judgement can enter outcome measurement or ascertainment. It distinguishes objective or record-based outcomes from participant-reported, clinician-assessed, adjudicated, threshold-dependent, or otherwise judgement-sensitive outcomes without pre-classifying outcomes from a fixed name list.

## Signaling-Question Raw Answer

A signaling-question raw answer is the validated LLM output for one signaling question before deterministic post-processing. It can contain only substantive answer codes or `NI`; structural `NA` is outside the raw answer and belongs to branching.

## Signaling-Question Answer

A signaling-question answer is the finalized answer record consumed by deterministic RoB 2 branching and judgment logic. It combines the answer code with verified quote evidence, deterministic page location, and advisory confidence metadata.

## Context Sufficiency

Context sufficiency is advisory metadata on a signaling-question answer that records whether the available source context appears capable of supporting a substantive answer. It helps distinguish a legitimate `NI` from a weakly grounded abstention, but it does not directly change deterministic RoB 2 answer codes.

## Faithfulness Score

A faithfulness score is advisory grounding metadata on a substantive signaling-question answer. It estimates whether the cited quote or justification is supported by the available source text and is used for confidence flagging and human-review routing, not deterministic RoB 2 branching.

## Entailment Score

An entailment score is advisory grounding metadata on a substantive signaling-question answer. It estimates whether the available source text entails the answer's cited claim and is emitted alongside the faithfulness score for confidence calibration.

## Signaling-Question Worker

A signaling-question worker is the assessment node that processes exactly one signaling question. It does not decide question ordering or domain judgments; those remain deterministic graph and algorithm responsibilities.

## Study Design

A study design is ARBITER's classification of the trial structure. Only an individually randomised parallel-group RCT is inside the v0.1 RoB 2-IRPG assessment scope; other designs are metadata for deterministic eligibility handling.

## Skip Record

A skip record is the audit artifact for an input trial that is outside ARBITER's v0.1 assessment scope. It records why no RoB 2 assessment was produced and uses sentinel trial-level output keys instead of fabricating nullable domain judgments.

## Reviewer-Facing Report

A reviewer-facing report is the Markdown audit artifact rendered from an existing assessment. It presents deterministic judgments and advisory signaling-question evidence for human inspection, but it does not introduce new extraction, new LLM reasoning, or a review gate.

## Unresolved Domain Judgment

An unresolved domain judgment is an auditable sentinel emitted when a domain's finalized signaling-question answers cannot be mapped by the deterministic RoB 2 IRPG decision table. It is not a RoB 2 risk judgment and forces human review so the pipeline can continue without treating the domain as Low, Some concerns, or High.

## Reachable Signaling-Question Answer Vector

A reachable signaling-question answer vector is a complete per-domain set of finalized signaling-question answer codes that can be emitted by the RoB 2 branching rules, including structural not-applicable answers for gated-out questions. Every reachable vector must map to a deterministic Low, Some concerns, or High domain judgment.

## QA Trace Bundle

A QA trace bundle is the read-only per-run observability artifact for live pipeline testing. It is written incrementally while a run executes and exposes pipeline inputs, outputs, intermediate artifacts, and full raw LLM prompt and response bodies without becoming part of the deterministic assessment record.

## Answer Budget

An answer budget is the output-token allowance reserved for the structured JSON object that ARBITER validates and records. It is distinct from any model-internal or provider-visible reasoning allowance.

## Reasoning Budget

A reasoning budget is the output-token allowance explicitly assigned to a reasoning-capable LLM for deliberation outside the structured answer. It must add to, not subtract from, the answer budget.

## Truncated Structured Output

A truncated structured output is a provider response that reports completion by token limit before the validated answer object is safely complete. It is a repairable provider-output failure, not a candidate answer.
