# Context

## Confidence Flag

A confidence flag is advisory metadata on a signaling-question answer. It helps reviewers spot weak retrieval or quote-verification cases, but it is not a RoB 2 answer code and never changes deterministic domain or overall judgments.

## ClinicalTrials.gov Record

A ClinicalTrials.gov record is the verbatim v2 registry JSON for a single NCT-numbered study. It is structured source evidence for downstream context assembly and metadata checks, not a normalized ARBITER model.

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
