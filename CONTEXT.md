# Context

## Confidence Flag

A confidence flag is advisory metadata on a signaling-question answer. It helps reviewers spot weak retrieval or quote-verification cases, but it is not a RoB 2 answer code and never changes deterministic domain or overall judgments.

## ClinicalTrials.gov Record

A ClinicalTrials.gov record is the verbatim v2 registry JSON for a single NCT-numbered study. It is structured source evidence for downstream context assembly and metadata checks, not a normalized ARBITER model.

## Study Design

A study design is ARBITER's classification of the trial structure. Only an individually randomised parallel-group RCT is inside the v0.1 RoB 2-IRPG assessment scope; other designs are metadata for deterministic eligibility handling.
