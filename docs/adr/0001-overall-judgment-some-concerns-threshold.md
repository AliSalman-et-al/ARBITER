# ADR 0001: Overall Judgment Some Concerns Threshold

## Status

Accepted

## Context

The RoB 2 IRPG domain decision tables produce deterministic domain judgments.
The overall guidance is partly deterministic: overall High applies when at
least one domain is High. It also says multiple domains with Some concerns can
substantially lower confidence enough to make the overall judgment High, but it
leaves that boundary to reviewer judgment.

ARBITER runs unattended, so this reviewer-judgment boundary must be made
deterministic and visibly owned by ARBITER rather than misattributed to the
Cochrane table logic.

## Decision

ARBITER sets `OVERALL_HIGH_SC_THRESHOLD = 3`.

The overall rollup is:

- Low when all five domains are Low.
- High when any domain is High.
- High when no domain is High and the number of Some concerns domains is at
  least `OVERALL_HIGH_SC_THRESHOLD`.
- Some concerns otherwise.

`requires_human_review` is true only for the two policy-driven multi-Some
concerns paths:

- `2 <= #Some concerns < OVERALL_HIGH_SC_THRESHOLD` with no High domains.
- `#Some concerns >= OVERALL_HIGH_SC_THRESHOLD` with no High domains.

A lone Some concerns domain is not flagged. An any-domain-High overall High is
not flagged by this policy because it follows the deterministic guidance rather
than ARBITER's threshold.

## Consequences

The threshold is a contract constant, not an environment knob. Changing it must
be reviewed as a policy change and should change the pipeline version used for
evaluation and persisted results.
