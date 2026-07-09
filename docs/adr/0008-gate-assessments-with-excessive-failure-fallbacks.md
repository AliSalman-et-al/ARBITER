# ADR 0008: Gate assessments with excessive failure fallbacks

## Status

Accepted

## Context

ARBITER records a flagged fallback answer when a signaling-question worker cannot obtain a valid structured model response after retries and repair. A small number of these failures can be reviewed at the SQ level, but an assessment dominated by failure-fallback answers is not an ordinary evidence read even if deterministic RoB 2 rollup can still compute a judgment.

## Decision

Each assessment carries an assessment-level reliability summary. Structural `NA` answers are excluded from the denominator. If failure-fallback signaling-question answers meet both the configured minimum count and fraction threshold, ARBITER keeps the domain-level records but replaces the overall judgment with `Unresolved`, sets `requires_human_review=True`, records a degradation event, and persists the reliability basis in JSON, Markdown, and SQLite outputs.

The default gate is at least two failure-fallback answers and at least 25% of non-structural signaling-question answers. The threshold is configurable through environment-backed settings so evaluation runs can tune the operational policy without changing deterministic RoB 2 decision tables.

## Consequences

Assessments with systemic provider or structured-output failure no longer look like normal Low, Some concerns, or High results. Reviewers still receive the computed domain details and per-SQ failure reasons for diagnosis, but downstream consumers can filter the outcome-level sentinel directly. The trade-off is that a transient provider incident can make an otherwise computable assessment unresolved until rerun.
