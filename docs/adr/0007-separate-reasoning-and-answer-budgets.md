# ADR 0007: Separate reasoning and answer budgets

## Status

Accepted

## Context

Reasoning-capable signaling-question models can spend thousands of completion tokens on deliberation before emitting the small JSON answer ARBITER records. When reasoning and answer text share one cap, the answer can be truncated while the pipeline still attempts to parse whatever partial object remains.

## Decision

OpenRouter signaling-question requests treat the configured signaling-question max tokens as the answer budget, assign reasoning-capable models a separate reasoning budget, and send the provider a top-level token cap equal to answer budget plus reasoning budget plus reserve. Provider-declared `finish_reason=length` is never salvaged as a valid answer; it enters the structured repair path so the answer object must be regenerated completely.

## Consequences

Raising the reasoning budget no longer shrinks the space available for the JSON answer, and truncated responses become visible repair attempts in trace data instead of silent partial parses. The trade-off is higher worst-case completion headroom per call, kept tunable through environment-backed config.
