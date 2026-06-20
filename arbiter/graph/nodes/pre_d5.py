"""Deterministic pre-D5 registered outcome comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rapidfuzz import fuzz

from arbiter.config import EnvSettings
from arbiter.models import OutcomeComparison

OUTCOME_COMPARISON_FIELDS = (
    "registered_outcome",
    "published_outcome",
    "outcome_similarity_score",
    "outcome_change_detected",
    "registered_as_primary",
)


def _empty_comparison() -> OutcomeComparison:
    return OutcomeComparison()


def _outcomes_module(ctgov_record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if ctgov_record is None:
        return None

    protocol_section = ctgov_record.get("protocolSection")
    if not isinstance(protocol_section, Mapping):
        return None

    outcomes_module = protocol_section.get("outcomesModule")
    if not isinstance(outcomes_module, Mapping):
        return None

    return outcomes_module


def _registered_outcome_measures(
    outcomes_module: Mapping[str, Any],
    key: str,
    *,
    registered_as_primary: bool,
) -> list[tuple[str, bool]]:
    outcomes = outcomes_module.get(key)
    if not isinstance(outcomes, list):
        return []

    measures: list[tuple[str, bool]] = []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        measure = outcome.get("measure")
        if isinstance(measure, str) and measure.strip():
            measures.append((measure.strip(), registered_as_primary))

    return measures


def compare_registered_outcome(
    *,
    assessed_outcome: str,
    ctgov_record: Mapping[str, Any] | None,
    threshold: float,
) -> OutcomeComparison:
    """Match an assessed outcome against all registered CT.gov outcomes."""

    outcomes_module = _outcomes_module(ctgov_record)
    if outcomes_module is None:
        return _empty_comparison()

    registered_outcomes = [
        *_registered_outcome_measures(outcomes_module, "primaryOutcomes", registered_as_primary=True),
        *_registered_outcome_measures(outcomes_module, "secondaryOutcomes", registered_as_primary=False),
    ]
    if not registered_outcomes:
        return _empty_comparison()

    best_measure: str | None = None
    best_score = -1.0
    best_registered_as_primary: bool | None = None
    for registered_measure, registered_as_primary in registered_outcomes:
        score = fuzz.ratio(assessed_outcome, registered_measure) / 100
        if score > best_score:
            best_measure = registered_measure
            best_score = score
            best_registered_as_primary = registered_as_primary

    rounded_score = round(best_score, 3)
    return OutcomeComparison(
        registered_outcome=best_measure,
        published_outcome=assessed_outcome,
        outcome_similarity_score=rounded_score,
        outcome_change_detected=rounded_score < threshold,
        registered_as_primary=best_registered_as_primary,
    )


def pre_d5_node_factory(threshold: float | None = None) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a LangGraph-compatible pre-D5 node."""

    outcome_match_threshold = EnvSettings().outcome_match_threshold if threshold is None else threshold

    def pre_d5_node(state: Mapping[str, Any]) -> dict[str, Any]:
        comparison = compare_registered_outcome(
            assessed_outcome=str(state.get("outcome", "")),
            ctgov_record=state.get("ctgov_record"),
            threshold=outcome_match_threshold,
        )
        return comparison.model_dump()

    return pre_d5_node


pre_d5_node = pre_d5_node_factory()

