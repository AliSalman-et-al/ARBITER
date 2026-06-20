from __future__ import annotations

from arbiter.graph.nodes.pre_d5 import compare_registered_outcome, pre_d5_node_factory


def _ctgov_record() -> dict:
    return {
        "protocolSection": {
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Overall Survival", "timeFrame": "36 months"},
                ],
                "secondaryOutcomes": [
                    {"measure": "Progression-Free Survival"},
                    {"measure": "Objective Response Rate"},
                ],
            }
        }
    }


def test_compare_registered_outcome_returns_empty_comparison_without_ctgov_data() -> None:
    comparison = compare_registered_outcome(
        assessed_outcome="Progression-Free Survival",
        ctgov_record=None,
        threshold=0.85,
    )

    assert comparison.registered_outcome is None
    assert comparison.published_outcome is None
    assert comparison.outcome_similarity_score is None
    assert comparison.outcome_change_detected is None
    assert comparison.registered_as_primary is None


def test_compare_registered_outcome_does_not_flag_registered_secondary_outcome() -> None:
    comparison = compare_registered_outcome(
        assessed_outcome="Progression-Free Survival",
        ctgov_record=_ctgov_record(),
        threshold=0.85,
    )

    assert comparison.registered_outcome == "Progression-Free Survival"
    assert comparison.published_outcome == "Progression-Free Survival"
    assert comparison.outcome_similarity_score == 1.0
    assert comparison.outcome_change_detected is False
    assert comparison.registered_as_primary is False


def test_compare_registered_outcome_flags_outcome_absent_from_registry() -> None:
    comparison = compare_registered_outcome(
        assessed_outcome="Time to Treatment Failure",
        ctgov_record=_ctgov_record(),
        threshold=0.85,
    )

    assert comparison.registered_outcome is not None
    assert comparison.published_outcome == "Time to Treatment Failure"
    assert comparison.outcome_similarity_score is not None
    assert comparison.outcome_similarity_score < 0.85
    assert comparison.outcome_change_detected is True


def test_compare_registered_outcome_tracks_primary_best_match() -> None:
    comparison = compare_registered_outcome(
        assessed_outcome="Overall Survival",
        ctgov_record=_ctgov_record(),
        threshold=0.85,
    )

    assert comparison.registered_outcome == "Overall Survival"
    assert comparison.outcome_change_detected is False
    assert comparison.registered_as_primary is True


def test_pre_d5_node_returns_flat_outcome_state_fields() -> None:
    node = pre_d5_node_factory(threshold=0.85)

    result = node(
        {
            "outcome": "Progression-Free Survival",
            "ctgov_record": _ctgov_record(),
        }
    )

    assert result == {
        "registered_outcome": "Progression-Free Survival",
        "published_outcome": "Progression-Free Survival",
        "outcome_similarity_score": 1.0,
        "outcome_change_detected": False,
        "registered_as_primary": False,
    }

