from __future__ import annotations

import pytest

from arbiter.graph.nodes.outcome_profile import (
    build_outcome_profile_messages,
    classify_outcome_measurement_profile,
    outcome_profile_node,
)
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import OutcomeMeasurementProfileType


def _ctgov_record() -> dict:
    return {
        "protocolSection": {
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Overall Survival",
                        "description": "Time from randomization until death from any cause.",
                        "timeFrame": "Up to 60 months",
                    },
                    {
                        "measure": "PSA Response",
                        "description": "At least a 50% decline in serum PSA from baseline.",
                        "timeFrame": "12 weeks",
                    },
                ],
                "secondaryOutcomes": [
                    {
                        "measure": "Progression-Free Survival",
                        "description": "Time to radiographic progression or death.",
                        "timeFrame": "Up to 60 months",
                    },
                    {
                        "measure": "Quality of Life",
                        "description": "Patient-reported FACT-P questionnaire score.",
                    },
                ],
            }
        }
    }


def _message_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(message["content"] for message in messages)


def test_outcome_profile_prompt_includes_taxonomy_and_registry_candidates() -> None:
    messages = build_outcome_profile_messages(
        assessed_outcome="Progression-free survival",
        registered_outcome_candidates=[
            {
                "measure": "Progression-Free Survival",
                "description": "Time to radiographic progression or death.",
                "timeFrame": "Up to 60 months",
                "match_score": 1.0,
            }
        ],
    )

    text = _message_text(messages)

    assert "vital-status" in text
    assert "clinician-composite" in text
    assert "Classify from measurement characteristics" in text
    assert "Progression-Free Survival" in text
    assert "radiographic progression or death" in text
    assert "composites that include death are clinician-composite" in text


@pytest.mark.asyncio
async def test_classifies_outcome_profile_with_aux_model() -> None:
    client = MockLLMClient(
        responses={
            "outcome_profile|Overall survival": {
                "profile": "vital-status",
                "basis": "The registry defines overall survival as time until death from any cause.",
                "matched_registered_outcome": "Overall Survival",
                "match_score": 1.0,
            }
        }
    )

    profile = await classify_outcome_measurement_profile(
        assessed_outcome="Overall survival",
        ctgov_record=_ctgov_record(),
        aux_model=client,
        config=object(),
    )

    assert profile.profile == OutcomeMeasurementProfileType.VITAL_STATUS
    assert profile.matched_registered_outcome == "Overall Survival"
    assert "death is the only event" in profile.definition
    assert client.calls == ["outcome_profile|Overall survival"]
    assert "death from any cause" in _message_text(client.trace_messages[0])


@pytest.mark.asyncio
async def test_profile_node_returns_flat_outcome_state_field() -> None:
    result = await outcome_profile_node(
        {
            "outcome": "Overall survival",
            "ctgov_record": _ctgov_record(),
            "aux_model": MockLLMClient(
                responses={
                    "outcome_profile|Overall survival": {
                        "profile": "vital-status",
                        "basis": "The outcome is single-criterion mortality.",
                        "matched_registered_outcome": "Overall Survival",
                    }
                }
            ),
        }
    )

    assert result["outcome_measurement_profile"]["profile"] == "vital-status"
    assert result["outcome_measurement_profile"]["matched_registered_outcome"] == (
        "Overall Survival"
    )


@pytest.mark.asyncio
async def test_profile_node_falls_back_to_unclear_when_aux_call_fails() -> None:
    result = await outcome_profile_node(
        {
            "outcome": "Overall survival",
            "ctgov_record": _ctgov_record(),
            "aux_model": MockLLMClient(),
        }
    )

    profile = result["outcome_measurement_profile"]
    assert profile["profile"] == "unclear"
    assert "classification failed" in profile["basis"]
