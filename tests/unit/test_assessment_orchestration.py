from __future__ import annotations

from pathlib import Path

import pytest

from arbiter import assess_trial
from arbiter.config import AssessmentConfig
from arbiter.graph.builder import build_outcome_graph
from arbiter.graph.state import AssessmentRuntime, TrialContext, base_ingestion_state
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import (
    AnswerCode,
    BlindingStatus,
    ConfidenceFlag,
    DocumentSection,
    EffectOfInterest,
    PageBox,
    SectionMap,
    TrialMetadata,
)
from arbiter.retrieval.supplement_index import SupplementIndex


def _section_map() -> SectionMap:
    text = (
        "The allocation sequence was random. Allocation was concealed. Baseline imbalances were not reported. "
        "Participants and personnel were aware of assignment. Deviations were balanced. "
        "The analysis was appropriate. Follow-up was complete. Outcome assessors were masked. "
        "The endpoint was prespecified."
    )
    return SectionMap(
        source_path="paper.pdf",
        full_text=text,
        sections=[
            DocumentSection(label="METHODS", pages=[0], char_start=0, char_end=len(text), text=text, domain_tags=[]),
            DocumentSection(label="RESULTS", pages=[1], char_start=0, char_end=len(text), text=text, domain_tags=[]),
        ],
        page_boxes=[PageBox(boxclass="text", text=text, bbox=(0, 0, 100, 100), page=0)],
    )


def _metadata(effect: EffectOfInterest = EffectOfInterest.ASSIGNMENT) -> TrialMetadata:
    return TrialMetadata(
        trial_id="T1",
        title="Trial",
        intervention="Drug",
        comparator="Placebo",
        primary_outcome="Overall survival",
        all_outcomes=["Overall survival", "Progression-free survival"],
        effect_of_interest=effect,
        blinding=BlindingStatus.DOUBLE_BLIND,
        nct_number="NCT00000001",
    )


def _ctx(client: MockLLMClient, effect: EffectOfInterest = EffectOfInterest.ASSIGNMENT) -> TrialContext:
    return TrialContext(
        config_summary={"effect_of_interest": effect.value},
        trial_metadata=_metadata(effect),
        section_map=_section_map(),
        raw_char_stream=_section_map().full_text,
        supplement_index=SupplementIndex.empty(),
        ct_gov_data={
            "protocolSection": {
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Overall survival"}],
                    "secondaryOutcomes": [{"measure": "Progression-free survival"}],
                }
            }
        },
        shared_prefix_text="Trial prefix.",
        ct_gov_block="[ClinicalTrials.gov]",
        llm_client_sq=client,
        llm_client_aux=MockLLMClient(),
    )


def _assignment_responses() -> dict[str, dict[str, str]]:
    return {
        "1.1|assignment": _raw("Y", "The allocation sequence was random."),
        "1.2|assignment": _raw("Y", "Allocation was concealed."),
        "1.3|assignment": _raw("N", "Baseline imbalances were not reported."),
        "2.1|assignment": _raw("Y", "Participants and personnel were aware of assignment."),
        "2.2|assignment": _raw("N", "Deviations were balanced."),
        "2.3|assignment": _raw("N", "Deviations were balanced."),
        "2.6|assignment": _raw("N", "The analysis was appropriate."),
        "2.7|assignment": _raw("N", "The analysis was appropriate."),
        "3.1|assignment": _raw("Y", "Follow-up was complete."),
        "4.1|assignment": _raw("N", "Outcome assessors were masked."),
        "4.2|assignment": _raw("N", "Outcome assessors were masked."),
        "4.3|assignment": _raw("N", "Outcome assessors were masked."),
        "5.1|assignment": _raw("Y", "The endpoint was prespecified."),
    }


def _adhering_responses() -> dict[str, dict[str, str]]:
    return {
        "2.1|adhering": _raw("Y", "Participants and personnel were aware of assignment."),
        "2.2|adhering": _raw("N", "Deviations were balanced."),
        "2.3|adhering": _raw("N", "Deviations were balanced."),
        "2.4|adhering": _raw("N", "Deviations were balanced."),
        "2.5|adhering": _raw("N", "Deviations were balanced."),
        "2.6|adhering": _raw("Y", "The analysis was appropriate."),
        "3.1|adhering": _raw("Y", "Follow-up was complete."),
        "4.1|adhering": _raw("N", "Outcome assessors were masked."),
        "4.2|adhering": _raw("N", "Outcome assessors were masked."),
        "4.3|adhering": _raw("N", "Outcome assessors were masked."),
        "5.1|adhering": _raw("Y", "The endpoint was prespecified."),
    }


def _raw(answer: str, quote: str) -> dict[str, str]:
    return {"answer": answer, "quote": quote, "justification": "The quoted text supports the answer."}


@pytest.mark.asyncio
async def test_assess_trial_reuses_d1_and_sorts_domains_for_each_outcome() -> None:
    client = MockLLMClient(responses=_assignment_responses())
    config = AssessmentConfig(paper_path=Path("paper.pdf"), outcomes=["Overall survival", "Progression-free survival"])

    assessments = await assess_trial(_ctx(client), config)

    assert [assessment.outcome for assessment in assessments] == ["Overall survival", "Progression-free survival"]
    assert [judgment.domain for judgment in assessments[0].domain_judgments] == ["D1", "D2", "D3", "D4", "D5"]
    assert assessments[0].domain_judgments[0].model_dump() == assessments[1].domain_judgments[0].model_dump()
    assert client.calls.count("1.1|assignment") == 1
    assert client.calls.count("2.1|assignment") == 2
    assert "2.7|assignment" in client.calls


@pytest.mark.asyncio
async def test_assess_trial_records_flagged_ni_when_signaling_question_call_fails(tmp_path: Path) -> None:
    responses = _assignment_responses()
    responses["1.2|assignment"] = TimeoutError("provider timed out after retries")
    client = MockLLMClient(responses=responses)
    config = AssessmentConfig(
        paper_path=Path("paper.pdf"),
        outcomes=["Overall survival"],
        output_dir=tmp_path,
        db_path=tmp_path / "assessments.sqlite",
    )

    assessment = (await assess_trial(_ctx(client), config))[0]

    d1 = next(domain for domain in assessment.domain_judgments if domain.domain == "D1")
    sq12 = next(answer for answer in d1.sq_answers if answer.sq_id == "1.2")
    assert sq12.answer == AnswerCode.NI
    assert sq12.confidence.flag == ConfidenceFlag.FLAGGED
    assert assessment.requires_human_review is True
    assert "1.2 signaling-question call failed: TimeoutError: provider timed out after retries" in assessment.errors
    assert list(tmp_path.glob("**/data.json"))
    assert (tmp_path / "assessments.sqlite").exists()


@pytest.mark.asyncio
async def test_outcome_graph_adhering_effect_only_structurally_nas_2_7() -> None:
    client = MockLLMClient(responses=_adhering_responses())
    config = AssessmentConfig(paper_path=Path("paper.pdf"), effect_of_interest="adhering")
    ctx = _ctx(client, EffectOfInterest.ADHERING)
    state = {
        **base_ingestion_state(ctx, config),
        "outcome": "Overall survival",
        "trial_domain_judgments": [
            {
                "domain": "D1",
                "scope": "trial",
                "judgment": "Low",
                "algorithm_rationale": "fixture",
                "sq_answers": [],
            }
        ],
        "domain_contexts": {},
        "sq_answers": {},
        "domain_judgments": [],
        "errors": [],
    }
    result = await build_outcome_graph().ainvoke(
        state,
        context=AssessmentRuntime(
            llm_client_sq=client,
            llm_client_aux=MockLLMClient(),
            supplement_index=SupplementIndex.empty(),
        ),
    )

    d2_answers = {sq_id: answer.answer.value for sq_id, answer in result["sq_answers"].items() if sq_id.startswith("2.")}
    assert d2_answers == {
        "2.1": "Y",
        "2.2": "N",
        "2.3": "N",
        "2.4": "N",
        "2.5": "N",
        "2.6": "Y",
        "2.7": "NA",
    }
    assert {"2.3|adhering", "2.4|adhering", "2.5|adhering", "2.6|adhering"} <= set(client.calls)
