from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import arbiter as arbiter_module
from arbiter import assess_trial
from arbiter.config import AssessmentConfig
from arbiter.graph import builder as builder_module
from arbiter.graph.builder import build_outcome_graph
from arbiter.graph.state import AssessmentRuntime, TrialContext, base_ingestion_state
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import (
    AnswerCode,
    BlindingStatus,
    ConfidenceFlag,
    DocumentSection,
    EffectOfInterest,
    Judgment,
    PageBox,
    ReliabilityStatus,
    SectionMap,
    TrialMetadata,
)
from arbiter.observability import RunTrace
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
            DocumentSection(
                label="METHODS",
                pages=[0],
                char_start=0,
                char_end=len(text),
                text=text,
                domain_tags=[],
            ),
            DocumentSection(
                label="RESULTS",
                pages=[1],
                char_start=0,
                char_end=len(text),
                text=text,
                domain_tags=[],
            ),
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


def _ctx(
    client: MockLLMClient, effect: EffectOfInterest = EffectOfInterest.ASSIGNMENT
) -> TrialContext:
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


def _review_flagged_ctx(client: MockLLMClient) -> TrialContext:
    ctx = _ctx(client)
    return replace(ctx, config_summary={"eligibility_requires_human_review": True})


def _assignment_responses() -> dict[str, Any]:
    return {
        "1.1|assignment": _raw("Y", "The allocation sequence was random."),
        "1.2|assignment": _raw("Y", "Allocation was concealed."),
        "1.3|assignment": _raw("N", "Baseline imbalances were not reported."),
        "2.1|assignment": _raw(
            "Y", "Participants and personnel were aware of assignment."
        ),
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
        "2.1|adhering": _raw(
            "Y", "Participants and personnel were aware of assignment."
        ),
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
    return {
        "answer": answer,
        "quote": quote,
        "justification": "The quoted text supports the answer.",
    }


@pytest.mark.asyncio
async def test_ingest_trial_reuses_one_docling_converter_for_paper_and_supplements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paper_path = tmp_path / "paper.pdf"
    supplement_paths = [tmp_path / "appendix.pdf", tmp_path / "protocol.pdf"]
    paper_path.write_bytes(b"paper")
    for path in supplement_paths:
        path.write_bytes(b"supplement")
    shared_converter = object()
    build_calls = 0
    converter_uses: list[object | None] = []

    def fake_build_docling_converter(_settings: object) -> object:
        nonlocal build_calls
        build_calls += 1
        return shared_converter

    def fake_ingest_paper(
        _path: Path,
        *,
        converter: object | None = None,
        force_refresh_cache: bool = False,
    ):
        assert force_refresh_cache is False
        converter_uses.append(converter)
        section_map = _section_map()
        return section_map, section_map.full_text

    async def fake_ingest_supplements(
        _paths: list[Path],
        _aux_client: MockLLMClient,
        *,
        converter: object | None = None,
        force_refresh_cache: bool = False,
        trace: object | None = None,
    ) -> SupplementIndex:
        assert force_refresh_cache is False
        assert trace is not None
        converter_uses.append(converter)
        return SupplementIndex.empty()

    async def fake_extract_metadata(*_args: object, **_kwargs: object) -> TrialMetadata:
        return _metadata()

    monkeypatch.setattr(
        arbiter_module, "build_docling_converter", fake_build_docling_converter
    )
    monkeypatch.setattr(arbiter_module, "ingest_paper", fake_ingest_paper)
    monkeypatch.setattr(arbiter_module, "ingest_supplements", fake_ingest_supplements)
    monkeypatch.setattr(arbiter_module, "fetch_ctgov", lambda _nct: None)
    monkeypatch.setattr(arbiter_module, "extract_metadata", fake_extract_metadata)
    monkeypatch.setattr(
        arbiter_module, "create_llm_client", lambda *_args, **_kwargs: MockLLMClient()
    )

    await arbiter_module.ingest_trial(
        AssessmentConfig(
            paper_path=paper_path,
            supplement_paths=supplement_paths,
            nct_number=None,
        )
    )

    assert build_calls == 1
    assert converter_uses == [shared_converter, shared_converter]


@pytest.mark.asyncio
async def test_assess_trial_reuses_d1_and_sorts_domains_for_each_outcome() -> None:
    client = MockLLMClient(responses=_assignment_responses())
    config = AssessmentConfig(
        paper_path=Path("paper.pdf"),
        outcomes=["Overall survival", "Progression-free survival"],
    )

    assessments = await assess_trial(_ctx(client), config)

    assert [assessment.outcome for assessment in assessments] == [
        "Overall survival",
        "Progression-free survival",
    ]
    assert [judgment.domain for judgment in assessments[0].domain_judgments] == [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    assert (
        assessments[0].domain_judgments[0].model_dump()
        == assessments[1].domain_judgments[0].model_dump()
    )
    assert assessments[0].errors == []
    assert assessments[1].errors == []
    assert client.calls.count("1.1|assignment") == 1
    assert client.calls.count("2.1|assignment") == 2
    assert "2.7|assignment" in client.calls


@pytest.mark.asyncio
async def test_assess_trial_records_flagged_ni_when_signaling_question_call_fails(
    tmp_path: Path,
) -> None:
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
    assert (
        "1.2 signaling-question call failed: TimeoutError: provider timed out after retries"
        in assessment.errors
    )
    assert list(tmp_path.glob("**/data.json"))
    assert (tmp_path / "assessments.sqlite").exists()


@pytest.mark.asyncio
async def test_assess_trial_unresolves_overall_when_failure_fallback_gate_trips(
    tmp_path: Path,
) -> None:
    responses = _assignment_responses()
    for label in ("1.2|assignment", "2.1|assignment", "4.1|assignment"):
        responses[label] = TimeoutError("provider timed out after retries")
    client = MockLLMClient(responses=responses)
    trace = RunTrace()
    config = AssessmentConfig(
        paper_path=Path("paper.pdf"),
        outcomes=["Overall survival"],
        output_dir=tmp_path,
        db_path=tmp_path / "assessments.sqlite",
    )
    config.env.failure_fallback_fraction_threshold = 0.20
    config.env.failure_fallback_min_count = 2

    assessment = (await assess_trial(replace(_ctx(client), trace=trace), config))[0]

    assert assessment.overall_judgment is Judgment.UNRESOLVED
    assert assessment.requires_human_review is True
    assert (
        assessment.reliability.status
        is ReliabilityStatus.FAILURE_FALLBACK_EXCESSIVE
    )
    assert assessment.reliability.failure_fallback_sq_count == 3
    assert assessment.reliability.sq_answer_count > 3
    assert any(
        "failure fallback signaling-question answers exceeded" in error
        for error in assessment.errors
    )
    assert "Deterministic rollup before the gate" in assessment.overall_rationale
    assert any(
        event["category"] == "assessment_reliability_gate"
        for event in trace.degradation_events
    )


@pytest.mark.asyncio
async def test_assess_trial_preserves_fail_open_eligibility_review_flag(
    tmp_path: Path,
) -> None:
    client = MockLLMClient(responses=_assignment_responses())
    config = AssessmentConfig(
        paper_path=Path("paper.pdf"),
        outcomes=["Overall survival"],
        output_dir=tmp_path,
        db_path=tmp_path / "assessments.sqlite",
    )

    assessment = (await assess_trial(_review_flagged_ctx(client), config))[0]

    assert assessment.requires_human_review is True


@pytest.mark.asyncio
async def test_outcome_graph_adhering_effect_only_structurally_nas_2_7() -> None:
    client = MockLLMClient(responses=_adhering_responses())
    config = AssessmentConfig(
        paper_path=Path("paper.pdf"), effect_of_interest="adhering"
    )
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

    d2_answers = {
        sq_id: answer.answer.value
        for sq_id, answer in result["sq_answers"].items()
        if sq_id.startswith("2.")
    }
    assert d2_answers == {
        "2.1": "Y",
        "2.2": "N",
        "2.3": "N",
        "2.4": "N",
        "2.5": "N",
        "2.6": "Y",
        "2.7": "NA",
    }
    assert {"2.3|adhering", "2.4|adhering", "2.5|adhering", "2.6|adhering"} <= set(
        client.calls
    )


@pytest.mark.asyncio
async def test_outcome_graph_degrades_unresolvable_domain_to_human_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MockLLMClient(responses=_assignment_responses())
    config = AssessmentConfig(paper_path=Path("paper.pdf"))
    ctx = _ctx(client)
    trace = RunTrace()
    original_judge = builder_module._judge_domain

    def raise_for_d3(domain: str, answers: Any, effect: str):
        if domain == "D3":
            raise ValueError("synthetic unresolved D3")
        return original_judge(domain, answers, effect)

    monkeypatch.setattr(builder_module, "_judge_domain", raise_for_d3)
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
            trace=trace,
        ),
    )

    d3 = next(
        judgment for judgment in result["domain_judgments"] if judgment.domain == "D3"
    )
    assert d3.judgment is Judgment.UNRESOLVED
    assert result["overall_judgment"] is Judgment.UNRESOLVED
    assert result["requires_human_review"] is True
    assert "D3 unresolved domain judgment: synthetic unresolved D3" in result["errors"]
    assert trace.degradation_events[0]["category"] == "unresolved_domain_judgment"
    assert trace.degradation_events[0]["severity"] == "error"
