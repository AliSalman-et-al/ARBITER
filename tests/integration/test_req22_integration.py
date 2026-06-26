from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from arbiter import assess_trial
from arbiter.config import AssessmentConfig, EffectOfInterest as EffectOfInterestValue
from arbiter.graph.state import TrialContext
from arbiter.llm.mock_client import MockLLMClient
from arbiter.manifest import run_batch
from arbiter.models import (
    BlindingStatus,
    DocumentSection,
    EffectOfInterest,
    PageBox,
    ParsingQuality,
    SectionMap,
    StudyDesign,
    TrialMetadata,
)
from arbiter.observability import QATraceBundle, RunTrace
from arbiter.retrieval.supplement_index import SupplementIndex


PAPER_TEXT = (
    "The allocation sequence was random. Allocation was concealed. "
    "Baseline imbalances were not reported. "
    "Participants and personnel were aware of assignment. Deviations were balanced. "
    "The analysis was appropriate. Follow-up was complete. Outcome assessors were masked. "
    "The endpoint was prespecified."
)


def _raw(answer: str, quote: str) -> dict[str, str]:
    return {"answer": answer, "quote": quote, "justification": "The quoted text supports the answer."}


def _assignment_responses(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    responses: dict[str, Any] = {
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
    if overrides:
        responses.update(overrides)
    return responses


def _adhering_responses() -> dict[str, Any]:
    return {
        "1.1|adhering": _raw("Y", "The allocation sequence was random."),
        "1.2|adhering": _raw("Y", "Allocation was concealed."),
        "1.3|adhering": _raw("N", "Baseline imbalances were not reported."),
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


def _section_map() -> SectionMap:
    return SectionMap(
        source_path="paper.pdf",
        full_text=PAPER_TEXT,
        sections=[
            DocumentSection(
                label="FULL_TEXT",
                pages=[0],
                char_start=0,
                char_end=len(PAPER_TEXT),
                text=PAPER_TEXT,
                domain_tags=[],
            )
        ],
        page_boxes=[
            PageBox(boxclass="text", text=PAPER_TEXT[:120], bbox=(0, 0, 100, 100), page=0),
            PageBox(boxclass="text", text=PAPER_TEXT[120:], bbox=(0, 110, 100, 200), page=0),
        ],
        parsing_quality=ParsingQuality.STANDARD,
    )


def _metadata(effect: EffectOfInterest = EffectOfInterest.ASSIGNMENT) -> TrialMetadata:
    return TrialMetadata(
        trial_id="trial-1",
        title="Fixture Trial",
        intervention="Drug",
        comparator="Placebo",
        primary_outcome="Overall Survival",
        all_outcomes=["Overall Survival", "Progression-free Survival"],
        effect_of_interest=effect,
        blinding=BlindingStatus.DOUBLE_BLIND,
        nct_number="NCT00000001",
        study_design=StudyDesign.PARALLEL_RCT,
    )


def _ctx(
    client: MockLLMClient,
    *,
    effect: EffectOfInterest = EffectOfInterest.ASSIGNMENT,
    trace: RunTrace | None = None,
) -> TrialContext:
    section_map = _section_map()
    return TrialContext(
        config_summary={"inputs_hash": "hash-1", "effect_of_interest": effect.value},
        trial_metadata=_metadata(effect),
        section_map=section_map,
        raw_char_stream=section_map.full_text,
        supplement_index=SupplementIndex.empty(),
        ct_gov_data={
            "protocolSection": {
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Overall Survival"}],
                    "secondaryOutcomes": [{"measure": "Progression-free Survival"}],
                }
            }
        },
        shared_prefix_text="Fixture trial prefix.",
        ct_gov_block="[ClinicalTrials.gov]",
        llm_client_sq=client,
        llm_client_aux=MockLLMClient(trace=trace),
        trace=trace,
    )


def _config(tmp_path: Path, *, outcomes: list[str] | None = None, effect: str = "assignment") -> AssessmentConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paper = tmp_path / "paper.pdf"
    paper.write_text(PAPER_TEXT, encoding="utf-8")
    return AssessmentConfig(
        paper_path=paper,
        outcomes=outcomes,
        effect_of_interest=cast(EffectOfInterestValue, effect),
        sq_model="mock",
        aux_model="mock",
        output_dir=tmp_path / "out",
        db_path=tmp_path / "arbiter.db",
        report_enabled=True,
    )


@pytest.mark.asyncio
async def test_full_trial_two_tier_reuse_writes_json_sqlite_and_report(tmp_path: Path) -> None:
    client = MockLLMClient(responses=_assignment_responses())
    config = _config(tmp_path, outcomes=["Overall Survival", "Progression-free Survival"])

    assessments = await assess_trial(_ctx(client), config)

    assert [assessment.outcome for assessment in assessments] == ["Overall Survival", "Progression-free Survival"]
    assert assessments[0].domain_judgments[0].model_dump() == assessments[1].domain_judgments[0].model_dump()
    assert client.calls.count("1.1|assignment") == 1
    assert client.calls.count("2.1|assignment") == 2
    assert (tmp_path / "out" / "trial-1" / "overall_survival__assignment" / "data.json").exists()
    assert (tmp_path / "out" / "trial-1" / "overall_survival__assignment" / "report.md").exists()
    with sqlite3.connect(config.db_path) as conn:
        rows = conn.execute("SELECT outcome FROM arbiter_assessments ORDER BY outcome").fetchall()
    assert rows == [("Overall Survival",), ("Progression-free Survival",)]


@pytest.mark.asyncio
async def test_llm_failure_on_one_sq_completes_with_flagged_ni_and_errors(tmp_path: Path) -> None:
    client = MockLLMClient(
        responses=_assignment_responses(
            {
                "3.1|assignment": RuntimeError("rate limit"),
                "3.2|assignment": _raw("N", "Follow-up was complete."),
                "3.3|assignment": _raw("N", "Follow-up was complete."),
            }
        )
    )

    assessment = (await assess_trial(_ctx(client), _config(tmp_path, outcomes=["Overall Survival"])))[0]

    d3 = next(domain for domain in assessment.domain_judgments if domain.domain == "D3")
    sq31 = next(answer for answer in d3.sq_answers if answer.sq_id == "3.1")
    assert sq31.answer.value == "NI"
    assert sq31.confidence.flag.value == "FLAGGED"
    assert assessment.errors


@pytest.mark.asyncio
async def test_quote_verification_failure_flags_answer(tmp_path: Path) -> None:
    client = MockLLMClient(
        responses=_assignment_responses({"1.1|assignment": _raw("Y", "This quote is absent from the paper.")})
    )

    assessment = (await assess_trial(_ctx(client), _config(tmp_path, outcomes=["Overall Survival"])))[0]

    d1 = next(domain for domain in assessment.domain_judgments if domain.domain == "D1")
    sq11 = next(answer for answer in d1.sq_answers if answer.sq_id == "1.1")
    assert sq11.confidence.quote_verified is False
    assert sq11.confidence.flag.value == "FLAGGED"


@pytest.mark.asyncio
async def test_d2_branching_assignment_and_adhering_effects(tmp_path: Path) -> None:
    assignment = MockLLMClient(responses=_assignment_responses())
    assignment_assessment = (await assess_trial(_ctx(assignment), _config(tmp_path / "a", outcomes=["Overall Survival"])))[0]

    adhering = MockLLMClient(responses=_adhering_responses())
    adhering_assessment = (
        await assess_trial(
            _ctx(adhering, effect=EffectOfInterest.ADHERING),
            _config(tmp_path / "b", outcomes=["Overall Survival"], effect="adhering"),
        )
    )[0]

    assignment_d2 = next(domain for domain in assignment_assessment.domain_judgments if domain.domain == "D2")
    adhering_d2 = next(domain for domain in adhering_assessment.domain_judgments if domain.domain == "D2")
    assert {answer.sq_id: answer.answer.value for answer in assignment_d2.sq_answers}["2.4"] == "NA"
    assert {answer.sq_id: answer.answer.value for answer in adhering_d2.sq_answers}["2.7"] == "NA"
    assert {"2.3|adhering", "2.4|adhering", "2.5|adhering", "2.6|adhering"} <= set(adhering.calls)


@pytest.mark.asyncio
async def test_d3_is_text_only_no_vision_call(tmp_path: Path) -> None:
    class VisionCountingMock(MockLLMClient):
        vision_calls = 0

        async def complete_vision(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.vision_calls += 1
            return await super().complete_vision(*args, **kwargs)

    client = VisionCountingMock(responses=_assignment_responses())

    assessment = (await assess_trial(_ctx(client), _config(tmp_path, outcomes=["Overall Survival"])))[0]

    assert client.vision_calls == 0
    assert "consort" not in assessment.model_dump(mode="json")


@pytest.mark.asyncio
async def test_trace_side_channel_and_trace_levels(tmp_path: Path) -> None:
    full_trace = RunTrace(trace_level="full")
    client = MockLLMClient(responses=_assignment_responses(), trace=full_trace)
    config = _config(tmp_path / "full", outcomes=["Overall Survival"])

    assessment = (await assess_trial(_ctx(client, trace=full_trace), config))[0]
    payload = json.loads((config.output_dir / "trial-1" / "trace.json").read_text(encoding="utf-8"))
    data = json.loads((config.output_dir / "trial-1" / "overall_survival__assignment" / "data.json").read_text())

    assert payload["trace_level"] == "full"
    assert payload["node_spans"]
    assert payload["llm_calls"][0]["messages"]
    assert (config.output_dir / "trial-1" / "artifacts").exists()
    assert "trace" not in data
    assert "trace" not in assessment.model_dump(mode="json")

    summary_trace = RunTrace(trace_level="summary")
    summary_config = _config(tmp_path / "summary", outcomes=["Overall Survival"])
    await assess_trial(
        _ctx(MockLLMClient(responses=_assignment_responses(), trace=summary_trace), trace=summary_trace),
        summary_config,
    )
    summary_payload = json.loads((summary_config.output_dir / "trial-1" / "trace.json").read_text(encoding="utf-8"))
    assert "messages" not in summary_payload["llm_calls"][0]
    assert not (summary_config.output_dir / "trial-1" / "artifacts").exists()

    off_trace = RunTrace(trace_level="off")
    off_config = _config(tmp_path / "off", outcomes=["Overall Survival"])
    await assess_trial(_ctx(MockLLMClient(responses=_assignment_responses(), trace=off_trace), trace=off_trace), off_config)
    assert not (off_config.output_dir / "trial-1" / "trace.json").exists()


@pytest.mark.asyncio
async def test_full_qa_trace_bundle_covers_assess_run_and_is_tail_safe(tmp_path: Path) -> None:
    config = _config(tmp_path, outcomes=["Overall Survival"])
    config.trace_level = "full"
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=["assess", "--trace", "full"],
        config=config,
    )
    config.qa_trace = bundle
    trace = RunTrace(trace_level="full", qa_trace=bundle)
    client = MockLLMClient(responses=_assignment_responses(), trace=trace)

    assessments = await assess_trial(_ctx(client, trace=trace), config)
    bundle.record_event(
        event_type="run.completed",
        status="completed",
        payload={"trial_id": assessments[0].trial_id},
    )
    bundle.close()

    lines = bundle.events_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    event_types = {event["event_type"] for event in events}
    artifact_roots = {ref.split("/", 1)[0] for event in events for ref in event["artifact_refs"]}

    assert len(lines) == len(events)
    assert {"llm_calls", "retrieval", "context", "quote_verification", "sq_answers", "judgments", "outputs"} <= artifact_roots
    assert {"llm.completed", "retrieval.completed", "context_assembly.completed", "sq.finalized"} <= event_types
    assert {"judgment.domain.completed", "judgment.overall.completed", "output.assessment_json.written"} <= event_types
    assert (bundle.root / "run_manifest.json").exists()
    assert all(json.loads(line) for line in lines)


@pytest.mark.asyncio
async def test_batch_idempotency_skips_second_run_and_force_reruns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    paper = tmp_path / "paper.pdf"
    paper.write_text(PAPER_TEXT, encoding="utf-8")
    manifest.write_text("main_paper,trial_label,outcomes\npaper.pdf,Trial 1,Overall Survival\n", encoding="utf-8")
    config = _config(tmp_path, outcomes=None)
    calls = {"ingest": 0, "assess": 0}

    async def fake_ingest(entry_config: AssessmentConfig) -> TrialContext:
        calls["ingest"] += 1
        return _ctx(MockLLMClient(responses=_assignment_responses()))

    async def fake_assess(ctx: TrialContext, entry_config: AssessmentConfig):
        calls["assess"] += 1
        return await assess_trial(ctx, entry_config)

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.manifest.assess_trial", fake_assess)

    first = await run_batch(manifest, config)
    second = await run_batch(manifest, config)
    forced = await run_batch(manifest, replace(config, force=True))

    assert first.assessed_pairs == 1
    assert second.skipped_entries == 1
    assert forced.assessed_pairs == 1
    assert calls == {"ingest": 2, "assess": 2}
