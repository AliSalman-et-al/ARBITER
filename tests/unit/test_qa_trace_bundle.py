from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import arbiter
from arbiter.config import AssessmentConfig
from arbiter.graph.builder import build_outcome_graph
from arbiter.graph.nodes.sq_node import sq_node
from arbiter.graph.state import AssessmentRuntime, TrialContext, base_ingestion_state
from arbiter.llm.mock_client import MockLLMClient
from arbiter.manifest import run_batch
from arbiter.models import (
    AnswerCode,
    BlindingStatus,
    DomainContext,
    DomainJudgment,
    DocumentSection,
    DocType,
    EffectOfInterest,
    Judgment,
    PageBox,
    ParsingQuality,
    SectionMap,
    SQAnswer,
    StudyDesign,
    SupplementSegment,
    TrialMetadata,
)
from arbiter.observability.qa_trace import QATraceBundle, create_qa_trace_bundle, generate_run_id
from arbiter.retrieval.supplement_index import SupplementIndex


def test_generate_run_id_uses_timestamp_and_short_id() -> None:
    run_id = generate_run_id()

    assert re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{8}", run_id)


def test_full_trace_bundle_writes_manifest_and_event_schema(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")
    supplement = tmp_path / "supplement.pdf"
    supplement.write_text("supplement", encoding="utf-8")
    config = AssessmentConfig.from_env(
        paper_path=paper,
        supplement_paths=[supplement],
        nct_number="NCT01234567",
        outcomes=["Overall survival"],
        sq_model="gpt-oss-120b-free",
        aux_model="gpt-oss-120b",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "arbiter.db",
        trace_level="full",
    )

    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=["assess", "--paper", str(paper), "--trace", "full"],
        config=config,
        input_manifest_path=None,
    )
    event = bundle.record_event(
        event_type="run.started",
        status="started",
        trial_id="NCT01234567",
        artifact_refs=["run_manifest.json"],
        payload={"ok": True},
    )
    bundle.close()

    assert bundle.root == tmp_path / "runs" / bundle.run_id / "qa_trace"
    manifest = json.loads((bundle.root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == bundle.run_id
    assert manifest["command"] == "assess"
    assert manifest["cli_args"] == ["assess", "--paper", str(paper), "--trace", "full"]
    assert manifest["trace_mode"] == "full"
    assert manifest["arbiter_version"] == "0.1.0"
    assert manifest["pipeline_version"] == "0.1.0"
    assert manifest["inputs"]["paper"]["path"] == str(paper)
    assert manifest["inputs"]["paper"]["sha256"]
    assert manifest["inputs"]["supplements"][0]["path"] == str(supplement)
    assert manifest["models"]["sq"]["name"] == "gpt-oss-120b-free"
    assert manifest["models"]["sq"]["provider"] == "openrouter"
    assert manifest["outputs"]["output_dir"] == str(tmp_path / "output")
    assert "api_key" not in json.dumps(manifest).lower()

    lines = (bundle.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload == {
        "schema_version": "1",
        "run_id": bundle.run_id,
        "event_id": event["event_id"],
        "parent_event_id": None,
        "timestamp": payload["timestamp"],
        "event_type": "run.started",
        "status": "started",
        "trial_id": "NCT01234567",
        "outcome": None,
        "domain": None,
        "sq_id": None,
        "artifact_refs": ["run_manifest.json"],
        "payload": {"ok": True},
    }


def test_atomic_artifact_write_uses_temp_then_rename(tmp_path: Path) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )

    path = bundle.write_json_artifact("artifacts/data.json", {"answer": "Y"})

    assert path == bundle.root / "artifacts" / "data.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"answer": "Y"}
    assert not list(path.parent.glob("*.tmp"))


def test_create_qa_trace_bundle_is_noop_for_summary_and_off(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config = AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="summary")

    assert create_qa_trace_bundle(config, command="assess", cli_args=[]) is None
    assert not Path("runs").exists()


def test_full_trace_setup_failure_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    config = AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full")

    def fail(*_args, **_kwargs):
        raise OSError("cannot create trace root")

    monkeypatch.setattr(Path, "mkdir", fail)

    with pytest.raises(OSError, match="cannot create trace root"):
        create_qa_trace_bundle(config, command="assess", cli_args=[])


@pytest.mark.asyncio
async def test_full_trace_records_source_artifacts_for_single_assess_ingestion(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    supplement = tmp_path / "supplement.pdf"
    paper.write_text("paper fixture", encoding="utf-8")
    supplement.write_text("supplement fixture", encoding="utf-8")
    config = AssessmentConfig(
        paper_path=paper,
        supplement_paths=[supplement],
        nct_number="NCT00000001",
        trace_level="full",
    )
    bundle = QATraceBundle.create(base_dir=tmp_path / "runs", command="assess", cli_args=[], config=config)
    config.qa_trace = bundle
    section_map = SectionMap(
        source_path=str(paper),
        full_text="Main paper parsed text.",
        sections=[
            DocumentSection(label="METHODS", pages=[0], char_start=0, char_end=23, text="Main paper parsed text.")
        ],
        page_boxes=[PageBox(boxclass="text", text="Main paper parsed text.", bbox=(0, 0, 1, 1), page=0)],
        parsing_quality=ParsingQuality.STANDARD,
        nct_number="NCT00000001",
    )
    segment = SupplementSegment(
        segment_id="supplement-1",
        source_file=str(supplement),
        doc_type=DocType.PROTOCOL,
        heading="Protocol",
        pages=[0],
        raw_text="Supplement parsed text.",
        annotation="Risk of bias relevant.",
        char_count=23,
    )

    monkeypatch.setattr(arbiter, "ingest_paper", lambda _path: (section_map, "raw char stream"))
    monkeypatch.setattr(arbiter, "ingest_supplements", lambda *_args: _async_value(SupplementIndex([segment])))
    monkeypatch.setattr(arbiter, "fetch_ctgov", lambda _nct: _async_value({"protocolSection": {"id": "NCT00000001"}}))
    monkeypatch.setattr(
        arbiter,
        "create_llm_client",
        lambda *_args, **_kwargs: MockLLMClient(
            responses={
                "metadata": {
                    "title": "Trial title",
                    "intervention": "Drug",
                    "comparator": "Placebo",
                    "primary_outcome": "Overall survival",
                    "all_outcomes": ["Overall survival"],
                    "blinding": BlindingStatus.DOUBLE_BLIND.value,
                    "nct_number": "NCT00000001",
                    "study_design": StudyDesign.PARALLEL_RCT.value,
                }
            }
        ),
    )

    await arbiter.ingest_trial(config)
    bundle.close()

    events = [json.loads(line) for line in (bundle.root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    main_ref = _single_event_ref(events, "ingestion.main_paper.completed")
    supplement_ref = _single_event_ref(events, "ingestion.supplements.completed")
    metadata_ref = _single_event_ref(events, "ingestion.metadata.completed")
    main_payload = json.loads((bundle.root / main_ref).read_text(encoding="utf-8"))
    supplement_payload = json.loads((bundle.root / supplement_ref).read_text(encoding="utf-8"))
    ctgov_payload = json.loads((bundle.root / "sources" / "ctgov" / "NCT00000001.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads((bundle.root / metadata_ref).read_text(encoding="utf-8"))

    assert main_payload["full_text"] == "Main paper parsed text."
    assert main_payload["page_boxes"][0]["page"] == 0
    assert supplement_payload["segments"][0]["raw_text"] == "Supplement parsed text."
    assert ctgov_payload["protocolSection"]["id"] == "NCT00000001"
    assert metadata_payload["trial_id"] == "NCT00000001"
    assert str(main_ref).replace("\\", "/").startswith("sources/main_paper/")
    assert str(supplement_ref).replace("\\", "/").startswith("sources/supplements/")
    assert _event_refs(events, "ingestion.ctgov.completed") == ["sources/ctgov/NCT00000001.json"]
    assert str(metadata_ref).replace("\\", "/") == "sources/metadata/NCT00000001.json"


async def _async_value(value):
    return value


def _event_refs(events: list[dict], event_type: str) -> list[str]:
    matches = [event for event in events if event["event_type"] == event_type]
    assert len(matches) == 1
    return matches[0]["artifact_refs"]


def _single_event_ref(events: list[dict], event_type: str) -> Path:
    refs = _event_refs(events, event_type)
    assert len(refs) == 1
    return Path(refs[0])


def _single_event(events: list[dict], event_type: str, *, domain: str | None = None, sq_id: str | None = None) -> dict:
    matches = [
        event
        for event in events
        if event["event_type"] == event_type
        and (domain is None or event["domain"] == domain)
        and (sq_id is None or event["sq_id"] == sq_id)
    ]
    assert len(matches) == 1
    return matches[0]


def _events(bundle: QATraceBundle) -> list[dict]:
    return [json.loads(line) for line in (bundle.root / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def _bundle(tmp_path: Path) -> QATraceBundle:
    return QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )


def _raw(answer: str, quote: str) -> dict[str, str]:
    return {"answer": answer, "quote": quote, "justification": "The quoted text supports the answer."}


@pytest.mark.asyncio
async def test_full_trace_records_sq_finalization_with_verified_quote(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    quote = "The allocation sequence was random."

    await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial prefix.",
            "domain_context": DomainContext(domain="D1", domain_specific_text=quote),
            "sq_model": MockLLMClient(responses={"1.1|assignment": _raw("Y", quote)}),
            "raw_char_stream": quote,
            "page_boxes": [PageBox(boxclass="text", text=quote, bbox=(0, 0, 1, 1), page=2)],
            "section_map": SectionMap(source_path="paper.pdf", full_text=quote, sections=[], page_boxes=[]),
            "trace": type("Trace", (), {"qa_trace": bundle})(),
        }
    )
    bundle.close()

    event = _single_event(_events(bundle), "sq.finalized")
    artifact = json.loads((bundle.root / event["artifact_refs"][0]).read_text(encoding="utf-8"))
    assert artifact["raw_answer"]["answer"] == "Y"
    assert artifact["final_answer"]["answer"] == "Y"
    assert artifact["quote_verification"]["normalized_quote"] == quote.casefold()
    assert artifact["quote_verification"]["matched_source_document"] == "paper.pdf"
    assert artifact["quote_verification"]["matched_page"] == 2
    assert artifact["quote_verification"]["match_strategy"] == "partial_ratio"
    assert artifact["quote_verification"]["match_score"] >= 85
    assert artifact["quote_verification"]["failure_reason"] is None
    assert artifact["confidence_flag"] == "CONFIDENT"


@pytest.mark.asyncio
async def test_full_trace_records_sq_finalization_with_unverified_quote(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial prefix.",
            "domain_context": DomainContext(domain="D1", domain_specific_text="The allocation sequence was random."),
            "sq_model": MockLLMClient(responses={"1.1|assignment": _raw("Y", "This quote is not in the source.")}),
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [
                PageBox(boxclass="text", text="The allocation sequence was random.", bbox=(0, 0, 1, 1), page=2)
            ],
            "trace": type("Trace", (), {"qa_trace": bundle})(),
        }
    )
    bundle.close()

    artifact = json.loads((bundle.root / _single_event_ref(_events(bundle), "sq.finalized")).read_text(encoding="utf-8"))
    assert artifact["quote_verification"]["verified"] is False
    assert artifact["quote_verification"]["matched_page"] is None
    assert artifact["quote_verification"]["failure_reason"] == "quote did not meet verification threshold"
    assert artifact["final_answer"]["answer"] == "NI"
    assert artifact["confidence_flag"] == "FLAGGED"


@pytest.mark.asyncio
async def test_full_trace_records_structural_na_domain_judgment_and_rollup(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source = (
        "No deviations occurred. The analysis was appropriate. Follow-up was complete. "
        "Outcome assessors were masked. The endpoint was prespecified."
    )
    client = MockLLMClient(
        responses={
            "2.1|assignment": _raw("N", "No deviations occurred."),
            "2.2|assignment": _raw("N", "No deviations occurred."),
            "2.6|assignment": _raw("Y", "The analysis was appropriate."),
            "3.1|assignment": _raw("Y", "Follow-up was complete."),
            "4.1|assignment": _raw("N", "Outcome assessors were masked."),
            "4.2|assignment": _raw("N", "Outcome assessors were masked."),
            "4.3|assignment": _raw("N", "Outcome assessors were masked."),
            "5.1|assignment": _raw("Y", "The endpoint was prespecified."),
        }
    )
    config = AssessmentConfig(paper_path=tmp_path / "paper.pdf")
    ctx = TrialContext(
        config_summary={},
        trial_metadata=TrialMetadata(
            trial_id="T1",
            title="Trial",
            intervention="Drug",
            comparator="Placebo",
            primary_outcome="Overall survival",
            all_outcomes=["Overall survival"],
            effect_of_interest=EffectOfInterest.ASSIGNMENT,
            blinding=BlindingStatus.DOUBLE_BLIND,
        ),
        section_map=SectionMap(
            source_path="paper.pdf",
            full_text=source,
            sections=[],
            page_boxes=[PageBox(boxclass="text", text=source, bbox=(0, 0, 1, 1), page=0)],
        ),
        raw_char_stream=source,
        supplement_index=SupplementIndex.empty(),
        ct_gov_data=None,
        shared_prefix_text="Trial prefix.",
        ct_gov_block=None,
        llm_client_sq=client,
        llm_client_aux=MockLLMClient(),
    )
    state = {
        **base_ingestion_state(ctx, config),
        "outcome": "Overall survival",
        "trial_domain_judgments": [
            DomainJudgment(
                domain="D1",
                scope="trial",
                judgment=Judgment.LOW,
                algorithm_rationale="fixture",
                sq_answers=[SQAnswer(sq_id="1.1", answer=AnswerCode.Y)],
            )
        ],
        "domain_contexts": {},
        "sq_answers": {},
        "domain_judgments": [],
        "errors": [],
    }

    await build_outcome_graph().ainvoke(
        state,
        context=AssessmentRuntime(
            llm_client_sq=client,
            llm_client_aux=MockLLMClient(),
            supplement_index=SupplementIndex.empty(),
            trace=type("Trace", (), {"qa_trace": bundle})(),
        ),
    )
    bundle.close()

    events = _events(bundle)
    branching = _single_event(events, "branching.resolved", sq_id="2.3")
    assert branching["payload"]["structurally_na"] is True
    assert branching["payload"]["asked_sqs"] == ["2.1", "2.2", "2.6"]

    domain_event = _single_event(events, "judgment.domain.completed", domain="D2")
    assert domain_event["payload"]["input_sq_answers"]["2.3"] == "NA"
    assert domain_event["payload"]["output_judgment"] == "Low"

    rollup_event = _single_event(events, "judgment.overall.completed")
    assert rollup_event["payload"]["domain_judgments"]["D1"] == "Low"
    assert rollup_event["payload"]["rollup_policy"] == "ADR-0001"
    assert rollup_event["payload"]["output_judgment"] == "Low"
    assert rollup_event["payload"]["requires_human_review_basis"] is None


@pytest.mark.asyncio
async def test_full_trace_records_source_artifacts_for_batch_entry_ingestion(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper fixture", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper,nct_number,trial_label\npaper.pdf,NCT00000002,Trial 2\n", encoding="utf-8")
    config = AssessmentConfig(
        paper_path=manifest,
        output_dir=tmp_path / "out",
        db_path=tmp_path / "arbiter.db",
        trace_level="full",
    )
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="batch",
        cli_args=["batch", str(manifest)],
        config=config,
        input_manifest_path=manifest,
    )
    config.qa_trace = bundle
    section_map = SectionMap(
        source_path=str(paper),
        full_text="Batch paper parsed text.",
        sections=[
            DocumentSection(label="FULL_TEXT", pages=[0], char_start=0, char_end=24, text="Batch paper parsed text.")
        ],
        page_boxes=[],
        parsing_quality=ParsingQuality.DEGRADED,
        nct_number="NCT00000002",
    )

    monkeypatch.setattr(arbiter, "ingest_paper", lambda _path: (section_map, "batch raw stream"))
    monkeypatch.setattr(arbiter, "ingest_supplements", lambda *_args: _async_value(SupplementIndex.empty()))
    monkeypatch.setattr(arbiter, "fetch_ctgov", lambda _nct: _async_value({"protocolSection": {"id": "NCT00000002"}}))
    monkeypatch.setattr(
        arbiter,
        "create_llm_client",
        lambda *_args, **_kwargs: MockLLMClient(
            responses={
                "metadata": {
                    "title": "Batch trial",
                    "intervention": "Drug",
                    "comparator": "Placebo",
                    "primary_outcome": "Overall survival",
                    "all_outcomes": ["Overall survival"],
                    "blinding": BlindingStatus.UNCLEAR.value,
                    "nct_number": "NCT00000002",
                    "study_design": StudyDesign.CLUSTER_RCT.value,
                    "study_design_basis": "Cluster allocation reported.",
                }
            }
        ),
    )

    await run_batch(manifest, config)
    bundle.close()

    events = [json.loads(line) for line in (bundle.root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    main_ref = _single_event_ref(events, "ingestion.main_paper.completed")
    assert json.loads((bundle.root / main_ref).read_text(encoding="utf-8"))["full_text"] == "Batch paper parsed text."
    assert json.loads((bundle.root / "sources" / "ctgov" / "NCT00000002.json").read_text(encoding="utf-8"))[
        "protocolSection"
    ]["id"] == "NCT00000002"
    assert str(main_ref).replace("\\", "/").startswith("sources/main_paper/")
    assert _event_refs(events, "ingestion.ctgov.completed") == ["sources/ctgov/NCT00000002.json"]
