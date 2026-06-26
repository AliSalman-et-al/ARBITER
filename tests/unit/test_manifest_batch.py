from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from arbiter.config import AssessmentConfig
from arbiter.graph.state import TrialContext
from arbiter.manifest import BatchSummary, check_eligibility, load_manifest, run_batch
from arbiter.models import BlindingStatus, EffectOfInterest, StudyDesign, TrialMetadata
from arbiter.observability.qa_trace import QATraceBundle
from arbiter.retrieval.supplement_index import SupplementIndex
from tests.unit.test_output_writers import _assessment


def test_load_manifest_parses_csv_lists_and_paths(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("NCT00000001", encoding="utf-8")
    supplements = tmp_path / "supplements"
    supplements.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "main_paper,supplements,nct_number,outcomes,trial_label\n"
        '"paper.pdf","supplements",NCT00000001,"Overall Survival;Adverse Events",Trial A\n',
        encoding="utf-8",
    )

    loaded = load_manifest(manifest)

    assert loaded.entries[0].main_paper == paper
    assert loaded.entries[0].supplements == [supplements]
    assert loaded.entries[0].outcomes == ["Overall Survival", "Adverse Events"]


@pytest.mark.asyncio
async def test_run_batch_skips_completed_enumerated_entry_before_ingestion(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "main_paper,trial_label,outcomes\npaper.pdf,Trial 1,Overall Survival\n",
        encoding="utf-8",
    )
    config = AssessmentConfig(paper_path=manifest, output_dir=tmp_path / "out", db_path=tmp_path / "arbiter.db")
    assessment = _assessment().model_copy(
        update={"trial_id": "trial-1", "outcome": "Overall Survival", "model_sq": config.sq_model}
    )
    from arbiter.output.sqlite_writer import write_assessment_sqlite

    write_assessment_sqlite(assessment, config.db_path, json_path=tmp_path / "data.json")

    async def fail_ingest(_config):
        raise AssertionError("ingest should not be called")

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fail_ingest)

    summary = await run_batch(manifest, config)

    assert summary == BatchSummary(processed_entries=1, skipped_entries=1, skipped_pairs=1)


@pytest.mark.asyncio
async def test_run_batch_ingests_once_and_assesses_only_missing_outcomes(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "main_paper,trial_label,outcomes\npaper.pdf,Trial 1,Overall Survival;Adverse Events\n",
        encoding="utf-8",
    )
    config = AssessmentConfig(paper_path=manifest, output_dir=tmp_path / "out", db_path=tmp_path / "arbiter.db")
    from arbiter.output.sqlite_writer import write_assessment_sqlite

    write_assessment_sqlite(
        _assessment().model_copy(update={"trial_id": "trial-1", "model_sq": config.sq_model}),
        config.db_path,
    )
    calls: dict[str, Any] = {"ingest": 0, "outcomes": None}

    async def fake_ingest(entry_config):
        calls["ingest"] += 1
        return _ctx(entry_config)

    async def fake_assess(_ctx, entry_config):
        calls["outcomes"] = entry_config.outcomes
        return [_assessment().model_copy(update={"outcome": "Adverse Events"})]

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.manifest.assess_trial", fake_assess)

    summary = await run_batch(manifest, config)

    assert calls == {"ingest": 1, "outcomes": ["Adverse Events"]}
    assert summary.assessed_pairs == 1
    assert summary.skipped_pairs == 1


@pytest.mark.asyncio
async def test_run_batch_writes_skip_record_for_ineligible_trial(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper,trial_label\npaper.pdf,Trial 1\n", encoding="utf-8")
    config = AssessmentConfig(paper_path=manifest, output_dir=tmp_path / "out", db_path=tmp_path / "arbiter.db")

    async def fake_ingest(entry_config):
        ctx = _ctx(entry_config)
        metadata = ctx.trial_metadata.model_copy(
            update={"study_design": StudyDesign.CLUSTER_RCT, "study_design_basis": "Cluster randomisation."}
        )
        return replace(ctx, trial_metadata=metadata)

    async def fail_assess(_ctx, _config):
        raise AssertionError("ineligible trial should not be assessed")

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.manifest.assess_trial", fail_assess)

    await run_batch(manifest, config)

    assert (tmp_path / "out" / "trial-1" / "skip.json").exists()
    with sqlite3.connect(config.db_path) as conn:
        row = conn.execute("SELECT outcome, overall_judgment, requires_human_review FROM arbiter_assessments").fetchone()
    assert row == ("__TRIAL__", None, 1)


@pytest.mark.asyncio
async def test_run_batch_continues_after_corrupt_entry(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "main_paper,trial_label,outcomes\nmissing.pdf,Bad,Overall Survival\nmissing2.pdf,Good,Overall Survival\n",
        encoding="utf-8",
    )
    config = AssessmentConfig(paper_path=manifest, output_dir=tmp_path / "out", db_path=tmp_path / "arbiter.db")
    seen: list[str | None] = []

    async def fake_ingest(entry_config):
        seen.append(entry_config.trial_label)
        if entry_config.trial_label == "Bad":
            raise RuntimeError("bad pdf")
        return _ctx(entry_config)

    async def fake_assess(_ctx, _config):
        return [_assessment()]

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.manifest.assess_trial", fake_assess)

    summary = await run_batch(manifest, config)

    assert seen == ["Bad", "Good"]
    assert summary.error_count == 1
    assert summary.assessed_pairs == 1


@pytest.mark.asyncio
async def test_run_batch_full_trace_records_error_and_skipped_entries(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "main_paper,trial_label,outcomes\nbad.pdf,Bad,Overall Survival\nskip.pdf,Skip,Overall Survival\n",
        encoding="utf-8",
    )
    config = AssessmentConfig(
        paper_path=manifest,
        output_dir=tmp_path / "out",
        db_path=tmp_path / "arbiter.db",
        trace_level="full",
    )
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="batch",
        cli_args=["batch", str(manifest), "--trace", "full"],
        config=config,
        input_manifest_path=manifest,
    )
    config.qa_trace = bundle

    async def fake_ingest(entry_config):
        if entry_config.trial_label == "Bad":
            raise RuntimeError("bad pdf")
        ctx = _ctx(entry_config)
        metadata = ctx.trial_metadata.model_copy(
            update={"study_design": StudyDesign.CLUSTER_RCT, "study_design_basis": "Cluster randomisation."}
        )
        return replace(ctx, trial_metadata=metadata)

    async def fail_assess(_ctx, _config):
        raise AssertionError("bad or ineligible entries should not be assessed")

    monkeypatch.setattr("arbiter.manifest.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.manifest.assess_trial", fail_assess)

    summary = await run_batch(manifest, config)
    bundle.close()

    events = [json.loads(line) for line in bundle.events_path.read_text(encoding="utf-8").splitlines()]
    error_event = _single_event(events, "batch.entry.error")
    skipped_event = _single_event(events, "batch.entry.skipped")
    assert summary.error_count == 1
    assert summary.processed_entries == 2
    assert error_event["status"] == "failed"
    assert error_event["payload"]["trial_label"] == "Bad"
    assert error_event["payload"]["error"] == "RuntimeError: bad pdf"
    assert skipped_event["status"] == "skipped"
    assert skipped_event["trial_id"] == "trial-1"
    assert skipped_event["payload"]["reason"] == "ineligible"


def _single_event(events: list[dict], event_type: str) -> dict:
    matches = [event for event in events if event["event_type"] == event_type]
    assert len(matches) == 1
    return matches[0]


def test_check_eligibility_skips_positive_out_of_scope_design() -> None:
    metadata = _metadata(study_design=StudyDesign.CLUSTER_RCT)
    config = AssessmentConfig(paper_path=Path("paper.pdf"))

    skip = check_eligibility(metadata, config)

    assert skip is not None
    assert skip.trial_id == "trial-1"
    assert skip.errors == ["ineligible study_design=cluster_rct: positive metadata evidence indicates an out-of-scope design."]


def test_check_eligibility_allows_unclear_metadata_when_registry_confirms_parallel_rct() -> None:
    metadata = _metadata(study_design=StudyDesign.UNCLEAR).model_copy(update={"study_design_basis": None})
    config = AssessmentConfig(paper_path=Path("paper.pdf"))
    ct_gov_data = {
        "protocolSection": {
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL"},
            }
        }
    }

    skip = check_eligibility(metadata, config, ct_gov_data=ct_gov_data)

    assert skip is None


def test_check_eligibility_fails_open_on_uncertainty_without_registry_or_paper_signal() -> None:
    metadata = _metadata(study_design=StudyDesign.UNCLEAR)
    config = AssessmentConfig(paper_path=Path("paper.pdf"))

    skip = check_eligibility(metadata, config, raw_char_stream="Design details were not available in the excerpt.")

    assert skip is None


def _ctx(config: AssessmentConfig) -> TrialContext:
    from arbiter.llm.mock_client import MockLLMClient
    from arbiter.models import DocumentSection, ParsingQuality, SectionMap

    section_map = SectionMap(
        source_path=str(config.paper_path),
        full_text="paper",
        sections=[DocumentSection(label="FULL_TEXT", pages=[0], char_start=0, char_end=5, text="paper")],
        page_boxes=[],
        parsing_quality=ParsingQuality.DEGRADED,
    )
    return TrialContext(
        config_summary={"inputs_hash": "hash-1"},
        trial_metadata=_metadata(),
        section_map=section_map,
        raw_char_stream="paper",
        supplement_index=SupplementIndex.empty(),
        ct_gov_data=None,
        shared_prefix_text="prefix",
        ct_gov_block="",
        llm_client_sq=MockLLMClient(responses={}),
        llm_client_aux=MockLLMClient(responses={}),
        trace=None,
    )


def _metadata(study_design: StudyDesign = StudyDesign.PARALLEL_RCT) -> TrialMetadata:
    return TrialMetadata(
        trial_id="trial-1",
        title="Trial title",
        intervention="Drug",
        comparator="Placebo",
        primary_outcome="Overall Survival",
        all_outcomes=["Overall Survival", "Adverse Events"],
        effect_of_interest=EffectOfInterest.ASSIGNMENT,
        blinding=BlindingStatus.DOUBLE_BLIND,
        nct_number="NCT00000001",
        study_design=study_design,
    )
