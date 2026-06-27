from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from arbiter.cli import cli
from arbiter.config import AssessmentConfig
from arbiter.manifest import BatchSummary
from arbiter.models import Judgment
from tests.unit.test_manifest_batch import _ctx
from tests.unit.test_output_writers import _assessment


def test_assess_cli_accepts_req_18_flags(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    supplement = tmp_path / "supplement.pdf"
    paper.write_text("paper", encoding="utf-8")
    supplement.write_text("sap", encoding="utf-8")
    captured: dict[str, AssessmentConfig] = {}

    async def fake_ingest(config: AssessmentConfig):
        captured["config"] = config
        return _ctx(config)

    async def fake_assess(_ctx, _config):
        return [_assessment().model_copy(update={"overall_judgment": Judgment.LOW})]

    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.assess_trial", fake_assess)

    result = CliRunner().invoke(
        cli,
        [
            "assess",
            "--paper",
            str(paper),
            "--supplement",
            str(supplement),
            "--nct",
            "NCT01234567",
            "--outcome",
            "Overall Survival",
            "--effect",
            "adhering",
            "--sq-model",
            "sq-test",
            "--aux-model",
            "aux-test",
            "--output-dir",
            str(tmp_path / "out"),
            "--db",
            str(tmp_path / "arbiter.db"),
            "--trace-level",
            "summary",
            "--no-report",
        ],
    )

    assert result.exit_code == 0
    assert "Trial trial-1" in result.output
    assert "Overall Survival: Low" in result.output
    config = captured["config"]
    assert config.nct_number == "NCT01234567"
    assert config.supplement_paths == [supplement]
    assert config.outcomes == ["Overall Survival"]
    assert config.effect_of_interest == "adhering"
    assert config.sq_model == "sq-test"
    assert config.aux_model == "aux-test"
    assert config.report_enabled is False


def test_assess_cli_full_trace_creates_run_level_qa_bundle(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")

    async def fake_ingest(config: AssessmentConfig):
        return _ctx(config)

    async def fake_assess(_ctx, _config):
        return [_assessment().model_copy(update={"overall_judgment": Judgment.LOW})]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.assess_trial", fake_assess)

    result = CliRunner().invoke(
        cli,
        [
            "assess",
            "--paper",
            str(paper),
            "--trace",
            "full",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0
    roots = list((tmp_path / "runs").glob("*/qa_trace"))
    assert len(roots) == 1
    assert (roots[0] / "run_manifest.json").exists()
    assert (roots[0] / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2
    events = [json.loads(line) for line in (roots[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == ["run.started", "run.completed"]


def test_assess_cli_summary_and_off_do_not_create_qa_bundle(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")

    async def fake_ingest(config: AssessmentConfig):
        return _ctx(config)

    async def fake_assess(_ctx, _config):
        return [_assessment().model_copy(update={"overall_judgment": Judgment.LOW})]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.assess_trial", fake_assess)

    summary = CliRunner().invoke(cli, ["assess", "--paper", str(paper), "--trace", "summary"])
    off = CliRunner().invoke(cli, ["assess", "--paper", str(paper), "--trace", "off"])

    assert summary.exit_code == 0
    assert off.exit_code == 0
    assert not (tmp_path / "runs").exists()


def test_assess_cli_full_trace_write_failure_fails_run(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")

    async def fake_ingest(_config: AssessmentConfig):
        raise AssertionError("assessment should not start after trace setup failure")

    def fail_create(*_args, **_kwargs):
        raise OSError("trace disk unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.create_qa_trace_bundle", fail_create)

    result = CliRunner().invoke(cli, ["assess", "--paper", str(paper), "--trace", "full"])

    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)
    assert "trace disk unavailable" in str(result.exception)


def test_batch_cli_accepts_manifest_option_and_model_flags(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper\npaper.pdf\n", encoding="utf-8")
    captured: dict[str, AssessmentConfig] = {}

    async def fake_run_batch(manifest_path: Path, config: AssessmentConfig, progress_callback=None):
        captured["config"] = config
        assert manifest_path == manifest
        if progress_callback is not None:
            progress_callback("[1] trial-1: skipped")
        return BatchSummary(processed_entries=1, skipped_entries=1)

    monkeypatch.setattr("arbiter.cli.run_batch", fake_run_batch)

    result = CliRunner().invoke(
        cli,
        [
            "batch",
            "--manifest",
            str(manifest),
            "--sq-model",
            "sq-test",
            "--aux-model",
            "aux-test",
            "--max-concurrency",
            "7",
            "--db",
            str(tmp_path / "arbiter.db"),
        ],
    )

    assert result.exit_code == 0
    assert "[1] trial-1: skipped" in result.output
    assert "Processed 1 entries" in result.output
    config = captured["config"]
    assert config.sq_model == "sq-test"
    assert config.aux_model == "aux-test"
    assert config.env.max_concurrency == 7


def test_batch_cli_full_trace_creates_one_bundle_for_batch(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper\npaper.pdf\n", encoding="utf-8")

    async def fake_run_batch(_manifest_path: Path, _config: AssessmentConfig, progress_callback=None):
        if progress_callback is not None:
            progress_callback("[1] trial-1: skipped")
        return BatchSummary(processed_entries=1, skipped_entries=1)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.run_batch", fake_run_batch)

    result = CliRunner().invoke(cli, ["batch", str(manifest), "--trace", "full"])

    assert result.exit_code == 0
    roots = list((tmp_path / "runs").glob("*/qa_trace"))
    assert len(roots) == 1
    assert (roots[0] / "run_manifest.json").exists()
    assert (roots[0] / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_assess_full_trace_announces_path_and_writes_latest_pointer(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")

    async def fake_ingest(config: AssessmentConfig):
        return _ctx(config)

    async def fake_assess(_ctx, _config):
        return [_assessment().model_copy(update={"overall_judgment": Judgment.LOW})]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.assess_trial", fake_assess)

    result = CliRunner().invoke(cli, ["assess", "--paper", str(paper), "--trace", "full"])

    assert result.exit_code == 0
    assert "QA trace:" in result.output
    assert "run_id" in result.output
    pointer = tmp_path / "runs" / "latest.txt"
    root = pointer.read_text(encoding="utf-8").strip()
    assert (tmp_path / "runs").glob("*/qa_trace")
    assert Path(root).name == "qa_trace"
    assert (Path(root) / "run_manifest.json").exists()


def test_trace_path_command_prints_latest_bundle(monkeypatch, tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper", encoding="utf-8")

    async def fake_ingest(config: AssessmentConfig):
        return _ctx(config)

    async def fake_assess(_ctx, _config):
        return [_assessment().model_copy(update={"overall_judgment": Judgment.LOW})]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.ingest_trial", fake_ingest)
    monkeypatch.setattr("arbiter.cli.assess_trial", fake_assess)
    CliRunner().invoke(cli, ["assess", "--paper", str(paper), "--trace", "full"])

    result = CliRunner().invoke(cli, ["trace", "path"])
    assert result.exit_code == 0
    printed = result.output.strip().splitlines()[-1]
    assert Path(printed).name == "qa_trace"
    assert (Path(printed) / "events.jsonl").exists()


def test_trace_path_command_errors_without_pointer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["trace", "path"])
    assert result.exit_code != 0
    assert "No trace bundle pointer" in result.output


def test_batch_cli_summary_and_off_do_not_create_qa_bundle(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper\npaper.pdf\n", encoding="utf-8")

    async def fake_run_batch(_manifest_path: Path, _config: AssessmentConfig, progress_callback=None):
        return BatchSummary(processed_entries=1, skipped_entries=1)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("arbiter.cli.run_batch", fake_run_batch)

    summary = CliRunner().invoke(cli, ["batch", str(manifest), "--trace", "summary"])
    off = CliRunner().invoke(cli, ["batch", str(manifest), "--trace", "off"])

    assert summary.exit_code == 0
    assert off.exit_code == 0
    assert not (tmp_path / "runs").exists()
