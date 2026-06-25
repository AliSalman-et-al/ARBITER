from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from arbiter.cli import cli
from arbiter.config import AssessmentConfig
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


def test_batch_cli_accepts_manifest_option_and_model_flags(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("main_paper\npaper.pdf\n", encoding="utf-8")
    captured: dict[str, AssessmentConfig] = {}

    async def fake_run_batch(manifest_path: Path, config: AssessmentConfig, progress_callback=None):
        from arbiter.manifest import BatchSummary

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
