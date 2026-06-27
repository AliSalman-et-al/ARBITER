from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

from arbiter.manifest import BatchSummary
from arbiter.output.sqlite_writer import write_assessment_sqlite
from tests.unit.test_output_writers import _assessment


def _load_run_eval():
    path = Path(__file__).resolve().parents[2] / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("arbiter_eval_run_eval", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rollup_normalized_uses_arbiter_policy() -> None:
    run_eval = _load_run_eval()

    assert run_eval.rollup_normalized(["Low", "Some concerns", "Low", "Low", "Low"]) == "Some concerns"
    assert run_eval.rollup_normalized(["Low", "Some concerns", "Some concerns", "Some concerns", "Low"]) == "High"
    assert run_eval.rollup_normalized(["Low", "High", "Low", "Low", "Low"]) == "High"


def test_agreement_report_includes_triad_and_confusion_matrix() -> None:
    run_eval = _load_run_eval()

    report = run_eval.agreement_report(
        [("Low", "Low"), ("Some concerns", "Low"), ("High", "High")],
        labels=["Low", "Some concerns", "High"],
    )

    assert report["n"] == 3
    assert report["percent_agreement"] == 2 / 3
    assert report["cohen_kappa"] is not None
    assert report["gwet_ac2"] is not None
    assert report["confusion_matrix"]["Some concerns"]["Low"] == 1


def test_derive_pipeline_version_changes_when_aux_model_changes() -> None:
    run_eval = _load_run_eval()

    open_arm = run_eval.EvalArm(name="a", sq_model="same", aux_model="open")
    frontier_arm = run_eval.EvalArm(name="b", sq_model="same", aux_model="frontier")

    assert run_eval.derive_pipeline_version("0.1.0", open_arm) != run_eval.derive_pipeline_version(
        "0.1.0", frontier_arm
    )


def test_smoke_arm_uses_native_schema_free_default() -> None:
    run_eval = _load_run_eval()

    arm = run_eval.smoke_arm()

    assert arm.name == "smoke-free-nemotron-3-super-120b-a12b"
    assert arm.sq_model == "nemotron-3-super-120b-a12b-free"
    assert arm.aux_model == "nemotron-3-super-120b-a12b-free"
    assert arm.execution_mode == "dev-free-tier"


def test_load_predictions_joins_sqlite_rows_to_manifest_trial_labels(tmp_path: Path) -> None:
    run_eval = _load_run_eval()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "trial_label,main_paper,supplements,nct_number,outcomes\n"
        "Trial A,paper.pdf,,NCT00000001,Overall Survival\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "arbiter.db"
    write_assessment_sqlite(
        _assessment().model_copy(update={"trial_id": "NCT00000001", "outcome": "Overall Survival"}),
        db_path,
        json_path=tmp_path / "data.json",
    )

    predictions = run_eval.load_predictions(db_path, manifest)

    assert len(predictions) == 1
    assert predictions[0].trial_label == "Trial A"
    assert predictions[0].domain_pred["D1"] == "Low"


def test_run_smoke_eval_uses_arm_pipeline_version(monkeypatch, tmp_path: Path) -> None:
    run_eval = _load_run_eval()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.csv").write_text(
        "trial_label,main_paper,supplements,nct_number,outcomes\n"
        "Trial A,paper.pdf,,NCT00000001,Overall Survival\n",
        encoding="utf-8",
    )
    for name in ["overall_survival.csv", "progression_free_survival.csv", "adverse_events.csv"]:
        (reference / name).write_text("Trial,D1,D2,D3,D4,D5,Overall Risk\n", encoding="utf-8")
    (reference / "overall_survival.csv").write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTrial A,L,L,L,L,L,Low\n",
        encoding="utf-8",
    )

    async def fake_run_batch(_manifest, config, progress_callback=None):
        assert config.pipeline_version.startswith("0.1.0+eval.")
        if progress_callback is not None:
            progress_callback("[1] Trial A: assessed 1 pair(s), skipped 0")
        write_assessment_sqlite(
            _assessment().model_copy(
                update={
                    "trial_id": "NCT00000001",
                    "outcome": "Overall Survival",
                    "pipeline_version": config.pipeline_version,
                    "model_sq": config.sq_model,
                    "model_aux": config.aux_model,
                }
            ),
            config.db_path,
            json_path=config.output_dir / "NCT00000001" / "overall_survival__assignment" / "data.json",
        )
        return BatchSummary(processed_entries=1, assessed_pairs=1)

    report = run_eval.asyncio.run(
        run_eval.run_smoke_eval(reference_dir=reference, run_id="test-run", run_batch_fn=fake_run_batch)
    )

    assert report["dev_only"] is True
    assert report["n_joined"] == 1
    assert report["overall_rollup_normalized"]["percent_agreement"] == 1.0
    with sqlite3.connect(reference / "runs" / "test-run" / "arbiter.db") as conn:
        row = conn.execute("SELECT pipeline_version FROM arbiter_assessments").fetchone()
    assert row[0] == report["pipeline_version"]
