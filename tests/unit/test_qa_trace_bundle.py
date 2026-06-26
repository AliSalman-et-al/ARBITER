from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from arbiter.config import AssessmentConfig
from arbiter.observability.qa_trace import QATraceBundle, create_qa_trace_bundle, generate_run_id


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
