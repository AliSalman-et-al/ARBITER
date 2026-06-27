"""Run-level QA trace bundle writer."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arbiter.config import AssessmentConfig, MODEL_REGISTRY

SCHEMA_VERSION = "1"


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def latest_pointer_path(base_dir: Path = Path("runs")) -> Path:
    """Location of the stable pointer to the most recent trace bundle."""

    return base_dir / "latest.txt"


def read_latest_pointer(base_dir: Path = Path("runs")) -> Path | None:
    """Return the most recent trace bundle root recorded in the latest pointer."""

    pointer = latest_pointer_path(base_dir)
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(text) if text else None


def _write_latest_pointer(base_dir: Path, root: Path) -> None:
    """Record the newest bundle root so `runs/latest.txt` always points at the live run.

    The run id is UTC-stamped, so sorting `runs/` by name or mtime is an unreliable
    way to find the active run from a local-time shell. This pointer gives engineers
    and agents a stable handle: `tail -f "$(cat runs/latest.txt)/events.jsonl"`.
    """

    try:
        latest_pointer_path(base_dir).write_text(f"{root}\n", encoding="utf-8")
    except OSError:
        pass


def create_qa_trace_bundle(
    config: AssessmentConfig,
    *,
    command: str,
    cli_args: list[str],
    input_manifest_path: Path | None = None,
    base_dir: Path = Path("runs"),
) -> "QATraceBundle | None":
    if config.trace_level != "full":
        return None
    return QATraceBundle.create(
        base_dir=base_dir,
        command=command,
        cli_args=cli_args,
        config=config,
        input_manifest_path=input_manifest_path,
    )


@dataclass
class QATraceBundle:
    run_id: str
    root: Path
    command: str
    cli_args: list[str]
    events_path: Path
    _events_handle: Any

    @classmethod
    def create(
        cls,
        *,
        base_dir: Path,
        command: str,
        cli_args: list[str],
        config: AssessmentConfig,
        input_manifest_path: Path | None = None,
    ) -> "QATraceBundle":
        run_id = generate_run_id()
        root = base_dir / run_id / "qa_trace"
        root.mkdir(parents=True, exist_ok=False)
        for dirname in [
            "sources",
            "llm_calls",
            "retrieval",
            "context",
            "quote_verification",
            "sq_answers",
            "judgments",
            "outputs",
            "artifacts",
        ]:
            (root / dirname).mkdir()
        events_path = root / "events.jsonl"
        events_handle = events_path.open("a", encoding="utf-8", buffering=1)
        _write_latest_pointer(base_dir, root)
        bundle = cls(
            run_id=run_id,
            root=root,
            command=command,
            cli_args=cli_args,
            events_path=events_path,
            _events_handle=events_handle,
        )
        bundle.write_json_artifact(
            "run_manifest.json",
            _manifest_payload(
                run_id=run_id,
                command=command,
                cli_args=cli_args,
                config=config,
                input_manifest_path=input_manifest_path,
            ),
        )
        return bundle

    def record_event(
        self,
        *,
        event_type: str,
        status: str,
        parent_event_id: str | None = None,
        trial_id: str | None = None,
        outcome: str | None = None,
        domain: str | None = None,
        sq_id: str | None = None,
        artifact_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "event_id": f"evt_{uuid.uuid4().hex}",
            "parent_event_id": parent_event_id,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
            "status": status,
            "trial_id": trial_id,
            "outcome": outcome,
            "domain": domain,
            "sq_id": sq_id,
            "artifact_refs": artifact_refs or [],
            "payload": payload or {},
        }
        self._events_handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._events_handle.flush()
        return event

    def write_json_artifact(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
        return path

    def write_source_artifact(
        self,
        relative_path: str | Path,
        payload: Any,
        *,
        event_type: str,
        status: str = "completed",
        trial_id: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> str:
        ref = _relative_ref(relative_path)
        self.write_json_artifact(ref, payload)
        self.record_event(
            event_type=event_type,
            status=status,
            trial_id=trial_id,
            artifact_refs=[ref],
            payload=event_payload,
        )
        return ref

    def close(self) -> None:
        self._events_handle.close()


def _manifest_payload(
    *,
    run_id: str,
    command: str,
    cli_args: list[str],
    config: AssessmentConfig,
    input_manifest_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "command": command,
        "cli_args": cli_args,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_at_local": datetime.now().astimezone().isoformat(),
        "arbiter_version": _package_version(),
        "pipeline_version": config.pipeline_version,
        "git": _git_metadata(),
        "inputs": {
            "manifest": _path_record(input_manifest_path) if input_manifest_path is not None else None,
            "paper": _path_record(config.paper_path),
            "supplements": [_path_record(path) for path in config.supplement_paths],
            "nct_number": config.nct_number,
            "trial_label": config.trial_label,
            "outcomes": config.outcomes,
            "effect_of_interest": config.effect_of_interest,
        },
        "models": {
            "sq": _model_record(config.sq_model),
            "aux": _model_record(config.aux_model),
            "vision": _model_record(config.vision_model) if config.vision_model else None,
        },
        "settings": {
            "sq_max_tokens": config.sq_max_tokens,
            "schema_repair_max_retries": config.env.schema_repair_max_retries,
            "network_max_retries": config.env.network_max_retries,
            "llm_request_timeout_s": config.env.llm_request_timeout_s,
            "max_concurrency": config.env.max_concurrency,
        },
        "trace_mode": config.trace_level,
        "outputs": {
            "output_dir": str(config.output_dir),
            "db_path": str(config.db_path),
        },
    }


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("arbiter")
    except Exception:
        return "0.1.0"


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "dirty": _git(["status", "--porcelain"]) not in {None, ""},
    }


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _path_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _sha256(path: Path) -> str | None:
    digest = __import__("hashlib").sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _relative_ref(path: str | Path) -> str:
    return Path(path).as_posix()


def _model_record(name: str) -> dict[str, Any]:
    info = MODEL_REGISTRY.get(name, {})
    return {
        "name": name,
        "provider": info.get("provider"),
        "model_id": info.get("model_id"),
        "supports_cache": info.get("supports_cache"),
        "supports_native_schema": info.get("supports_native_schema"),
        "supports_vision": info.get("supports_vision"),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.value
    return repr(value)
