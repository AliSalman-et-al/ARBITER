"""Batch manifest parsing and unattended runner."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, cast
from uuid import uuid4

import pymupdf
from pydantic import BaseModel, Field, field_validator

from arbiter import assess_trial, ingest_trial
from arbiter.config import AssessmentConfig
from arbiter.eligibility import decide_eligibility
from arbiter.ingestion.metadata_extractor import build_trial_id, normalize_nct, slugify
from arbiter.models import SkipRecord, StudyDesign, TrialMetadata
from arbiter.output.sqlite_writer import SKIP_EFFECT, SKIP_OUTCOME, write_skip_record

NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


class ManifestEntry(BaseModel):
    """One manifest row."""

    main_paper: Path
    supplements: list[Path] = Field(default_factory=list)
    nct_number: str | None = None
    outcomes: list[str] | None = None
    trial_label: str | None = None

    @field_validator("nct_number")
    @classmethod
    def normalize_nct_number(cls, value: str | None) -> str | None:
        return normalize_nct(value)


class BatchManifest(BaseModel):
    entries: list[ManifestEntry]


class BatchSummary(BaseModel):
    processed_entries: int = 0
    skipped_entries: int = 0
    assessed_pairs: int = 0
    skipped_pairs: int = 0
    error_count: int = 0
    total_wall_time_s: float = 0.0
    total_llm_latency_s: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int | None = 0
    total_cost: float | None = 0.0
    slowest_trials: list[dict[str, Any]] = Field(default_factory=list)


def load_manifest(path: Path) -> BatchManifest:
    """Load a CSV or JSON batch manifest."""

    base_dir = path.parent
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("entries", payload) if isinstance(payload, dict) else payload
        return BatchManifest(entries=[_entry_from_mapping(row, base_dir) for row in rows])

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return BatchManifest(entries=[_entry_from_mapping(row, base_dir) for row in rows])


async def run_batch(
    manifest_path: Path,
    base_config: AssessmentConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> BatchSummary:
    """Run all manifest entries, continuing after per-entry failures."""

    manifest = load_manifest(manifest_path)
    summary = BatchSummary()
    for index, entry in enumerate(manifest.entries, start=1):
        summary.processed_entries += 1
        try:
            result = await _run_entry(entry, base_config)
            summary.skipped_entries += int(result["entry_skipped"])
            summary.assessed_pairs += int(result["assessed_pairs"])
            summary.skipped_pairs += int(result["skipped_pairs"])
            if timing := result.get("timing_summary"):
                _merge_timing(summary, str(result.get("trial_id") or f"entry-{index}"), timing)
            if progress_callback is not None:
                progress_callback(_progress_line(index, entry, result))
        except Exception as exc:
            summary.error_count += 1
            _write_manifest_error(entry, index, exc, base_config)
            _record_batch_entry_event(
                base_config,
                event_type="batch.entry.error",
                status="failed",
                entry=entry,
                index=index,
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
            if progress_callback is not None:
                progress_callback(f"[{index}] {entry.trial_label or entry.main_paper.name}: error {type(exc).__name__}")
    summary.slowest_trials = sorted(summary.slowest_trials, key=lambda item: item["wall_time_s"], reverse=True)[:5]
    return summary


def check_eligibility(
    trial_metadata: TrialMetadata,
    config: AssessmentConfig,
    *,
    ct_gov_data: dict | None = None,
    section_map: Any | None = None,
    raw_char_stream: str | None = None,
) -> SkipRecord | None:
    """Return a skip artifact for trials outside the v0.1 parallel-RCT scope."""

    decision = decide_eligibility(
        trial_metadata,
        ct_gov_data=ct_gov_data,
        section_map=section_map,
        raw_char_stream=raw_char_stream,
    )
    if decision.eligible:
        return None
    return SkipRecord(
        assessment_id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        trial_id=trial_metadata.trial_id,
        nct_number=trial_metadata.nct_number,
        study_design=decision.study_design,
        study_design_basis=decision.basis,
        model_sq=config.sq_model,
        model_aux=config.aux_model,
        pipeline_version=config.pipeline_version,
        errors=[f"ineligible study_design={decision.study_design.value}: {decision.basis}"],
    )


def completed_pair_exists(
    db_path: Path,
    *,
    trial_id: str,
    outcome: str,
    effect_of_interest: str,
    model_sq: str,
    pipeline_version: str = "0.1.0",
) -> bool:
    return _row_exists(db_path, trial_id, outcome, effect_of_interest, model_sq, pipeline_version)


def skip_row_exists(
    db_path: Path,
    *,
    trial_id: str,
    model_sq: str,
    pipeline_version: str = "0.1.0",
) -> bool:
    return _row_exists(db_path, trial_id, SKIP_OUTCOME, SKIP_EFFECT, model_sq, pipeline_version)


async def _run_entry(entry: ManifestEntry, base_config: AssessmentConfig) -> dict[str, Any]:
    config = _config_for_entry(entry, base_config)
    cheap_trial_id = _cheap_trial_id(entry)
    if cheap_trial_id and not config.force:
        if skip_row_exists(
            config.db_path,
            trial_id=cheap_trial_id,
            model_sq=config.sq_model,
            pipeline_version=config.pipeline_version,
        ):
            _record_batch_entry_event(
                config,
                event_type="batch.entry.skipped",
                status="skipped",
                entry=entry,
                payload={"reason": "existing_skip_record", "trial_id": cheap_trial_id},
            )
            return {"entry_skipped": True, "assessed_pairs": 0, "skipped_pairs": 0, "trial_id": cheap_trial_id}
        if entry.outcomes and all(
            completed_pair_exists(
                config.db_path,
                trial_id=cheap_trial_id,
                outcome=outcome,
                effect_of_interest=config.effect_of_interest,
                model_sq=config.sq_model,
                pipeline_version=config.pipeline_version,
            )
            for outcome in entry.outcomes
        ):
            _record_batch_entry_event(
                config,
                event_type="batch.entry.skipped",
                status="skipped",
                entry=entry,
                payload={"reason": "all_requested_pairs_completed", "trial_id": cheap_trial_id},
            )
            return {
                "entry_skipped": True,
                "assessed_pairs": 0,
                "skipped_pairs": len(entry.outcomes),
                "trial_id": cheap_trial_id,
            }

    ctx = await ingest_trial(config)
    skip = check_eligibility(
        ctx.trial_metadata,
        config,
        ct_gov_data=ctx.ct_gov_data,
        section_map=ctx.section_map,
        raw_char_stream=ctx.raw_char_stream,
    )
    if skip is not None:
        skip = skip.model_copy(update={"inputs_hash": ctx.config_summary.get("inputs_hash")})
        write_skip_record(skip, config.output_dir, config.db_path)
        _record_batch_entry_event(
            config,
            event_type="batch.entry.skipped",
            status="skipped",
            entry=entry,
            payload={
                "reason": "ineligible",
                "trial_id": ctx.trial_metadata.trial_id,
                "errors": skip.errors,
            },
        )
        return {
            "entry_skipped": False,
            "assessed_pairs": 0,
            "skipped_pairs": 0,
            "trial_id": ctx.trial_metadata.trial_id,
            "timing_summary": _trace_timing_summary(ctx.trace),
        }

    outcomes = list(config.outcomes or ctx.trial_metadata.all_outcomes or [ctx.trial_metadata.primary_outcome])
    missing = [
        outcome
        for outcome in outcomes
        if config.force
        or not completed_pair_exists(
            config.db_path,
            trial_id=ctx.trial_metadata.trial_id,
            outcome=outcome,
            effect_of_interest=config.effect_of_interest,
            model_sq=config.sq_model,
            pipeline_version=config.pipeline_version,
        )
    ]
    if not missing:
        _record_batch_entry_event(
            config,
            event_type="batch.entry.skipped",
            status="skipped",
            entry=entry,
            payload={"reason": "all_pairs_completed", "trial_id": ctx.trial_metadata.trial_id},
        )
        return {
            "entry_skipped": False,
            "assessed_pairs": 0,
            "skipped_pairs": len(outcomes),
            "trial_id": ctx.trial_metadata.trial_id,
            "timing_summary": _trace_timing_summary(ctx.trace),
        }

    assessments = await assess_trial(ctx, replace(config, outcomes=missing))
    return {
        "entry_skipped": False,
        "assessed_pairs": len(assessments),
        "skipped_pairs": len(outcomes) - len(missing),
        "trial_id": ctx.trial_metadata.trial_id,
        "timing_summary": _trace_timing_summary(ctx.trace),
    }


def _progress_line(index: int, entry: ManifestEntry, result: dict[str, Any]) -> str:
    label = str(result.get("trial_id") or entry.trial_label or entry.main_paper.name)
    if result.get("entry_skipped"):
        return f"[{index}] {label}: skipped"
    if int(result.get("assessed_pairs") or 0) == 0 and int(result.get("skipped_pairs") or 0) == 0:
        return f"[{index}] {label}: ineligible"
    return (
        f"[{index}] {label}: assessed {int(result.get('assessed_pairs') or 0)} pair(s), "
        f"skipped {int(result.get('skipped_pairs') or 0)}"
    )


def _trace_timing_summary(trace: object | None) -> dict[str, Any] | None:
    if trace is None or not hasattr(trace, "timing_summary"):
        return None
    return cast(Any, trace).timing_summary()


def _merge_timing(summary: BatchSummary, trial_id: str, timing: dict[str, Any]) -> None:
    wall_time = float(timing.get("wall_time_s") or 0.0)
    summary.total_wall_time_s += wall_time
    summary.total_llm_latency_s += float(timing.get("llm_latency_s") or 0.0)
    summary.total_llm_calls += int(timing.get("llm_call_count") or 0)
    token_count = _timing_token_count(timing)
    if token_count is None:
        summary.total_tokens = None
    elif summary.total_tokens is not None:
        summary.total_tokens += token_count
    if timing.get("pricing_unknown") or timing.get("total_cost") is None:
        summary.total_cost = None
    elif summary.total_cost is not None:
        summary.total_cost += float(timing["total_cost"])
    summary.slowest_trials.append({"trial_id": trial_id, "wall_time_s": wall_time})


def _timing_token_count(timing: dict[str, Any]) -> int | None:
    values = [
        timing.get("input_token_count"),
        timing.get("output_token_count"),
        timing.get("cache_read_token_count"),
        timing.get("cache_write_token_count"),
    ]
    known = [int(value) for value in values if value is not None]
    return sum(known) if known else None


def _entry_from_mapping(row: dict[str, Any], base_dir: Path) -> ManifestEntry:
    return ManifestEntry(
        main_paper=_resolve_path(str(row.get("main_paper") or ""), base_dir),
        supplements=_parse_supplements(row.get("supplements"), base_dir),
        nct_number=_clean(row.get("nct_number")),
        outcomes=_parse_outcomes(row.get("outcomes")),
        trial_label=_clean(row.get("trial_label")),
    )


def _config_for_entry(entry: ManifestEntry, base_config: AssessmentConfig) -> AssessmentConfig:
    supplements = _expand_supplements(entry.supplements)
    return replace(
        base_config,
        paper_path=entry.main_paper,
        supplement_paths=supplements,
        nct_number=entry.nct_number,
        trial_label=entry.trial_label,
        outcomes=entry.outcomes,
    )


def _cheap_trial_id(entry: ManifestEntry) -> str | None:
    nct_number = entry.nct_number or _scan_pdf_for_nct(entry.main_paper)
    if nct_number:
        return nct_number
    if entry.trial_label and (slug := slugify(entry.trial_label)):
        return slug
    if entry.main_paper.exists():
        return build_trial_id(nct_number=None, trial_label=None, paper_path=entry.main_paper, fallback_text="")
    return None


def _scan_pdf_for_nct(path: Path) -> str | None:
    try:
        with pymupdf.open(path) as doc:
            for page_index in range(len(doc)):
                match = NCT_PATTERN.search(doc.load_page(page_index).get_text())
                if match:
                    return match.group(0).upper()
    except Exception:
        try:
            match = NCT_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore"))
            return match.group(0).upper() if match else None
        except OSError:
            return None
    return None


def _row_exists(
    db_path: Path,
    trial_id: str,
    outcome: str,
    effect_of_interest: str,
    model_sq: str,
    pipeline_version: str,
) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        try:
            row = conn.execute(
                """
                SELECT 1 FROM arbiter_assessments
                WHERE trial_id = ? AND outcome = ? AND effect_of_interest = ?
                  AND model_sq = ? AND pipeline_version = ?
                LIMIT 1
                """,
                (trial_id, outcome, effect_of_interest, model_sq, pipeline_version),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
    return row is not None


def _write_manifest_error(entry: ManifestEntry, index: int, exc: Exception, config: AssessmentConfig) -> None:
    trial_id = _cheap_trial_id(entry) or f"manifest-entry-{index}"
    skip = SkipRecord(
        assessment_id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        trial_id=trial_id,
        nct_number=entry.nct_number,
        study_design=StudyDesign.UNCLEAR,
        study_design_basis="manifest entry failed before assessment",
        model_sq=config.sq_model,
        model_aux=config.aux_model,
        pipeline_version=config.pipeline_version,
        errors=[f"manifest entry {index} failed: {type(exc).__name__}: {exc}"],
    )
    write_skip_record(skip, config.output_dir, config.db_path)


def _record_batch_entry_event(
    config: AssessmentConfig,
    *,
    event_type: str,
    status: str,
    entry: ManifestEntry,
    index: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if config.qa_trace is None:
        return
    config.qa_trace.record_event(
        event_type=event_type,
        status=status,
        trial_id=(payload or {}).get("trial_id") if payload else None,
        payload={
            "entry_index": index,
            "main_paper": str(entry.main_paper),
            "trial_label": entry.trial_label,
            "nct_number": entry.nct_number,
            **(payload or {}),
        },
    )


def _parse_supplements(value: Any, base_dir: Path) -> list[Path]:
    cleaned = _clean(value)
    if not cleaned:
        return []
    return [_resolve_path(part, base_dir) for part in cleaned.split(";") if part.strip()]


def _parse_outcomes(value: Any) -> list[str] | None:
    cleaned = _clean(value)
    if not cleaned:
        return None
    outcomes = [" ".join(part.split()) for part in cleaned.split(";") if part.strip()]
    return outcomes or None


def _expand_supplements(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pdf")))
        else:
            expanded.append(path)
    return expanded


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value.strip())
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
