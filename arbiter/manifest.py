"""Batch manifest parsing and unattended runner."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf
from pydantic import BaseModel, Field, field_validator

from arbiter import PIPELINE_VERSION, assess_trial, ingest_trial
from arbiter.config import AssessmentConfig
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


async def run_batch(manifest_path: Path, base_config: AssessmentConfig) -> BatchSummary:
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
        except Exception as exc:
            summary.error_count += 1
            _write_manifest_error(entry, index, exc, base_config)
    return summary


def check_eligibility(trial_metadata: TrialMetadata, config: AssessmentConfig) -> SkipRecord | None:
    """Return a skip artifact for trials outside the v0.1 parallel-RCT scope."""

    if trial_metadata.study_design == StudyDesign.PARALLEL_RCT:
        return None
    basis = trial_metadata.study_design_basis or "study design was not confirmed as a parallel-group RCT"
    return SkipRecord(
        assessment_id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        trial_id=trial_metadata.trial_id,
        nct_number=trial_metadata.nct_number,
        study_design=trial_metadata.study_design,
        study_design_basis=trial_metadata.study_design_basis,
        model_sq=config.sq_model,
        model_aux=config.aux_model,
        pipeline_version=PIPELINE_VERSION,
        errors=[f"ineligible study_design={trial_metadata.study_design.value}: {basis}"],
    )


def completed_pair_exists(
    db_path: Path,
    *,
    trial_id: str,
    outcome: str,
    effect_of_interest: str,
    model_sq: str,
    pipeline_version: str = PIPELINE_VERSION,
) -> bool:
    return _row_exists(db_path, trial_id, outcome, effect_of_interest, model_sq, pipeline_version)


def skip_row_exists(
    db_path: Path,
    *,
    trial_id: str,
    model_sq: str,
    pipeline_version: str = PIPELINE_VERSION,
) -> bool:
    return _row_exists(db_path, trial_id, SKIP_OUTCOME, SKIP_EFFECT, model_sq, pipeline_version)


async def _run_entry(entry: ManifestEntry, base_config: AssessmentConfig) -> dict[str, int | bool]:
    config = _config_for_entry(entry, base_config)
    cheap_trial_id = _cheap_trial_id(entry)
    if cheap_trial_id and not config.force:
        if skip_row_exists(config.db_path, trial_id=cheap_trial_id, model_sq=config.sq_model):
            return {"entry_skipped": True, "assessed_pairs": 0, "skipped_pairs": 0}
        if entry.outcomes and all(
            completed_pair_exists(
                config.db_path,
                trial_id=cheap_trial_id,
                outcome=outcome,
                effect_of_interest=config.effect_of_interest,
                model_sq=config.sq_model,
            )
            for outcome in entry.outcomes
        ):
            return {"entry_skipped": True, "assessed_pairs": 0, "skipped_pairs": len(entry.outcomes)}

    ctx = await ingest_trial(config)
    skip = check_eligibility(ctx.trial_metadata, config)
    if skip is not None:
        skip = skip.model_copy(update={"inputs_hash": ctx.config_summary.get("inputs_hash")})
        write_skip_record(skip, config.output_dir, config.db_path)
        return {"entry_skipped": False, "assessed_pairs": 0, "skipped_pairs": 0}

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
        )
    ]
    if not missing:
        return {"entry_skipped": False, "assessed_pairs": 0, "skipped_pairs": len(outcomes)}

    assessments = await assess_trial(ctx, replace(config, outcomes=missing))
    return {
        "entry_skipped": False,
        "assessed_pairs": len(assessments),
        "skipped_pairs": len(outcomes) - len(missing),
    }


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
        pipeline_version=PIPELINE_VERSION,
        errors=[f"manifest entry {index} failed: {type(exc).__name__}: {exc}"],
    )
    write_skip_record(skip, config.output_dir, config.db_path)


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
