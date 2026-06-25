"""SQLite persistence for ARBITER assessment results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from arbiter.models import Assessment, ConfidenceFlag, Judgment, SkipRecord
from arbiter.output.json_writer import assessment_json_path, skip_json_path

SKIP_OUTCOME = "__TRIAL__"
SKIP_EFFECT = "__NA__"


def write_assessment_sqlite(assessment: Assessment, db_path: Path, *, json_path: Path | None = None) -> None:
    """Upsert one row for a trial-outcome assessment."""

    resolved_json_path = json_path or assessment_json_path(assessment, Path(""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(_UPSERT_SQL, _assessment_row(assessment, resolved_json_path))


def write_skip_record(skip: SkipRecord, output_dir: Path, db_path: Path) -> Path:
    """Persist an ineligible-trial skip artifact and sentinel SQLite row."""

    path = skip_json_path(skip, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(skip.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(_UPSERT_SQL, _skip_row(skip, path))
    return path


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS arbiter_assessments (
            assessment_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            trial_id TEXT NOT NULL,
            nct_number TEXT,
            title TEXT,
            outcome TEXT NOT NULL,
            effect_of_interest TEXT NOT NULL,
            overall_judgment TEXT,
            d1_judgment TEXT,
            d2_judgment TEXT,
            d3_judgment TEXT,
            d4_judgment TEXT,
            d5_judgment TEXT,
            flagged_sq_count INTEGER,
            uncertain_sq_count INTEGER,
            requires_human_review INTEGER NOT NULL,
            study_design TEXT,
            model_sq TEXT NOT NULL,
            model_aux TEXT,
            pipeline_version TEXT NOT NULL,
            inputs_hash TEXT,
            json_path TEXT,
            errors TEXT NOT NULL,
            UNIQUE (trial_id, outcome, effect_of_interest, model_sq, pipeline_version)
        )
        """
    )


def _assessment_row(assessment: Assessment, json_path: Path) -> dict[str, Any]:
    domain_judgments = {judgment.domain.lower(): judgment.judgment for judgment in assessment.domain_judgments}
    return {
        "assessment_id": assessment.assessment_id,
        "created_at": assessment.created_at,
        "trial_id": assessment.trial_id,
        "nct_number": assessment.nct_number,
        "title": assessment.trial_metadata.title,
        "outcome": assessment.outcome,
        "effect_of_interest": assessment.trial_metadata.effect_of_interest.value,
        "overall_judgment": assessment.overall_judgment.value,
        "d1_judgment": _judgment_value(domain_judgments.get("d1")),
        "d2_judgment": _judgment_value(domain_judgments.get("d2")),
        "d3_judgment": _judgment_value(domain_judgments.get("d3")),
        "d4_judgment": _judgment_value(domain_judgments.get("d4")),
        "d5_judgment": _judgment_value(domain_judgments.get("d5")),
        "flagged_sq_count": _confidence_count(assessment, ConfidenceFlag.FLAGGED),
        "uncertain_sq_count": _confidence_count(assessment, ConfidenceFlag.UNCERTAIN),
        "requires_human_review": int(assessment.requires_human_review),
        "study_design": assessment.trial_metadata.study_design.value,
        "model_sq": assessment.model_sq,
        "model_aux": assessment.model_aux,
        "pipeline_version": assessment.pipeline_version,
        "inputs_hash": _inputs_hash(assessment),
        "json_path": str(json_path),
        "errors": json.dumps(assessment.errors),
    }


def _skip_row(skip: SkipRecord, json_path: Path) -> dict[str, Any]:
    return {
        "assessment_id": skip.assessment_id,
        "created_at": skip.created_at,
        "trial_id": skip.trial_id,
        "nct_number": skip.nct_number,
        "title": None,
        "outcome": SKIP_OUTCOME,
        "effect_of_interest": SKIP_EFFECT,
        "overall_judgment": None,
        "d1_judgment": None,
        "d2_judgment": None,
        "d3_judgment": None,
        "d4_judgment": None,
        "d5_judgment": None,
        "flagged_sq_count": None,
        "uncertain_sq_count": None,
        "requires_human_review": int(skip.requires_human_review),
        "study_design": skip.study_design.value,
        "model_sq": skip.model_sq,
        "model_aux": skip.model_aux,
        "pipeline_version": skip.pipeline_version,
        "inputs_hash": skip.inputs_hash,
        "json_path": str(json_path),
        "errors": json.dumps(skip.errors),
    }


def _judgment_value(judgment: Judgment | None) -> str | None:
    return judgment.value if judgment is not None else None


def _confidence_count(assessment: Assessment, flag: ConfidenceFlag) -> int:
    return sum(
        1
        for domain in assessment.domain_judgments
        for answer in domain.sq_answers
        if answer.confidence.flag == flag
    )


def _inputs_hash(assessment: Assessment) -> str | None:
    value = assessment.config_summary.get("inputs_hash")
    return str(value) if value is not None else None


_COLUMNS = [
    "assessment_id",
    "created_at",
    "trial_id",
    "nct_number",
    "title",
    "outcome",
    "effect_of_interest",
    "overall_judgment",
    "d1_judgment",
    "d2_judgment",
    "d3_judgment",
    "d4_judgment",
    "d5_judgment",
    "flagged_sq_count",
    "uncertain_sq_count",
    "requires_human_review",
    "study_design",
    "model_sq",
    "model_aux",
    "pipeline_version",
    "inputs_hash",
    "json_path",
    "errors",
]

_UPSERT_SQL = f"""
INSERT INTO arbiter_assessments ({", ".join(_COLUMNS)})
VALUES ({", ".join(":" + column for column in _COLUMNS)})
ON CONFLICT(trial_id, outcome, effect_of_interest, model_sq, pipeline_version) DO UPDATE SET
{", ".join(f"{column}=excluded.{column}" for column in _COLUMNS if column not in {"trial_id", "outcome", "effect_of_interest", "model_sq", "pipeline_version"})}
"""
