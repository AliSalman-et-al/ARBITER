"""JSON artifact writer for ARBITER assessments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from arbiter.models import Assessment, DomainJudgment, SkipRecord


def write_assessment_json(assessment: Assessment, output_dir: Path) -> Path:
    """Write one per-trial/per-outcome assessment JSON artifact."""

    path = assessment_json_path(assessment, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_assessment_payload(assessment), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def assessment_json_path(assessment: Assessment, output_dir: Path) -> Path:
    effect = assessment.trial_metadata.effect_of_interest.value
    return output_dir / assessment.trial_id / f"{_slugify(assessment.outcome)}__{effect}" / "data.json"


def skip_json_path(skip: SkipRecord, output_dir: Path) -> Path:
    return output_dir / skip.trial_id / "skip.json"


def _assessment_payload(assessment: Assessment) -> dict[str, Any]:
    domains = {
        judgment.domain: _domain_payload(judgment)
        for judgment in sorted(assessment.domain_judgments, key=lambda item: item.domain)
    }
    return {
        "identifiers": {
            "assessment_id": assessment.assessment_id,
            "created_at": assessment.created_at,
            "trial_id": assessment.trial_id,
            "nct_number": assessment.nct_number,
            "outcome": assessment.outcome,
            "effect_of_interest": assessment.trial_metadata.effect_of_interest.value,
            "pipeline_version": assessment.pipeline_version,
        },
        "models": {
            "sq": assessment.model_sq,
            "aux": assessment.model_aux,
            "vision": assessment.model_vision,
        },
        "requires_human_review": assessment.requires_human_review,
        "config_summary": assessment.config_summary,
        "trial_metadata": assessment.trial_metadata.model_dump(mode="json"),
        "ct_gov_data": assessment.ct_gov_data,
        "outcome_comparison": (
            assessment.outcome_comparison.model_dump(mode="json") if assessment.outcome_comparison else None
        ),
        "domains": domains,
        "overall": {
            "judgment": assessment.overall_judgment.value,
            "rationale": assessment.overall_rationale,
        },
        "sources_manifest": assessment.sources_manifest.model_dump(mode="json"),
        "errors": assessment.errors,
    }


def _domain_payload(judgment: DomainJudgment) -> dict[str, Any]:
    return {
        "judgment": judgment.judgment.value,
        "scope": judgment.scope,
        "rationale": judgment.algorithm_rationale,
        "sq_answers": {
            answer.sq_id: {
                "answer": answer.answer.value,
                "quote": answer.quote,
                "page": answer.page,
                "justification": answer.justification,
                "confidence": answer.confidence.model_dump(mode="json"),
            }
            for answer in judgment.sq_answers
        },
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "outcome"
