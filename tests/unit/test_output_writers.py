from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arbiter.models import (
    AnswerCode,
    BlindingStatus,
    ConfidenceFlag,
    ConfidenceSignals,
    DomainJudgment,
    EffectOfInterest,
    Judgment,
    OutcomeComparison,
    ParsingQuality,
    SQAnswer,
    SkipRecord,
    SourcesManifest,
    StudyDesign,
    TrialMetadata,
    Assessment,
)
from arbiter.output import write_assessment_json, write_assessment_sqlite, write_skip_record
from arbiter.output.report_writer import write_assessment_report


def test_write_assessment_json_uses_nested_layout_and_prd_shape(tmp_path: Path) -> None:
    assessment = _assessment()

    path = write_assessment_json(assessment, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "trial-1" / "overall_survival__assignment" / "data.json"
    assert payload["identifiers"]["trial_id"] == "trial-1"
    assert payload["models"] == {"sq": "sq-model", "aux": "aux-model", "vision": None}
    assert list(payload["domains"]) == ["D1", "D2", "D3", "D4", "D5"]
    assert payload["domains"]["D1"]["sq_answers"]["1.1"]["answer"] == "Y"
    assert "trace" not in payload


def test_write_assessment_report_renders_reviewer_markdown(tmp_path: Path) -> None:
    assessment = _assessment_with_report_details()

    path = write_assessment_report(
        assessment,
        tmp_path,
        timing_summary={"outcome_cost": 0.0123, "trial_tier_cost": None, "wall_time_s": 1.5},
    )
    markdown = path.read_text(encoding="utf-8")

    assert path == tmp_path / "trial-1" / "overall_survival__assignment" / "report.md"
    assert "**Overall judgment: Some concerns**" in markdown
    assert "Assessment is marked `requires_human_review=True`." in markdown
    assert "D2 2.1: `FLAGGED` - Quote could not be verified." in markdown
    assert "D3 3.1: `UNCERTAIN` - Weak retrieved supplement passage." in markdown
    assert "| Domain | Scope | Judgment | Deterministic algorithm rationale |" in markdown
    assert "D5 fixture rationale." in markdown
    assert "Justification (LLM-authored)" in markdown
    assert "FLAGGED: Quote could not be verified." in markdown
    assert "Outcome Comparison" in markdown
    assert "Registered OS" in markdown
    assert "Shared trial-tier cost counted once per trial" in markdown
    assert "hazard ratio" not in markdown.lower()


def test_write_assessment_report_omits_timing_footer_when_not_supplied(tmp_path: Path) -> None:
    path = write_assessment_report(_assessment(), tmp_path)
    markdown = path.read_text(encoding="utf-8")

    assert "## Cost And Timing" not in markdown
    assert "No flagged or uncertain signaling questions." not in markdown


def test_write_assessment_sqlite_creates_schema_and_upserts_unique_key(tmp_path: Path) -> None:
    db_path = tmp_path / "arbiter.db"
    assessment = _assessment(assessment_id="first", errors=["old"])

    write_assessment_sqlite(assessment, db_path, json_path=tmp_path / "data.json")
    write_assessment_sqlite(_assessment(assessment_id="second", errors=["new"]), db_path, json_path=tmp_path / "data.json")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT assessment_id, trial_id, outcome, effect_of_interest, overall_judgment,
                   d1_judgment, flagged_sq_count, uncertain_sq_count, requires_human_review,
                   inputs_hash, errors
            FROM arbiter_assessments
            """
        ).fetchall()

    assert rows == [
        (
            "second",
            "trial-1",
            "Overall Survival",
            "assignment",
            "Low",
            "Low",
            1,
            1,
            1,
            "hash-1",
            '["new"]',
        )
    ]


def test_write_skip_record_writes_skip_json_and_sentinel_sqlite_row(tmp_path: Path) -> None:
    skip = SkipRecord(
        assessment_id="skip-1",
        created_at="2026-01-01T00:00:00+00:00",
        trial_id="trial-1",
        nct_number="NCT00000001",
        study_design=StudyDesign.UNCLEAR,
        study_design_basis="No randomisation wording found.",
        model_sq="sq-model",
        model_aux="aux-model",
        pipeline_version="0.1.0",
        inputs_hash="hash-1",
        errors=["ineligible study_design=unclear: No randomisation wording found."],
    )

    path = write_skip_record(skip, tmp_path / "out", tmp_path / "arbiter.db")
    write_skip_record(skip.model_copy(update={"assessment_id": "skip-2"}), tmp_path / "out", tmp_path / "arbiter.db")

    assert path == tmp_path / "out" / "trial-1" / "skip.json"
    assert json.loads(path.read_text(encoding="utf-8"))["study_design"] == "unclear"
    with sqlite3.connect(tmp_path / "arbiter.db") as conn:
        rows = conn.execute(
            """
            SELECT assessment_id, title, outcome, effect_of_interest, overall_judgment,
                   requires_human_review, study_design, json_path
            FROM arbiter_assessments
            """
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][:7] == ("skip-2", None, "__TRIAL__", "__NA__", None, 1, "unclear")
    assert rows[0][7].endswith("skip.json")


def _assessment(assessment_id: str = "assessment-1", errors: list[str] | None = None) -> Assessment:
    metadata = TrialMetadata(
        trial_id="trial-1",
        title="Trial title",
        intervention="Drug",
        comparator="Placebo",
        primary_outcome="Overall Survival",
        all_outcomes=["Overall Survival"],
        effect_of_interest=EffectOfInterest.ASSIGNMENT,
        blinding=BlindingStatus.DOUBLE_BLIND,
        nct_number="NCT00000001",
        study_design=StudyDesign.PARALLEL_RCT,
    )
    return Assessment(
        assessment_id=assessment_id,
        created_at="2026-01-01T00:00:00+00:00",
        pipeline_version="0.1.0",
        model_sq="sq-model",
        model_aux="aux-model",
        trial_id="trial-1",
        nct_number="NCT00000001",
        outcome="Overall Survival",
        requires_human_review=True,
        config_summary={"inputs_hash": "hash-1"},
        trial_metadata=metadata,
        outcome_comparison=None,
        domain_judgments=[
            _domain("D1", "trial", "1.1", ConfidenceFlag.CONFIDENT),
            _domain("D2", "outcome", "2.1", ConfidenceFlag.FLAGGED),
            _domain("D3", "outcome", "3.1", ConfidenceFlag.UNCERTAIN),
            _domain("D4", "outcome", "4.1", ConfidenceFlag.CONFIDENT),
            _domain("D5", "outcome", "5.1", ConfidenceFlag.CONFIDENT),
        ],
        overall_judgment=Judgment.LOW,
        overall_rationale="All domains are low.",
        sources_manifest=SourcesManifest(
            main_paper="paper.pdf",
            supplements=["supplement.pdf"],
            ct_gov_retrieved=True,
            parsing_quality=ParsingQuality.STANDARD,
        ),
        errors=errors or [],
    )


def _domain(domain: str, scope: str, sq_id: str, flag: ConfidenceFlag) -> DomainJudgment:
    return DomainJudgment(
        domain=domain,
        scope=scope,  # type: ignore[arg-type]
        judgment=Judgment.LOW,
        algorithm_rationale=f"{domain} fixture rationale.",
        sq_answers=[
            SQAnswer(
                sq_id=sq_id,
                answer=AnswerCode.Y,
                quote="The allocation sequence was random.",
                page=0,
                justification="The quoted text supports the answer.",
                confidence=ConfidenceSignals(flag=flag),
            )
        ],
    )


def _assessment_with_report_details() -> Assessment:
    assessment = _assessment()
    domains = []
    for domain in assessment.domain_judgments:
        if domain.domain == "D2":
            answer = domain.sq_answers[0].model_copy(
                update={
                    "confidence": ConfidenceSignals(
                        flag=ConfidenceFlag.FLAGGED,
                        flag_reason="Quote could not be verified.",
                    )
                }
            )
            domains.append(domain.model_copy(update={"sq_answers": [answer]}))
        elif domain.domain == "D3":
            answer = domain.sq_answers[0].model_copy(
                update={
                    "confidence": ConfidenceSignals(
                        flag=ConfidenceFlag.UNCERTAIN,
                        flag_reason="Weak retrieved supplement passage.",
                    )
                }
            )
            domains.append(domain.model_copy(update={"sq_answers": [answer]}))
        else:
            domains.append(domain)
    return assessment.model_copy(
        update={
            "overall_judgment": Judgment.SOME_CONCERNS,
            "overall_rationale": "Multiple domains have some concerns.",
            "domain_judgments": domains,
            "outcome_comparison": OutcomeComparison(
                registered_outcome="Registered OS",
                published_outcome="Overall Survival",
                outcome_similarity_score=0.92,
                outcome_change_detected=False,
                registered_as_primary=True,
            ),
        }
    )
