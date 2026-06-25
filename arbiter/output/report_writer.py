"""Reviewer-facing Markdown report writer for ARBITER assessments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arbiter.models import (
    Assessment,
    ConfidenceFlag,
    DomainJudgment,
    EffectOfInterest,
    OutcomeComparison,
    SQAnswer,
)
from arbiter.output.json_writer import assessment_json_path
from arbiter.prompts.sq_prompts import get_sq_prompt


def write_assessment_report(
    assessment: Assessment,
    output_dir: Path,
    timing_summary: dict | None = None,
) -> Path:
    """Write the reviewer-facing Markdown report for one assessment."""

    path = assessment_json_path(assessment, output_dir).with_name("report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(assessment, timing_summary), encoding="utf-8")
    return path


def _render_report(assessment: Assessment, timing_summary: dict | None) -> str:
    lines: list[str] = [
        f"# ARBITER RoB 2 Report: {_text(assessment.outcome)}",
        "",
        f"**Overall judgment: {_text(assessment.overall_judgment.value)}**",
        "",
        (
            "LLM-authored content is limited to signaling-question answers, quotes, "
            "and per-SQ justifications. Domain and overall rationales are deterministic "
            "RoB 2 algorithm outputs."
        ),
        "",
        "## Header",
        "",
        f"- Trial ID: `{_text(assessment.trial_id)}`",
        f"- NCT: {_text(assessment.nct_number or 'Not recorded')}",
        f"- Outcome: {_text(assessment.outcome)}",
        f"- Effect: {_effect_label(assessment.trial_metadata.effect_of_interest)}",
        f"- Created: {_text(assessment.created_at)}",
        f"- SQ model: `{_text(assessment.model_sq)}`",
        f"- Aux model: `{_text(assessment.model_aux)}`",
        f"- Vision model: `{_text(assessment.model_vision or 'None')}`",
        f"- Pipeline version: `{_text(assessment.pipeline_version)}`",
        f"- Main paper: {_text(assessment.sources_manifest.main_paper)}",
        f"- Supplements: {_text(', '.join(assessment.sources_manifest.supplements) or 'None')}",
        f"- ClinicalTrials.gov retrieved: {_yes_no(assessment.sources_manifest.ct_gov_retrieved)}",
        f"- Parsing quality: {_text(assessment.sources_manifest.parsing_quality.value)}",
        "",
        "## Needs Attention",
        "",
    ]

    attention = _needs_attention(assessment)
    if attention:
        lines.extend(f"- {item}" for item in attention)
    else:
        lines.append("- No flagged or uncertain signaling questions.")

    lines.extend(
        [
            "",
            "## Domain Summary",
            "",
            "| Domain | Scope | Judgment | Deterministic algorithm rationale |",
            "| --- | --- | --- | --- |",
        ]
    )
    for domain in _sorted_domains(assessment):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(domain.domain),
                    _cell(domain.scope),
                    _cell(domain.judgment.value),
                    _cell(domain.algorithm_rationale),
                ]
            )
            + " |"
        )
    lines.append(
        "| "
        + " | ".join(["Overall", "-", _cell(assessment.overall_judgment.value), _cell(assessment.overall_rationale)])
        + " |"
    )

    for domain in _sorted_domains(assessment):
        lines.extend(_render_domain(assessment, domain))

    lines.extend(_render_sources_and_quality(assessment))
    if timing_summary is not None:
        lines.extend(_render_timing_summary(timing_summary))

    lines.extend(
        [
            "",
            "---",
            "",
            "v0.1 advisory: this is a first-pass draft; confidence flags are advisory and uncalibrated.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_domain(assessment: Assessment, domain: DomainJudgment) -> list[str]:
    lines = [
        "",
        f"## {domain.domain} Detail",
        "",
        f"**Judgment:** {_text(domain.judgment.value)}",
        "",
        f"**Algorithm rationale (deterministic):** {_text(domain.algorithm_rationale)}",
        "",
    ]
    if domain.domain == "D5" and assessment.outcome_comparison is not None:
        lines.extend(_render_outcome_comparison(assessment.outcome_comparison))

    lines.extend(
        [
            "| SQ | Question | Answer | Confidence flag | Quote | Page | Justification (LLM-authored) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for answer in domain.sq_answers:
        lines.append(_render_sq_row(answer, assessment.trial_metadata.effect_of_interest))
    return lines


def _render_sq_row(answer: SQAnswer, effect: EffectOfInterest) -> str:
    prompt = get_sq_prompt(answer.sq_id, effect)
    quote = answer.quote or "-"
    if answer.page is not None:
        page = str(answer.page)
    else:
        page = "-"
    return (
        "| "
        + " | ".join(
            [
                _cell(answer.sq_id),
                _cell(prompt.question_text),
                _cell(answer.answer.value),
                _cell(_confidence_label(answer)),
                _cell(quote),
                _cell(page),
                _cell(answer.justification or "-"),
            ]
        )
        + " |"
    )


def _render_outcome_comparison(comparison: OutcomeComparison) -> list[str]:
    return [
        "### Outcome Comparison",
        "",
        "| Registered outcome | Published outcome | Similarity | Change detected | Registered primary |",
        "| --- | --- | --- | --- | --- |",
        "| "
        + " | ".join(
            [
                _cell(comparison.registered_outcome or "Not found"),
                _cell(comparison.published_outcome or "Not recorded"),
                _cell(_optional_number(comparison.outcome_similarity_score)),
                _cell(_optional_bool(comparison.outcome_change_detected)),
                _cell(_optional_bool(comparison.registered_as_primary)),
            ]
        )
        + " |",
        "",
    ]


def _render_sources_and_quality(assessment: Assessment) -> list[str]:
    supplements = assessment.sources_manifest.supplements
    errors = assessment.errors
    lines = [
        "",
        "## Sources And Parsing Quality",
        "",
        f"- Main paper: {_text(assessment.sources_manifest.main_paper)}",
        f"- Supplements consulted: {_text(', '.join(supplements) if supplements else 'None')}",
        f"- ClinicalTrials.gov retrieved: {_yes_no(assessment.sources_manifest.ct_gov_retrieved)}",
        f"- Parsing quality: {_text(assessment.sources_manifest.parsing_quality.value)}",
    ]
    if errors:
        lines.append(f"- Errors: {_text('; '.join(errors))}")
    return lines


def _render_timing_summary(timing_summary: dict[str, Any]) -> list[str]:
    outcome_cost = timing_summary.get("outcome_cost")
    trial_cost = timing_summary.get("trial_tier_cost")
    lines = [
        "",
        "## Cost And Timing",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| This outcome tier cost | {_cell(_money_or_unknown(outcome_cost))} |",
        f"| Shared trial-tier cost counted once per trial | {_cell(_money_or_unknown(trial_cost))} |",
    ]
    for key, value in timing_summary.items():
        if key in {"outcome_cost", "trial_tier_cost"}:
            continue
        lines.append(f"| {_cell(str(key))} | {_cell(str(value))} |")
    return lines


def _needs_attention(assessment: Assessment) -> list[str]:
    items: list[str] = []
    if assessment.requires_human_review:
        items.append("Assessment is marked `requires_human_review=True`.")
    for domain in _sorted_domains(assessment):
        for answer in domain.sq_answers:
            if answer.confidence.flag in {ConfidenceFlag.FLAGGED, ConfidenceFlag.UNCERTAIN}:
                reason = answer.confidence.flag_reason or "No flag reason recorded."
                items.append(
                    f"{domain.domain} {answer.sq_id}: `{answer.confidence.flag.value}` - {_text(reason)}"
                )
    return items


def _confidence_label(answer: SQAnswer) -> str:
    reason = answer.confidence.flag_reason
    label = {
        ConfidenceFlag.CONFIDENT: "CONFIDENT",
        ConfidenceFlag.UNCERTAIN: "UNCERTAIN",
        ConfidenceFlag.FLAGGED: "FLAGGED",
    }[answer.confidence.flag]
    if reason:
        return f"{label}: {reason}"
    return label


def _sorted_domains(assessment: Assessment) -> list[DomainJudgment]:
    return sorted(assessment.domain_judgments, key=lambda item: item.domain)


def _effect_label(effect: EffectOfInterest) -> str:
    if effect is EffectOfInterest.ASSIGNMENT:
        return "Effect of assignment to intervention (intention-to-treat)"
    return "Effect of adhering to intervention (per-protocol)"


def _money_or_unknown(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, int | float):
        return f"${value:.4f}"
    return str(value)


def _optional_number(value: float | None) -> str:
    return "Unknown" if value is None else f"{value:.3f}"


def _optional_bool(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return _yes_no(value)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _cell(value: str) -> str:
    return _text(value).replace("|", "\\|").replace("\n", "<br>")


def _text(value: str) -> str:
    return str(value)
