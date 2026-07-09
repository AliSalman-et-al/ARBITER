"""Assessment-level reliability checks."""

from __future__ import annotations

from collections.abc import Iterable

from arbiter.models import (
    AnswerCode,
    AssessmentReliability,
    DomainJudgment,
    ReliabilityStatus,
    SQFallbackKind,
)


def summarize_assessment_reliability(
    domain_judgments: Iterable[DomainJudgment],
    *,
    failure_fallback_threshold: float,
    failure_fallback_min_count: int,
) -> AssessmentReliability:
    """Summarize whether failure fallbacks dominate the assessment evidence."""

    answers = [
        answer
        for domain in domain_judgments
        for answer in domain.sq_answers
        if answer.answer != AnswerCode.NA
    ]
    failure_fallback_count = sum(
        1
        for answer in answers
        if answer.confidence.fallback_kind == SQFallbackKind.SQ_CALL_FAILED
    )
    answer_count = len(answers)
    fraction = failure_fallback_count / answer_count if answer_count else 0.0
    threshold = _clamp_fraction(failure_fallback_threshold)
    min_count = max(1, int(failure_fallback_min_count))
    excessive = failure_fallback_count >= min_count and fraction >= threshold
    basis = (
        "failure fallback signaling-question answers exceeded the assessment "
        f"reliability gate ({failure_fallback_count}/{answer_count}="
        f"{fraction:.1%}; threshold {threshold:.1%}, minimum {min_count})"
        if excessive
        else None
    )
    return AssessmentReliability(
        status=(
            ReliabilityStatus.FAILURE_FALLBACK_EXCESSIVE
            if excessive
            else ReliabilityStatus.OK
        ),
        sq_answer_count=answer_count,
        failure_fallback_sq_count=failure_fallback_count,
        failure_fallback_fraction=fraction,
        failure_fallback_threshold=threshold,
        failure_fallback_min_count=min_count,
        basis=basis,
    )


def _clamp_fraction(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
