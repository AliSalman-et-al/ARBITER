"""Deterministic confidence flagging for signaling-question answers."""

from __future__ import annotations

import os

from arbiter.models import AnswerCode, ConfidenceFlag, ConfidenceSignals

DEFAULT_RETRIEVAL_UNCERTAIN_THRESHOLD = 0.35


def compute_confidence(
    answer: AnswerCode | str,
    quote_verified: bool,
    segments_retrieved: int,
    segments_available: int,
    retrieval_top_score: float | None,
) -> ConfidenceSignals:
    """Compute advisory confidence metadata from verification and retrieval signals."""
    answer_code = AnswerCode(answer)
    threshold = _retrieval_uncertain_threshold()
    weak_retrieval = retrieval_top_score is not None and retrieval_top_score < threshold

    flag = ConfidenceFlag.CONFIDENT
    flag_reason: str | None = None

    if answer_code == AnswerCode.NA:
        flag = ConfidenceFlag.CONFIDENT
    elif answer_code not in {AnswerCode.NI, AnswerCode.NA} and not quote_verified:
        flag = ConfidenceFlag.FLAGGED
        flag_reason = "supporting quote could not be verified in the source text"
    elif answer_code == AnswerCode.NI and segments_available > 0 and weak_retrieval:
        flag = ConfidenceFlag.FLAGGED
        flag_reason = "answer is NI despite available supplements and a weak best retrieved passage"
    elif weak_retrieval:
        flag = ConfidenceFlag.UNCERTAIN
        flag_reason = "best retrieved passage is below the relevance threshold"
    elif answer_code == AnswerCode.NI and segments_available == 0:
        flag = ConfidenceFlag.UNCERTAIN
        flag_reason = "answer is NI with no domain-relevant supplementary material available"

    return ConfidenceSignals(
        supplement_segments_retrieved=segments_retrieved,
        supplement_segments_available=segments_available,
        retrieval_top_score=retrieval_top_score,
        quote_verified=quote_verified,
        flag=flag,
        flag_reason=flag_reason,
    )


def _retrieval_uncertain_threshold() -> float:
    value = os.getenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD")
    return DEFAULT_RETRIEVAL_UNCERTAIN_THRESHOLD if value is None or value == "" else float(value)
