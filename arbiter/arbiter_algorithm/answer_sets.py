"""RoB 2 signaling-question answer-set contract."""

from __future__ import annotations

from arbiter.models import AnswerCode


LLM_ANSWER_CODES = (AnswerCode.Y, AnswerCode.PY, AnswerCode.PN, AnswerCode.N, AnswerCode.NI)

SQ_VALID_ANSWER_CODES: dict[str, tuple[AnswerCode, ...]] = {
    "3.2": (AnswerCode.Y, AnswerCode.PY, AnswerCode.PN, AnswerCode.N),
}

SQ_DISALLOWED_FALLBACKS: dict[str, dict[AnswerCode, AnswerCode]] = {
    "3.2": {AnswerCode.NI: AnswerCode.N},
}


def valid_answer_codes(sq_id: str) -> tuple[AnswerCode, ...]:
    """Return the non-structural answer codes RoB 2 permits for an SQ."""

    return SQ_VALID_ANSWER_CODES.get(sq_id, LLM_ANSWER_CODES)


def normalize_answer_for_sq(sq_id: str, answer: AnswerCode) -> AnswerCode:
    """Normalize impossible LLM answer codes to a deterministic fallback."""

    if answer == AnswerCode.NA or answer in valid_answer_codes(sq_id):
        return answer
    fallback = SQ_DISALLOWED_FALLBACKS.get(sq_id, {}).get(answer)
    if fallback is not None:
        return fallback
    return AnswerCode.NI if AnswerCode.NI in valid_answer_codes(sq_id) else valid_answer_codes(sq_id)[-1]
