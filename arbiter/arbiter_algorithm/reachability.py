"""Enumerate signaling-question answer vectors reachable through branching."""

from __future__ import annotations

from itertools import product

from arbiter.arbiter_algorithm.answer_sets import valid_answer_codes
from arbiter.arbiter_algorithm.branching import (
    DOMAIN_SQS,
    get_applicable_sqs,
    get_na_sqs,
)
from arbiter.models import AnswerCode, EffectOfInterest

AnswerVector = dict[str, AnswerCode]


def reachable_terminal_answer_vectors(
    domain: str, effect: EffectOfInterest | str
) -> tuple[AnswerVector, ...]:
    """Return complete answer vectors that branching can emit for a domain.

    Structural ``NA`` answers are included for gated-out questions, so every
    returned vector has an entry for each signaling question in the domain.
    """

    normalized_domain = _domain(domain)
    effect_value = EffectOfInterest(effect)
    terminal_vectors: list[AnswerVector] = []

    def walk(current: AnswerVector) -> None:
        with_structural_na = {
            **current,
            **{
                sq_id: AnswerCode.NA
                for sq_id in get_na_sqs(normalized_domain, effect_value, current)
                if sq_id not in current
            },
        }
        applicable = get_applicable_sqs(
            normalized_domain, effect_value, with_structural_na
        )
        if not applicable:
            terminal_vectors.append(_ordered_vector(normalized_domain, with_structural_na))
            return

        for values in product(*(valid_answer_codes(sq_id) for sq_id in applicable)):
            walk(
                {
                    **with_structural_na,
                    **dict(zip(applicable, values, strict=True)),
                }
            )

    walk({})
    return tuple(terminal_vectors)


def _domain(value: str) -> str:
    normalized = value.upper()
    if normalized in DOMAIN_SQS:
        return normalized
    if normalized.isdigit():
        domain = f"D{normalized}"
        if domain in DOMAIN_SQS:
            return domain
    raise ValueError(f"Unknown RoB 2 domain: {value!r}")


def _ordered_vector(domain: str, answers: AnswerVector) -> AnswerVector:
    missing = [sq_id for sq_id in DOMAIN_SQS[domain] if sq_id not in answers]
    if missing:
        raise ValueError(
            f"{domain} branching terminated without SQ answer(s): {', '.join(missing)}"
        )
    return {sq_id: answers[sq_id] for sq_id in DOMAIN_SQS[domain]}
