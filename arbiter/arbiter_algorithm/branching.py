"""Conditional RoB 2 signaling-question routing.

The predicates in this module encode the IRPG branching contract pinned in
`docs/rob2/` and summarized by REQ-08. The functions return unanswered SQs
that are ready to ask, plus SQs that are structurally not applicable given the
answers collected so far.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arbiter.models import AnswerCode, EffectOfInterest


Answers = Mapping[str, Any]

YES = {AnswerCode.Y, AnswerCode.PY}
NO = {AnswerCode.N, AnswerCode.PN}
NO_OR_NI = {AnswerCode.N, AnswerCode.PN, AnswerCode.NI}
YES_OR_NI = {AnswerCode.Y, AnswerCode.PY, AnswerCode.NI}

DOMAIN_SQS = {
    "D1": ("1.1", "1.2", "1.3"),
    "D2": ("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"),
    "D3": ("3.1", "3.2", "3.3", "3.4"),
    "D4": ("4.1", "4.2", "4.3", "4.4", "4.5"),
    "D5": ("5.1", "5.2", "5.3"),
}


def get_applicable_sqs(domain: str, effect: EffectOfInterest | str, current_answers: Answers) -> list[str]:
    """Return unanswered SQ IDs whose gates are satisfied now."""

    applicable, _ = _route(domain, effect, current_answers)
    return [sq_id for sq_id in applicable if sq_id not in current_answers]


def get_na_sqs(domain: str, effect: EffectOfInterest | str, current_answers: Answers) -> list[str]:
    """Return SQ IDs that are structurally not applicable now."""

    _, na_sqs = _route(domain, effect, current_answers)
    return na_sqs


def _route(domain: str, effect: EffectOfInterest | str, answers: Answers) -> tuple[list[str], list[str]]:
    normalized = _domain(domain)
    effect_value = EffectOfInterest(effect)

    if normalized == "D1":
        return list(DOMAIN_SQS["D1"]), []
    if normalized == "D2":
        if effect_value is EffectOfInterest.ASSIGNMENT:
            return _route_d2_assignment(answers)
        return _route_d2_adhering(answers)
    if normalized == "D3":
        return _route_d3(answers)
    if normalized == "D4":
        return _route_d4(answers)
    if normalized == "D5":
        return _route_d5(answers)
    raise ValueError(f"Unknown RoB 2 domain: {domain!r}")


def _route_d2_assignment(answers: Answers) -> tuple[list[str], list[str]]:
    applicable = ["2.1", "2.2", "2.6"]
    na_sqs: list[str] = []

    if _any_gate(answers, ("2.1", "2.2"), YES_OR_NI):
        applicable.append("2.3")
        if _gate(answers, "2.3", YES):
            applicable.append("2.4")
            if _gate(answers, "2.4", YES_OR_NI):
                applicable.append("2.5")
            elif _known(answers, "2.4"):
                na_sqs.append("2.5")
        elif _known(answers, "2.3"):
            na_sqs.extend(["2.4", "2.5"])
    elif _all_known(answers, "2.1", "2.2"):
        na_sqs.extend(["2.3", "2.4", "2.5"])

    if _gate(answers, "2.6", NO_OR_NI):
        applicable.append("2.7")
    elif _known(answers, "2.6"):
        na_sqs.append("2.7")

    return _ordered_unique(applicable, "D2"), _ordered_unique(na_sqs, "D2")


def _route_d2_adhering(answers: Answers) -> tuple[list[str], list[str]]:
    applicable = ["2.1", "2.2", "2.4", "2.5"]
    na_sqs = ["2.7"]

    if _any_gate(answers, ("2.1", "2.2"), YES_OR_NI):
        applicable.append("2.3")
    elif _all_known(answers, "2.1", "2.2"):
        na_sqs.append("2.3")

    if (
        _gate(answers, "2.3", NO_OR_NI)
        or _gate(answers, "2.4", YES_OR_NI)
        or _gate(answers, "2.5", YES_OR_NI)
    ):
        applicable.append("2.6")
    elif _d2_adhering_26_operands_resolved(answers):
        na_sqs.append("2.6")

    return _ordered_unique(applicable, "D2"), _ordered_unique(na_sqs, "D2")


def _route_d3(answers: Answers) -> tuple[list[str], list[str]]:
    applicable = ["3.1"]
    na_sqs: list[str] = []

    if _gate(answers, "3.1", NO_OR_NI):
        applicable.append("3.2")
        if _gate(answers, "3.2", NO):
            applicable.append("3.3")
            if _gate(answers, "3.3", YES_OR_NI):
                applicable.append("3.4")
            elif _known(answers, "3.3"):
                na_sqs.append("3.4")
        elif _known(answers, "3.2"):
            na_sqs.extend(["3.3", "3.4"])
    elif _known(answers, "3.1"):
        na_sqs.extend(["3.2", "3.3", "3.4"])

    return _ordered_unique(applicable, "D3"), _ordered_unique(na_sqs, "D3")


def _route_d4(answers: Answers) -> tuple[list[str], list[str]]:
    applicable = ["4.1", "4.2"]
    na_sqs: list[str] = []

    if _all_gate(answers, ("4.1", "4.2"), NO_OR_NI):
        applicable.append("4.3")
        if _gate(answers, "4.3", YES_OR_NI):
            applicable.append("4.4")
            if _gate(answers, "4.4", YES_OR_NI):
                applicable.append("4.5")
            elif _known(answers, "4.4"):
                na_sqs.append("4.5")
        elif _known(answers, "4.3"):
            na_sqs.extend(["4.4", "4.5"])
    elif _all_known(answers, "4.1", "4.2") and (_code(answers["4.1"]) in YES or _code(answers["4.2"]) in YES):
        na_sqs.extend(["4.3", "4.4", "4.5"])

    return _ordered_unique(applicable, "D4"), _ordered_unique(na_sqs, "D4")


def _route_d5(answers: Answers) -> tuple[list[str], list[str]]:
    applicable = ["5.1"]
    na_sqs: list[str] = []

    if _gate(answers, "5.1", NO_OR_NI):
        applicable.extend(["5.2", "5.3"])
    elif _gate(answers, "5.1", YES):
        na_sqs.extend(["5.2", "5.3"])

    return _ordered_unique(applicable, "D5"), _ordered_unique(na_sqs, "D5")


def _d2_adhering_26_operands_resolved(answers: Answers) -> bool:
    if not _all_known(answers, "2.4", "2.5"):
        return False
    if _any_gate(answers, ("2.1", "2.2"), YES_OR_NI):
        return _known(answers, "2.3")
    return _all_known(answers, "2.1", "2.2")


def _gate(answers: Answers, sq_id: str, allowed: set[AnswerCode]) -> bool:
    return _known(answers, sq_id) and _code(answers[sq_id]) in allowed


def _any_gate(answers: Answers, sq_ids: tuple[str, ...], allowed: set[AnswerCode]) -> bool:
    return any(_gate(answers, sq_id, allowed) for sq_id in sq_ids)


def _all_gate(answers: Answers, sq_ids: tuple[str, ...], allowed: set[AnswerCode]) -> bool:
    return all(_gate(answers, sq_id, allowed) for sq_id in sq_ids)


def _known(answers: Answers, sq_id: str) -> bool:
    return sq_id in answers


def _all_known(answers: Answers, *sq_ids: str) -> bool:
    return all(_known(answers, sq_id) for sq_id in sq_ids)


def _code(value: Any) -> AnswerCode:
    if hasattr(value, "answer"):
        value = value.answer
    return value if isinstance(value, AnswerCode) else AnswerCode(value)


def _domain(value: str) -> str:
    normalized = value.upper()
    if normalized in DOMAIN_SQS:
        return normalized
    if normalized.isdigit():
        return f"D{normalized}"
    raise ValueError(f"Unknown RoB 2 domain: {value!r}")


def _ordered_unique(values: list[str], domain: str) -> list[str]:
    value_set = set(values)
    return [sq_id for sq_id in DOMAIN_SQS[domain] if sq_id in value_set]
