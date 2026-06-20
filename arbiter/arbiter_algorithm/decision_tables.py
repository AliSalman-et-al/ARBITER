"""Pure Python transcription of the RoB 2 IRPG domain decision tables.

The branch predicates are transcribed from the pinned `ROB2_IRPG_beta_v9`
workbook in `docs/rob2/rob2_irpg_algorithm.xlsm`, specifically the
`SugButton1_Click` ... `SugButton5_Click` VBA handlers in `UserForm3.frm`.
The workbook is git-ignored; this module is the committed, hermetic
translation used at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arbiter.models import AnswerCode, EffectOfInterest, Judgment


Answers = Mapping[str, Any]

YES = {AnswerCode.Y, AnswerCode.PY}
NO = {AnswerCode.N, AnswerCode.PN}
NO_OR_NI = {AnswerCode.N, AnswerCode.PN, AnswerCode.NI}
YES_OR_NI = {AnswerCode.Y, AnswerCode.PY, AnswerCode.NI}
ANY_INFORMATIVE = {AnswerCode.Y, AnswerCode.PY, AnswerCode.PN, AnswerCode.N, AnswerCode.NI}


def judge_domain_1(answers: Answers) -> tuple[Judgment, str]:
    a11, a12, a13 = _answers(answers, "1.1", "1.2", "1.3")

    if a12 in YES:
        if a11 in {AnswerCode.Y, AnswerCode.PY, AnswerCode.NI} and a13 in NO_OR_NI:
            return _result(Judgment.LOW, "1.2=Y/PY, 1.1=Y/PY/NI, 1.3=N/PN/NI")
        if a11 in NO and a13 in ANY_INFORMATIVE:
            return _result(Judgment.SOME_CONCERNS, "1.2=Y/PY, 1.1=N/PN")
        if a11 in {AnswerCode.Y, AnswerCode.PY, AnswerCode.NI} and a13 in YES:
            return _result(Judgment.SOME_CONCERNS, "1.2=Y/PY, 1.3=Y/PY")
    if a12 == AnswerCode.NI:
        if a11 in ANY_INFORMATIVE and a13 in NO_OR_NI:
            return _result(Judgment.SOME_CONCERNS, "1.2=NI, 1.3=N/PN/NI")
        if a11 in ANY_INFORMATIVE and a13 in YES:
            return _result(Judgment.HIGH, "1.2=NI, 1.3=Y/PY")
    if a12 in NO:
        return _result(Judgment.HIGH, "1.2=N/PN")
    raise ValueError("Domain 1 answers do not reach a RoB 2 IRPG judgment")


def judge_domain_2(answers: Answers, effect: EffectOfInterest | str) -> tuple[Judgment, str]:
    effect_value = EffectOfInterest(effect)
    if effect_value is EffectOfInterest.ASSIGNMENT:
        return _judge_domain_2_assignment(answers)
    return _judge_domain_2_adhering(answers)


def judge_domain_3(answers: Answers) -> tuple[Judgment, str]:
    a31, a32, a33, a34 = _answers(answers, "3.1", "3.2", "3.3", "3.4")

    if a31 in YES:
        return _result(Judgment.LOW, "3.1=Y/PY")
    if a31 in NO_OR_NI:
        if a32 in YES:
            return _result(Judgment.LOW, "3.1=N/PN/NI, 3.2=Y/PY")
        if a32 in NO_OR_NI:
            if a33 in NO:
                return _result(Judgment.LOW, "3.2=N/PN/NI, 3.3=N/PN")
            if a33 in YES_OR_NI:
                if a34 in NO:
                    return _result(Judgment.SOME_CONCERNS, "3.3=Y/PY/NI, 3.4=N/PN")
                if a34 in YES_OR_NI:
                    return _result(Judgment.HIGH, "3.3=Y/PY/NI, 3.4=Y/PY/NI")
    raise ValueError("Domain 3 answers do not reach a RoB 2 IRPG judgment")


def judge_domain_4(answers: Answers) -> tuple[Judgment, str]:
    a41, a42, a43, a44, a45 = _answers(answers, "4.1", "4.2", "4.3", "4.4", "4.5")

    if a41 in YES or a42 in YES:
        return _result(Judgment.HIGH, "4.1=Y/PY or 4.2=Y/PY")
    if a41 in NO_OR_NI:
        if a42 in NO:
            if a43 in NO:
                return _result(Judgment.LOW, "4.1=N/PN/NI, 4.2=N/PN, 4.3=N/PN")
            if a43 in YES_OR_NI:
                if a44 in NO:
                    return _result(Judgment.LOW, "4.3=Y/PY/NI, 4.4=N/PN")
                if a44 in YES_OR_NI:
                    if a45 in NO:
                        return _result(Judgment.SOME_CONCERNS, "4.4=Y/PY/NI, 4.5=N/PN")
                    if a45 in YES_OR_NI:
                        return _result(Judgment.HIGH, "4.4=Y/PY/NI, 4.5=Y/PY/NI")
        if a42 == AnswerCode.NI:
            if a43 in NO:
                return _result(Judgment.SOME_CONCERNS, "4.2=NI, 4.3=N/PN")
            if a43 in YES_OR_NI:
                if a44 in NO:
                    return _result(Judgment.SOME_CONCERNS, "4.2=NI, 4.4=N/PN")
                if a44 in YES_OR_NI:
                    if a45 in NO:
                        return _result(Judgment.SOME_CONCERNS, "4.2=NI, 4.5=N/PN")
                    if a45 in YES_OR_NI:
                        return _result(Judgment.HIGH, "4.2=NI, 4.5=Y/PY/NI")
    raise ValueError("Domain 4 answers do not reach a RoB 2 IRPG judgment")


def judge_domain_5(answers: Answers) -> tuple[Judgment, str]:
    a51, a52, a53 = _answers(answers, "5.1", "5.2", "5.3")

    if a52 in NO and a53 in NO:
        if a51 in YES:
            return _result(Judgment.LOW, "5.2=N/PN, 5.3=N/PN, 5.1=Y/PY")
        if a51 in NO_OR_NI:
            return _result(Judgment.SOME_CONCERNS, "5.2=N/PN, 5.3=N/PN, 5.1=N/PN/NI")
    if a52 in YES or a53 in YES:
        return _result(Judgment.HIGH, "5.2=Y/PY or 5.3=Y/PY")
    if a51 in ANY_INFORMATIVE and a52 in ANY_INFORMATIVE and a53 in ANY_INFORMATIVE:
        return _result(Judgment.SOME_CONCERNS, "5.1/5.2/5.3 completed without Low or High pattern")
    raise ValueError("Domain 5 answers do not reach a RoB 2 IRPG judgment")


def _judge_domain_2_assignment(answers: Answers) -> tuple[Judgment, str]:
    a21, a22, a23, a24, a25, a26, a27 = _answers(answers, "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7")

    part1 = ""
    if a21 in NO and a22 in NO:
        part1 = "Low"
    elif a21 in YES_OR_NI or a22 in YES_OR_NI:
        if a23 in NO:
            part1 = "Low"
        elif a23 == AnswerCode.NI:
            part1 = "Some concerns"
        elif a23 in YES:
            if a24 in NO:
                part1 = "Some concerns"
            elif a24 in YES_OR_NI:
                if a25 in YES:
                    part1 = "Some concerns"
                elif a25 in NO_OR_NI:
                    part1 = "High"

    part2 = ""
    if a26 in YES:
        part2 = "Low"
    elif a26 in NO_OR_NI:
        if a27 in NO:
            part2 = "Some concerns"
        elif a27 in YES_OR_NI:
            part2 = "High"

    if part1 == "Low" and part2 == "Low":
        return _result(Judgment.LOW, "assignment: deviation component=Low, analysis component=Low")
    if part1 and part2 and (part1 == "High" or part2 == "High"):
        return _result(Judgment.HIGH, f"assignment: deviation component={part1}, analysis component={part2}")
    if part1 and part2:
        return _result(Judgment.SOME_CONCERNS, f"assignment: deviation component={part1}, analysis component={part2}")
    raise ValueError("Domain 2 assignment answers do not reach a RoB 2 IRPG judgment")


def _judge_domain_2_adhering(answers: Answers) -> tuple[Judgment, str]:
    a21, a22, a23, a24, a25, a26 = _answers(answers, "2.1", "2.2", "2.3", "2.4", "2.5", "2.6")
    no_failure_or_nonadherence = a24 in {AnswerCode.N, AnswerCode.PN, AnswerCode.NA} and a25 in {
        AnswerCode.N,
        AnswerCode.PN,
        AnswerCode.NA,
    }

    if a21 in NO and a22 in NO:
        if no_failure_or_nonadherence:
            return _result(Judgment.LOW, "adhering: 2.1=N/PN, 2.2=N/PN, 2.4/2.5=N/PN/NA")
        if a24 in YES_OR_NI or a25 in YES_OR_NI:
            if a26 in YES:
                return _result(Judgment.SOME_CONCERNS, "adhering: 2.4/2.5=Y/PY/NI, 2.6=Y/PY")
            if a26 in NO_OR_NI:
                return _result(Judgment.HIGH, "adhering: 2.4/2.5=Y/PY/NI, 2.6=N/PN/NI")

    if a21 in YES_OR_NI or a22 in YES_OR_NI:
        if a23 in {AnswerCode.Y, AnswerCode.PY, AnswerCode.NA}:
            if no_failure_or_nonadherence:
                return _result(Judgment.LOW, "adhering: 2.3=Y/PY/NA, 2.4/2.5=N/PN/NA")
            if a24 in YES_OR_NI or a25 in YES_OR_NI:
                if a26 in YES:
                    return _result(Judgment.SOME_CONCERNS, "adhering: 2.4/2.5=Y/PY/NI, 2.6=Y/PY")
                if a26 in NO_OR_NI:
                    return _result(Judgment.HIGH, "adhering: 2.4/2.5=Y/PY/NI, 2.6=N/PN/NI")
        if a23 in NO_OR_NI:
            if a26 in YES:
                return _result(Judgment.SOME_CONCERNS, "adhering: 2.3=N/PN/NI, 2.6=Y/PY")
            if a26 in NO_OR_NI:
                return _result(Judgment.HIGH, "adhering: 2.3=N/PN/NI, 2.6=N/PN/NI")

    raise ValueError("Domain 2 adhering answers do not reach a RoB 2 IRPG judgment")


def _answers(answers: Answers, *sq_ids: str) -> tuple[AnswerCode, ...]:
    missing = [sq_id for sq_id in sq_ids if sq_id not in answers]
    if missing:
        raise KeyError(f"Missing SQ answer(s): {', '.join(missing)}")
    return tuple(_code(answers[sq_id].answer) for sq_id in sq_ids)


def _code(value: AnswerCode | str) -> AnswerCode:
    return value if isinstance(value, AnswerCode) else AnswerCode(value)


def _result(judgment: Judgment, rationale: str) -> tuple[Judgment, str]:
    return judgment, f"{rationale} -> {judgment.value}"
