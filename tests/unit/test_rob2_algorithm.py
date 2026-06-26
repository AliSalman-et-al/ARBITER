from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pytest

from arbiter.arbiter_algorithm import branching, decision_tables, rollup
from arbiter.models import AnswerCode as A
from arbiter.models import ConfidenceFlag, ConfidenceSignals, DomainJudgment, EffectOfInterest, Judgment, SQAnswer


@dataclass(frozen=True)
class StubAnswer:
    answer: A


def ans(**values: A) -> dict[str, StubAnswer]:
    return {key.replace("_", "."): StubAnswer(value) for key, value in values.items()}


def domain(domain: str, judgment: Judgment) -> DomainJudgment:
    return DomainJudgment(domain=domain, scope="trial" if domain == "D1" else "outcome", judgment=judgment, algorithm_rationale="")


def domain_with_answer(domain: str, judgment: Judgment, answer: SQAnswer) -> DomainJudgment:
    return DomainJudgment(
        domain=domain,
        scope="trial" if domain == "D1" else "outcome",
        judgment=judgment,
        algorithm_rationale="",
        sq_answers=[answer],
    )


def test_d1_3_direction_is_not_inverted() -> None:
    low, _ = decision_tables.judge_domain_1(ans(**{"1_1": A.Y, "1_2": A.Y, "1_3": A.N}))
    some_concerns, _ = decision_tables.judge_domain_1(ans(**{"1_1": A.Y, "1_2": A.Y, "1_3": A.Y}))
    high, _ = decision_tables.judge_domain_1(ans(**{"1_1": A.Y, "1_2": A.NI, "1_3": A.Y}))

    assert low is Judgment.LOW
    assert some_concerns is Judgment.SOME_CONCERNS
    assert high is Judgment.HIGH


def test_domain_1_matches_vba_transcription() -> None:
    for a11, a12, a13 in product(list(A)[:-1], repeat=3):
        answers = ans(**{"1_1": a11, "1_2": a12, "1_3": a13})
        assert decision_tables.judge_domain_1(answers)[0] is _oracle_d1(a11, a12, a13)


def test_domain_2_assignment_matches_vba_transcription() -> None:
    keys = ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6", "2_7"]
    for values in product(list(A), repeat=7):
        expected = _oracle_d2_assignment(*values)
        if expected is not None:
            assert decision_tables.judge_domain_2(ans(**dict(zip(keys, values, strict=True))), EffectOfInterest.ASSIGNMENT)[0] is expected


def test_domain_2_adhering_matches_vba_transcription() -> None:
    keys = ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6"]
    for values in product(list(A), repeat=6):
        expected = _oracle_d2_adhering(*values)
        if expected is not None:
            assert decision_tables.judge_domain_2(ans(**dict(zip(keys, values, strict=True))), EffectOfInterest.ADHERING)[0] is expected


def test_domain_3_matches_vba_transcription() -> None:
    keys = ["3_1", "3_2", "3_3", "3_4"]
    for values in product(list(A), repeat=4):
        expected = _oracle_d3(*values)
        if expected is not None:
            assert decision_tables.judge_domain_3(ans(**dict(zip(keys, values, strict=True))))[0] is expected


def test_domain_4_matches_vba_transcription() -> None:
    keys = ["4_1", "4_2", "4_3", "4_4", "4_5"]
    for values in product(list(A), repeat=5):
        expected = _oracle_d4(*values)
        if expected is not None:
            assert decision_tables.judge_domain_4(ans(**dict(zip(keys, values, strict=True))))[0] is expected


def test_domain_5_matches_vba_transcription() -> None:
    keys = ["5_1", "5_2", "5_3"]
    for values in product(list(A), repeat=3):
        expected = _oracle_d5(*values)
        if expected is not None:
            assert decision_tables.judge_domain_5(ans(**dict(zip(keys, values, strict=True))))[0] is expected


def test_overall_rollup_all_combinations() -> None:
    for combo in product(list(Judgment), repeat=5):
        actual, _, review = rollup.compute_overall_judgment([domain(f"D{i}", judgment) for i, judgment in enumerate(combo, 1)])
        some_count = combo.count(Judgment.SOME_CONCERNS)
        if Judgment.HIGH in combo:
            assert (actual, review) == (Judgment.HIGH, False)
        elif some_count == 0:
            assert (actual, review) == (Judgment.LOW, False)
        elif some_count >= rollup.OVERALL_HIGH_SC_THRESHOLD:
            assert (actual, review) == (Judgment.HIGH, True)
        else:
            assert (actual, review) == (Judgment.SOME_CONCERNS, 2 <= some_count < rollup.OVERALL_HIGH_SC_THRESHOLD)


def test_overall_rollup_review_band_moves_with_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollup, "OVERALL_HIGH_SC_THRESHOLD", 4)
    two_sc = [
        domain("D1", Judgment.SOME_CONCERNS),
        domain("D2", Judgment.SOME_CONCERNS),
        domain("D3", Judgment.LOW),
        domain("D4", Judgment.LOW),
        domain("D5", Judgment.LOW),
    ]
    three_sc = [
        domain("D1", Judgment.SOME_CONCERNS),
        domain("D2", Judgment.SOME_CONCERNS),
        domain("D3", Judgment.SOME_CONCERNS),
        domain("D4", Judgment.LOW),
        domain("D5", Judgment.LOW),
    ]
    four_sc = [
        domain("D1", Judgment.SOME_CONCERNS),
        domain("D2", Judgment.SOME_CONCERNS),
        domain("D3", Judgment.SOME_CONCERNS),
        domain("D4", Judgment.SOME_CONCERNS),
        domain("D5", Judgment.LOW),
    ]

    assert rollup.compute_overall_judgment(two_sc)[2] is True
    assert rollup.compute_overall_judgment(three_sc)[2] is True
    assert rollup.compute_overall_judgment(four_sc) == (
        Judgment.HIGH,
        "4 domains Some concerns >= threshold 4 -> High",
        True,
    )


def test_overall_rollup_requires_review_for_reliability_signals_without_changing_judgment() -> None:
    flagged = SQAnswer(
        sq_id="1.1",
        answer=A.NI,
        confidence=ConfidenceSignals(
            quote_verified=False,
            flag=ConfidenceFlag.FLAGGED,
            flag_reason="supporting quote could not be verified in the source text",
        ),
    )
    judgments = [
        domain_with_answer("D1", Judgment.LOW, flagged),
        domain("D2", Judgment.LOW),
        domain("D3", Judgment.LOW),
        domain("D4", Judgment.LOW),
        domain("D5", Judgment.LOW),
    ]

    overall, rationale, requires_review = rollup.compute_overall_judgment(judgments)

    assert (overall, rationale, requires_review) == (Judgment.LOW, "all domains Low -> Low", True)
    assert rollup.compute_human_review_basis(judgments, rationale) == (
        "flagged SQ answer(s): D1 1.1; unverified quote(s): D1 1.1"
    )


def test_d2_assignment_branching_waits_for_chain_predecessors() -> None:
    answers = ans(**{"2_1": A.Y, "2_2": A.N})

    assert branching.get_applicable_sqs("D2", EffectOfInterest.ASSIGNMENT, answers) == ["2.3", "2.6"]
    assert "2.4" not in branching.get_applicable_sqs("D2", EffectOfInterest.ASSIGNMENT, answers)
    assert branching.get_na_sqs("D2", EffectOfInterest.ASSIGNMENT, {}) == []


def test_d2_adhering_only_2_7_is_effect_exclusive_na_initially() -> None:
    assert branching.get_na_sqs("D2", EffectOfInterest.ADHERING, {}) == ["2.7"]
    assert branching.get_applicable_sqs("D2", EffectOfInterest.ADHERING, {}) == ["2.1", "2.2", "2.4", "2.5"]


def test_d2_adhering_compound_gate_treats_na_as_not_satisfied() -> None:
    answers = ans(**{"2_1": A.Y, "2_2": A.N, "2_3": A.Y, "2_4": A.NA, "2_5": A.N})

    assert "2.6" not in branching.get_applicable_sqs("D2", EffectOfInterest.ADHERING, answers)
    assert set(branching.get_na_sqs("D2", EffectOfInterest.ADHERING, answers)) == {"2.6", "2.7"}


def test_d4_branching_keeps_4_1_and_4_2_ungated_and_skips_chain_on_problem_state() -> None:
    answers = ans(**{"4_1": A.Y, "4_2": A.N})

    assert branching.get_applicable_sqs("D4", EffectOfInterest.ASSIGNMENT, {}) == ["4.1", "4.2"]
    assert branching.get_applicable_sqs("D4", EffectOfInterest.ASSIGNMENT, answers) == []
    assert branching.get_na_sqs("D4", EffectOfInterest.ASSIGNMENT, answers) == ["4.3", "4.4", "4.5"]


def test_d5_branching_and_structural_na_judgment() -> None:
    answers = ans(**{"5_1": A.Y})
    judgment_answers = ans(**{"5_1": A.Y, "5_2": A.NA, "5_3": A.NA})

    assert branching.get_applicable_sqs("D5", EffectOfInterest.ASSIGNMENT, answers) == []
    assert branching.get_na_sqs("D5", EffectOfInterest.ASSIGNMENT, answers) == ["5.2", "5.3"]
    assert decision_tables.judge_domain_5(judgment_answers)[0] is Judgment.LOW


def _oracle_d1(a11: A, a12: A, a13: A) -> Judgment:
    if a12 in {A.Y, A.PY}:
        if a11 in {A.Y, A.PY, A.NI} and a13 in {A.N, A.PN, A.NI}:
            return Judgment.LOW
        if a11 in {A.N, A.PN} and a13 in {A.Y, A.PY, A.N, A.PN, A.NI}:
            return Judgment.SOME_CONCERNS
        if a11 in {A.Y, A.PY, A.NI} and a13 in {A.Y, A.PY}:
            return Judgment.SOME_CONCERNS
    if a12 == A.NI:
        if a11 in {A.Y, A.PY, A.PN, A.N, A.NI} and a13 in {A.N, A.PN, A.NI}:
            return Judgment.SOME_CONCERNS
        if a11 in {A.Y, A.PY, A.PN, A.N, A.NI} and a13 in {A.Y, A.PY}:
            return Judgment.HIGH
    if a12 in {A.N, A.PN}:
        return Judgment.HIGH
    raise AssertionError("unreachable for non-NA D1 oracle")


def _oracle_d2_assignment(a21: A, a22: A, a23: A, a24: A, a25: A, a26: A, a27: A) -> Judgment | None:
    part1 = ""
    part2 = ""
    if (a21 in {A.N, A.PN}) and (a22 in {A.N, A.PN}):
        part1 = "Low risk"
    elif a21 in {A.Y, A.PY, A.NI} or a22 in {A.Y, A.PY, A.NI}:
        if a23 in {A.N, A.PN}:
            part1 = "Low risk"
        elif a23 == A.NI:
            part1 = "Some concerns"
        elif a23 in {A.Y, A.PY}:
            if a24 in {A.N, A.PN}:
                part1 = "Some concerns"
            elif a24 in {A.NI, A.Y, A.PY}:
                if a25 in {A.Y, A.PY}:
                    part1 = "Some concerns"
                elif a25 in {A.N, A.PN, A.NI}:
                    part1 = "High risk"
    if a26 in {A.Y, A.PY}:
        part2 = "Low risk"
    elif a26 in {A.N, A.PN, A.NI}:
        if a27 in {A.N, A.PN}:
            part2 = "Some concerns"
        elif a27 in {A.Y, A.PY, A.NI}:
            part2 = "High risk"
    if part1 == "Low risk" and part2 == "Low risk":
        return Judgment.LOW
    if part1 and part2 and (part1 == "High risk" or part2 == "High risk"):
        return Judgment.HIGH
    if part1 and part2:
        return Judgment.SOME_CONCERNS
    return None


def _oracle_d2_adhering(a21: A, a22: A, a23: A, a24: A, a25: A, a26: A) -> Judgment | None:
    no_2425 = a24 in {A.N, A.PN, A.NA} and a25 in {A.N, A.PN, A.NA}
    if a21 in {A.N, A.PN} and a22 in {A.N, A.PN}:
        if no_2425:
            return Judgment.LOW
        if a24 in {A.Y, A.PY, A.NI} or a25 in {A.Y, A.PY, A.NI}:
            if a26 in {A.Y, A.PY}:
                return Judgment.SOME_CONCERNS
            if a26 in {A.N, A.PN, A.NI}:
                return Judgment.HIGH
    elif a21 in {A.Y, A.PY, A.NI} or a22 in {A.Y, A.PY, A.NI}:
        if a23 in {A.Y, A.PY, A.NA}:
            if no_2425:
                return Judgment.LOW
            if a24 in {A.Y, A.PY, A.NI} or a25 in {A.Y, A.PY, A.NI}:
                if a26 in {A.Y, A.PY}:
                    return Judgment.SOME_CONCERNS
                if a26 in {A.N, A.PN, A.NI}:
                    return Judgment.HIGH
        elif a23 in {A.N, A.PN, A.NI}:
            if a26 in {A.PY, A.Y}:
                return Judgment.SOME_CONCERNS
            if a26 in {A.N, A.PN, A.NI}:
                return Judgment.HIGH
    return None


def _oracle_d3(a31: A, a32: A, a33: A, a34: A) -> Judgment | None:
    if a31 in {A.Y, A.PY}:
        return Judgment.LOW
    if a31 in {A.N, A.PN, A.NI}:
        if a32 in {A.Y, A.PY}:
            return Judgment.LOW
        if a32 in {A.N, A.PN, A.NI}:
            if a33 in {A.N, A.PN}:
                return Judgment.LOW
            if a33 in {A.Y, A.PY, A.NI}:
                if a34 in {A.N, A.PN}:
                    return Judgment.SOME_CONCERNS
                if a34 in {A.Y, A.PY, A.NI}:
                    return Judgment.HIGH
    return None


def _oracle_d4(a41: A, a42: A, a43: A, a44: A, a45: A) -> Judgment | None:
    if a41 in {A.PY, A.Y} or a42 in {A.PY, A.Y}:
        return Judgment.HIGH
    if a41 in {A.N, A.PN, A.NI}:
        if a42 in {A.N, A.PN}:
            if a43 in {A.N, A.PN}:
                return Judgment.LOW
            if a43 in {A.Y, A.PY, A.NI}:
                if a44 in {A.N, A.PN}:
                    return Judgment.LOW
                if a44 in {A.Y, A.PY, A.NI}:
                    if a45 in {A.N, A.PN}:
                        return Judgment.SOME_CONCERNS
                    if a45 in {A.Y, A.PY, A.NI}:
                        return Judgment.HIGH
        if a42 == A.NI:
            if a43 in {A.N, A.PN}:
                return Judgment.SOME_CONCERNS
            if a43 in {A.Y, A.PY, A.NI}:
                if a44 in {A.N, A.PN}:
                    return Judgment.SOME_CONCERNS
                if a44 in {A.Y, A.PY, A.NI}:
                    if a45 in {A.N, A.PN}:
                        return Judgment.SOME_CONCERNS
                    if a45 in {A.Y, A.PY, A.NI}:
                        return Judgment.HIGH
    return None


def _oracle_d5(a51: A, a52: A, a53: A) -> Judgment | None:
    if a52 in {A.N, A.PN} and a53 in {A.N, A.PN}:
        if a51 in {A.Y, A.PY}:
            return Judgment.LOW
        if a51 in {A.N, A.PN, A.NI}:
            return Judgment.SOME_CONCERNS
    if a52 in {A.Y, A.PY} or a53 in {A.Y, A.PY}:
        return Judgment.HIGH
    if a51 in {A.Y, A.PY, A.N, A.PN, A.NI} and a52 in {A.Y, A.PY, A.N, A.PN, A.NI} and a53 in {
        A.Y,
        A.PY,
        A.N,
        A.PN,
        A.NI,
    }:
        return Judgment.SOME_CONCERNS
    return None
