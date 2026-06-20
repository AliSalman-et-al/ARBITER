from __future__ import annotations

import pytest

from arbiter.arbiter_algorithm.branching import DOMAIN_SQS
from arbiter.models import EffectOfInterest
from arbiter.prompts.sq_prompts import SQ_PROMPTS, SQPromptTemplate, get_sq_prompt


def test_sq_prompts_cover_22_positions_across_26_templates() -> None:
    expected_positions = {sq_id for sq_ids in DOMAIN_SQS.values() for sq_id in sq_ids}

    assert len(expected_positions) == 22
    assert {sq_id for sq_id, _ in SQ_PROMPTS} == expected_positions
    assert len(SQ_PROMPTS) == 26


def test_d2_shared_ids_are_keyed_by_effect() -> None:
    for sq_id in ("2.3", "2.4", "2.5", "2.6"):
        assert (sq_id, "assignment") in SQ_PROMPTS
        assert (sq_id, "adhering") in SQ_PROMPTS
        assert SQ_PROMPTS[(sq_id, "assignment")].question_text != SQ_PROMPTS[(sq_id, "adhering")].question_text


def test_d2_4_adhering_uses_per_protocol_wording() -> None:
    adhering = get_sq_prompt("2.4", EffectOfInterest.ADHERING)
    assignment = get_sq_prompt("2.4", EffectOfInterest.ASSIGNMENT)

    assert "failures in implementing the intervention" in adhering.question_text
    assert "deviations likely to have affected the outcome" in assignment.question_text
    assert adhering.effect == "adhering"
    assert assignment.effect == "assignment"


def test_both_fallback_for_effect_invariant_questions() -> None:
    assert get_sq_prompt("1.1", EffectOfInterest.ASSIGNMENT) is SQ_PROMPTS[("1.1", "both")]
    assert get_sq_prompt("5.3", EffectOfInterest.ADHERING) is SQ_PROMPTS[("5.3", "both")]


def test_2_7_is_template_keyed_both_but_not_adhering_specific() -> None:
    assert ("2.7", "both") in SQ_PROMPTS
    assert ("2.7", "adhering") not in SQ_PROMPTS
    assert ("2.7", "assignment") not in SQ_PROMPTS


def test_templates_have_required_fields_and_no_scope_field() -> None:
    for key, template in SQ_PROMPTS.items():
        assert key == (template.sq_id, template.effect)
        assert isinstance(template, SQPromptTemplate)
        assert template.question_text.strip()
        assert template.answer_definitions.strip()
        assert template.key_terms
        assert not hasattr(template, "applies_to")


@pytest.mark.parametrize("code", ["Y:", "PY:", "PN:", "N:", "NI:"])
def test_answer_definitions_include_each_answer_code(code: str) -> None:
    for template in SQ_PROMPTS.values():
        assert code in template.answer_definitions


def test_answer_definitions_include_py_pn_bridge_and_forbid_na() -> None:
    for template in SQ_PROMPTS.values():
        definitions = template.answer_definitions
        assert "reasonable inference" in definitions
        assert "Reserve NI for genuine textual silence" in definitions
        assert "Do not answer NA" in definitions

