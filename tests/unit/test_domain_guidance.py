from __future__ import annotations

from arbiter.prompts.domain_guidance import domain_reasoning_guidance


def test_domain_reasoning_guidance_is_empty_outside_domain_4() -> None:
    assert domain_reasoning_guidance("3.1") == ""


def test_domain_4_guidance_covers_outcome_measurement_characteristics() -> None:
    guidance = domain_reasoning_guidance("4.3")

    assert "objective/hard endpoint" in guidance
    assert "participant-reported endpoint" in guidance
    assert "clinician- or assessor-judged endpoint" in guidance
    assert "threshold choices" in guidance
    assert "Do not infer measurement bias from lack of blinding alone" in guidance


def test_domain_4_2_guidance_distinguishes_visit_cadence_from_measurement_method() -> None:
    guidance = domain_reasoning_guidance("4.2")

    assert "Different clinic visit frequency" in guidance
    assert "not enough by itself" in guidance
    assert "measurement method, source, threshold, cutoff, or timing window" in guidance
