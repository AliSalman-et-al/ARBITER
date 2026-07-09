from __future__ import annotations

from arbiter.prompts.domain_guidance import domain_reasoning_guidance


def test_domain_reasoning_guidance_leaves_domain_1_unguided() -> None:
    assert domain_reasoning_guidance("1.1") == ""


def test_domain_2_guidance_keeps_effect_of_interest_in_view() -> None:
    guidance = domain_reasoning_guidance("2.6")

    assert "effect of assignment to intervention" in guidance
    assert "effect of adhering to intervention" in guidance
    assert "Do not infer bias from open-label design alone" in guidance
    assert "target estimand" in guidance


def test_domain_2_guidance_treats_ni_as_last_resort() -> None:
    guidance = domain_reasoning_guidance("2.3")

    assert "NI is a last resort" in guidance
    assert "infer PN when routine, protocol-consistent clinical management" in guidance
    assert "Reserve NI for cases where a deviation is actually described" in guidance


def test_domain_2_3_guidance_excludes_protocol_consistent_care() -> None:
    guidance = domain_reasoning_guidance("2.3")

    assert "inconsistent with the protocol" in guidance
    assert "arose because of the trial context" in guidance
    assert "Do not count protocol-consistent care" in guidance
    assert "dose reduction or cessation for toxicity" in guidance
    assert "Awareness of assignment in an open-label trial is not itself a deviation" in guidance


def test_domain_2_4_guidance_requires_relevant_deviation_and_outcome() -> None:
    guidance = domain_reasoning_guidance("2.4")

    assert "Answer only about deviations identified in 2.3" in guidance
    assert "specific assessed outcome" in guidance
    assert "Evidence about a different topic" in guidance
    assert "if no relevant deviation bears on the outcome, PN/N is appropriate" in guidance


def test_domain_2_6_guidance_identifies_appropriate_assignment_analyses() -> None:
    guidance = domain_reasoning_guidance("2.6")

    assert "Strict intention-to-treat" in guidance
    assert "modified ITT that excludes only participants with missing outcome data" in guidance
    assert "Naive per-protocol and as-treated analyses are inappropriate" in guidance
    assert "Post-randomization exclusion of participants later found ineligible" in guidance


def test_domain_3_guidance_distinguishes_missing_data_from_bias() -> None:
    guidance = domain_reasoning_guidance("3.3")

    assert "observed outcome data" in guidance
    assert "Do not infer bias from any missing data alone" in guidance
    assert "could depend on the true outcome value" in guidance
    assert "deterioration" in guidance


def test_domain_4_guidance_covers_outcome_measurement_characteristics() -> None:
    guidance = domain_reasoning_guidance("4.3")

    assert "objective/hard endpoint" in guidance
    assert "participant-reported endpoint" in guidance
    assert "clinician- or assessor-judged endpoint" in guidance
    assert "threshold choices" in guidance
    assert "Do not infer measurement bias from lack of blinding alone" in guidance


def test_domain_4_2_guidance_distinguishes_visit_cadence_from_measurement_method() -> (
    None
):
    guidance = domain_reasoning_guidance("4.2")

    assert "Different clinic visit frequency" in guidance
    assert "not enough by itself" in guidance
    assert "measurement method, source, threshold, cutoff, or timing window" in guidance


def test_domain_4_2_guidance_anchors_visit_detected_vs_independent_ascertainment() -> (
    None
):
    guidance = domain_reasoning_guidance("4.2")

    assert "outcomes detected at study visits" in guidance
    assert "all-cause mortality / death from any cause" in guidance
    assert "continuous vital-status follow-up or registry linkage" in guidance
    assert "Extra visits that exist only to administer the experimental treatment" in guidance


def test_domain_4_3_guidance_infers_open_label_assessor_awareness_when_unrebutted() -> (
    None
):
    guidance = domain_reasoning_guidance("4.3")

    assert "open-label" in guidance
    assert "central blinded adjudication committee" in guidance
    assert "answer PY" in guidance
    assert "Reserve NI" in guidance


def test_domain_4_4_guidance_separates_fixed_values_from_judgement() -> None:
    guidance = domain_reasoning_guidance("4.4")

    assert "recorded value does not depend on who assesses them" in guidance
    assert "all-cause mortality/death" in guidance
    assert "laboratory values" in guidance
    assert "centrally and blindly adjudicated events" in guidance
    assert "participant-reported outcomes" in guidance


def test_domain_4_5_guidance_requires_concrete_influence_mechanism() -> None:
    guidance = domain_reasoning_guidance("4.5")

    assert "could have been influenced" in guidance
    assert "likely was influenced" in guidance
    assert "strong beliefs about benefit/harm" in guidance
    assert "assessor who also delivered the intervention" in guidance
    assert "standardized criteria" in guidance


def test_domain_5_guidance_focuses_on_result_selection_not_multiplicity_alone() -> None:
    guidance = domain_reasoning_guidance("5.3")

    assert "specific assessed outcome/result" in guidance
    assert "Do not infer reporting bias just because multiple outcomes" in guidance
    assert "analysis methods or populations" in guidance
    assert "because of the results" in guidance
