from __future__ import annotations

from arbiter.prompts.domain_guidance import domain_reasoning_guidance


def test_domain_1_guidance_judges_questions_separately() -> None:
    guidance = domain_reasoning_guidance("1.1")

    assert "Domain 1 reasoning guidance" in guidance
    assert "sequence generation (1.1)" in guidance
    assert "allocation concealment (1.2)" in guidance
    assert "baseline imbalance (1.3)" in guidance
    assert "Use NI only when PY/PN would be unreasonable" in guidance


def test_domain_1_2_guidance_anchors_central_and_restricted_allocation() -> None:
    guidance = domain_reasoning_guidance("1.2")

    assert "centrally controlled" in guidance
    assert "access to the sequence was restricted" in guidance
    assert "large multicentre trial with stratified randomization" in guidance
    assert "balanced baseline characteristics" in guidance
    assert "Disclosure to investigators after enrolment" in guidance


def test_domain_1_guidance_maps_registry_randomization_without_detail_to_py() -> None:
    guidance = domain_reasoning_guidance("1.1")

    assert "ClinicalTrials.gov Allocation: RANDOMIZED" in guidance
    assert "supports PY rather than Y for 1.1" in guidance
    assert "not evidence of allocation concealment" in guidance


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


def test_domain_2_guidance_maps_registry_masking_none_to_awareness() -> None:
    guidance = domain_reasoning_guidance("2.1")

    assert "ClinicalTrials.gov Masking: NONE" in guidance
    assert "participants and carers or intervention deliverers were aware" in guidance
    assert "supports Y/PY for 2.1 and 2.2" in guidance


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


def test_domain_3_1_guidance_treats_imputation_and_early_censoring_as_missing() -> None:
    guidance = domain_reasoning_guidance("3.1")

    assert "Imputed values count as missing outcome data" in guidance
    assert "administratively censored at end of follow-up are not missing" in guidance
    assert "withdrew, were lost to follow-up, or switched treatment" in guidance
    assert "early non-event censoring is large relative to observed events" in guidance


def test_domain_3_4_guidance_checks_censoring_imbalance() -> None:
    guidance = domain_reasoning_guidance("3.4")

    assert "censoring rates differ between arms" in guidance
    assert "reasons for missingness that differ between groups" in guidance
    assert "relate to prognosis/the outcome" in guidance


def test_domain_4_guidance_covers_outcome_measurement_characteristics() -> None:
    guidance = domain_reasoning_guidance("4.3")

    assert "objective/hard endpoint" in guidance
    assert "participant-reported endpoint" in guidance
    assert "clinician- or assessor-judged endpoint" in guidance
    assert "threshold choices" in guidance
    assert "Do not infer measurement bias from lack of blinding alone" in guidance


def test_domain_4_guidance_balances_objectivity_assumptions() -> None:
    guidance = domain_reasoning_guidance("4.4")

    assert "do not assume objectivity from generic terms" in guidance
    assert "progression, response, clinical event, or composite endpoint" in guidance
    assert "whether adjudication was blinded" in guidance


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


def test_domain_5_1_guidance_treats_registration_as_prespecification_signal() -> None:
    guidance = domain_reasoning_guidance("5.1")

    assert "trial registration that predates the primary analysis" in guidance
    assert "endpoints or the statistical analysis plan were pre-specified" in guidance
    assert "Do not require every statistical detail" in guidance
    assert "plan changed after unblinding" in guidance


def test_domain_5_2_guidance_separates_composites_from_measurement_multiplicity() -> None:
    guidance = domain_reasoning_guidance("5.2")

    assert "pre-specified composite endpoint" in guidance
    assert "pre-specified co-primary endpoints" in guidance
    assert "not multiple eligible outcome measurements" in guidance
    assert "separately pre-specified alternatives" in guidance
    assert "distinct registry endpoint is a different outcome" in guidance
