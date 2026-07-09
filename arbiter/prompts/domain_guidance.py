"""Domain-specific reasoning guidance for signaling-question prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arbiter.models import OutcomeMeasurementProfile


def assessed_outcome_block(outcome: str) -> str:
    cleaned = " ".join(outcome.split())
    if not cleaned:
        return ""
    return f"[Assessed outcome]\nAssessed outcome: {cleaned}"


def outcome_measurement_profile_block(
    profile: OutcomeMeasurementProfile | Mapping[str, Any] | None,
    sq_id: str,
) -> str:
    domain = sq_id.split(".", 1)[0]
    if domain not in {"4", "5"}:
        return ""
    coerced = _coerce_outcome_measurement_profile(profile)
    if coerced is None:
        return ""

    lines = [
        "[Outcome measurement profile]",
        (
            "This derived profile is non-citable orientation and an advisory "
            "reasoning frame; do not copy it into quote."
        ),
        f"Outcome type: {coerced.profile.value}",
        f"Definition: {coerced.definition}",
    ]
    if coerced.matched_registered_outcome:
        lines.append(f"Matched registered outcome: {coerced.matched_registered_outcome}")
    if coerced.basis:
        lines.append(f"Basis: {coerced.basis}")
    return "\n".join(lines)


def _coerce_outcome_measurement_profile(
    profile: OutcomeMeasurementProfile | Mapping[str, Any] | None,
) -> OutcomeMeasurementProfile | None:
    if profile is None:
        return None
    if isinstance(profile, OutcomeMeasurementProfile):
        return profile
    if isinstance(profile, Mapping):
        return OutcomeMeasurementProfile.model_validate(profile)
    return None


def domain_reasoning_guidance(sq_id: str) -> str:
    domain = sq_id.split(".", 1)[0]
    if domain == "1":
        return _domain_1_guidance(sq_id)
    if domain == "2":
        return _domain_2_guidance(sq_id)
    if domain == "3":
        return _domain_3_guidance(sq_id)
    if domain == "4":
        return _domain_4_guidance(sq_id)
    if domain == "5":
        return _domain_5_guidance(sq_id)
    return ""


def _domain_1_guidance(sq_id: str) -> str:
    guidance = [
        "[Domain 1 reasoning guidance]",
        (
            "Judge sequence generation (1.1), allocation concealment (1.2), and "
            "baseline imbalance (1.3) separately. Use NI only when PY/PN would be "
            "unreasonable."
        ),
    ]
    if sq_id == "1.2":
        guidance.append(
            "For 1.2, score Y/PY when allocation was centrally controlled, such "
            "as central, pharmacy, telephone, or web randomization, or access to "
            'the sequence was restricted, such as "only the data manager held the '
            'list" or "the sequence was not disclosed until after enrolment." For '
            "a large multicentre trial with stratified randomization and balanced "
            "baseline characteristics, answer PY rather than NI even if the exact "
            'concealment mechanism is unnamed. Reserve NI for reports that only say '
            '"randomized" with no infrastructure, stratification, or baseline-balance '
            "context. Disclosure to investigators after enrolment is expected and "
            "does not imply the allocation was known before enrolment."
        )
    return "\n".join(guidance)


def _domain_2_guidance(sq_id: str) -> str:
    guidance = [
        "[Domain 2 reasoning guidance]",
        (
            "Keep the effect of interest in view. For the effect of assignment to intervention, "
            "count deviations only when they are inconsistent with the intended intervention, "
            "arose because of the experimental context, and could affect the assessed outcome."
        ),
        (
            "For the effect of adhering to intervention, focus on protocol-relevant non-protocol "
            "interventions, implementation failures, or non-adherence that could affect the "
            "assessed outcome, whether or not they arose because of the experimental context."
        ),
        (
            "Do not infer bias from open-label design alone. Link any Y/PY answer to source text "
            "showing awareness, a deviation or non-adherence mechanism, likely outcome impact, "
            "imbalance between groups, or an analysis that does not match the effect of interest."
        ),
        (
            "NI is a last resort. Do not answer NI merely because the report omits an explicit "
            "statement that ordinary care was unrelated to the trial context; infer PN when "
            "routine, protocol-consistent clinical management is described. Reserve NI for cases "
            "where a deviation is actually described but its origin genuinely cannot be judged."
        ),
    ]
    if sq_id == "2.3":
        guidance.append(
            "For 2.3, count only changes from the assigned intervention that are inconsistent "
            "with the protocol and arose because of the trial context, such as recruitment, "
            "engagement, unblinding, or trial personnel undermining the protocol in ways that "
            "would not happen in routine care. Do not count protocol-consistent care: dose "
            "reduction or cessation for toxicity, treatment changes made after an outcome event, "
            "or additional interventions used to manage side effects of the assigned treatment. "
            "Awareness of assignment in an open-label trial is not itself a deviation."
        )
    if sq_id == "2.4":
        guidance.append(
            "For 2.4: Answer only about deviations identified in 2.3, and only in relation "
            "to the specific assessed outcome. Evidence about a different topic, such as how "
            "adverse events were documented, is not a basis for judging whether intervention "
            "deviations affected this outcome; if no relevant deviation bears on the outcome, "
            "PN/N is appropriate."
        )
    if sq_id == "2.6":
        guidance.append(
            "For analysis questions, distinguish the target estimand: analyses preserving randomized "
            "assignment usually support assignment effects, while adherence effects need analyses "
            "that address post-randomization adherence or protocol deviations."
        )
        guidance.append(
            "For 2.6: Strict intention-to-treat and modified ITT that excludes only participants "
            "with missing outcome data are appropriate for the effect of assignment. Naive "
            "per-protocol and as-treated analyses are inappropriate. Post-randomization exclusion "
            "of participants later found ineligible is appropriate only when eligibility could not "
            "have been influenced by the assigned group."
        )
    return "\n".join(guidance)


def _domain_3_guidance(sq_id: str) -> str:
    guidance = [
        "[Domain 3 reasoning guidance]",
        (
            "Focus on whether outcome data for the assessed result were actually available for "
            "randomized participants. Imputed or modelled values can address missingness, but they "
            "are not the same as observed outcome data."
        ),
        (
            "Missing outcome data create bias when the missingness could depend on the true outcome "
            "value and the analysis does not adequately address that risk. Do not infer bias from "
            "any missing data alone; assess extent, reasons, balance between groups, and sensitivity "
            "analyses."
        ),
    ]
    if sq_id in {"3.3", "3.4"}:
        guidance.append(
            "For true-value dependence, look for reasons such as deterioration, adverse events, lack "
            "of efficacy, death, withdrawal due to symptoms, or loss to follow-up patterns that make "
            "the unobserved outcome likely different from observed outcomes."
        )
    if sq_id == "3.1":
        guidance.append(
            "For 3.1: Imputed values count as missing outcome data for this question. "
            "For time-to-event outcomes, participants administratively censored at "
            "end of follow-up are not missing; participants censored early because "
            "they withdrew, were lost to follow-up, or switched treatment may be missing. "
            "If early non-event censoring is large relative to observed events, do "
            "not answer Y without checking whether censoring is informative."
        )
    if sq_id == "3.4":
        guidance.append(
            "For 3.4 and time-to-event outcomes, check whether censoring rates differ "
            "between arms; a meaningful difference, or reasons for missingness that "
            "differ between groups or relate to prognosis/the outcome, supports Y/PY."
        )
    if sq_id == "3.2":
        guidance.append(
            "For evidence that the result was not biased, prefer direct sensitivity analyses, robust "
            "assumptions, or very small/balanced missingness over generic statements that missing "
            "data were handled."
        )
    return "\n".join(guidance)


def _domain_4_guidance(sq_id: str) -> str:
    guidance = [
        "[Domain 4 reasoning guidance]",
        (
            "First consider the outcome's measurement characteristics from the assessed outcome "
            "and source text: objective/hard endpoint, laboratory or record-based endpoint, "
            "clinician- or assessor-judged endpoint, participant-reported endpoint, or endpoint "
            "with adjudication/threshold choices."
        ),
        (
            "Objective outcomes usually have less room for assessor judgement; participant-reported, "
            "clinician-assessed, imaging-assessed, composite, and threshold-dependent outcomes usually "
            "have more room for measurement or assessor influence."
        ),
        (
            "Do not infer measurement bias from lack of blinding alone. Link any Y/PY answer to a "
            "plausible mechanism in the source text: different method, source, threshold, cutoff, "
            "timing protocol, assessor awareness, or participant/assessor judgement affecting the "
            "recorded outcome."
        ),
    ]
    if sq_id == "4.2":
        guidance.append(
            "For 4.2, compare outcome measurement or ascertainment between intervention groups. "
            "Different clinic visit frequency, assessment schedule, or follow-up cadence is not "
            "enough by itself; use it only if the source supports that it changed the measurement "
            "method, source, threshold, cutoff, or timing window for the outcome."
        )
        guidance.append(
            "Differential clinic-visit frequency changes ascertainment only for outcomes detected "
            "at study visits, such as asymptomatic or imaging-detected progression, investigator- "
            "or screen-detected events. For outcomes ascertained independently of the visit "
            "schedule - all-cause mortality / death from any cause, and events captured through "
            "continuous vital-status follow-up or registry linkage - more frequent visits in one "
            "arm do not change whether or when the outcome is recorded, so answer N/PN unless "
            "the source shows the measurement method, criteria, or thresholds themselves differed "
            "between arms. Extra visits that exist only to administer the experimental treatment "
            "are not additional outcome-assessment occasions when both arms use the same "
            "pre-specified assessment protocol."
        )
    if sq_id == "4.3":
        guidance.append(
            "If the trial is open-label, with participants or personnel unblinded, and the report "
            "describes no central blinded adjudication committee or independent blinded assessor, "
            "answer PY because assessors were likely aware. Reserve NI for cases where assessor "
            "blinding genuinely cannot be inferred."
        )
    if sq_id == "4.4":
        guidance.append(
            "For outcomes whose recorded value does not depend on who assesses them - all-cause "
            "mortality/death, laboratory values, automated measurements, centrally and blindly "
            "adjudicated events - answer N/PN because knowledge of assignment cannot change the "
            "recorded value. Reserve Y/PY for participant-reported outcomes and observer or "
            "clinician assessments that involve judgement."
        )
    if sq_id == "4.5":
        guidance.append(
            "Distinguish could have been influenced, which supports Some concerns, from likely was "
            "influenced, which supports High. Answer Y/PY only with a concrete mechanism, such as "
            "strong beliefs about benefit/harm or an assessor who also delivered the intervention. "
            "When standardized criteria are applied without such a mechanism, answer N/PN."
        )
    return "\n".join(guidance)


def _domain_5_guidance(sq_id: str) -> str:
    guidance = [
        "[Domain 5 reasoning guidance]",
        (
            "Assess selection of the reported result for the specific assessed outcome/result, not "
            "whether the study reported many outcomes or analyses in general."
        ),
        (
            "Use protocols, registries, statistical analysis plans, and methods text to decide "
            "whether the reported measurement, time point, analysis population, model, covariate "
            "adjustment, or effect estimate was pre-specified before unblinded outcome data were "
            "available."
        ),
        (
            "Do not infer reporting bias just because multiple outcomes, time points, or analyses "
            "exist. Link any Y/PY answer to source text showing multiple eligible choices for this "
            "result and a plausible basis for selecting the reported one because of the results."
        ),
    ]
    if sq_id == "5.1":
        guidance.append(
            "For pre-specification, distinguish a finalized prospective analysis plan from post hoc "
            "methods, late registry edits, or incomplete plans that do not identify this result's "
            "analysis."
        )
        guidance.append(
            "For 5.1, a cited trial registration that predates the primary analysis, "
            "or an explicit statement that endpoints or the statistical analysis plan "
            "were pre-specified, supports Y/PY. Do not require every statistical "
            "detail, such as covariates, imputation, or sensitivity analyses, to be "
            "reprinted in the paper. Reserve PN for specific evidence that the plan "
            "changed after unblinding, or no registration and no pre-specification "
            "anywhere."
        )
    if sq_id in {"5.2", "5.3"}:
        guidance.append(
            "For multiplicity questions, identify the eligible alternatives for the same outcome "
            "domain: measurements or time points for 5.2, and analysis methods or populations for "
            "5.3."
        )
    if sq_id == "5.2":
        guidance.append(
            "For 5.2, a pre-specified composite endpoint or pre-specified co-primary "
            "endpoints are not multiple eligible outcome measurements of one another. "
            "Answer Y/PY only when one scale, definition, component, or time point "
            "was chosen from several separately pre-specified alternatives on the "
            "basis of the observed results. A distinct registry endpoint is a "
            "different outcome, not an alternative measurement of this one."
        )
    return "\n".join(guidance)
