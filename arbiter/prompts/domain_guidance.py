"""Domain-specific reasoning guidance for signaling-question prompts."""

from __future__ import annotations


def assessed_outcome_block(outcome: str) -> str:
    cleaned = " ".join(outcome.split())
    if not cleaned:
        return ""
    return f"[Assessed outcome]\nAssessed outcome: {cleaned}"


def domain_reasoning_guidance(sq_id: str) -> str:
    domain = sq_id.split(".", 1)[0]
    if domain == "2":
        return _domain_2_guidance(sq_id)
    if domain == "3":
        return _domain_3_guidance(sq_id)
    if domain == "4":
        return _domain_4_guidance(sq_id)
    if domain == "5":
        return _domain_5_guidance(sq_id)
    return ""


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
    if sq_id in {"5.2", "5.3"}:
        guidance.append(
            "For multiplicity questions, identify the eligible alternatives for the same outcome "
            "domain: measurements or time points for 5.2, and analysis methods or populations for "
            "5.3."
        )
    return "\n".join(guidance)
