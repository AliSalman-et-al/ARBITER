"""RoB 2 signaling-question prompt templates.

Scope is owned by ``arbiter_algorithm.branching``. This module only stores the
wording discriminator used by the SQ worker: ``(sq_id, effect)`` with a
``both`` fallback for questions whose text does not vary by effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arbiter.models import EffectOfInterest


SQEffect = Literal["assignment", "adhering", "both"]
SQPromptKey = tuple[str, SQEffect]

ANSWER_BRIDGE = (
    "Use probable answers when the source supports a reasonable inference even "
    "without stating it outright. Reserve NI for genuine textual silence: no "
    "provided source gives any basis for answering. Do not answer NA; structural "
    "not-applicable decisions are set deterministically by branching."
)


@dataclass(frozen=True)
class SQPromptTemplate:
    sq_id: str
    effect: SQEffect
    question_text: str
    answer_definitions: str
    key_terms: list[str]


def get_sq_prompt(sq_id: str, effect: EffectOfInterest | str) -> SQPromptTemplate:
    """Return the effect-specific template, falling back to effect='both'."""

    effect_value = EffectOfInterest(effect).value
    key = (sq_id, effect_value)
    if key in SQ_PROMPTS:
        return SQ_PROMPTS[key]
    return SQ_PROMPTS[(sq_id, "both")]


def _defs(yes: str, no: str) -> str:
    return "\n".join(
        [
            f"Y: The source directly supports that {yes}.",
            f"PY: The source probably supports that {yes}, even if indirect or not fully explicit.",
            f"PN: The source probably supports that {no}, even if indirect or not fully explicit.",
            f"N: The source directly supports that {no}.",
            "NI: The provided source text gives no basis for answering this signaling question.",
            ANSWER_BRIDGE,
        ]
    )


def _template(
    sq_id: str,
    effect: SQEffect,
    question_text: str,
    yes: str,
    no: str,
    key_terms: list[str],
) -> SQPromptTemplate:
    return SQPromptTemplate(
        sq_id=sq_id,
        effect=effect,
        question_text=question_text,
        answer_definitions=_defs(yes, no),
        key_terms=key_terms,
    )


SQ_PROMPTS: dict[SQPromptKey, SQPromptTemplate] = {
    ("1.1", "both"): _template(
        "1.1",
        "both",
        "Was the allocation sequence random?",
        "the allocation sequence was random",
        "the allocation sequence was not random",
        ["random", "randomized", "randomised", "sequence generation", "computer generated", "random number", "minimisation", "stratified"],
    ),
    ("1.2", "both"): _template(
        "1.2",
        "both",
        "Was the allocation sequence concealed until participants were enrolled and assigned to interventions?",
        "the allocation sequence was concealed until assignment",
        "the allocation sequence was not concealed until assignment",
        ["concealment", "allocation concealment", "sealed envelope", "opaque envelope", "central pharmacy", "IWRS", "central randomisation"],
    ),
    ("1.3", "both"): _template(
        "1.3",
        "both",
        "Did baseline differences between intervention groups suggest a problem with the randomisation process?",
        "baseline differences suggested a problem with randomisation",
        "baseline differences did not suggest a problem with randomisation",
        ["baseline characteristics", "table 1", "demographics", "imbalance", "covariate"],
    ),
    ("2.1", "both"): _template(
        "2.1",
        "both",
        "Were participants aware of their assigned intervention during the trial?",
        "participants were aware of their assigned intervention",
        "participants were not aware of their assigned intervention",
        ["blinding", "masking", "open-label", "placebo", "participant", "assigned intervention"],
    ),
    ("2.2", "both"): _template(
        "2.2",
        "both",
        "Were carers and people delivering the interventions aware of participants' assigned intervention during the trial?",
        "carers or intervention deliverers were aware of assigned intervention",
        "carers or intervention deliverers were not aware of assigned intervention",
        ["blinding", "masking", "open-label", "placebo", "carer", "clinician", "investigator"],
    ),
    ("2.3", "assignment"): _template(
        "2.3",
        "assignment",
        "[If Y/PY/NI to 2.1 or 2.2] Were there deviations from the intended intervention that arose because of the experimental context?",
        "deviations from intended intervention arose because of the experimental context",
        "deviations from intended intervention did not arise because of the experimental context",
        ["deviation", "protocol deviation", "experimental context", "co-intervention", "concomitant", "crossover", "discontinued"],
    ),
    ("2.4", "assignment"): _template(
        "2.4",
        "assignment",
        "[If Y/PY to 2.3] Were these deviations likely to have affected the outcome?",
        "the deviations were likely to have affected the outcome",
        "the deviations were unlikely to have affected the outcome",
        ["deviation", "affected outcome", "outcome impact", "protocol deviation", "crossover", "discontinued"],
    ),
    ("2.5", "assignment"): _template(
        "2.5",
        "assignment",
        "[If Y/PY/NI to 2.4] Were these deviations from intended intervention balanced between groups?",
        "deviations from intended intervention were balanced between groups",
        "deviations from intended intervention were not balanced between groups",
        ["deviation", "balanced between groups", "imbalance", "intervention groups", "protocol deviation"],
    ),
    ("2.6", "assignment"): _template(
        "2.6",
        "assignment",
        "Was an appropriate analysis used to estimate the effect of assignment to intervention?",
        "the analysis appropriately estimated the effect of assignment to intervention",
        "the analysis did not appropriately estimate the effect of assignment to intervention",
        ["intention-to-treat", "ITT", "modified ITT", "full analysis set", "randomised", "randomized", "assigned group"],
    ),
    ("2.7", "both"): _template(
        "2.7",
        "both",
        "[If N/PN/NI to 2.6] Was there potential for a substantial impact (on the result) of the failure to analyse participants in the group to which they were randomised?",
        "failure to analyse participants in their randomised group could substantially impact the result",
        "failure to analyse participants in their randomised group could not substantially impact the result",
        ["intention-to-treat", "ITT", "as treated", "per-protocol", "excluded", "randomised group", "impact"],
    ),
    ("2.3", "adhering"): _template(
        "2.3",
        "adhering",
        "[If Y/PY/NI to 2.1 or 2.2] Were important non-protocol interventions balanced across intervention groups?",
        "important non-protocol interventions were balanced across intervention groups",
        "important non-protocol interventions were not balanced across intervention groups",
        ["non-protocol intervention", "co-intervention", "concomitant", "balanced", "intervention groups"],
    ),
    ("2.4", "adhering"): _template(
        "2.4",
        "adhering",
        "[If applicable] Were there failures in implementing the intervention that could have affected the outcome?",
        "there were failures in implementing the intervention that could have affected the outcome",
        "there were no failures in implementing the intervention that could have affected the outcome",
        ["implementation", "fidelity", "dose received", "intervention delivery", "protocol", "affected outcome"],
    ),
    ("2.5", "adhering"): _template(
        "2.5",
        "adhering",
        "[If applicable] Was there non-adherence to the assigned intervention regimen that could have affected participants' outcomes?",
        "there was non-adherence that could have affected participants' outcomes",
        "there was no non-adherence that could have affected participants' outcomes",
        ["adherence", "compliance", "non-adherence", "assigned regimen", "dose received", "discontinued"],
    ),
    ("2.6", "adhering"): _template(
        "2.6",
        "adhering",
        "[If N/PN/NI to 2.3, or Y/PY/NI to 2.4 or 2.5] Was an appropriate analysis used to estimate the effect of adhering to the intervention?",
        "the analysis appropriately estimated the effect of adhering to the intervention",
        "the analysis did not appropriately estimate the effect of adhering to the intervention",
        ["per-protocol", "adherence-adjusted", "instrumental variable", "adhering", "compliance", "as treated"],
    ),
    ("3.1", "both"): _template(
        "3.1",
        "both",
        "Were data for this outcome available for all, or nearly all, randomised participants?",
        "outcome data were available for all or nearly all randomised participants",
        "outcome data were not available for all or nearly all randomised participants",
        ["missing data", "lost to follow-up", "dropout", "withdrawal", "analysed", "completeness"],
    ),
    ("3.2", "both"): _template(
        "3.2",
        "both",
        "[If N/PN/NI to 3.1] Is there evidence that the result was not biased by missing outcome data?",
        "there is evidence the result was not biased by missing outcome data",
        "there is not evidence the result was not biased by missing outcome data",
        ["missing data", "sensitivity analysis", "imputation", "tipping point", "complete case", "bias"],
    ),
    ("3.3", "both"): _template(
        "3.3",
        "both",
        "[If N/PN to 3.2] Could missingness in the outcome depend on its true value?",
        "missingness in the outcome could depend on its true value",
        "missingness in the outcome could not depend on its true value",
        ["missingness", "true value", "MAR", "MNAR", "MCAR", "reason for withdrawal", "informative censoring"],
    ),
    ("3.4", "both"): _template(
        "3.4",
        "both",
        "[If Y/PY/NI to 3.3] Is it likely that missingness in the outcome depended on its true value?",
        "missingness in the outcome likely depended on its true value",
        "missingness in the outcome likely did not depend on its true value",
        ["missingness", "true value", "informative censoring", "disease progression", "death", "withdrawal reason"],
    ),
    ("4.1", "both"): _template(
        "4.1",
        "both",
        "Was the method of measuring the outcome inappropriate?",
        "the outcome measurement method was inappropriate",
        "the outcome measurement method was appropriate",
        ["outcome measure", "validated instrument", "scale", "endpoint definition", "measurement method"],
    ),
    ("4.2", "both"): _template(
        "4.2",
        "both",
        "Could measurement or ascertainment of the outcome have differed between intervention groups?",
        "outcome measurement or ascertainment could have differed between intervention groups",
        "outcome measurement or ascertainment could not have differed between intervention groups",
        ["assessment schedule", "ascertainment", "measurement", "differed between groups", "intervention groups"],
    ),
    ("4.3", "both"): _template(
        "4.3",
        "both",
        "[If N/PN/NI to 4.1 and 4.2] Were outcome assessors aware of the intervention received by study participants?",
        "outcome assessors were aware of the intervention received",
        "outcome assessors were not aware of the intervention received",
        ["outcome assessor", "blinded assessor", "masked assessor", "adjudication", "endpoint committee", "central review"],
    ),
    ("4.4", "both"): _template(
        "4.4",
        "both",
        "[If Y/PY/NI to 4.3] Could assessment of the outcome have been influenced by knowledge of the intervention received?",
        "outcome assessment could have been influenced by knowledge of intervention received",
        "outcome assessment could not have been influenced by knowledge of intervention received",
        ["subjective", "objective", "patient-reported", "clinician-assessed", "hard endpoint", "mortality"],
    ),
    ("4.5", "both"): _template(
        "4.5",
        "both",
        "[If Y/PY/NI to 4.4] Is it likely that assessment of the outcome was influenced by knowledge of the intervention received?",
        "outcome assessment was likely influenced by knowledge of intervention received",
        "outcome assessment was likely not influenced by knowledge of intervention received",
        ["subjective", "objective", "blinding", "assessment influence", "patient-reported", "clinician-assessed"],
    ),
    ("5.1", "both"): _template(
        "5.1",
        "both",
        "Were the data that produced this result analysed in accordance with a pre-specified analysis plan that was finalised before unblinded outcome data were available?",
        "the result was analysed according to a pre-specified analysis plan finalised before unblinded outcome data were available",
        "the result was not analysed according to a pre-specified analysis plan finalised before unblinded outcome data were available",
        ["pre-specified", "pre-registered", "protocol", "statistical analysis plan", "SAP", "registry", "ClinicalTrials.gov"],
    ),
    ("5.2", "both"): _template(
        "5.2",
        "both",
        "[Is the result] selected from multiple eligible outcome measurements on the basis of the results?",
        "the result was selected from multiple eligible outcome measurements on the basis of the results",
        "the result was not selected from multiple eligible outcome measurements on the basis of the results",
        ["outcome switching", "selective reporting", "multiple measurements", "scales", "definitions", "timepoints"],
    ),
    ("5.3", "both"): _template(
        "5.3",
        "both",
        "[Is the result] selected from multiple eligible analyses of the data on the basis of the results?",
        "the result was selected from multiple eligible analyses of the data on the basis of the results",
        "the result was not selected from multiple eligible analyses of the data on the basis of the results",
        ["post hoc", "unplanned analysis", "multiplicity", "subgroup", "multiple analyses", "selective reporting"],
    ),
}

