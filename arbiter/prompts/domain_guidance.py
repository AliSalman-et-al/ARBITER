"""Domain-specific reasoning guidance for signaling-question prompts."""

from __future__ import annotations


def assessed_outcome_block(outcome: str) -> str:
    cleaned = " ".join(outcome.split())
    if not cleaned:
        return ""
    return "[Assessed outcome]\n" f"Assessed outcome: {cleaned}"


def domain_reasoning_guidance(sq_id: str) -> str:
    if not sq_id.startswith("4."):
        return ""

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
    return "\n".join(guidance)
