"""Overall RoB 2 judgment rollup for ARBITER.

Policy note: see `docs/adr/0001-overall-judgment-some-concerns-threshold.md`.
The domain judgments are verbatim RoB 2 IRPG table outputs; the threshold for
turning multiple Some concerns domains into an overall High judgment is an
ARBITER unattended-assessment policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from arbiter.models import ConfidenceFlag, DomainJudgment, Judgment


OVERALL_HIGH_SC_THRESHOLD = 3


def compute_overall_judgment(domain_judgments: Sequence[DomainJudgment]) -> tuple[Judgment, str, bool]:
    """Compute overall judgment using ADR-0001's deterministic rollup policy.

    Low if all domains are Low; High if any domain is High or if the count of
    Some concerns domains is at least `OVERALL_HIGH_SC_THRESHOLD`; otherwise
    Some concerns. `requires_human_review` is true only on policy-driven
    multi-Some-concerns paths derived from the same threshold.
    """

    if len(domain_judgments) != 5:
        raise ValueError("Exactly five domain judgments (D1..D5) are required")

    judgments = [_judgment(item.judgment) for item in domain_judgments]
    high_count = judgments.count(Judgment.HIGH)
    some_concerns_count = judgments.count(Judgment.SOME_CONCERNS)

    reliability_review_basis = compute_reliability_review_basis(domain_judgments)

    if high_count:
        return Judgment.HIGH, ">=1 domain High -> High", reliability_review_basis is not None
    if some_concerns_count == 0:
        return Judgment.LOW, "all domains Low -> Low", reliability_review_basis is not None
    if some_concerns_count >= OVERALL_HIGH_SC_THRESHOLD:
        return (
            Judgment.HIGH,
            f"{some_concerns_count} domains Some concerns >= threshold {OVERALL_HIGH_SC_THRESHOLD} -> High",
            True,
        )

    requires_review = 2 <= some_concerns_count < OVERALL_HIGH_SC_THRESHOLD
    return (
        Judgment.SOME_CONCERNS,
        f"{some_concerns_count} domains Some concerns < threshold {OVERALL_HIGH_SC_THRESHOLD} -> Some concerns",
        requires_review or reliability_review_basis is not None,
    )


def compute_human_review_basis(domain_judgments: Sequence[DomainJudgment], rollup_rationale: str) -> str | None:
    """Explain why an assessment should be routed to human review, if any."""

    reliability_basis = compute_reliability_review_basis(domain_judgments)
    policy_basis = _policy_review_basis(domain_judgments, rollup_rationale)
    if policy_basis and reliability_basis:
        return f"{policy_basis}; {reliability_basis}"
    return policy_basis or reliability_basis


def compute_reliability_review_basis(domain_judgments: Sequence[DomainJudgment]) -> str | None:
    """Return a human-review basis for weak SQ reliability signals."""

    flagged: list[str] = []
    uncertain: list[str] = []
    unverified: list[str] = []

    for domain in domain_judgments:
        for answer in domain.sq_answers:
            label = f"{domain.domain} {answer.sq_id}"
            if answer.confidence.flag == ConfidenceFlag.FLAGGED:
                flagged.append(label)
            elif answer.confidence.flag == ConfidenceFlag.UNCERTAIN:
                uncertain.append(label)
            if not answer.confidence.quote_verified:
                unverified.append(label)

    parts = []
    if flagged:
        parts.append(f"flagged SQ answer(s): {', '.join(flagged)}")
    if uncertain:
        parts.append(f"uncertain SQ answer(s): {', '.join(uncertain)}")
    if unverified:
        parts.append(f"unverified quote(s): {', '.join(unverified)}")

    return "; ".join(parts) if parts else None


def _policy_review_basis(domain_judgments: Sequence[DomainJudgment], rollup_rationale: str) -> str | None:
    judgments = [_judgment(item.judgment) for item in domain_judgments]
    if judgments.count(Judgment.HIGH):
        return None
    some_concerns_count = judgments.count(Judgment.SOME_CONCERNS)
    if some_concerns_count >= 2:
        return rollup_rationale
    return None


def _judgment(value: Judgment | str) -> Judgment:
    return value if isinstance(value, Judgment) else Judgment(value)
