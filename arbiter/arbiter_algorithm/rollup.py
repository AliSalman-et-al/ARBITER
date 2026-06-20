"""Overall RoB 2 judgment rollup for ARBITER.

Policy note: see `docs/adr/0001-overall-judgment-some-concerns-threshold.md`.
The domain judgments are verbatim RoB 2 IRPG table outputs; the threshold for
turning multiple Some concerns domains into an overall High judgment is an
ARBITER unattended-assessment policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from arbiter.models import DomainJudgment, Judgment


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

    if high_count:
        return Judgment.HIGH, ">=1 domain High -> High", False
    if some_concerns_count == 0:
        return Judgment.LOW, "all domains Low -> Low", False
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
        requires_review,
    )


def _judgment(value: Judgment | str) -> Judgment:
    return value if isinstance(value, Judgment) else Judgment(value)
