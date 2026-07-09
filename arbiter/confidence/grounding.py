"""Deterministic grounding signals for signaling-question answers."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from arbiter.confidence.quote_verifier import QuoteSource
from arbiter.models import AnswerCode

DEFAULT_CONTEXT_SUFFICIENT_RETRIEVAL_THRESHOLD = 0.35
DEFAULT_ENTAILMENT_ACCEPT_THRESHOLD = 0.72
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GroundingSignals:
    context_sufficient: bool
    context_sufficiency_reason: str
    entailment_score: float | None
    faithfulness_score: float | None
    grounding_method: Literal["quote_verification", "lexical_overlap", "not_applicable"]


def assess_grounding(
    *,
    answer: AnswerCode | str,
    quote: str,
    justification: str,
    sources: list[QuoteSource],
    context_text: str | None = None,
    quote_verified: bool,
    retrieval_top_score: float | None,
    segments_available: int,
) -> GroundingSignals:
    """Score whether available context supports the answer.

    This is deliberately deterministic: quote verification is treated as direct
    support, while lexical overlap is a conservative fallback signal for
    unverified or missing quotes. It emits sufficiency and grounding metadata for
    routing without changing the RoB 2 answer code by itself.
    """

    answer_code = AnswerCode(answer)
    source_text = _source_text(sources)
    sufficiency_text = source_text if context_text is None else context_text
    context_sufficient, reason = _context_sufficiency(
        source_text=sufficiency_text,
        retrieval_top_score=retrieval_top_score,
        segments_available=segments_available,
    )

    if answer_code in {AnswerCode.NI, AnswerCode.NA}:
        return GroundingSignals(
            context_sufficient=context_sufficient,
            context_sufficiency_reason=reason,
            entailment_score=None,
            faithfulness_score=None,
            grounding_method="not_applicable",
        )

    if quote_verified:
        return GroundingSignals(
            context_sufficient=True,
            context_sufficiency_reason="supporting quote verifies against available source text",
            entailment_score=1.0,
            faithfulness_score=1.0,
            grounding_method="quote_verification",
        )

    score = _lexical_support_score(
        claim_text=" ".join(part for part in (quote, justification) if part.strip()),
        source_text=source_text,
    )
    accepted = score >= _entailment_accept_threshold()
    return GroundingSignals(
        context_sufficient=context_sufficient or accepted,
        context_sufficiency_reason=(
            "answer text has lexical support in available source context"
            if accepted
            else reason
        ),
        entailment_score=score,
        faithfulness_score=score,
        grounding_method="lexical_overlap",
    )


def _context_sufficiency(
    *,
    source_text: str,
    retrieval_top_score: float | None,
    segments_available: int,
) -> tuple[bool, str]:
    if not source_text.strip():
        return False, "no source context was available to answer the signaling question"
    if retrieval_top_score is not None:
        if retrieval_top_score >= _context_sufficient_retrieval_threshold():
            return True, "retrieval score indicates sufficient domain-relevant context"
        return False, "retrieval score indicates weak domain-relevant context"
    if segments_available > 0:
        return True, "domain-relevant supplementary context was available"
    return True, "main or registry source context was available"


def _lexical_support_score(*, claim_text: str, source_text: str) -> float:
    claim = _normalize_text(claim_text)
    source = _normalize_text(source_text)
    if not claim or not source:
        return 0.0

    ratio = fuzz.token_set_ratio(claim, source) / 100.0
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return ratio
    source_tokens = _content_tokens(source)
    coverage = len(claim_tokens & source_tokens) / len(claim_tokens)
    return round((ratio + coverage) / 2.0, 4)


def _source_text(sources: list[QuoteSource]) -> str:
    return "\n".join(
        source.raw_char_stream for source in sources if source.raw_char_stream.strip()
    )


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text) if len(token) > 2}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    normalized = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _context_sufficient_retrieval_threshold() -> float:
    value = os.getenv("ARBITER_CONTEXT_SUFFICIENT_RETRIEVAL_THRESHOLD")
    return (
        DEFAULT_CONTEXT_SUFFICIENT_RETRIEVAL_THRESHOLD
        if value is None or value == ""
        else float(value)
    )


def _entailment_accept_threshold() -> float:
    value = os.getenv("ARBITER_ENTAILMENT_ACCEPT_THRESHOLD")
    return (
        DEFAULT_ENTAILMENT_ACCEPT_THRESHOLD
        if value is None or value == ""
        else float(value)
    )
