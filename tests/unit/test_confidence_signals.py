from __future__ import annotations

from arbiter.confidence.signals import compute_confidence
from arbiter.models import AnswerCode, ConfidenceFlag, ConfidenceSignals


def test_na_answer_is_always_confident(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")

    confidence = compute_confidence(
        AnswerCode.NA,
        quote_verified=False,
        segments_retrieved=3,
        segments_available=5,
        retrieval_top_score=0.1,
    )

    assert confidence.flag == ConfidenceFlag.CONFIDENT
    assert confidence.flag_reason is None


def test_unverified_quote_on_substantive_answer_is_flagged() -> None:
    confidence = compute_confidence(
        AnswerCode.Y,
        quote_verified=False,
        segments_retrieved=2,
        segments_available=2,
        retrieval_top_score=0.8,
    )

    assert confidence.flag == ConfidenceFlag.FLAGGED
    assert confidence.flag_reason is not None
    assert "quote" in confidence.flag_reason


def test_ni_with_available_supplements_and_weak_retrieval_is_flagged(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")

    confidence = compute_confidence(
        AnswerCode.NI,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=4,
        retrieval_top_score=0.2,
    )

    assert confidence.flag == ConfidenceFlag.FLAGGED
    assert confidence.flag_reason is not None
    assert "supplements" in confidence.flag_reason


def test_ni_with_sufficient_context_is_flagged() -> None:
    confidence = compute_confidence(
        AnswerCode.NI,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=1,
        retrieval_top_score=0.9,
        context_sufficient=True,
        context_sufficiency_reason="retrieval score indicates sufficient domain-relevant context",
    )

    assert confidence.context_sufficient is True
    assert confidence.flag == ConfidenceFlag.FLAGGED
    assert confidence.flag_reason == "answer is NI despite sufficient source context"


def test_low_retrieval_score_is_uncertain(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")

    confidence = compute_confidence(
        AnswerCode.PY,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=1,
        retrieval_top_score=0.34,
    )

    assert confidence.flag == ConfidenceFlag.UNCERTAIN
    assert confidence.flag_reason is not None
    assert "relevance" in confidence.flag_reason


def test_retrieval_threshold_default_is_reanchored_for_embeddinggemma(monkeypatch) -> None:
    monkeypatch.delenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", raising=False)

    below_default = compute_confidence(
        AnswerCode.PY,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=1,
        retrieval_top_score=0.24,
    )
    at_default = compute_confidence(
        AnswerCode.PY,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=1,
        retrieval_top_score=0.25,
    )

    assert below_default.flag == ConfidenceFlag.UNCERTAIN
    assert at_default.flag == ConfidenceFlag.CONFIDENT


def test_low_supplement_retrieval_score_does_not_uncertain_main_paper_quote(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")

    confidence = compute_confidence(
        AnswerCode.PY,
        quote_verified=True,
        segments_retrieved=1,
        segments_available=1,
        retrieval_top_score=0.34,
        quote_source_type="main_paper",
    )

    assert confidence.flag == ConfidenceFlag.CONFIDENT
    assert confidence.flag_reason is None


def test_ni_with_no_supplementary_material_is_uncertain() -> None:
    confidence = compute_confidence(
        AnswerCode.NI,
        quote_verified=True,
        segments_retrieved=0,
        segments_available=0,
        retrieval_top_score=None,
    )

    assert confidence.flag == ConfidenceFlag.UNCERTAIN
    assert confidence.flag_reason is not None
    assert "no domain-relevant" in confidence.flag_reason


def test_verified_quote_with_adequate_retrieval_is_confident(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")

    confidence = compute_confidence(
        AnswerCode.N,
        quote_verified=True,
        segments_retrieved=2,
        segments_available=2,
        retrieval_top_score=0.35,
    )

    assert confidence.flag == ConfidenceFlag.CONFIDENT
    assert confidence.flag_reason is None


def test_missing_retrieval_score_does_not_trigger_score_based_flags() -> None:
    confidence = compute_confidence(
        AnswerCode.PN,
        quote_verified=True,
        segments_retrieved=0,
        segments_available=3,
        retrieval_top_score=None,
    )

    assert confidence.flag == ConfidenceFlag.CONFIDENT


def test_model_has_no_answer_consistency_field() -> None:
    assert "answer_consistency" not in ConfidenceSignals.model_fields


def test_model_emits_grounding_fields_for_phase_two_signals() -> None:
    assert {"context_sufficient", "entailment_score", "faithfulness_score"} <= set(
        ConfidenceSignals.model_fields
    )
