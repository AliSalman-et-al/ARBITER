from __future__ import annotations

import pytest

from arbiter.graph.nodes.sq_node import finalize_sq_answer, sq_node
from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMAuthenticationError, LLMClient
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import (
    AnswerCode,
    ConfidenceFlag,
    DomainContext,
    PageBox,
    SQRawAnswer,
)


def box(page: int, text: str) -> PageBox:
    return PageBox(boxclass="text", text=text, bbox=(0.0, 0.0, 100.0, 100.0), page=page)


def context() -> DomainContext:
    return DomainContext(
        domain="D1",
        domain_specific_text="The allocation sequence was random.",
        supplement_block="",
        retrieval_top_score=0.8,
        segments_retrieved=0,
        segments_available=0,
    )


class FailingLLMClient(LLMClient):
    async def complete_structured(self, *args, **kwargs):
        raise TimeoutError("provider timed out after retries")

    def supports_prompt_caching(self) -> bool:
        return False

    def supports_native_schema(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return False


class AuthFailingLLMClient(FailingLLMClient):
    async def complete_structured(self, *args, **kwargs):
        raise LLMAuthenticationError("fake authentication failed")


def test_finalize_sq_answer_resolves_page_and_confidence() -> None:
    raw = SQRawAnswer(
        answer="Y",
        quote="The allocation sequence was random.",
        justification="The methods section directly reports random allocation.",
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        context(),
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
    )

    assert answer.sq_id == "1.1"
    assert answer.answer == AnswerCode.Y
    assert answer.page == 2
    assert answer.confidence.quote_verified is True
    assert answer.confidence.flag == ConfidenceFlag.CONFIDENT


def test_sq_raw_answer_normalizes_common_shape_drift() -> None:
    raw = SQRawAnswer.model_validate(
        {
            "answer": "Y",
            "quotes": [
                "The allocation sequence was random.",
                "Allocation used blocks.",
            ],
            "reasoning": ["The methods section reports random allocation."],
        }
    )

    assert raw.quote == "The allocation sequence was random.\nAllocation used blocks."
    assert raw.justification == "The methods section reports random allocation."


def test_finalize_sq_answer_ni_short_circuits_quote_and_page() -> None:
    raw = SQRawAnswer(
        answer="NI",
        quote="The allocation sequence was random.",
        justification="No relevant text was found.",
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        context(),
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
    )

    assert answer.answer == AnswerCode.NI
    assert answer.quote == ""
    assert answer.page is None
    assert answer.confidence.quote_verified is True


def test_finalize_sq_answer_unverified_substantive_answer_becomes_flagged_ni() -> None:
    raw = SQRawAnswer(
        answer="Y",
        quote="This quote is not in the source.",
        justification="The quoted text supports random allocation.",
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        context(),
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
    )

    assert answer.answer == AnswerCode.NI
    assert answer.quote == ""
    assert answer.page is None
    assert answer.confidence.quote_verified is False
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED
    assert (
        answer.confidence.flag_reason
        == "supporting quote could not be verified in the source text"
    )


def test_finalize_sq_answer_verifies_quote_from_supplement_block() -> None:
    raw = SQRawAnswer(
        answer="Y",
        quote="Participants were randomly assigned with stratification according to extent of disease.",
        justification="The protocol supplement directly reports stratified random assignment.",
    )
    ctx = DomainContext(
        domain="D1",
        domain_specific_text="The main paper says allocation was randomized.",
        supplement_block=(
            "[Supplement: protocol.pdf; heading: Randomization; pages: 7]\n"
            "Participants were randomly assigned with stratification according to extent of disease."
        ),
        retrieval_top_score=0.8,
        segments_retrieved=1,
        segments_available=1,
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        ctx,
        raw_char_stream="The main paper says allocation was randomized.",
        page_boxes=[box(2, "The main paper says allocation was randomized.")],
    )

    assert answer.answer == AnswerCode.Y
    assert answer.quote == raw.quote
    assert answer.page == 7
    assert answer.confidence.quote_verified is True


def test_finalize_sq_answer_verifies_short_quote_from_registry_block(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_QUOTE_MIN_VERIFY_CHARS", "15")
    raw = SQRawAnswer(
        answer="N",
        quote="Masking: NONE",
        justification="The registry reports no masking.",
    )

    answer = finalize_sq_answer(
        raw,
        "4.3",
        context(),
        raw_char_stream="The main paper does not describe masking.",
        page_boxes=[box(2, "The main paper does not describe masking.")],
        ct_gov_block="[ClinicalTrials.gov]\nMasking: NONE",
    )

    assert answer.answer == AnswerCode.N
    assert answer.quote == "Masking: NONE"
    assert answer.page is None
    assert answer.confidence.quote_verified is True
    assert answer.confidence.flag == ConfidenceFlag.CONFIDENT


def test_finalize_sq_answer_soft_truncates_after_verification(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_SQ_QUOTE_SOFT_LIMIT", "10")
    raw = SQRawAnswer(
        answer="Y",
        quote="The allocation sequence was random.",
        justification="The methods section directly reports random allocation.",
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        context(),
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
    )

    assert answer.quote == "The alloca"
    assert answer.page == 2
    assert answer.confidence.quote_verified is True


@pytest.mark.asyncio
async def test_sq_node_calls_sq_model_once_and_returns_answer_map() -> None:
    client = MockLLMClient(
        responses={
            "1.1|assignment": {
                "answer": "Y",
                "quote": "The allocation sequence was random.",
                "justification": "The text directly supports random sequence generation.",
            }
        }
    )
    config = AssessmentConfig(paper_path="paper.pdf")

    result = await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": context(),
            "sq_model": client,
            "config": config,
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [box(4, "The allocation sequence was random.")],
        }
    )

    assert client.calls == ["1.1|assignment"]
    assert client.max_tokens == [config.sq_max_tokens]
    assert set(result["sq_answers"]) == {"1.1"}
    answer = result["sq_answers"]["1.1"]
    assert answer.answer == AnswerCode.Y
    assert answer.page == 4


@pytest.mark.asyncio
async def test_sq_node_converts_llm_failure_to_flagged_ni() -> None:
    result = await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": context(),
            "sq_model": FailingLLMClient("fake"),
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [box(4, "The allocation sequence was random.")],
        }
    )

    answer = result["sq_answers"]["1.1"]
    assert answer.answer == AnswerCode.NI
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED
    assert result["errors"] == [
        "1.1 signaling-question call failed: TimeoutError: provider timed out after retries"
    ]


@pytest.mark.asyncio
async def test_sq_node_does_not_swallow_auth_errors() -> None:
    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        await sq_node(
            {
                "sq_id": "1.1",
                "effect_of_interest": "assignment",
                "domain_context": context(),
                "sq_model": AuthFailingLLMClient("fake"),
            }
        )
