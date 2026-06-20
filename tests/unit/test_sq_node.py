from __future__ import annotations

import pytest

from arbiter.graph.nodes.sq_node import finalize_sq_answer, sq_node
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


def test_finalize_sq_answer_ni_short_circuits_quote_and_page() -> None:
    raw = SQRawAnswer(answer="NI", quote="The allocation sequence was random.", justification="No relevant text was found.")

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

    result = await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": context(),
            "sq_model": client,
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [box(4, "The allocation sequence was random.")],
        }
    )

    assert client.calls == ["1.1|assignment"]
    assert set(result["sq_answers"]) == {"1.1"}
    answer = result["sq_answers"]["1.1"]
    assert answer.answer == AnswerCode.Y
    assert answer.page == 4
