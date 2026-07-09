from __future__ import annotations

from pathlib import Path

import pytest

from arbiter.observability.qa_trace import QATraceBundle
from arbiter.observability.trace import RunTrace
from arbiter.graph.nodes.sq_node import (
    build_sq_messages,
    finalize_sq_answer,
    sq_node,
    sq_raw_answer_schema_for_sq,
)
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


def _raw(answer: str, quote: str) -> dict[str, str]:
    return {
        "answer": answer,
        "quote": quote,
        "justification": "The quoted text supports the answer.",
    }


def _message_text(messages: list[dict]) -> str:
    content = messages[1]["content"]
    return "\n".join(part["text"] for part in content)


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
    assert answer.confidence.context_sufficient is True
    assert answer.confidence.entailment_score == 1.0
    assert answer.confidence.faithfulness_score == 1.0
    assert answer.confidence.grounding_method == "quote_verification"
    assert answer.confidence.flag == ConfidenceFlag.CONFIDENT


def test_finalize_sq_answer_ignores_weak_supplement_score_for_main_paper_quote(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", "0.35")
    raw = SQRawAnswer(
        answer="Y",
        quote="The allocation sequence was random.",
        justification="The methods section directly reports random allocation.",
    )
    ctx = DomainContext(
        domain="D1",
        domain_specific_text="The allocation sequence was random.",
        supplement_block="[Supplement: protocol.pdf; heading: Randomization; pages: 7]\nDifferent supplement text.",
        retrieval_top_score=0.2,
        segments_retrieved=1,
        segments_available=1,
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        ctx,
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
        source_document="paper.pdf",
    )

    assert answer.answer == AnswerCode.Y
    assert answer.confidence.quote_verified is True
    assert answer.confidence.flag == ConfidenceFlag.CONFIDENT
    assert answer.confidence.flag_reason is None


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


def test_sq_raw_answer_degrades_malformed_optional_fields() -> None:
    raw = SQRawAnswer.model_validate(
        {
            "answer": ".",
            "quote": None,
            "justification": ["Reasoning fragment.", None, "Second fragment."],
        }
    )

    assert raw.answer == "NI"
    assert raw.quote == ""
    assert raw.justification == "Reasoning fragment.\nSecond fragment."


def test_sq_raw_answer_truncates_overlong_model_fields() -> None:
    raw = SQRawAnswer.model_validate(
        {
            "answer": "Y",
            "quote": "q" * 4001,
            "justification": "j" * 1001,
        }
    )

    assert raw.quote == "q" * 4000
    assert raw.justification == "j" * 1000


def test_sq_raw_answer_schema_forbids_additional_properties() -> None:
    assert SQRawAnswer.model_json_schema()["additionalProperties"] is False


def test_d3_2_raw_answer_schema_excludes_ni() -> None:
    schema = sq_raw_answer_schema_for_sq("3.2").model_json_schema()

    assert schema["properties"]["answer"]["enum"] == ["Y", "PY", "PN", "N"]
    assert "answer" in schema["required"]


def test_build_sq_messages_guides_4_2_with_general_measurement_reasoning() -> None:
    messages = build_sq_messages(
        sq_id="4.2",
        effect="assignment",
        outcome="Overall survival",
        shared_prefix_text="Trial metadata prefix.",
        context=DomainContext(
            domain="D4",
            domain_specific_text=(
                "Participants in the docetaxel group were seen every 3 weeks and "
                "participants in the ADT-alone group every 3 months."
            ),
        ),
    )

    text = _message_text(messages)

    assert "Assessed outcome: Overall survival" in text
    assert "objective/hard endpoint" in text
    assert "participant-reported endpoint" in text
    assert "Different clinic visit frequency" in text
    assert "not enough by itself" in text


def test_build_sq_messages_applies_same_d4_reasoning_to_different_outcome_names() -> (
    None
):
    messages = build_sq_messages(
        sq_id="4.2",
        effect="assignment",
        outcome="Progression-free survival",
        shared_prefix_text="Trial metadata prefix.",
        context=DomainContext(
            domain="D4",
            domain_specific_text="Tumor progression was assessed by investigators.",
        ),
    )

    text = _message_text(messages)

    assert "Assessed outcome: Progression-free survival" in text
    assert "objective/hard endpoint" in text
    assert "participant-reported endpoint" in text
    assert "Different clinic visit frequency" in text


def test_build_sq_messages_adds_domain_specific_guidance_without_crossing_domains() -> (
    None
):
    messages = build_sq_messages(
        sq_id="3.1",
        effect="assignment",
        outcome="Overall survival",
        shared_prefix_text="Trial metadata prefix.",
        context=DomainContext(
            domain="D3",
            domain_specific_text="Follow-up was complete.",
        ),
    )

    text = _message_text(messages)

    assert "Assessed outcome: Overall survival" in text
    assert "Domain 3 reasoning guidance" in text
    assert "Do not infer bias from any missing data alone" in text
    assert "Domain 4 reasoning guidance" not in text
    assert "objective/hard endpoint" not in text


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


def test_finalize_sq_answer_maps_d3_2_ni_to_no() -> None:
    raw = SQRawAnswer(
        answer="NI",
        quote="",
        justification="No evidence was found that missing outcome data did not bias the result.",
    )

    answer = finalize_sq_answer(
        raw,
        "3.2",
        DomainContext(domain="D3", domain_specific_text="", supplement_block=""),
        raw_char_stream="",
        page_boxes=[],
    )

    assert answer.answer == AnswerCode.N
    assert answer.quote == ""
    assert answer.page is None
    assert answer.justification == raw.justification
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED
    assert answer.confidence.flag_reason == "3.2 does not permit NI; normalized to N"


def test_finalize_sq_answer_flags_ni_when_context_is_sufficient() -> None:
    raw = SQRawAnswer(
        answer="NI",
        quote="",
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
    assert answer.confidence.context_sufficient is True
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED
    assert (
        answer.confidence.flag_reason
        == "answer is NI despite sufficient source context"
    )


def test_finalize_sq_answer_does_not_treat_full_paper_as_sufficient_context_for_ni() -> (
    None
):
    raw = SQRawAnswer(
        answer="NI",
        quote="",
        justification="No relevant text was found.",
    )
    empty_context = DomainContext(
        domain="D1",
        domain_specific_text="",
        supplement_block="",
        retrieval_top_score=None,
        segments_retrieved=0,
        segments_available=0,
    )

    answer = finalize_sq_answer(
        raw,
        "1.1",
        empty_context,
        raw_char_stream="The allocation sequence was random.",
        page_boxes=[box(2, "The allocation sequence was random.")],
    )

    assert answer.answer == AnswerCode.NI
    assert answer.confidence.context_sufficient is False
    assert answer.confidence.flag == ConfidenceFlag.UNCERTAIN


def test_finalize_sq_answer_unverified_substantive_answer_is_kept_and_flagged() -> None:
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

    assert answer.answer == AnswerCode.Y
    assert answer.quote == "This quote is not in the source."
    assert answer.page is None
    assert answer.confidence.quote_verified is False
    assert answer.confidence.entailment_score is not None
    assert answer.confidence.faithfulness_score == answer.confidence.entailment_score
    assert answer.confidence.grounding_method == "lexical_overlap"
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


def test_finalize_sq_answer_verifies_short_quote_from_registry_block(
    monkeypatch,
) -> None:
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
async def test_sq_node_repairs_empty_quote_once_for_substantive_answer() -> None:
    client = MockLLMClient(
        responses={
            "1.1|assignment": _raw("Y", ""),
            "1.1|assignment|quote_repair": {
                "quote": "The allocation sequence was random."
            },
        }
    )
    config = AssessmentConfig(paper_path=Path("paper.pdf"))
    config.env.quote_repair_max_tokens = 3456

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

    answer = result["sq_answers"]["1.1"]
    assert client.calls == ["1.1|assignment", "1.1|assignment|quote_repair"]
    assert client.max_tokens == [config.sq_max_tokens, 3456]
    assert answer.answer == AnswerCode.Y
    assert answer.quote == "The allocation sequence was random."
    assert answer.page == 4
    assert answer.confidence.quote_verified is True


@pytest.mark.asyncio
async def test_sq_node_keeps_substantive_answer_when_empty_quote_repair_fails() -> None:
    client = MockLLMClient(responses={"1.1|assignment": _raw("Y", "")})

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

    answer = result["sq_answers"]["1.1"]
    assert client.calls == ["1.1|assignment", "1.1|assignment|quote_repair"]
    assert answer.answer == AnswerCode.Y
    assert answer.quote == ""
    assert answer.page is None
    assert answer.confidence.quote_verified is False
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED


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
    config = AssessmentConfig(paper_path=Path("paper.pdf"))

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
async def test_sq_node_normalizes_failed_d3_2_call_to_no() -> None:
    result = await sq_node(
        {
            "sq_id": "3.2",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": DomainContext(domain="D3", domain_specific_text=""),
            "sq_model": FailingLLMClient("fake"),
            "raw_char_stream": "",
            "page_boxes": [],
        }
    )

    answer = result["sq_answers"]["3.2"]
    assert answer.answer == AnswerCode.N
    assert answer.confidence.flag == ConfidenceFlag.FLAGGED
    assert "3.2 does not permit NI; normalized to N" in answer.confidence.flag_reason


@pytest.mark.asyncio
async def test_sq_node_records_degradation_for_failed_call(tmp_path) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)

    await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": context(),
            "sq_model": FailingLLMClient("fake"),
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [box(4, "The allocation sequence was random.")],
            "trace": trace,
            "outcome": "Overall survival",
        }
    )

    assert trace.degradation_events[0]["category"] == "sq_call_failed_to_ni"
    assert trace.degradation_events[0]["domain"] == "D1"
    assert trace.degradation_events[0]["sq_id"] == "1.1"


@pytest.mark.asyncio
async def test_sq_node_does_not_record_degradation_for_unverified_quote(
    tmp_path,
) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)

    await sq_node(
        {
            "sq_id": "1.1",
            "effect_of_interest": "assignment",
            "shared_prefix_text": "Trial metadata prefix.",
            "domain_context": context(),
            "sq_model": MockLLMClient(
                responses={
                    "1.1|assignment": _raw("Y", "This quote is not in the source.")
                }
            ),
            "raw_char_stream": "The allocation sequence was random.",
            "page_boxes": [box(4, "The allocation sequence was random.")],
            "trace": trace,
            "outcome": "Overall survival",
        }
    )

    assert not trace.degradation_events


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
