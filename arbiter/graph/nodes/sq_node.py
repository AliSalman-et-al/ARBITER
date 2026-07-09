"""Signaling-question worker node."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from arbiter.arbiter_algorithm.answer_sets import normalize_answer_for_sq
from arbiter.confidence.grounding import assess_grounding
from arbiter.confidence.quote_verifier import QuoteSource, describe_quote_verification_sources, resolve_quote_source
from arbiter.confidence.signals import QuoteSourceType, compute_confidence
from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMAuthenticationError, LLMClient
from arbiter.models import AnswerCode, ConfidenceFlag, ConfidenceSignals, DomainContext, PageBox, SQAnswer, SQRawAnswer
from arbiter.prompts.domain_guidance import assessed_outcome_block, domain_reasoning_guidance
from arbiter.prompts.sq_prompts import ANSWER_BRIDGE, get_sq_prompt

DEFAULT_QUOTE_SOFT_LIMIT = 1200
DEFAULT_JUSTIFICATION_SOFT_LIMIT = 500


class SQQuoteRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = ""


async def sq_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run one signaling question and return a mergeable SQ answer map."""

    sq_id = _require_str(state, "sq_id")
    effect = _effect_from_state(state)
    context = _domain_context_from_state(state)
    sq_model = _sq_model_from_state(state)
    config = _config_from_state(state)

    try:
        raw = await sq_model.complete_structured(
            build_sq_messages(
                sq_id=sq_id,
                effect=effect,
                outcome=str(state.get("outcome") or ""),
                shared_prefix_text=str(state.get("shared_prefix_text") or ""),
                context=context,
            ),
            SQRawAnswer,
            temperature=0.0,
            max_tokens=getattr(config, "sq_max_tokens", 2048),
            call_label=f"{sq_id}|{effect}",
        )
    except LLMAuthenticationError:
        raise
    except Exception as exc:
        _record_degradation(
            state,
            category="sq_call_failed_to_ni",
            reason=f"signaling-question call failed: {type(exc).__name__}: {exc}",
            severity="error",
            domain=_domain_for_sq(sq_id),
            sq_id=sq_id,
            payload={
                "fallback_answer": "NI",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        return {
            "sq_answers": {sq_id: _failed_sq_answer(sq_id, exc)},
            "errors": [f"{sq_id} signaling-question call failed: {type(exc).__name__}: {exc}"],
        }

    if not isinstance(raw, SQRawAnswer):
        raw = SQRawAnswer.model_validate(raw)

    raw = await _repair_unquoted_substantive_answer(
        raw,
        sq_id=sq_id,
        effect=effect,
        outcome=str(state.get("outcome") or ""),
        shared_prefix_text=str(state.get("shared_prefix_text") or ""),
        context=context,
        sq_model=sq_model,
        config=config,
    )

    answer = finalize_sq_answer(
        raw,
        sq_id,
        context,
        raw_char_stream=_raw_char_stream_from_state(state),
        page_boxes=_page_boxes_from_state(state),
        source_document=_source_document_from_state(state),
        ct_gov_block=str(state.get("ct_gov_block") or ""),
    )
    _record_sq_finalization_trace(state, sq_id, context, raw, answer)
    return {"sq_answers": {sq_id: answer}}


def _failed_sq_answer(sq_id: str, exc: Exception) -> SQAnswer:
    return SQAnswer(
        sq_id=sq_id,
        answer=AnswerCode.NI,
        quote="",
        page=None,
        justification="No information was recorded because the signaling-question call failed.",
        confidence=ConfidenceSignals(
            quote_verified=True,
            flag=ConfidenceFlag.FLAGGED,
            flag_reason=f"signaling-question call failed: {type(exc).__name__}: {exc}",
        ),
    )

def build_sq_messages(
    *,
    sq_id: str,
    effect: str,
    outcome: str = "",
    shared_prefix_text: str,
    context: DomainContext,
) -> list[dict[str, Any]]:
    template = get_sq_prompt(sq_id, effect)
    dynamic_suffix = "\n\n".join(
        part
        for part in (
            "[Domain source text]\n" + context.domain_specific_text.strip(),
            "[Supplement source text]\n" + (context.supplement_block or "").strip(),
            assessed_outcome_block(outcome),
            domain_reasoning_guidance(sq_id),
            "[Signaling question]\n" + template.question_text,
            "[Answer definitions]\n" + template.answer_definitions,
            "[Task]\n"
            "Find the most relevant verbatim sentence or sentences in the SOURCE TEXT, "
            "copy them exactly into quote, choose one answer code, and write exactly "
            "one justification sentence. Do not provide a page number. "
            "Only answer NI when no relevant text exists in any provided source. "
            + ANSWER_BRIDGE,
        )
        if part.strip()
    )
    return [
        {
            "role": "system",
            "content": "You answer one Cochrane RoB 2 signaling question. You never make risk-of-bias judgments.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[Static trial prefix]\n" + shared_prefix_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_suffix},
            ],
        },
    ]


async def _repair_unquoted_substantive_answer(
    raw: SQRawAnswer,
    *,
    sq_id: str,
    effect: str,
    outcome: str,
    shared_prefix_text: str,
    context: DomainContext,
    sq_model: LLMClient,
    config: AssessmentConfig | object,
) -> SQRawAnswer:
    if AnswerCode(raw.answer) == AnswerCode.NI or raw.quote.strip():
        return raw

    try:
        repair = cast(
            SQQuoteRepair,
            await sq_model.complete_structured(
                build_quote_repair_messages(
                    sq_id=sq_id,
                    effect=effect,
                    outcome=outcome,
                    shared_prefix_text=shared_prefix_text,
                    context=context,
                    raw=raw,
                ),
                SQQuoteRepair,
                temperature=0.0,
                max_tokens=min(getattr(config, "sq_max_tokens", 2048), 512),
                call_label=f"{sq_id}|{effect}|quote_repair",
            ),
        )
    except Exception:
        return raw

    quote = repair.quote.strip()
    if not quote:
        return raw
    return raw.model_copy(update={"quote": quote[:4000]})


def build_quote_repair_messages(
    *,
    sq_id: str,
    effect: str,
    outcome: str,
    shared_prefix_text: str,
    context: DomainContext,
    raw: SQRawAnswer,
) -> list[dict[str, Any]]:
    template = get_sq_prompt(sq_id, effect)
    dynamic_suffix = "\n\n".join(
        part
        for part in (
            "[Domain source text]\n" + context.domain_specific_text.strip(),
            "[Supplement source text]\n" + (context.supplement_block or "").strip(),
            assessed_outcome_block(outcome),
            domain_reasoning_guidance(sq_id),
            "[Signaling question]\n" + template.question_text,
            "[Previous answer]\n"
            f"answer: {raw.answer}\n"
            f"justification: {raw.justification}\n",
            "[Task]\n"
            "The previous response gave a substantive answer but omitted the quote. "
            "Return the single most relevant verbatim supporting sentence from the SOURCE TEXT. "
            "If no supporting sentence appears in the provided source text, return an empty quote.",
        )
        if part.strip()
    )
    return [
        {
            "role": "system",
            "content": "You locate verbatim source support for one Cochrane RoB 2 signaling-question answer.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[Static trial prefix]\n" + shared_prefix_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_suffix},
            ],
        },
    ]


def finalize_sq_answer(
    raw: SQRawAnswer,
    sq_id: str,
    context: DomainContext,
    *,
    raw_char_stream: str,
    page_boxes: list[PageBox],
    source_document: str | None = None,
    ct_gov_block: str = "",
) -> SQAnswer:
    """Turn a validated LLM payload into the deterministic SQ answer record."""

    raw_answer_code = AnswerCode(raw.answer)
    answer_code = normalize_answer_for_sq(sq_id, raw_answer_code)
    answer_was_normalized = answer_code != raw_answer_code
    quote = raw.quote
    justification = raw.justification
    quote_sources = _quote_sources(context, raw_char_stream, page_boxes, source_document, ct_gov_block)

    if answer_was_normalized:
        quote = ""
        page = None
        quote_verified = True
        quote_source_type = None
    elif answer_code == AnswerCode.NI:
        quote = ""
        page = None
        quote_verified = True
        quote_source_type = None
    else:
        quote_verified, page, matched_source_document = resolve_quote_source(
            quote,
            quote_sources,
        )
        quote_source_type = _quote_source_type(matched_source_document, source_document) if quote_verified else None

    grounding = assess_grounding(
        answer=AnswerCode.NA if answer_was_normalized else answer_code,
        quote=raw.quote,
        justification=raw.justification,
        sources=quote_sources,
        context_text=_grounding_context_text(context, ct_gov_block),
        quote_verified=quote_verified,
        retrieval_top_score=context.retrieval_top_score,
        segments_available=context.segments_available,
    )
    confidence = compute_confidence(
        answer_code,
        quote_verified=quote_verified,
        segments_retrieved=context.segments_retrieved,
        segments_available=context.segments_available,
        retrieval_top_score=context.retrieval_top_score,
        quote_source_type=quote_source_type,
        context_sufficient=grounding.context_sufficient,
        context_sufficiency_reason=grounding.context_sufficiency_reason,
        entailment_score=grounding.entailment_score,
        faithfulness_score=grounding.faithfulness_score,
        grounding_method=grounding.grounding_method,
    )
    if answer_was_normalized:
        confidence.flag = ConfidenceFlag.FLAGGED
        confidence.flag_reason = f"{sq_id} does not permit {raw_answer_code.value}; normalized to {answer_code.value}"

    return SQAnswer(
        sq_id=sq_id,
        answer=answer_code,
        quote=_soft_truncate(quote, _env_int("ARBITER_SQ_QUOTE_SOFT_LIMIT", DEFAULT_QUOTE_SOFT_LIMIT)),
        page=page,
        justification=_soft_truncate(
            justification,
            _env_int("ARBITER_SQ_JUSTIFICATION_SOFT_LIMIT", DEFAULT_JUSTIFICATION_SOFT_LIMIT),
        ),
        confidence=confidence,
    )

def _domain_context_from_state(state: Mapping[str, Any]) -> DomainContext:
    context = state.get("domain_context")
    if isinstance(context, DomainContext):
        return context
    if isinstance(context, Mapping):
        return DomainContext.model_validate(context)

    sq_id = state.get("sq_id")
    domain = _domain_for_sq(str(sq_id)) if sq_id else None
    contexts = state.get("domain_contexts")
    if isinstance(contexts, Mapping) and domain and domain in contexts:
        value = contexts[domain]
        return value if isinstance(value, DomainContext) else DomainContext.model_validate(value)

    raise TypeError("sq_node requires state['domain_context'] or state['domain_contexts'][domain]")


def _sq_model_from_state(state: Mapping[str, Any]) -> LLMClient:
    model = state.get("sq_model")
    if isinstance(model, LLMClient):
        return model
    runtime = state.get("runtime")
    runtime_model = getattr(runtime, "sq_model", None)
    if isinstance(runtime_model, LLMClient):
        return runtime_model
    raise TypeError("sq_node requires an sq_model LLMClient")


def _config_from_state(state: Mapping[str, Any]) -> AssessmentConfig | object:
    config = state.get("config")
    return config if config is not None else object()


def _effect_from_state(state: Mapping[str, Any]) -> str:
    effect = state.get("effect_of_interest")
    if effect is not None:
        return str(getattr(effect, "value", effect))
    config = state.get("config")
    config_effect = getattr(config, "effect_of_interest", None)
    if config_effect is not None:
        return str(getattr(config_effect, "value", config_effect))
    return "assignment"


def _page_boxes_from_state(state: Mapping[str, Any]) -> list[PageBox]:
    boxes = state.get("page_boxes")
    if isinstance(boxes, list):
        return [box if isinstance(box, PageBox) else PageBox.model_validate(box) for box in boxes]
    section_map = state.get("section_map")
    section_boxes = getattr(section_map, "page_boxes", None)
    if isinstance(section_boxes, list):
        return [box if isinstance(box, PageBox) else PageBox.model_validate(box) for box in section_boxes]
    return []


def _raw_char_stream_from_state(state: Mapping[str, Any]) -> str:
    stream = state.get("raw_char_stream")
    if stream is not None:
        return str(stream)
    section_map = state.get("section_map")
    return str(getattr(section_map, "full_text", "") or "")


def _record_sq_finalization_trace(
    state: Mapping[str, Any],
    sq_id: str,
    context: DomainContext,
    raw: SQRawAnswer,
    answer: SQAnswer,
) -> None:
    qa_trace = _qa_trace_from_state(state)
    if qa_trace is None:
        return
    domain = _domain_for_sq(sq_id)
    source_document = _source_document_from_state(state)
    quote_verification = (
        {
            "normalized_quote": "",
            "verified": True,
            "matched_source_document": None,
            "matched_page": None,
            "matched_span": None,
            "match_strategy": "not_applicable",
            "match_score": None,
            "verification_threshold": None,
            "failure_reason": None,
        }
        if AnswerCode(raw.answer) == AnswerCode.NI
        else describe_quote_verification_sources(
            raw.quote,
            _quote_sources(
                context,
                _raw_char_stream_from_state(state),
                _page_boxes_from_state(state),
                source_document,
                str(state.get("ct_gov_block") or ""),
            ),
        )
    )
    payload = {
        "sq_id": sq_id,
        "domain": domain,
        "raw_answer": raw.model_dump(mode="json"),
        "quote_verification": quote_verification,
        "final_answer": answer.model_dump(mode="json"),
        "confidence_flag": answer.confidence.flag.value,
        "soft_truncation": {
            "quote_truncated": raw.quote != answer.quote and answer.answer != AnswerCode.NI,
            "justification_truncated": raw.justification != answer.justification,
            "quote_original_length": len(raw.quote),
            "quote_final_length": len(answer.quote),
            "justification_original_length": len(raw.justification),
            "justification_final_length": len(answer.justification),
        },
        "fallback_details": None,
        "context_retrieval": {
            "segments_retrieved": context.segments_retrieved,
            "segments_available": context.segments_available,
            "retrieval_top_score": context.retrieval_top_score,
        },
    }
    quote_ref = f"quote_verification/{domain}/{sq_id.replace('.', '_')}.json"
    qa_trace.write_json_artifact(
        quote_ref,
        {
            "sq_id": sq_id,
            "domain": domain,
            "raw_quote": raw.quote,
            **quote_verification,
            "confidence_flag": answer.confidence.flag.value,
        },
    )
    qa_trace.record_event(
        event_type="quote_verification.completed",
        status="completed",
        trial_id=_trial_id_from_state(state),
        outcome=str(state.get("outcome")) if state.get("outcome") is not None else None,
        domain=domain,
        sq_id=sq_id,
        artifact_refs=[quote_ref],
        payload={
            "verified": quote_verification["verified"],
            "match_strategy": quote_verification["match_strategy"],
            "match_score": quote_verification["match_score"],
            "confidence_flag": answer.confidence.flag.value,
        },
    )
    artifact_ref = f"sq_answers/{domain}/{sq_id.replace('.', '_')}.finalization.json"
    qa_trace.write_json_artifact(artifact_ref, payload)
    qa_trace.record_event(
        event_type="sq.finalized",
        status="completed",
        trial_id=_trial_id_from_state(state),
        outcome=str(state.get("outcome")) if state.get("outcome") is not None else None,
        domain=domain,
        sq_id=sq_id,
        artifact_refs=[artifact_ref],
        payload={
            "raw_answer": raw.answer,
            "final_answer": answer.answer.value,
            "quote_verified": answer.confidence.quote_verified,
            "confidence_flag": answer.confidence.flag.value,
        },
    )


def _qa_trace_from_state(state: Mapping[str, Any]) -> Any | None:
    trace = state.get("trace")
    if trace is not None:
        qa_trace = getattr(trace, "qa_trace", None)
        if qa_trace is not None:
            return qa_trace
    config = state.get("config")
    return getattr(config, "qa_trace", None)


def _record_degradation(
    state: Mapping[str, Any],
    *,
    category: str,
    reason: str,
    severity: str,
    domain: str,
    sq_id: str,
    payload: dict[str, Any],
) -> None:
    trace = state.get("trace")
    if trace is None:
        config = state.get("config")
        trace = getattr(config, "trace", None)
    if trace is None or not hasattr(trace, "record_degradation"):
        return
    trace.record_degradation(
        category=category,
        reason=reason,
        severity=severity,
        trial_id=_trial_id_from_state(state),
        outcome=str(state.get("outcome")) if state.get("outcome") is not None else None,
        domain=domain,
        sq_id=sq_id,
        payload=payload,
    )


def _source_document_from_state(state: Mapping[str, Any]) -> str | None:
    section_map = state.get("section_map")
    source_path = getattr(section_map, "source_path", None)
    return str(source_path) if source_path is not None else None


def _quote_source_type(matched_source_document: str | None, main_source_document: str | None) -> QuoteSourceType:
    if matched_source_document == "ClinicalTrials.gov":
        return "registry"
    if matched_source_document == main_source_document:
        return "main_paper"
    return "supplement"


def _quote_sources(
    context: DomainContext,
    raw_char_stream: str,
    page_boxes: list[PageBox],
    source_document: str | None,
    ct_gov_block: str = "",
) -> list[QuoteSource]:
    sources = [QuoteSource(source_document=source_document, raw_char_stream=raw_char_stream, page_boxes=page_boxes)]
    if ct_gov_block.strip():
        sources.append(
            QuoteSource(
                source_document="ClinicalTrials.gov",
                raw_char_stream=ct_gov_block,
                page_boxes=[],
                page_required=False,
            )
        )
    sources.extend(_supplement_quote_sources(context.supplement_block))
    return sources


def _grounding_context_text(context: DomainContext, ct_gov_block: str) -> str:
    return "\n".join(
        part.strip()
        for part in (context.domain_specific_text, context.supplement_block, ct_gov_block)
        if part.strip()
    )


_SUPPLEMENT_HEADER_RE = re.compile(
    r"^\[Supplement:\s*(?P<source>.*?);.*?pages:\s*(?P<pages>[^\]]+)\]\s*$",
    re.IGNORECASE,
)


def _supplement_quote_sources(supplement_block: str) -> list[QuoteSource]:
    sources: list[QuoteSource] = []
    current_source: str | None = None
    current_page: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_source is None or not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        page = current_page if current_page is not None else 0
        sources.append(
            QuoteSource(
                source_document=current_source,
                raw_char_stream=text,
                page_boxes=[PageBox(boxclass="text", text=text, bbox=(0.0, 0.0, 0.0, 0.0), page=page)],
            )
        )

    for line in supplement_block.splitlines():
        match = _SUPPLEMENT_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_source = match.group("source").strip()
            current_page = _first_page(match.group("pages"))
            current_lines = []
        elif current_source is not None:
            current_lines.append(line)
    flush()
    return sources


def _first_page(pages: str) -> int | None:
    match = re.search(r"\d+", pages)
    return int(match.group(0)) if match else None


def _trial_id_from_state(state: Mapping[str, Any]) -> str | None:
    metadata = state.get("trial_metadata")
    trial_id = getattr(metadata, "trial_id", None)
    return str(trial_id) if trial_id is not None else None


def _domain_for_sq(sq_id: str) -> str:
    return f"D{sq_id.split('.', 1)[0]}"


def _require_str(state: Mapping[str, Any], key: str) -> str:
    value = state.get(key)
    if value is None:
        raise KeyError(f"sq_node requires state[{key!r}]")
    return str(value)


def _soft_truncate(text: str, limit: int) -> str:
    if limit < 0:
        return text
    return text if len(text) <= limit else text[:limit]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)
