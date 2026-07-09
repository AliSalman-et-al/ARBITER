"""Signaling-question worker node."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, create_model

from arbiter.arbiter_algorithm.answer_sets import (
    normalize_answer_for_sq,
    valid_answer_codes,
)
from arbiter.confidence.grounding import assess_grounding
from arbiter.confidence.quote_verifier import (
    QuoteSource,
    describe_quote_verification_sources,
    resolve_quote_source,
)
from arbiter.confidence.signals import QuoteSourceType, compute_confidence
from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMAuthenticationError, LLMClient
from arbiter.models import (
    AnswerCode,
    ConfidenceFlag,
    ConfidenceSignals,
    DomainContext,
    OutcomeMeasurementProfile,
    PageBox,
    SQFallbackKind,
    SQAnswer,
    SQRawAnswer,
    SQRawAnswerWithCompleteness,
)
from arbiter.prompts.domain_guidance import (
    assessed_outcome_block,
    domain_reasoning_guidance,
    outcome_anchoring_block,
    outcome_measurement_profile_block,
)
from arbiter.prompts.sq_prompts import ANSWER_BRIDGE, get_sq_prompt

DEFAULT_QUOTE_SOFT_LIMIT = 1200
DEFAULT_JUSTIFICATION_SOFT_LIMIT = 500


class SQQuoteRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = ""


@dataclass(frozen=True)
class OrientationQuoteRepairResult:
    original_quote: str
    failure_kind: str
    matched_orientation: bool
    repair_attempted: bool = False
    repair_quote: str = ""
    repair_verified: bool = False
    repair_matched_source_document: str | None = None
    repair_matched_page: int | None = None
    repair_failure_reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "original_quote": self.original_quote,
            "failure_kind": self.failure_kind,
            "matched_orientation": self.matched_orientation,
            "repair_attempted": self.repair_attempted,
            "repair_quote": self.repair_quote,
            "repair_verified": self.repair_verified,
            "repair_matched_source_document": self.repair_matched_source_document,
            "repair_matched_page": self.repair_matched_page,
            "repair_failure_reason": self.repair_failure_reason,
        }


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
                outcome_measurement_profile=_outcome_measurement_profile_from_state(
                    state
                ),
                trial_orientation_text=_trial_orientation_text_from_state(state),
                shared_source_prefix_text=_shared_source_prefix_text_from_state(state),
                context=context,
            ),
            sq_raw_answer_schema_for_sq(sq_id),
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
            "errors": [
                f"{sq_id} signaling-question call failed: {type(exc).__name__}: {exc}"
            ],
        }

    if not isinstance(raw, SQRawAnswer):
        raw = SQRawAnswer.model_validate(raw)

    raw = await _repair_unquoted_substantive_answer(
        raw,
        sq_id=sq_id,
        effect=effect,
        outcome=str(state.get("outcome") or ""),
        outcome_measurement_profile=_outcome_measurement_profile_from_state(state),
        shared_source_prefix_text=_shared_source_prefix_text_from_state(state),
        context=context,
        sq_model=sq_model,
        config=config,
    )
    raw_for_trace = raw
    raw, orientation_repair = await _repair_non_citable_orientation_quote(
        raw,
        sq_id=sq_id,
        effect=effect,
        outcome=str(state.get("outcome") or ""),
        outcome_measurement_profile=_outcome_measurement_profile_from_state(state),
        trial_orientation_text=_trial_orientation_text_from_state(state),
        shared_source_prefix_text=_shared_source_prefix_text_from_state(state),
        context=context,
        raw_char_stream=_raw_char_stream_from_state(state),
        page_boxes=_page_boxes_from_state(state),
        source_document=_source_document_from_state(state),
        ct_gov_block=str(state.get("ct_gov_block") or ""),
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
    if orientation_repair is not None and not orientation_repair.repair_verified:
        answer.confidence.flag = ConfidenceFlag.FLAGGED
        answer.confidence.flag_reason = "quote matched non-citable trial orientation"
    _record_sq_finalization_trace(
        state,
        sq_id,
        context,
        raw_for_trace,
        answer,
        verification_raw=raw,
        orientation_quote_repair=orientation_repair,
    )
    return {"sq_answers": {sq_id: answer}}


@lru_cache(maxsize=None)
def sq_raw_answer_schema_for_sq(sq_id: str) -> type[SQRawAnswer]:
    """Return the raw-answer schema with the SQ-specific answer enum."""

    if sq_id == "3.1":
        return SQRawAnswerWithCompleteness

    codes = tuple(code.value for code in valid_answer_codes(sq_id))
    if codes == ("Y", "PY", "PN", "N", "NI"):
        return SQRawAnswer

    answer_type = Literal.__getitem__(codes)  # type: ignore[attr-defined]
    return create_model(
        f"SQRawAnswer_{sq_id.replace('.', '_')}",
        __base__=SQRawAnswer,
        answer=(answer_type, ...),
    )


def _failed_sq_answer(sq_id: str, exc: Exception) -> SQAnswer:
    raw_fallback = AnswerCode.NI
    answer = normalize_answer_for_sq(sq_id, raw_fallback)
    flag_reason = f"signaling-question call failed: {type(exc).__name__}: {exc}"
    if answer != raw_fallback:
        flag_reason = (
            f"{flag_reason}; {sq_id} does not permit {raw_fallback.value}; "
            f"normalized to {answer.value}"
        )
    return SQAnswer(
        sq_id=sq_id,
        answer=answer,
        quote="",
        page=None,
        justification="No information was recorded because the signaling-question call failed.",
        confidence=ConfidenceSignals(
            quote_verified=True,
            flag=ConfidenceFlag.FLAGGED,
            flag_reason=flag_reason,
            fallback_kind=SQFallbackKind.SQ_CALL_FAILED,
        ),
    )


def _render_static_prompt_prefix(
    *,
    trial_orientation_text: str,
    shared_source_prefix_text: str,
    legacy_shared_prefix_text: str,
) -> str:
    if trial_orientation_text.strip() or shared_source_prefix_text.strip():
        parts = [
            trial_orientation_text.strip(),
            "[Citable source text]\n" + shared_source_prefix_text.strip()
            if shared_source_prefix_text.strip()
            else "",
        ]
        return "\n\n".join(part for part in parts if part)
    return "[Citable source text]\n" + legacy_shared_prefix_text.strip()


def _evidence_tiering_block(context: DomainContext) -> str:
    if not context.domain_specific_text.strip() and not (
        context.supplement_block or ""
    ).strip():
        return ""
    return "\n".join(
        [
            "[Evidence tiers]",
            (
                "The domain source text is the primary source for this signaling "
                "question. Retrieved supplement passages are supplementary evidence."
            ),
            (
                "Trust the primary tier first when the tiers appear to conflict, "
                "and use supplementary evidence to fill gaps or corroborate the "
                "primary source."
            ),
        ]
    )


def _sq_3_1_completeness_scaffold(sq_id: str) -> str:
    if sq_id != "3.1":
        return ""
    return "\n".join(
        [
            "[Completeness calculation]",
            (
                "For 3.1, explicitly calculate outcome-data completeness when "
                "participant counts are available, using the form "
                "completeness_calculation: 234/249 = 94.0%."
            ),
            (
                "Treat missing arithmetic as a support concern, not a mechanical "
                "answer gate; still choose the answer code from the available source "
                "evidence and RoB 2 guidance."
            ),
        ]
    )


def build_sq_messages(
    *,
    sq_id: str,
    effect: str,
    outcome: str = "",
    outcome_measurement_profile: (
        OutcomeMeasurementProfile | Mapping[str, Any] | None
    ) = None,
    shared_prefix_text: str = "",
    trial_orientation_text: str = "",
    shared_source_prefix_text: str = "",
    context: DomainContext,
) -> list[dict[str, Any]]:
    template = get_sq_prompt(sq_id, effect)
    prefix_text = _render_static_prompt_prefix(
        trial_orientation_text=trial_orientation_text,
        shared_source_prefix_text=shared_source_prefix_text,
        legacy_shared_prefix_text=shared_prefix_text,
    )
    dynamic_suffix = "\n\n".join(
        part
        for part in (
            "[Domain source text]\n" + context.domain_specific_text.strip(),
            "[Supplement source text]\n" + (context.supplement_block or "").strip(),
            _evidence_tiering_block(context),
            assessed_outcome_block(outcome),
            outcome_anchoring_block(outcome, sq_id),
            outcome_measurement_profile_block(outcome_measurement_profile, sq_id),
            domain_reasoning_guidance(sq_id),
            _sq_3_1_completeness_scaffold(sq_id),
            "[Signaling question]\n" + template.question_text,
            "[Answer definitions]\n" + template.answer_definitions,
            "[Task]\n"
            "Find the most relevant verbatim sentence or sentences in the citable source text, "
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
                    "text": prefix_text,
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
    outcome_measurement_profile: OutcomeMeasurementProfile | Mapping[str, Any] | None,
    shared_source_prefix_text: str,
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
                    outcome_measurement_profile=outcome_measurement_profile,
                    shared_source_prefix_text=shared_source_prefix_text,
                    context=context,
                    raw=raw,
                ),
                SQQuoteRepair,
                temperature=0.0,
                max_tokens=_quote_repair_max_tokens(config),
                call_label=f"{sq_id}|{effect}|quote_repair",
            ),
        )
    except Exception:
        return raw

    quote = repair.quote.strip()
    if not quote:
        return raw
    return raw.model_copy(update={"quote": quote[:4000]})


async def _repair_non_citable_orientation_quote(
    raw: SQRawAnswer,
    *,
    sq_id: str,
    effect: str,
    outcome: str,
    outcome_measurement_profile: OutcomeMeasurementProfile | Mapping[str, Any] | None,
    trial_orientation_text: str,
    shared_source_prefix_text: str,
    context: DomainContext,
    raw_char_stream: str,
    page_boxes: list[PageBox],
    source_document: str | None,
    ct_gov_block: str,
    sq_model: LLMClient,
    config: AssessmentConfig | object,
) -> tuple[SQRawAnswer, OrientationQuoteRepairResult | None]:
    if AnswerCode(raw.answer) == AnswerCode.NI or not raw.quote.strip():
        return raw, None

    quote_sources = _quote_sources(
        context, raw_char_stream, page_boxes, source_document, ct_gov_block
    )
    quote_verified, _, _ = resolve_quote_source(raw.quote, quote_sources)
    if quote_verified:
        return raw, None

    if not _matches_non_citable_orientation(raw.quote, trial_orientation_text):
        return raw, None

    base_result = OrientationQuoteRepairResult(
        original_quote=raw.quote,
        failure_kind="non_citable_orientation_quote",
        matched_orientation=True,
    )
    try:
        repair = cast(
            SQQuoteRepair,
            await sq_model.complete_structured(
                build_orientation_quote_repair_messages(
                    sq_id=sq_id,
                    effect=effect,
                    outcome=outcome,
                    outcome_measurement_profile=outcome_measurement_profile,
                    shared_source_prefix_text=shared_source_prefix_text,
                    context=context,
                    raw=raw,
                ),
                SQQuoteRepair,
                temperature=0.0,
                max_tokens=_quote_repair_max_tokens(config),
                call_label=f"{sq_id}|{effect}|orientation_quote_repair",
            ),
        )
    except Exception as exc:
        return raw, replace(
            base_result,
            repair_attempted=True,
            repair_failure_reason=f"{type(exc).__name__}: {exc}",
        )

    repair_quote = repair.quote.strip()[:4000]
    if not repair_quote:
        return raw, replace(
            base_result,
            repair_attempted=True,
            repair_failure_reason="repair returned empty quote",
        )

    repair_verified, repair_page, repair_source = resolve_quote_source(
        repair_quote, quote_sources
    )
    repair_result = replace(
        base_result,
        repair_attempted=True,
        repair_quote=repair_quote,
        repair_verified=repair_verified,
        repair_matched_source_document=repair_source if repair_verified else None,
        repair_matched_page=repair_page if repair_verified else None,
        repair_failure_reason=None
        if repair_verified
        else "repair quote could not be verified in citable source text",
    )
    if not repair_verified:
        return raw, repair_result
    return raw.model_copy(update={"quote": repair_quote}), repair_result


def build_quote_repair_messages(
    *,
    sq_id: str,
    effect: str,
    outcome: str,
    outcome_measurement_profile: (
        OutcomeMeasurementProfile | Mapping[str, Any] | None
    ) = None,
    shared_prefix_text: str = "",
    shared_source_prefix_text: str = "",
    context: DomainContext,
    raw: SQRawAnswer,
) -> list[dict[str, Any]]:
    template = get_sq_prompt(sq_id, effect)
    prefix_text = _render_static_prompt_prefix(
        trial_orientation_text="",
        shared_source_prefix_text=shared_source_prefix_text,
        legacy_shared_prefix_text=shared_prefix_text,
    )
    dynamic_suffix = "\n\n".join(
        part
        for part in (
            "[Domain source text]\n" + context.domain_specific_text.strip(),
            "[Supplement source text]\n" + (context.supplement_block or "").strip(),
            _evidence_tiering_block(context),
            assessed_outcome_block(outcome),
            outcome_anchoring_block(outcome, sq_id),
            outcome_measurement_profile_block(outcome_measurement_profile, sq_id),
            domain_reasoning_guidance(sq_id),
            _sq_3_1_completeness_scaffold(sq_id),
            "[Signaling question]\n" + template.question_text,
            "[Previous answer]\n"
            f"answer: {raw.answer}\n"
            f"justification: {raw.justification}\n",
            "[Task]\n"
            "The previous response gave a substantive answer but omitted the quote. "
            "Return the single most relevant verbatim supporting sentence from the citable source text. "
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
                    "text": prefix_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic_suffix},
            ],
        },
    ]


def build_orientation_quote_repair_messages(
    *,
    sq_id: str,
    effect: str,
    outcome: str,
    shared_source_prefix_text: str,
    outcome_measurement_profile: (
        OutcomeMeasurementProfile | Mapping[str, Any] | None
    ) = None,
    context: DomainContext,
    raw: SQRawAnswer,
) -> list[dict[str, Any]]:
    template = get_sq_prompt(sq_id, effect)
    prefix_text = _render_static_prompt_prefix(
        trial_orientation_text="",
        shared_source_prefix_text=shared_source_prefix_text,
        legacy_shared_prefix_text="",
    )
    dynamic_suffix = "\n\n".join(
        part
        for part in (
            "[Domain source text]\n" + context.domain_specific_text.strip(),
            "[Supplement source text]\n" + (context.supplement_block or "").strip(),
            _evidence_tiering_block(context),
            assessed_outcome_block(outcome),
            outcome_anchoring_block(outcome, sq_id),
            outcome_measurement_profile_block(outcome_measurement_profile, sq_id),
            domain_reasoning_guidance(sq_id),
            _sq_3_1_completeness_scaffold(sq_id),
            "[Signaling question]\n" + template.question_text,
            "[Previous answer]\n"
            f"answer: {raw.answer}\n"
            f"non-citable quote: {raw.quote}\n"
            f"justification: {raw.justification}\n",
            "[Task]\n"
            "The previous response copied non-citable trial orientation into quote. "
            "Return only a replacement verbatim quote from the citable source text "
            "that supports the existing answer. Return an empty quote only if no "
            "supporting citable source text exists. Do not change the answer code or justification.",
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
                    "text": prefix_text,
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
    quote_sources = _quote_sources(
        context, raw_char_stream, page_boxes, source_document, ct_gov_block
    )

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
        quote_source_type = (
            _quote_source_type(matched_source_document, source_document)
            if quote_verified
            else None
        )

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
        quote=_soft_truncate(
            quote, _env_int("ARBITER_SQ_QUOTE_SOFT_LIMIT", DEFAULT_QUOTE_SOFT_LIMIT)
        ),
        page=page,
        justification=_soft_truncate(
            justification,
            _env_int(
                "ARBITER_SQ_JUSTIFICATION_SOFT_LIMIT", DEFAULT_JUSTIFICATION_SOFT_LIMIT
            ),
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
        return (
            value
            if isinstance(value, DomainContext)
            else DomainContext.model_validate(value)
        )

    raise TypeError(
        "sq_node requires state['domain_context'] or state['domain_contexts'][domain]"
    )


def _outcome_measurement_profile_from_state(
    state: Mapping[str, Any],
) -> OutcomeMeasurementProfile | None:
    profile = state.get("outcome_measurement_profile")
    if isinstance(profile, OutcomeMeasurementProfile):
        return profile
    if isinstance(profile, Mapping):
        return OutcomeMeasurementProfile.model_validate(profile)
    return None


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


def _quote_repair_max_tokens(config: AssessmentConfig | object) -> int:
    env = getattr(config, "env", None)
    configured = getattr(env, "quote_repair_max_tokens", None)
    if configured is not None:
        return int(configured)
    return int(getattr(config, "sq_max_tokens", 2048) or 2048)


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
        return [
            box if isinstance(box, PageBox) else PageBox.model_validate(box)
            for box in boxes
        ]
    section_map = state.get("section_map")
    section_boxes = getattr(section_map, "page_boxes", None)
    if isinstance(section_boxes, list):
        return [
            box if isinstance(box, PageBox) else PageBox.model_validate(box)
            for box in section_boxes
        ]
    return []


def _raw_char_stream_from_state(state: Mapping[str, Any]) -> str:
    stream = state.get("raw_char_stream")
    if stream is not None:
        return str(stream)
    section_map = state.get("section_map")
    return str(getattr(section_map, "full_text", "") or "")


def _trial_orientation_text_from_state(state: Mapping[str, Any]) -> str:
    return str(state.get("trial_orientation_text") or "")


def _shared_source_prefix_text_from_state(state: Mapping[str, Any]) -> str:
    source_prefix = str(state.get("shared_source_prefix_text") or "")
    if source_prefix.strip():
        return source_prefix
    return str(state.get("shared_prefix_text") or "")


def _matches_non_citable_orientation(quote: str, trial_orientation_text: str) -> bool:
    normalized_quote = _normalize_for_orientation_match(quote)
    normalized_orientation = _normalize_for_orientation_match(trial_orientation_text)
    if not normalized_quote or not normalized_orientation:
        return False
    if len(normalized_quote) < 12:
        return False
    if normalized_quote in normalized_orientation:
        return True
    quote_tokens = set(normalized_quote.split())
    orientation_tokens = set(normalized_orientation.split())
    if not quote_tokens:
        return False
    coverage = len(quote_tokens & orientation_tokens) / len(quote_tokens)
    return coverage >= 0.9


def _normalize_for_orientation_match(text: str) -> str:
    normalized = re.sub(r"[_:-]+", " ", text.casefold())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _record_sq_finalization_trace(
    state: Mapping[str, Any],
    sq_id: str,
    context: DomainContext,
    raw: SQRawAnswer,
    answer: SQAnswer,
    *,
    verification_raw: SQRawAnswer | None = None,
    orientation_quote_repair: OrientationQuoteRepairResult | None = None,
) -> None:
    qa_trace = _qa_trace_from_state(state)
    if qa_trace is None:
        return
    domain = _domain_for_sq(sq_id)
    source_document = _source_document_from_state(state)
    quote_raw = verification_raw or raw
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
        if AnswerCode(quote_raw.answer) == AnswerCode.NI
        else describe_quote_verification_sources(
            quote_raw.quote,
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
        "validated_answer": quote_raw.model_dump(mode="json"),
        "orientation_quote_repair": orientation_quote_repair.model_dump()
        if orientation_quote_repair is not None
        else None,
        "quote_verification": quote_verification,
        "final_answer": answer.model_dump(mode="json"),
        "confidence_flag": answer.confidence.flag.value,
        "soft_truncation": {
            "quote_truncated": quote_raw.quote != answer.quote
            and answer.answer != AnswerCode.NI,
            "justification_truncated": quote_raw.justification != answer.justification,
            "quote_original_length": len(quote_raw.quote),
            "quote_final_length": len(answer.quote),
            "justification_original_length": len(quote_raw.justification),
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
            "raw_quote": quote_raw.quote,
            "original_raw_quote": raw.quote,
            "failure_kind": orientation_quote_repair.failure_kind
            if orientation_quote_repair is not None
            else None,
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
            "failure_kind": orientation_quote_repair.failure_kind
            if orientation_quote_repair is not None
            else None,
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


def _quote_source_type(
    matched_source_document: str | None, main_source_document: str | None
) -> QuoteSourceType:
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
    sources = [
        QuoteSource(
            source_document=source_document,
            raw_char_stream=raw_char_stream,
            page_boxes=page_boxes,
        )
    ]
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
        for part in (
            context.domain_specific_text,
            context.supplement_block,
            ct_gov_block,
        )
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
                page_boxes=[
                    PageBox(
                        boxclass="text", text=text, bbox=(0.0, 0.0, 0.0, 0.0), page=page
                    )
                ],
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
