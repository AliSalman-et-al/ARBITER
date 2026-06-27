"""Serializable LangGraph state and runtime handles for assessment graphs."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from typing_extensions import TypedDict

from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMClient
from arbiter.models import (
    DomainContext,
    DomainJudgment,
    Judgment,
    SQAnswer,
    SectionMap,
    TrialMetadata,
)
from arbiter.retrieval.supplement_index import SupplementIndex


def merge_dict(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge graph maps while rejecting conflicting duplicate keys."""

    merged = dict(left or {})
    for key, value in dict(right or {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting graph state value for key {key!r}")
        merged[key] = value
    return merged


class IngestionState(TypedDict, total=False):
    config: AssessmentConfig
    config_summary: dict[str, Any]
    trial_metadata: TrialMetadata
    section_map: SectionMap
    raw_char_stream: str
    ct_gov_data: dict[str, Any] | None
    ctgov_record: dict[str, Any] | None
    shared_prefix_text: str
    ct_gov_block: str | None
    effect_of_interest: str


class TrialState(IngestionState, total=False):
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]
    errors: Annotated[list[str], operator.add]
    consort: None


class OutcomeState(IngestionState, total=False):
    outcome: str
    trial_domain_judgments: list[DomainJudgment]
    outcome_change_detected: bool | None
    registered_outcome: str | None
    published_outcome: str | None
    outcome_similarity_score: float | None
    registered_as_primary: bool | None
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]
    overall_judgment: Judgment | None
    overall_rationale: str | None
    requires_human_review: bool | None
    errors: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class AssessmentRuntime:
    """Non-serializable graph handles supplied through LangGraph context."""

    llm_client_sq: LLMClient
    llm_client_aux: LLMClient
    supplement_index: SupplementIndex
    trace: object | None = None
    llm_client_vision: LLMClient | None = None

    @property
    def sq_model(self) -> LLMClient:
        return self.llm_client_sq

    @property
    def aux_model(self) -> LLMClient:
        return self.llm_client_aux


@dataclass
class TrialContext:
    """Phase-1 trial bundle consumed by assess_trial without re-ingestion."""

    config_summary: dict[str, Any]
    trial_metadata: TrialMetadata
    section_map: SectionMap
    raw_char_stream: str
    supplement_index: SupplementIndex
    ct_gov_data: dict[str, Any] | None
    shared_prefix_text: str
    ct_gov_block: str | None
    llm_client_sq: LLMClient
    llm_client_aux: LLMClient
    trace: object | None = None


def base_ingestion_state(ctx: TrialContext, config: AssessmentConfig) -> IngestionState:
    effect = str(getattr(config.effect_of_interest, "value", config.effect_of_interest))
    return {
        "config": config,
        "config_summary": dict(ctx.config_summary),
        "trial_metadata": ctx.trial_metadata,
        "section_map": ctx.section_map,
        "raw_char_stream": ctx.raw_char_stream,
        "ct_gov_data": ctx.ct_gov_data,
        "ctgov_record": ctx.ct_gov_data,
        "shared_prefix_text": ctx.shared_prefix_text,
        "ct_gov_block": ctx.ct_gov_block,
        "effect_of_interest": effect,
    }
