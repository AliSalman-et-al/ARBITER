"""Configuration and model registry for ARBITER."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv

load_dotenv()

EffectOfInterest = Literal["assignment", "adhering"]
TraceLevel = Literal["off", "summary", "full"]
Provider = Literal["anthropic", "openai", "openrouter"]


class ModelInfo(TypedDict, total=False):
    provider: Provider
    model_id: str
    base_url: str
    supports_cache: bool | str
    supports_native_schema: bool | str
    supports_vision: bool
    supports_reasoning: bool
    context_window: int
    max_output_tokens: int
    input_modalities: list[str]
    output_modalities: list[str]
    price_per_mtok_in: float | None
    price_per_mtok_out: float | None
    price_per_mtok_cache_read: float | None
    price_per_mtok_cache_write: float | None


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gpt-oss-120b": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "supports_cache": False,
        "supports_native_schema": True,
        "supports_vision": False,
        "supports_reasoning": True,
        "context_window": 131072,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.039,
        "price_per_mtok_out": 0.18,
    },
    "gpt-oss-20b": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-20b",
        "supports_cache": False,
        "supports_native_schema": True,
        "supports_vision": False,
        "supports_reasoning": True,
        "context_window": 131072,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.029,
        "price_per_mtok_out": 0.14,
    },
    "qwen3-32b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-32b",
        "supports_cache": False,
        "supports_native_schema": True,
        "supports_vision": False,
        "supports_reasoning": True,
        "context_window": 131072,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.08,
        "price_per_mtok_out": 0.28,
    },
    "llama-3.3-70b-instruct": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "supports_cache": False,
        "supports_native_schema": True,
        "supports_vision": False,
        "supports_reasoning": False,
        "context_window": 131072,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.10,
        "price_per_mtok_out": 0.32,
    },
    "qwen3-235b-a22b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b",
        "supports_cache": False,
        "supports_native_schema": "json_object_only",
        "supports_vision": False,
        "supports_reasoning": True,
        "context_window": 131072,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.455,
        "price_per_mtok_out": 1.82,
    },
    "claude-sonnet-4.6": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "supports_cache": True,
        "supports_native_schema": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 3.0,
        "price_per_mtok_out": 15.0,
        "price_per_mtok_cache_read": 0.30,
        "price_per_mtok_cache_write": 3.75,
    },
    "gpt-5.5": {
        "provider": "openai",
        "model_id": "gpt-5.5",
        "supports_cache": True,
        "supports_native_schema": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 5.0,
        "price_per_mtok_out": 30.0,
        "price_per_mtok_cache_read": 0.50,
    },
    "google/gemma-4-31b-it:free": {
        "provider": "openrouter",
        "model_id": "google/gemma-4-31b-it:free",
        "supports_cache": False,
        "supports_native_schema": "json_object_only",
        "supports_vision": True,
        "supports_reasoning": True,
        "context_window": 262144,
        "input_modalities": ["text", "image", "video"],
        "output_modalities": ["text"],
        "price_per_mtok_in": 0.0,
        "price_per_mtok_out": 0.0,
    },
}


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(_env_str(name, default) or default)


@dataclass
class EnvSettings:
    anthropic_api_key: str | None = field(default_factory=lambda: _env_str("ANTHROPIC_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    openrouter_api_key: str | None = field(default_factory=lambda: _env_str("OPENROUTER_API_KEY"))
    prefix_token_budget: int = field(default_factory=lambda: _env_int("ARBITER_PREFIX_TOKEN_BUDGET", 4000))
    supplement_token_budget: int = field(default_factory=lambda: _env_int("ARBITER_SUPPLEMENT_TOKEN_BUDGET", 2000))
    metadata_token_budget: int = field(default_factory=lambda: _env_int("ARBITER_METADATA_TOKEN_BUDGET", 3000))
    max_outcomes: int = field(default_factory=lambda: _env_int("ARBITER_MAX_OUTCOMES", 10))
    domain_text_min_chars: int = field(default_factory=lambda: _env_int("ARBITER_DOMAIN_TEXT_MIN_CHARS", 500))
    domain_text_token_budget: int = field(default_factory=lambda: _env_int("ARBITER_DOMAIN_TEXT_TOKEN_BUDGET", 1500))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("ARBITER_RETRIEVAL_TOP_K", 5))
    small_segment_token_threshold: int = field(
        default_factory=lambda: _env_int("ARBITER_SMALL_SEGMENT_TOKEN_THRESHOLD", 1500)
    )
    large_segment_char_threshold: int = field(default_factory=lambda: _env_int("ARBITER_LARGE_SEGMENT_CHAR_THRESHOLD", 6000))
    retrieval_uncertain_threshold: float = field(
        default_factory=lambda: _env_float("ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD", 0.35)
    )
    supplement_parse_window: int = field(default_factory=lambda: _env_int("ARBITER_SUPPLEMENT_PARSE_WINDOW", 20))
    doctype_scan_pages: int = field(default_factory=lambda: _env_int("ARBITER_DOCTYPE_SCAN_PAGES", 10))
    min_segments: int = field(default_factory=lambda: _env_int("ARBITER_MIN_SEGMENTS", 3))
    domain_tag_scan_chars: int = field(default_factory=lambda: _env_int("ARBITER_DOMAIN_TAG_SCAN_CHARS", 300))
    quote_verify_threshold: int = field(default_factory=lambda: _env_int("ARBITER_QUOTE_VERIFY_THRESHOLD", 85))
    quote_min_verify_chars: int = field(default_factory=lambda: _env_int("ARBITER_QUOTE_MIN_VERIFY_CHARS", 15))
    outcome_match_threshold: float = field(default_factory=lambda: _env_float("ARBITER_OUTCOME_MATCH_THRESHOLD", 0.85))
    schema_repair_max_retries: int = field(default_factory=lambda: _env_int("ARBITER_SCHEMA_REPAIR_MAX_RETRIES", 2))
    network_max_retries: int = field(default_factory=lambda: _env_int("ARBITER_NETWORK_MAX_RETRIES", 3))
    max_annotations_per_doc: int = field(default_factory=lambda: _env_int("ARBITER_MAX_ANNOTATIONS_PER_DOC", 40))
    annotation_preamble_tokens: int = field(default_factory=lambda: _env_int("ARBITER_ANNOTATION_PREAMBLE_TOKENS", 500))
    consort_detect_threshold: float = field(default_factory=lambda: _env_float("ARBITER_CONSORT_DETECT_THRESHOLD", 0.80))
    consort_enabled: bool = field(default_factory=lambda: _env_bool("ARBITER_CONSORT_ENABLED", False))
    max_concurrency: int = field(default_factory=lambda: _env_int("ARBITER_MAX_CONCURRENCY", 2))


@dataclass
class AssessmentConfig:
    paper_path: Path
    supplement_paths: list[Path] = field(default_factory=list)
    nct_number: str | None = None
    trial_label: str | None = None
    outcomes: list[str] | None = None
    effect_of_interest: EffectOfInterest = "assignment"
    sq_model: str = field(default_factory=lambda: _env_str("ARBITER_SQ_MODEL", "gpt-oss-120b") or "gpt-oss-120b")
    aux_model: str = field(default_factory=lambda: _env_str("ARBITER_AUX_MODEL", "gpt-oss-120b") or "gpt-oss-120b")
    vision_model: str | None = field(default_factory=lambda: _env_str("ARBITER_VISION_MODEL"))
    consort_vision_enabled: bool = field(default_factory=lambda: _env_bool("ARBITER_CONSORT_ENABLED", False))
    sq_max_tokens: int = field(default_factory=lambda: _env_int("ARBITER_SQ_MAX_TOKENS", 2048))
    output_dir: Path = field(default_factory=lambda: _env_path("ARBITER_OUTPUT_DIR", "./output"))
    db_path: Path = field(default_factory=lambda: _env_path("ARBITER_DB_PATH", "./arbiter.db"))
    force: bool = False
    trace_level: TraceLevel = field(
        default_factory=lambda: cast(TraceLevel, _env_str("ARBITER_TRACE_LEVEL", "full") or "full")
    )
    report_enabled: bool = field(default_factory=lambda: _env_bool("ARBITER_REPORT_ENABLED", True))
    env: EnvSettings = field(default_factory=EnvSettings)

    @classmethod
    def from_env(
        cls,
        *,
        paper_path: Path,
        supplement_paths: list[Path] | None = None,
        nct_number: str | None = None,
        trial_label: str | None = None,
        outcomes: list[str] | None = None,
        effect_of_interest: EffectOfInterest = "assignment",
        sq_model: str | None = None,
        aux_model: str | None = None,
        output_dir: Path | None = None,
        db_path: Path | None = None,
        max_concurrency: int | None = None,
        force: bool = False,
        trace_level: TraceLevel | None = None,
        report_enabled: bool | None = None,
    ) -> "AssessmentConfig":
        config = cls(
            paper_path=paper_path,
            supplement_paths=supplement_paths or [],
            nct_number=nct_number,
            trial_label=trial_label,
            outcomes=outcomes,
            effect_of_interest=effect_of_interest,
            force=force,
        )
        if sq_model is not None:
            config.sq_model = sq_model
        if aux_model is not None:
            config.aux_model = aux_model
        if output_dir is not None:
            config.output_dir = output_dir
        if db_path is not None:
            config.db_path = db_path
        if max_concurrency is not None:
            config.env.max_concurrency = max_concurrency
        if trace_level is not None:
            config.trace_level = trace_level
        if report_enabled is not None:
            config.report_enabled = report_enabled
        return config
