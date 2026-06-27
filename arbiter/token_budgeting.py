"""Context-window-aware token budgeting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from arbiter.config import AssessmentConfig, EnvSettings, MODEL_REGISTRY

ZoneName = Literal["shared_prefix", "domain_text", "supplement_block", "metadata"]


@dataclass(frozen=True)
class TrimReport:
    zone: ZoneName
    original_tokens: int
    kept_tokens: int
    dropped_tokens: int
    budget_tokens: int

    @property
    def trimmed(self) -> bool:
        return self.dropped_tokens > 0


@dataclass(frozen=True)
class BudgetedText:
    text: str
    report: TrimReport


@dataclass(frozen=True)
class InputTokenBudget:
    context_window: int
    reserved_output_tokens: int
    input_budget: int
    zone_budgets: dict[ZoneName, int]


ZONE_SHARES: dict[ZoneName, float] = {
    "shared_prefix": 0.60,
    "domain_text": 0.25,
    "supplement_block": 0.15,
    "metadata": 0.30,
}


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def cap_text_to_tokens(text: str, token_budget: int, zone: ZoneName) -> BudgetedText:
    if token_budget <= 0 or not text:
        original = count_tokens(text)
        return BudgetedText(
            text="",
            report=TrimReport(
                zone=zone,
                original_tokens=original,
                kept_tokens=0,
                dropped_tokens=original,
                budget_tokens=max(0, token_budget),
            ),
        )
    encoding = _encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= token_budget:
        cleaned = text.strip()
        kept = count_tokens(cleaned)
        return BudgetedText(
            text=cleaned,
            report=TrimReport(
                zone=zone,
                original_tokens=len(tokens),
                kept_tokens=kept,
                dropped_tokens=max(0, len(tokens) - kept),
                budget_tokens=token_budget,
            ),
        )
    trimmed = encoding.decode(tokens[:token_budget]).rstrip()
    kept = count_tokens(trimmed)
    return BudgetedText(
        text=trimmed,
        report=TrimReport(
            zone=zone,
            original_tokens=len(tokens),
            kept_tokens=kept,
            dropped_tokens=max(0, len(tokens) - kept),
            budget_tokens=token_budget,
        ),
    )


def input_token_budget(
    *,
    config: AssessmentConfig | None = None,
    settings: EnvSettings | None = None,
) -> InputTokenBudget:
    active_settings = settings or getattr(config, "env", None) or EnvSettings()
    context_window = _context_window(config)
    reserved_output = _reserved_output_tokens(config, active_settings)
    input_budget = max(1, context_window - reserved_output)
    return InputTokenBudget(
        context_window=context_window,
        reserved_output_tokens=reserved_output,
        input_budget=input_budget,
        zone_budgets={
            zone: max(1, int(input_budget * share))
            for zone, share in ZONE_SHARES.items()
        },
    )


def zone_budget(
    zone: ZoneName,
    *,
    config: AssessmentConfig | None = None,
    settings: EnvSettings | None = None,
) -> int:
    budget = input_token_budget(config=config, settings=settings).zone_budgets[zone]
    legacy = _legacy_zone_budget(zone, settings or getattr(config, "env", None))
    return min(budget, legacy) if legacy is not None else budget


def trim_reports_payload(*reports: TrimReport) -> dict[str, Any]:
    return {
        report.zone: {
            "budget_tokens": report.budget_tokens,
            "original_tokens": report.original_tokens,
            "kept_tokens": report.kept_tokens,
            "dropped_tokens": report.dropped_tokens,
            "trimmed": report.trimmed,
        }
        for report in reports
    }


def _reserved_output_tokens(config: AssessmentConfig | None, settings: EnvSettings) -> int:
    sq_max = int(getattr(config, "sq_max_tokens", 0) or 0)
    return max(1, sq_max + settings.reasoning_max_tokens + settings.reasoning_output_reserve_tokens)


def _context_window(config: AssessmentConfig | None) -> int:
    if config is None:
        return int(MODEL_REGISTRY["gpt-oss-120b"]["context_window"])
    model_info = MODEL_REGISTRY.get(config.sq_model, {})
    return int(model_info.get("context_window") or MODEL_REGISTRY["gpt-oss-120b"]["context_window"])


def _legacy_zone_budget(zone: ZoneName, settings: EnvSettings | None) -> int | None:
    if settings is None:
        return None
    if zone == "shared_prefix":
        return settings.prefix_token_budget
    if zone == "domain_text":
        return settings.domain_text_token_budget
    if zone == "supplement_block":
        return settings.supplement_token_budget
    if zone == "metadata":
        return settings.metadata_token_budget
    return None


@lru_cache(maxsize=1)
def _encoding() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_harmony")
    except Exception:
        try:
            import tiktoken

            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return _WhitespaceEncoding()


class _WhitespaceEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)
