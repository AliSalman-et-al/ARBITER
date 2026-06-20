"""Factory for model-registry-backed LLM clients."""

from __future__ import annotations

from arbiter.config import MODEL_REGISTRY, EnvSettings
from arbiter.llm.anthropic_client import AnthropicLLMClient
from arbiter.llm.base import LLMClient
from arbiter.llm.openai_client import OpenAILLMClient
from arbiter.llm.openrouter_client import OpenRouterLLMClient


def create_llm_client(
    model: str,
    trace: object | None = None,
    *,
    settings: EnvSettings | None = None,
) -> LLMClient:
    """Create the configured provider client for ``model``."""

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown LLM model {model!r}.")

    info = MODEL_REGISTRY[model]
    kwargs = {
        "model_id": info["model_id"],
        "supports_cache": info.get("supports_cache", False),
        "supports_schema": info.get("supports_native_schema", False),
        "supports_vision": info.get("supports_vision", False),
        "trace": trace,
        "settings": settings,
    }

    provider = info["provider"]
    if provider == "anthropic":
        return AnthropicLLMClient(model, **kwargs)
    if provider == "openai":
        return OpenAILLMClient(model, **kwargs)
    if provider == "openrouter":
        return OpenRouterLLMClient(model, **kwargs)

    raise ValueError(f"Unsupported provider {provider!r} for model {model!r}.")
