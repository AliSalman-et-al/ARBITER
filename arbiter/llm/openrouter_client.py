"""OpenRouter LangChain client wrapper."""

from __future__ import annotations

from typing import Any

from arbiter.llm.base import LangChainLLMClient


class OpenRouterLLMClient(LangChainLLMClient):
    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise ImportError("Install ARBITER's openrouter extra to use OpenRouter models.") from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "timeout": self.settings.llm_request_timeout_s,
        }
        if self.settings.openrouter_api_key:
            kwargs["api_key"] = self.settings.openrouter_api_key
        return ChatOpenRouter(**kwargs)
