"""Anthropic LangChain client wrapper."""

from __future__ import annotations

from typing import Any

from arbiter.llm.base import LangChainLLMClient


class AnthropicLLMClient(LangChainLLMClient):
    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Install ARBITER's anthropic extra to use Anthropic models.") from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "timeout": self.settings.llm_request_timeout_s,
        }
        if self.settings.anthropic_api_key:
            kwargs["api_key"] = self.settings.anthropic_api_key
        return ChatAnthropic(**kwargs)
