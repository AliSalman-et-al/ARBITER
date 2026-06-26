"""Vanilla OpenAI LangChain client wrapper."""

from __future__ import annotations

from typing import Any

from arbiter.llm.base import LangChainLLMClient, strip_cache_control


class OpenAILLMClient(LangChainLLMClient):
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return strip_cache_control(messages)

    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Install ARBITER's openai extra to use OpenAI models.") from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "timeout": self.settings.llm_request_timeout_s,
        }
        if self.settings.openai_api_key:
            kwargs["api_key"] = self.settings.openai_api_key
        return ChatOpenAI(**kwargs)
