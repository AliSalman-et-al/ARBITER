"""OpenRouter client wrapper."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from arbiter.llm.base import LangChainLLMClient

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _make_transport() -> httpx.AsyncBaseTransport | None:
    return None


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

    async def _call_langchain_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> Any:
        response = await self._post_chat_completion(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            method=method,
        )
        content = _extract_message_content(response)
        try:
            parsed = schema.model_validate_json(content)
            parsing_error = None
        except (ValidationError, ValueError, TypeError) as exc:
            parsed = None
            parsing_error = exc
        return {"parsed": parsed, "raw": response, "parsing_error": parsing_error}

    async def _post_chat_completion(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> dict[str, Any]:
        if not self.settings.openrouter_api_key:
            raise PermissionError("OPENROUTER_API_KEY is required for OpenRouter models")

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": _response_format_for_schema(schema, method),
            "provider": {"require_parameters": True},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            transport=_make_transport(),
            timeout=self.settings.llm_request_timeout_s,
        ) as client:
            response = await client.post(OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


def _response_format_for_schema(schema: type[BaseModel], method: str) -> dict[str, Any]:
    if method == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
    return {"type": "json_object"}


def _extract_message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter response did not contain choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    return json.dumps(content)
