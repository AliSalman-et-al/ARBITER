"""OpenRouter client wrapper."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from arbiter.llm.base import LangChainLLMClient, strip_cache_control

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterTransientResponseError(RuntimeError):
    """Raised when OpenRouter returns a retryable response without usable content."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _ProviderErrorResponse(status_code, response_body)


class _ProviderErrorResponse:
    def __init__(self, status_code: int | None, body: Any | None) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


def _make_transport() -> httpx.AsyncBaseTransport | None:
    return None


class OpenRouterLLMClient(LangChainLLMClient):
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return strip_cache_control(messages)

    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise ImportError(
                "Install ARBITER's openrouter extra to use OpenRouter models."
            ) from exc

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
            raise PermissionError(
                "OPENROUTER_API_KEY is required for OpenRouter models"
            )

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": _response_format_for_schema(schema, method),
            "plugins": [{"id": "response-healing"}],
        }
        session_id = _openrouter_session_id(
            self.settings.openrouter_session_id, self.trace
        )
        if session_id is not None:
            payload["session_id"] = session_id
        reasoning = _reasoning_config(
            max_tokens=max_tokens,
            requested_reasoning_tokens=self.settings.reasoning_max_tokens,
            output_reserve_tokens=self.settings.reasoning_output_reserve_tokens,
        )
        if self._supports_reasoning and reasoning is not None:
            payload["reasoning"] = reasoning
        if method == "json_schema":
            payload["provider"] = {"require_parameters": True}
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_response_cache:
            headers["X-OpenRouter-Cache"] = "true"
        async with httpx.AsyncClient(
            transport=_make_transport(),
            timeout=self.settings.llm_request_timeout_s,
        ) as client:
            response = await client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
            response.raise_for_status()
            self._last_cache_hit = _cache_hit_from_header(
                response.headers.get("X-OpenRouter-Cache-Status")
            )
            payload = response.json()
            _raise_for_retryable_error_envelope(payload)
            return payload


def _reasoning_config(
    *,
    max_tokens: int,
    requested_reasoning_tokens: int,
    output_reserve_tokens: int,
) -> dict[str, Any] | None:
    if max_tokens <= 1 or requested_reasoning_tokens <= 0:
        return None
    reserve = max(1, output_reserve_tokens)
    ceiling = min(requested_reasoning_tokens, max_tokens - reserve)
    if ceiling <= 0:
        ceiling = max_tokens - 1
    return {"max_tokens": ceiling, "exclude": False}


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


def _openrouter_session_id(configured: str | None, trace: object | None) -> str | None:
    if configured:
        return configured
    trial_id = getattr(trace, "trial_id", None)
    if trial_id:
        return f"arbiter:{trial_id}"
    return None


def _cache_hit_from_header(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "hit":
        return True
    if normalized == "miss":
        return False
    return None


def _extract_message_content(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterTransientResponseError(
            "OpenRouter response did not contain choices[0].message.content"
        ) from exc
    content = message.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return _salvage_json_text(stripped)
    fallback = _reasoning_content(message)
    if fallback:
        return _salvage_json_text(fallback)
    if isinstance(content, str):
        raise OpenRouterTransientResponseError(
            "OpenRouter response contained empty choices[0].message.content"
        )
    return json.dumps(content)


def _raise_for_retryable_error_envelope(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    error = payload.get("error")
    if error is None:
        return
    status_code = _error_envelope_status_code(error)
    message = _error_envelope_message(error)
    if status_code is not None and status_code < 500 and status_code != 429:
        return
    if status_code is None and not _message_describes_retryable_provider_error(message):
        return
    raise OpenRouterTransientResponseError(
        f"OpenRouter returned retryable provider error envelope: {message}",
        status_code=status_code,
        response_body=payload,
    )


def _error_envelope_status_code(error: Any) -> int | None:
    if not isinstance(error, dict):
        return None
    for key in ("status", "status_code", "code"):
        value = error.get(key)
        try:
            status_code = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= status_code <= 599:
            return status_code
    return None


def _error_envelope_message(error: Any) -> str:
    if isinstance(error, dict):
        for key in ("message", "detail", "type", "code"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(error)


def _message_describes_retryable_provider_error(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "rate",
            "too many requests",
            "temporar",
            "server",
            "unavailable",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _reasoning_content(message: dict[str, Any]) -> str | None:
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_think_blocks(value.strip())
    details = message.get("reasoning_details")
    if isinstance(details, list):
        text = "\n".join(_iter_text_fragments(details)).strip()
        if text:
            return _strip_think_blocks(text)
    return None


def _iter_text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_iter_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for key in ("text", "content", "output", "reasoning"):
            fragments.extend(_iter_text_fragments(value.get(key)))
        return fragments
    return []


def _strip_think_blocks(text: str) -> str:
    return re.sub(
        r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()


def _salvage_json_text(text: str) -> str:
    candidates = [text, _strip_code_fence(text), _first_balanced_object(text)]
    for candidate in list(candidates):
        if candidate:
            candidates.append(_unicode_unescape(candidate))
    for candidate in candidates:
        if not candidate:
            continue
        stripped = candidate.strip()
        if _is_json_object_text(stripped):
            return stripped
    return text


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else text


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _unicode_unescape(text: str) -> str:
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def _is_json_object_text(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False
