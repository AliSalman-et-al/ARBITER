"""Provider-neutral LLM client contract for structured ARBITER calls."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from arbiter.config import EnvSettings


class LLMAuthenticationError(RuntimeError):
    """Raised when a provider rejects credentials or request authorization."""


class LLMInvalidRequestError(RuntimeError):
    """Raised when a provider rejects a non-retryable request."""


class LLMClient(ABC):
    """Abstract client that always returns validated Pydantic output."""

    def __init__(
        self,
        model: str,
        *,
        trace: object | None = None,
        settings: EnvSettings | None = None,
    ) -> None:
        self.model = model
        self.trace = trace
        self.settings = settings or EnvSettings()

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        call_label: str | None = None,
    ) -> BaseModel:
        """Return a validated instance of ``schema``."""

    @abstractmethod
    def supports_prompt_caching(self) -> bool:
        """Whether provider-specific prompt cache directives are supported."""

    @abstractmethod
    def supports_native_schema(self) -> bool:
        """Whether the provider/model can enforce the schema at decode time."""

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether the model can receive image content in principle."""

    async def complete_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        raise NotImplementedError("Vision (CONSORT extraction) lands in v0.2.")


class LangChainLLMClient(LLMClient):
    """Common structured-output implementation for LangChain chat wrappers."""

    native_method = "json_schema"
    repair_method = "json_mode"

    def __init__(
        self,
        model: str,
        *,
        model_id: str,
        supports_cache: bool | str,
        supports_schema: bool | str,
        supports_vision: bool,
        trace: object | None = None,
        settings: EnvSettings | None = None,
    ) -> None:
        super().__init__(model, trace=trace, settings=settings)
        self.model_id = model_id
        self._supports_cache = supports_cache
        self._supports_schema = supports_schema
        self._supports_vision = supports_vision
        self._last_repair_attempts: list[dict[str, Any]] = []
        self._last_network_attempts = 0
        self._last_transient_errors: list[str] = []
        self._last_usage: dict[str, int | None] = {}
        self._last_raw_response: Any | None = None

    @abstractmethod
    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        """Construct the provider LangChain chat model with retries disabled."""

    def supports_prompt_caching(self) -> bool:
        return bool(self._supports_cache)

    def supports_native_schema(self) -> bool:
        return self._supports_schema is True

    def supports_vision(self) -> bool:
        return self._supports_vision

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        call_label: str | None = None,
    ) -> BaseModel:
        call_messages = self._prepare_messages(messages)
        self._last_repair_attempts = []
        self._last_network_attempts = 0
        self._last_transient_errors = []
        self._last_usage = {}
        self._last_raw_response = None
        started = time.perf_counter()
        error: Exception | None = None
        result: BaseModel | None = None
        if self.supports_native_schema():
            method = self.native_method
            try:
                result = await self._invoke_structured(
                    call_messages,
                    schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    method=method,
                )
                return result
            except Exception as exc:
                error = exc
                raise
            finally:
                self._record_trace(
                    messages=call_messages,
                    schema=schema,
                    method=method,
                    call_label=call_label,
                    latency_s=time.perf_counter() - started,
                    error=error,
                    result=result,
                )

        method = self.repair_method
        try:
            result = await self._invoke_with_repair_ladder(
                call_messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            self._record_trace(
                messages=call_messages,
                schema=schema,
                method=method,
                call_label=call_label,
                latency_s=time.perf_counter() - started,
                error=error,
                result=result,
            )

    async def _invoke_with_repair_ladder(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
    ) -> BaseModel:
        repair_messages = list(messages)
        last_error: Exception | None = None
        max_retries = self.settings.schema_repair_max_retries

        for attempt in range(max_retries + 1):
            repair_prompt = _repair_prompt_from_messages(repair_messages, original_count=len(messages))
            try:
                result = await self._invoke_structured(
                    repair_messages,
                    schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    method=self.repair_method,
                )
                self._last_repair_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "validated": True,
                        "error": None,
                        "repair_prompt": repair_prompt,
                        "request_messages": repair_messages,
                        "raw_response": self._last_raw_response,
                        "parsed_response": result,
                        "validation_result": {
                            "schema": schema.__name__,
                            "validated": True,
                            "error": None,
                        },
                    }
                )
                return result
            except ValueError as exc:
                last_error = exc
                self._last_repair_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "validated": False,
                        "error": str(exc),
                        "repair_prompt": repair_prompt,
                        "request_messages": repair_messages,
                        "raw_response": self._last_raw_response,
                        "parsed_response": None,
                        "validation_result": {
                            "schema": schema.__name__,
                            "validated": False,
                            "error": str(exc),
                        },
                    }
                )
                if attempt >= max_retries:
                    break
                repair_messages = [
                    *repair_messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not validate against the required JSON schema.\n"
                            f"Validation/parsing error:\n{exc}\n\n"
                            "Return only corrected JSON for the same task."
                        ),
                    },
                ]

        raise ValueError(
            f"{self.model} failed to produce valid {schema.__name__} after "
            f"{max_retries + 1} schema attempts: {last_error}"
        )

    async def _invoke_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> BaseModel:
        self._last_raw_response = None
        result = await self._invoke_with_network_retries(
            lambda: self._call_langchain_structured(
                messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
                method=method,
            )
        )
        self._last_raw_response = result
        self._last_usage = _extract_usage(result)
        return _coerce_structured_result(result, schema)

    async def _call_langchain_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> Any:
        chat_model = self._make_chat_model(temperature=temperature, max_tokens=max_tokens)
        structured = chat_model.with_structured_output(schema, method=method, include_raw=True)
        return await structured.ainvoke(messages)

    async def _invoke_with_network_retries(self, call: Callable[[], Any]) -> Any:
        attempts = max(1, self.settings.network_max_retries)
        for attempt in range(attempts):
            self._last_network_attempts = attempt + 1
            try:
                return await call()
            except Exception as exc:
                if _is_auth_error(exc):
                    raise LLMAuthenticationError(f"{self.model} authentication failed: {exc}") from exc
                if _is_invalid_request_error(exc):
                    raise LLMInvalidRequestError(f"{self.model} request was rejected: {exc}") from exc
                if not _is_transient_error(exc):
                    raise
                self._last_transient_errors.append(f"{type(exc).__name__}: {exc}")
                if attempt == attempts - 1:
                    raise
                delay = min(0.25 * (2**attempt), 2.0)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
        raise AssertionError("unreachable")

    def _record_trace(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        method: str,
        call_label: str | None,
        latency_s: float,
        error: Exception | None,
        result: BaseModel | None,
    ) -> None:
        if self.trace is None or not hasattr(self.trace, "record_llm_call"):
            return
        self.trace.record_llm_call(
            model=self.model,
            call_label=call_label,
            messages=messages,
            schema_name=schema.__name__,
            method=method,
            input_tokens=self._last_usage.get("input_tokens"),
            output_tokens=self._last_usage.get("output_tokens"),
            cache_read_tokens=self._last_usage.get("cache_read_tokens"),
            cache_write_tokens=self._last_usage.get("cache_write_tokens"),
            latency_s=latency_s,
            repair_attempts=self._last_repair_attempts,
            network_attempts=self._last_network_attempts or None,
            transient_errors=self._last_transient_errors,
            error=str(error) if error is not None else None,
            cache_hit=None if not self.supports_prompt_caching() else False,
            raw_response=self._last_raw_response,
            parsed_response=result,
            validation_result={
                "schema": schema.__name__,
                "validated": error is None and result is not None,
                "error": str(error) if error is not None else None,
            },
            final_result=result,
        )


def _coerce_structured_result(result: Any, schema: type[BaseModel]) -> BaseModel:
    if isinstance(result, dict) and {"parsed", "raw", "parsing_error"} <= set(result):
        parsing_error = result.get("parsing_error")
        parsed = result.get("parsed")
        if parsing_error is not None:
            raise ValueError(str(parsing_error))
        if parsed is None:
            raise ValueError("structured output parser returned no parsed value")
        return _validate_schema_instance(parsed, schema)

    return _validate_schema_instance(result, schema)


def _repair_prompt_from_messages(messages: list[dict[str, Any]], *, original_count: int) -> str | None:
    if len(messages) <= original_count:
        return None
    content = messages[-1].get("content")
    return content if isinstance(content, str) else None


def _validate_schema_instance(value: Any, schema: type[BaseModel]) -> BaseModel:
    if isinstance(value, schema):
        return value
    return schema.model_validate(value)


def _extract_usage(result: Any) -> dict[str, int | None]:
    raw = result.get("raw") if isinstance(result, dict) else result
    metadata = getattr(raw, "usage_metadata", None)
    if not isinstance(metadata, dict):
        response_metadata = getattr(raw, "response_metadata", None)
        if isinstance(response_metadata, dict):
            metadata = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(metadata, dict):
        return {}
    details = metadata.get("input_token_details") or metadata.get("prompt_token_details") or {}
    return {
        "input_tokens": _int_or_none(metadata.get("input_tokens") or metadata.get("prompt_tokens")),
        "output_tokens": _int_or_none(metadata.get("output_tokens") or metadata.get("completion_tokens")),
        "cache_read_tokens": _int_or_none(details.get("cache_read") or details.get("cached_tokens")),
        "cache_write_tokens": _int_or_none(details.get("cache_write")),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strip_cache_control(value: Any) -> Any:
    """Remove provider-specific cache directives from nested message content."""

    if isinstance(value, list):
        return [strip_cache_control(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_cache_control(item) for key, item in value.items() if key != "cache_control"}
    return value


def _is_transient_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "rate",
        "timeout",
        "temporar",
        "connection",
        "connect",
        "server",
        "unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in name or marker in text for marker in markers)


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "authentication",
        "permissiondenied",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "incorrect api key",
        "api key",
        "401",
        "403",
    )
    return any(marker in name or marker in text for marker in markers)


def _is_invalid_request_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "badrequest",
        "invalidrequest",
        "unprocessable",
        "context length",
        "maximum context",
        "400",
        "422",
    )
    return any(marker in name or marker in text for marker in markers)
