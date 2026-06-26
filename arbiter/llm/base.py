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


class LLMRequestTimeoutError(TimeoutError):
    """Raised when an ARBITER-bounded provider request times out."""


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
        self._last_provider_error: dict[str, Any] | None = None

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
        self._last_provider_error = None
        started = time.perf_counter()
        error: Exception | None = None
        result: BaseModel | None = None
        if self.supports_native_schema():
            method = self.native_method
            self._record_trace_start(messages=call_messages, schema=schema, method=method, call_label=call_label)
            try:
                result = await self._invoke_structured(
                    call_messages,
                    schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    method=method,
                    call_label=call_label,
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
        self._record_trace_start(messages=call_messages, schema=schema, method=method, call_label=call_label)
        try:
            result = await self._invoke_with_repair_ladder(
                call_messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
                call_label=call_label,
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
        call_label: str | None,
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
                    call_label=call_label,
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
        call_label: str | None,
    ) -> BaseModel:
        self._last_raw_response = None
        result = await self._invoke_with_network_retries(
            lambda: self._call_langchain_structured(
                messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
                method=method,
            ),
            messages=messages,
            schema=schema,
            method=method,
            call_label=call_label,
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

    async def _invoke_with_network_retries(
        self,
        call: Callable[[], Any],
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        method: str,
        call_label: str | None,
    ) -> Any:
        attempts = max(1, self.settings.network_max_retries)
        for attempt in range(attempts):
            self._last_network_attempts = attempt + 1
            attempt_started = time.perf_counter()
            self._record_network_attempt_start(
                messages=messages,
                schema=schema,
                method=method,
                call_label=call_label,
                attempt=attempt + 1,
                max_attempts=attempts,
            )
            try:
                return await asyncio.wait_for(call(), timeout=self.settings.llm_request_timeout_s)
            except TimeoutError as exc:
                timeout_error = LLMRequestTimeoutError(
                    f"{self.model} timed out after {self.settings.llm_request_timeout_s:g} seconds"
                )
                self._last_provider_error = provider_error_summary(timeout_error)
                transient_error = f"{type(timeout_error).__name__}: {timeout_error}"
                self._last_transient_errors.append(transient_error)
                self._record_network_attempt_failure(
                    messages=messages,
                    schema=schema,
                    method=method,
                    call_label=call_label,
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    elapsed_s=time.perf_counter() - attempt_started,
                    transient_error=transient_error,
                    provider_error=self._last_provider_error,
                    retrying=attempt < attempts - 1,
                )
                if attempt == attempts - 1:
                    raise timeout_error from exc
                delay = min(0.25 * (2**attempt), 2.0)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
            except Exception as exc:
                self._last_provider_error = provider_error_summary(exc)
                if _is_auth_error(exc):
                    raise LLMAuthenticationError(f"{self.model} authentication failed: {exc}") from exc
                if _is_invalid_request_error(exc):
                    raise LLMInvalidRequestError(f"{self.model} request was rejected: {exc}") from exc
                if not _is_transient_error(exc):
                    raise
                transient_error = f"{type(exc).__name__}: {exc}"
                self._last_transient_errors.append(transient_error)
                self._record_network_attempt_failure(
                    messages=messages,
                    schema=schema,
                    method=method,
                    call_label=call_label,
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    elapsed_s=time.perf_counter() - attempt_started,
                    transient_error=transient_error,
                    provider_error=self._last_provider_error,
                    retrying=attempt < attempts - 1,
                )
                if attempt == attempts - 1:
                    raise
                delay = min(0.25 * (2**attempt), 2.0)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
        raise AssertionError("unreachable")

    def _record_network_attempt_start(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        method: str,
        call_label: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        if self.trace is None or not hasattr(self.trace, "start_llm_network_attempt"):
            return
        self.trace.start_llm_network_attempt(
            model=self.model,
            call_label=call_label,
            messages=messages,
            schema_name=schema.__name__,
            method=method,
            attempt=attempt,
            max_attempts=max_attempts,
        )

    def _record_network_attempt_failure(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        method: str,
        call_label: str | None,
        attempt: int,
        max_attempts: int,
        elapsed_s: float,
        transient_error: str,
        provider_error: dict[str, Any] | None,
        retrying: bool,
    ) -> None:
        if self.trace is None or not hasattr(self.trace, "fail_llm_network_attempt"):
            return
        self.trace.fail_llm_network_attempt(
            model=self.model,
            call_label=call_label,
            messages=messages,
            schema_name=schema.__name__,
            method=method,
            attempt=attempt,
            max_attempts=max_attempts,
            elapsed_s=elapsed_s,
            transient_error=transient_error,
            provider_error=provider_error,
            retrying=retrying,
        )

    def _record_trace_start(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        method: str,
        call_label: str | None,
    ) -> None:
        if self.trace is None or not hasattr(self.trace, "start_llm_call"):
            return
        self.trace.start_llm_call(
            model=self.model,
            call_label=call_label,
            messages=messages,
            schema_name=schema.__name__,
            method=method,
        )

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
            provider_error=self._last_provider_error,
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


def provider_error_summary(exc: Exception) -> dict[str, Any]:
    """Extract safe provider diagnostics from common SDK exception shapes."""

    response = getattr(exc, "response", None)
    status_code = _first_present(
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
        getattr(response, "status", None),
    )
    headers = _sanitized_headers(getattr(response, "headers", None) or getattr(exc, "headers", None))
    request_id = _first_present(
        getattr(exc, "request_id", None),
        getattr(exc, "requestid", None),
        getattr(exc, "x_request_id", None),
        headers.get("x-request-id"),
        headers.get("request-id"),
    )
    normalized_status = _int_or_none(status_code)
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "status_code": normalized_status,
        "retryable": _is_retryable_provider_error(exc, normalized_status),
        "request_id": request_id,
        "headers": headers,
        "response_body": _response_body(response),
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _sanitized_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    safe_names = {
        "request-id",
        "x-request-id",
        "x-correlation-id",
        "cf-ray",
        "retry-after",
        "openai-processing-ms",
    }
    safe: dict[str, str] = {}
    for key, value in items:
        normalized = str(key).lower()
        if normalized in safe_names:
            safe[normalized] = str(value)
    return safe


def _response_body(response: Any) -> Any:
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return json_method()
        except Exception:
            pass
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)[:4000]
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")[:4000]
    if content is not None:
        return str(content)[:4000]
    return None


def _is_retryable_provider_error(exc: Exception, status_code: int | None) -> bool:
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status_code is not None and 400 <= status_code < 500:
        return False
    return _is_transient_error(exc)


def _is_transient_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "rate",
        "toomanyrequests",
        "too many requests",
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
