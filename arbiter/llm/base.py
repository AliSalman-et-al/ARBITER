"""Provider-neutral LLM client contract for structured ARBITER calls."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from arbiter.config import EnvSettings


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
        if self.supports_native_schema():
            return await self._invoke_structured(
                call_messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
                method=self.native_method,
            )

        return await self._invoke_with_repair_ladder(
            call_messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
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
            try:
                return await self._invoke_structured(
                    repair_messages,
                    schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    method=self.repair_method,
                )
            except ValueError as exc:
                last_error = exc
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
        result = await self._invoke_with_network_retries(
            lambda: self._call_langchain_structured(
                messages,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
                method=method,
            )
        )
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
            try:
                return await call()
            except Exception as exc:
                if not _is_transient_error(exc) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(0.25 * (2**attempt), 2.0))
        raise AssertionError("unreachable")


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


def _validate_schema_instance(value: Any, schema: type[BaseModel]) -> BaseModel:
    if isinstance(value, schema):
        return value
    return schema.model_validate(value)


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
