"""Network-free deterministic LLM client for tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from arbiter.llm.base import LLMClient


class MockLLMClient(LLMClient):
    """Return fixture responses keyed by ``call_label``."""

    def __init__(
        self,
        model: str = "mock",
        *,
        responses: Mapping[str, Any] | None = None,
        native_schema: bool = True,
        prompt_caching: bool = False,
        vision: bool = False,
    ) -> None:
        super().__init__(model)
        self.responses = dict(responses or {})
        self._native_schema = native_schema
        self._prompt_caching = prompt_caching
        self._vision = vision
        self.calls: list[str | None] = []

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        call_label: str | None = None,
    ) -> BaseModel:
        self.calls.append(call_label)
        if call_label is None:
            raise KeyError("MockLLMClient requires call_label-keyed fixtures.")
        if call_label not in self.responses:
            raise KeyError(f"No mock LLM fixture for call_label {call_label!r}.")

        value = self.responses[call_label]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, schema):
            return value
        return schema.model_validate(value)

    def supports_prompt_caching(self) -> bool:
        return self._prompt_caching

    def supports_native_schema(self) -> bool:
        return self._native_schema

    def supports_vision(self) -> bool:
        return self._vision
