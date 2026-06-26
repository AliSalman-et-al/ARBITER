from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from arbiter.config import EnvSettings
from arbiter.llm.base import LLMAuthenticationError, LangChainLLMClient, strip_cache_control
from arbiter.llm.factory import create_llm_client
from arbiter.llm.mock_client import MockLLMClient
from arbiter.llm.openai_client import OpenAILLMClient
from arbiter.llm.openrouter_client import OpenRouterLLMClient


class ToyResponse(BaseModel):
    answer: str
    quote: str = ""


class TraceRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeLangChainClient(LangChainLLMClient):
    def __init__(
        self,
        *,
        results: list[Any],
        native_schema: bool,
        settings: EnvSettings | None = None,
    ) -> None:
        super().__init__(
            "fake",
            model_id="fake/provider-model",
            supports_cache=False,
            supports_schema=native_schema,
            supports_vision=False,
            settings=settings,
        )
        self.results = list(results)
        self.methods: list[str] = []

    def _make_chat_model(self, *, temperature: float, max_tokens: int) -> Any:
        raise AssertionError("FakeLangChainClient bypasses LangChain construction.")

    async def _call_langchain_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> Any:
        self.methods.append(method)
        if not self.results:
            raise AssertionError("No fake result left.")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


@pytest.mark.asyncio
async def test_native_structured_output_returns_validated_model() -> None:
    client = FakeLangChainClient(
        native_schema=True,
        results=[{"parsed": ToyResponse(answer="Y"), "raw": object(), "parsing_error": None}],
    )

    response = await client.complete_structured([], ToyResponse, call_label="1.1|assignment")

    assert response == ToyResponse(answer="Y")
    assert client.methods == ["json_schema"]


@pytest.mark.asyncio
async def test_langchain_trace_preserves_raw_response_and_parsed_result() -> None:
    trace = TraceRecorder()
    raw = {"id": "raw-call", "content": [{"text": '{"answer":"Y"}'}]}
    client = FakeLangChainClient(
        native_schema=True,
        results=[{"parsed": {"answer": "Y"}, "raw": raw, "parsing_error": None}],
    )
    client.trace = trace

    response = await client.complete_structured([{"role": "user", "content": "prompt"}], ToyResponse)

    assert response == ToyResponse(answer="Y")
    assert trace.calls[0]["raw_response"] == {"parsed": {"answer": "Y"}, "raw": raw, "parsing_error": None}
    assert trace.calls[0]["parsed_response"] == ToyResponse(answer="Y")
    assert trace.calls[0]["validation_result"] == {
        "schema": "ToyResponse",
        "validated": True,
        "error": None,
    }


@pytest.mark.asyncio
async def test_non_native_structured_output_repairs_after_parse_error() -> None:
    client = FakeLangChainClient(
        native_schema=False,
        results=[
            {"parsed": None, "raw": object(), "parsing_error": ValueError("missing answer")},
            {"parsed": {"answer": "PY", "quote": "reported centrally"}, "raw": object(), "parsing_error": None},
        ],
    )

    response = await client.complete_structured([], ToyResponse, call_label="1.2|assignment")

    assert response == ToyResponse(answer="PY", quote="reported centrally")
    assert client.methods == ["json_mode", "json_mode"]


@pytest.mark.asyncio
async def test_non_native_structured_output_raises_after_bounded_retries() -> None:
    settings = EnvSettings()
    settings.schema_repair_max_retries = 1
    client = FakeLangChainClient(
        native_schema=False,
        settings=settings,
        results=[
            {"parsed": None, "raw": object(), "parsing_error": ValueError("bad json")},
            {"parsed": None, "raw": object(), "parsing_error": ValueError("still bad")},
        ],
    )

    with pytest.raises(ValueError, match="failed to produce valid ToyResponse after 2 schema attempts"):
        await client.complete_structured([], ToyResponse, call_label="1.3|assignment")


@pytest.mark.asyncio
async def test_network_rate_limit_retries_then_succeeds(monkeypatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("arbiter.llm.base.asyncio.sleep", no_sleep)
    settings = EnvSettings()
    settings.network_max_retries = 3
    client = FakeLangChainClient(
        native_schema=True,
        settings=settings,
        results=[
            RateLimitError("429 rate limit"),
            RateLimitError("429 rate limit"),
            {"parsed": {"answer": "Y"}, "raw": object(), "parsing_error": None},
        ],
    )

    response = await client.complete_structured([], ToyResponse, call_label="1.1|assignment")

    assert response == ToyResponse(answer="Y")
    assert client.methods == ["json_schema", "json_schema", "json_schema"]
    assert client._last_network_attempts == 3
    assert len(client._last_transient_errors) == 2


@pytest.mark.asyncio
async def test_auth_error_aborts_without_retry() -> None:
    client = FakeLangChainClient(
        native_schema=True,
        results=[AuthenticationError("401 invalid api key"), {"parsed": {"answer": "Y"}, "raw": object()}],
    )

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        await client.complete_structured([], ToyResponse, call_label="1.1|assignment")

    assert client.methods == ["json_schema"]


@pytest.mark.asyncio
async def test_mock_client_uses_call_label_keyed_fixtures() -> None:
    client = MockLLMClient(responses={"metadata": {"answer": "N"}})

    assert await client.complete_structured([], ToyResponse, call_label="metadata") == ToyResponse(answer="N")
    assert client.calls == ["metadata"]

    with pytest.raises(KeyError):
        await client.complete_structured([], ToyResponse, call_label="1.1|assignment")


@pytest.mark.asyncio
async def test_complete_vision_is_v0_1_stub() -> None:
    client = MockLLMClient(responses={})

    with pytest.raises(NotImplementedError, match="v0.2"):
        await client.complete_vision(b"fake", "extract flow", ToyResponse)


def test_openai_strips_cache_control_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "cacheable prefix",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]

    assert strip_cache_control(messages) == [{"role": "user", "content": [{"type": "text", "text": "cacheable prefix"}]}]


def test_factory_dispatches_openrouter_gpt_oss_paid_slug_to_native_client() -> None:
    client = create_llm_client("gpt-oss-120b")

    assert isinstance(client, OpenRouterLLMClient)
    assert client.supports_native_schema() is True


def test_factory_dispatches_vanilla_openai_client() -> None:
    client = create_llm_client("gpt-5.5")

    assert isinstance(client, OpenAILLMClient)
    assert client.supports_native_schema() is True


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown LLM model"):
        create_llm_client("not-a-model")
