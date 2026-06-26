from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from arbiter.config import AssessmentConfig, EnvSettings
from arbiter.llm.base import LLMAuthenticationError, LLMRequestTimeoutError, LangChainLLMClient, strip_cache_control
from arbiter.llm.factory import create_llm_client
from arbiter.llm.mock_client import MockLLMClient
from arbiter.llm.openai_client import OpenAILLMClient
from arbiter.llm.openrouter_client import OpenRouterLLMClient
from arbiter.observability.qa_trace import QATraceBundle
from arbiter.observability.trace import RunTrace


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


class BlockingLangChainClient(FakeLangChainClient):
    def __init__(self) -> None:
        super().__init__(
            native_schema=True,
            results=[{"parsed": {"answer": "Y"}, "raw": {"id": "raw-call"}, "parsing_error": None}],
        )
        self.provider_entered = asyncio.Event()
        self.release_provider = asyncio.Event()

    async def _call_langchain_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> Any:
        self.provider_entered.set()
        await self.release_provider.wait()
        return await super()._call_langchain_structured(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            method=method,
        )


class HungLangChainClient(FakeLangChainClient):
    def __init__(self, *, settings: EnvSettings) -> None:
        super().__init__(native_schema=True, results=[], settings=settings)
        self.provider_entered = asyncio.Event()

    async def _call_langchain_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
        method: str,
    ) -> Any:
        self.provider_entered.set()
        await asyncio.Event().wait()


class RateLimitError(Exception):
    pass


class TooManyRequestsResponseError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ProviderResponse:
    status_code = 429
    text = '{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}'
    headers = {"x-request-id": "req_123", "authorization": "Bearer secret"}

    def json(self) -> dict[str, Any]:
        return {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}}


class GenericProviderError(Exception):
    status_code = 429
    response = ProviderResponse()
    request_id = "req_attr_456"


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
async def test_full_qa_trace_writes_llm_started_before_provider_returns(tmp_path) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)
    client = BlockingLangChainClient()
    client.trace = trace

    task = asyncio.create_task(
        client.complete_structured([{"role": "user", "content": "prompt"}], ToyResponse, call_label="1.1|assignment")
    )
    await client.provider_entered.wait()

    events = [json.loads(line) for line in bundle.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == ["llm.started", "llm.network_attempt.started"]
    assert events[0]["trial_id"] == "T1"
    assert events[0]["domain"] == "D1"
    assert events[0]["sq_id"] == "1.1"
    assert events[0]["payload"]["model"] == "fake"
    assert events[1]["parent_event_id"] == events[0]["event_id"]
    assert events[1]["payload"]["attempt"] == 1

    client.release_provider.set()
    assert await task == ToyResponse(answer="Y")
    bundle.close()


@pytest.mark.asyncio
async def test_non_native_structured_output_repairs_after_parse_error() -> None:
    trace = TraceRecorder()
    client = FakeLangChainClient(
        native_schema=False,
        results=[
            {"parsed": None, "raw": {"content": "not json"}, "parsing_error": ValueError("missing answer")},
            {
                "parsed": {"answer": "PY", "quote": "reported centrally"},
                "raw": {"content": '{"answer":"PY","quote":"reported centrally"}'},
                "parsing_error": None,
            },
        ],
    )
    client.trace = trace

    response = await client.complete_structured([], ToyResponse, call_label="1.2|assignment")

    assert response == ToyResponse(answer="PY", quote="reported centrally")
    assert client.methods == ["json_mode", "json_mode"]
    attempts = trace.calls[0]["repair_attempts"]
    assert attempts[0]["validated"] is False
    assert attempts[0]["raw_response"]["raw"] == {"content": "not json"}
    assert attempts[0]["validation_result"] == {
        "schema": "ToyResponse",
        "validated": False,
        "error": "missing answer",
    }
    assert "Validation/parsing error:\nmissing answer" in attempts[1]["repair_prompt"]
    assert attempts[1]["validated"] is True
    assert attempts[1]["parsed_response"] == ToyResponse(answer="PY", quote="reported centrally")


@pytest.mark.asyncio
async def test_non_native_structured_output_raises_after_bounded_retries() -> None:
    settings = EnvSettings()
    settings.schema_repair_max_retries = 1
    trace = TraceRecorder()
    client = FakeLangChainClient(
        native_schema=False,
        settings=settings,
        results=[
            {"parsed": None, "raw": {"content": "bad"}, "parsing_error": ValueError("bad json")},
            {"parsed": None, "raw": {"content": "still bad"}, "parsing_error": ValueError("still bad")},
        ],
    )
    client.trace = trace

    with pytest.raises(ValueError, match="failed to produce valid ToyResponse after 2 schema attempts"):
        await client.complete_structured([], ToyResponse, call_label="1.3|assignment")

    attempts = trace.calls[0]["repair_attempts"]
    assert [attempt["validated"] for attempt in attempts] == [False, False]
    assert attempts[0]["raw_response"]["raw"] == {"content": "bad"}
    assert attempts[1]["raw_response"]["raw"] == {"content": "still bad"}
    assert trace.calls[0]["validation_result"]["validated"] is False


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
async def test_network_too_many_requests_error_retries_then_succeeds(monkeypatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("arbiter.llm.base.asyncio.sleep", no_sleep)
    settings = EnvSettings()
    settings.network_max_retries = 2
    client = FakeLangChainClient(
        native_schema=True,
        settings=settings,
        results=[
            TooManyRequestsResponseError("Provider returned error"),
            {"parsed": {"answer": "Y"}, "raw": object(), "parsing_error": None},
        ],
    )

    response = await client.complete_structured([], ToyResponse, call_label="metadata")

    assert response == ToyResponse(answer="Y")
    assert client.methods == ["json_schema", "json_schema"]
    assert client._last_network_attempts == 2
    assert client._last_transient_errors == ["TooManyRequestsResponseError: Provider returned error"]


@pytest.mark.asyncio
async def test_provider_call_timeout_records_failed_full_trace(tmp_path) -> None:
    settings = EnvSettings()
    settings.network_max_retries = 1
    settings.llm_request_timeout_s = 0.01
    config = AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full")
    config.env = settings
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=config,
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)
    client = HungLangChainClient(settings=settings)
    client.trace = trace

    with pytest.raises(LLMRequestTimeoutError, match="timed out after 0.01 seconds"):
        await client.complete_structured(
            [{"role": "user", "content": "prompt"}],
            ToyResponse,
            call_label="supplement_annotation|WINDOW_3",
        )
    bundle.close()

    events = [json.loads(line) for line in bundle.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "llm.started",
        "llm.network_attempt.started",
        "llm.network_attempt.failed",
        "llm.failed",
    ]
    assert events[1]["parent_event_id"] == events[0]["event_id"]
    assert events[2]["parent_event_id"] == events[0]["event_id"]
    assert events[2]["payload"]["attempt"] == 1
    assert events[2]["payload"]["retrying"] is False
    assert events[3]["parent_event_id"] == events[0]["event_id"]
    artifact = json.loads((bundle.root / "llm_calls" / "llm_000001.json").read_text(encoding="utf-8"))
    assert artifact["error"] == "fake timed out after 0.01 seconds"
    assert artifact["provider_error"] == {
        "error_type": "LLMRequestTimeoutError",
        "message": "fake timed out after 0.01 seconds",
        "status_code": None,
        "retryable": True,
        "request_id": None,
        "headers": {},
        "response_body": None,
    }


@pytest.mark.asyncio
async def test_full_qa_trace_records_each_network_retry_attempt(monkeypatch, tmp_path) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("arbiter.llm.base.asyncio.sleep", no_sleep)
    monkeypatch.setattr("arbiter.llm.base.random.uniform", lambda *_args: 0.0)
    settings = EnvSettings()
    settings.network_max_retries = 3
    config = AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full")
    config.env = settings
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=config,
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)
    client = FakeLangChainClient(
        native_schema=True,
        settings=settings,
        results=[
            RateLimitError("429 rate limit"),
            TooManyRequestsResponseError("Provider returned error"),
            {"parsed": {"answer": "Y"}, "raw": object(), "parsing_error": None},
        ],
    )
    client.trace = trace

    response = await client.complete_structured(
        [{"role": "user", "content": "prompt"}],
        ToyResponse,
        call_label="1.1|assignment",
    )
    bundle.close()

    assert response == ToyResponse(answer="Y")
    events = [json.loads(line) for line in bundle.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "llm.started",
        "llm.network_attempt.started",
        "llm.network_attempt.failed",
        "llm.network_attempt.started",
        "llm.network_attempt.failed",
        "llm.network_attempt.started",
        "llm.completed",
    ]
    start_event = events[0]
    attempt_events = events[1:6]
    assert {event["parent_event_id"] for event in attempt_events} == {start_event["event_id"]}
    assert [event["payload"]["attempt"] for event in attempt_events if event["status"] == "started"] == [1, 2, 3]
    failed_attempts = [event for event in attempt_events if event["status"] == "failed"]
    assert [event["payload"]["attempt"] for event in failed_attempts] == [1, 2]
    assert all(event["payload"]["elapsed_s"] >= 0 for event in failed_attempts)
    assert failed_attempts[0]["payload"]["transient_error"] == "RateLimitError: 429 rate limit"
    assert failed_attempts[1]["payload"]["transient_error"] == (
        "TooManyRequestsResponseError: Provider returned error"
    )
    assert failed_attempts[0]["payload"]["retrying"] is True
    assert failed_attempts[1]["payload"]["retrying"] is True
    assert events[-1]["payload"]["call_id"] == start_event["payload"]["call_id"]


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
async def test_openrouter_fast_403_aborts_without_retry(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            403,
            request=request,
            json={"error": {"message": "Key limit exceeded (total limit)"}},
            headers={"x-request-id": "or_req_123"},
        )

    settings = EnvSettings()
    settings.openrouter_api_key = "test-key"
    settings.network_max_retries = 3
    settings.llm_request_timeout_s = 10
    client = OpenRouterLLMClient(
        "gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        supports_cache=False,
        supports_schema=True,
        supports_vision=False,
        settings=settings,
    )
    monkeypatch.setattr("arbiter.llm.openrouter_client._make_transport", lambda: httpx.MockTransport(handler))

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        await client.complete_structured(
            [{"role": "user", "content": "Return JSON."}],
            ToyResponse,
            call_label="supplement_annotation|WINDOW_3",
        )

    assert len(requests) == 1
    assert client._last_network_attempts == 1
    assert client._last_transient_errors == []
    assert client._last_provider_error is not None
    assert client._last_provider_error["status_code"] == 403
    assert client._last_provider_error["retryable"] is False
    assert client._last_provider_error["response_body"] == {"error": {"message": "Key limit exceeded (total limit)"}}


@pytest.mark.asyncio
async def test_openrouter_direct_post_returns_validated_structured_output(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"answer":"Y","quote":"central randomisation"}'}}]},
        )

    settings = EnvSettings()
    settings.openrouter_api_key = "test-key"
    client = OpenRouterLLMClient(
        "gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        supports_cache=False,
        supports_schema=True,
        supports_vision=False,
        settings=settings,
    )
    monkeypatch.setattr("arbiter.llm.openrouter_client._make_transport", lambda: httpx.MockTransport(handler))

    response = await client.complete_structured(
        [{"role": "user", "content": "Return JSON."}],
        ToyResponse,
        call_label="1.1|assignment",
    )

    assert response == ToyResponse(answer="Y", quote="central randomisation")
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["stream"] is False
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["type"] == "json_schema"
    assert requests[0].headers["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_full_qa_trace_records_actionable_provider_error_summary(tmp_path) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=[],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)
    client = FakeLangChainClient(
        native_schema=True,
        results=[GenericProviderError("Provider returned error")],
    )
    client.trace = trace

    with pytest.raises(GenericProviderError, match="Provider returned error"):
        await client.complete_structured(
            [{"role": "user", "content": "prompt"}],
            ToyResponse,
            call_label="supplement_annotation|WINDOW_3",
        )
    bundle.close()

    artifact = json.loads((bundle.root / "llm_calls" / "llm_000001.json").read_text(encoding="utf-8"))
    provider_error = artifact["provider_error"]
    assert artifact["error"] == "Provider returned error"
    assert provider_error["error_type"] == "GenericProviderError"
    assert provider_error["message"] == "Provider returned error"
    assert provider_error["status_code"] == 429
    assert provider_error["retryable"] is True
    assert provider_error["request_id"] == "req_attr_456"
    assert provider_error["response_body"] == {
        "error": {"message": "rate limit exceeded", "type": "rate_limit_error"}
    }
    assert provider_error["headers"] == {"x-request-id": "req_123"}


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
