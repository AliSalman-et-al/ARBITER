from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from arbiter.config import AssessmentConfig
from arbiter.observability import RunTrace, estimate_call_cost
from arbiter.observability.qa_trace import QATraceBundle


def test_estimate_call_cost_distinguishes_unknown_from_free() -> None:
    paid = estimate_call_cost("gpt-oss-120b", input_tokens=1_000_000, output_tokens=1_000_000)
    assert paid == {"cost": 0.219, "pricing_unknown": False}

    free = estimate_call_cost("google/gemma-4-31b-it:free", input_tokens=1_000_000, output_tokens=1_000_000)
    assert free == {"cost": 0.0, "pricing_unknown": False}

    unknown = estimate_call_cost("not-in-registry", input_tokens=1)
    assert unknown == {"cost": None, "pricing_unknown": True}

    missing_cache_price = estimate_call_cost("gpt-oss-120b", input_tokens=1, cache_read_tokens=1)
    assert missing_cache_price == {"cost": None, "pricing_unknown": True}


def test_summary_trace_writes_counts_without_bodies_or_artifacts(tmp_path) -> None:
    trace = RunTrace(trace_level="summary", trial_id="T1")
    prefix_hash = trace.register_prefix("cacheable prefix")

    with trace.node_span(tier="trial", node="context_D1"):
        time.sleep(0)
    trace.record_llm_call(
        model="gpt-oss-120b",
        call_label="1.1|assignment",
        messages=[{"role": "user", "content": "dynamic only"}],
        input_tokens=10,
        output_tokens=5,
        latency_s=0.01,
        cache_hit=None,
    )

    path = trace.flush(tmp_path, artifacts={"section_map": {"not": "written"}})
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["prefixes"] == {prefix_hash: ""}
    assert payload["node_spans"][0]["tier"] == "trial"
    assert payload["llm_calls"][0]["input_tokens"] == 10
    assert payload["llm_calls"][0]["cache_hit"] is None
    assert "messages" not in payload["llm_calls"][0]
    assert not (tmp_path / "T1" / "artifacts").exists()


def test_full_trace_writes_bodies_and_artifacts(tmp_path) -> None:
    trace = RunTrace(trace_level="full", trial_id="T1")
    trace.record_llm_call(
        model="gpt-oss-120b",
        call_label="1.1|assignment",
        messages=[{"role": "user", "content": "prompt body"}],
        raw_response={"answer": "Y"},
    )

    path = trace.flush(tmp_path, artifacts={"trial_metadata": {"trial_id": "T1"}})
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["llm_calls"][0]["messages"][0]["content"] == "prompt body"
    artifact = tmp_path / "T1" / "artifacts" / "trial_metadata.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"trial_id": "T1"}


def test_full_trace_mirrors_node_and_llm_events_to_run_level_bundle(tmp_path: Path) -> None:
    bundle = QATraceBundle.create(
        base_dir=tmp_path / "runs",
        command="assess",
        cli_args=["assess", "--trace", "full"],
        config=AssessmentConfig(paper_path=tmp_path / "paper.pdf", trace_level="full"),
    )
    trace = RunTrace(trace_level="full", trial_id="T1", qa_trace=bundle)

    with trace.node_span(tier="outcome", node="sq_worker_D2", outcome="Overall survival"):
        trace.record_llm_call(
            model="gpt-oss-120b",
            call_label="2.1|assignment",
            messages=[{"role": "user", "content": "full prompt body"}],
            schema_name="SQRawAnswer",
            method="json_schema",
            input_tokens=10,
            output_tokens=5,
            latency_s=0.25,
            repair_attempts=[
                {
                    "attempt": 1,
                    "validated": False,
                    "error": "missing answer",
                    "request_messages": [{"role": "user", "content": "full prompt body"}],
                    "raw_response": {"content": "not json"},
                    "validation_result": {"schema": "SQRawAnswer", "validated": False, "error": "missing answer"},
                },
                {
                    "attempt": 2,
                    "validated": True,
                    "error": None,
                    "repair_prompt": "Return only corrected JSON.",
                    "request_messages": [{"role": "user", "content": "Return only corrected JSON."}],
                    "raw_response": {"answer": "Y", "quote": "full response body"},
                    "parsed_response": {"answer": "Y", "quote": "full response body"},
                    "validation_result": {"schema": "SQRawAnswer", "validated": True, "error": None},
                },
            ],
            network_attempts=1,
            cache_hit=False,
            raw_response={"answer": "Y", "quote": "full response body"},
        )
    bundle.close()

    events = [
        json.loads(line)
        for line in (bundle.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "node.started",
        "llm.started",
        "llm.repair_attempt.failed",
        "llm.repair_attempt.completed",
        "llm.completed",
        "node.completed",
    ]
    assert events[0]["outcome"] == "Overall survival"
    assert events[0]["domain"] == "D2"
    assert events[2]["parent_event_id"] == events[1]["event_id"]
    assert events[3]["parent_event_id"] == events[1]["event_id"]
    assert events[4]["artifact_refs"] == ["llm_calls/llm_000001.json"]

    llm_call = json.loads((bundle.root / "llm_calls" / "llm_000001.json").read_text(encoding="utf-8"))
    assert llm_call["call_id"] == "llm_000001"
    assert llm_call["trial_id"] == "T1"
    assert llm_call["outcome"] == "Overall survival"
    assert llm_call["domain"] == "D2"
    assert llm_call["sq_id"] == "2.1"
    assert llm_call["model"] == "gpt-oss-120b"
    assert llm_call["provider"] == "openrouter"
    assert llm_call["temperature"] is None
    assert llm_call["prompt"]["messages"][0]["content"] == "full prompt body"
    assert llm_call["raw_response_body"] == {"answer": "Y", "quote": "full response body"}
    assert llm_call["parsed_response"] == {"answer": "Y", "quote": "full response body"}
    assert llm_call["validation_result"] == {"schema": "SQRawAnswer", "validated": True, "error": None}
    assert llm_call["repair_attempt_count"] == 2
    assert llm_call["repair_attempts"][0]["raw_response"] == {"content": "not json"}
    assert llm_call["repair_attempts"][0]["validation_result"]["error"] == "missing answer"
    assert llm_call["repair_attempts"][1]["repair_prompt"] == "Return only corrected JSON."
    assert llm_call["final_result"] == {"answer": "Y", "quote": "full response body"}
    assert llm_call["token_cost_metadata"]["input_tokens"] == 10
    assert llm_call["token_cost_metadata"]["output_tokens"] == 5
    assert llm_call["token_cost_metadata"]["cost"] == pytest.approx(0.00000129)


def test_off_trace_is_noop(tmp_path) -> None:
    trace = RunTrace(trace_level="off", trial_id="T1")
    trace.record_node_span(tier="trial", node="context_D1", outcome=None, duration_s=1.0)
    trace.record_llm_call(model="gpt-oss-120b", call_label="x", messages=[], latency_s=1.0)

    assert trace.node_spans == []
    assert trace.call_records == []
    assert trace.flush(tmp_path) is None
    assert not (tmp_path / "T1").exists()
