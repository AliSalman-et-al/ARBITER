from __future__ import annotations

import json
import time

from arbiter.observability import RunTrace, estimate_call_cost


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


def test_off_trace_is_noop(tmp_path) -> None:
    trace = RunTrace(trace_level="off", trial_id="T1")
    trace.record_node_span(tier="trial", node="context_D1", outcome=None, duration_s=1.0)
    trace.record_llm_call(model="gpt-oss-120b", call_label="x", messages=[], latency_s=1.0)

    assert trace.node_spans == []
    assert trace.call_records == []
    assert trace.flush(tmp_path) is None
    assert not (tmp_path / "T1").exists()
