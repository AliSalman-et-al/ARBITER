"""Run trace collector for side-channel observability."""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from arbiter.config import TraceLevel
from arbiter.observability.cost import estimate_call_cost

TraceLevelValue = Literal["off", "summary", "full"]
_active_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("arbiter_trace_span", default=None)


@dataclass
class RunTrace:
    """Per-trial trace collector.

    The object is a runtime handle. It is intentionally not part of LangGraph
    state or persisted assessment records.
    """

    trace_level: TraceLevel = "full"
    trial_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    node_spans: list[dict[str, Any]] = field(default_factory=list)
    call_records: list[dict[str, Any]] = field(default_factory=list)
    prefixes: dict[str, str] = field(default_factory=dict)

    def enabled(self) -> bool:
        return self.trace_level != "off"

    def is_full(self) -> bool:
        return self.trace_level == "full"

    def register_prefix(self, text: str | None) -> str | None:
        if not self.enabled() or not text:
            return None
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if self.is_full():
            self.prefixes.setdefault(digest, text)
        else:
            self.prefixes.setdefault(digest, "")
        return digest

    def node_span(self, *, tier: str, node: str, outcome: str | None = None) -> AbstractContextManager[None]:
        return _NodeSpanContext(self, tier=tier, node=node, outcome=outcome)

    def record_node_span(
        self,
        *,
        tier: str,
        node: str,
        outcome: str | None,
        duration_s: float,
        error: str | None = None,
    ) -> None:
        if not self.enabled():
            return
        self.node_spans.append(
            {
                "span_id": f"span_{len(self.node_spans) + 1}",
                "tier": tier,
                "node": node,
                "outcome": outcome,
                "duration_s": duration_s,
                "error": error,
            }
        )

    def record_llm_call(
        self,
        *,
        model: str,
        call_label: str | None,
        messages: list[dict[str, Any]] | None,
        schema_name: str | None = None,
        method: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        latency_s: float = 0.0,
        repair_attempts: list[dict[str, Any]] | None = None,
        network_attempts: int | None = None,
        transient_errors: list[str] | None = None,
        error: str | None = None,
        cache_hit: bool | None = None,
        raw_response: Any | None = None,
    ) -> None:
        if not self.enabled():
            return

        prefix_hash = self._prefix_hash_from_messages(messages or [])
        cost = estimate_call_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        record = {
            "span_id": _active_span.get(),
            "model": model,
            "call_label": call_label,
            "schema": schema_name,
            "method": method,
            "prefix_hash": prefix_hash,
            "latency_s": latency_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cache_hit": cache_hit,
            "cost": cost["cost"],
            "pricing_unknown": cost["pricing_unknown"],
            "repair_attempt_count": len(repair_attempts or []),
            "network_attempts": network_attempts,
            "transient_errors": transient_errors or [],
            "error": error,
        }
        if self.is_full():
            record["messages"] = messages
            record["raw_response"] = _jsonable(raw_response)
            record["repair_attempts"] = repair_attempts or []
        self.call_records.append(record)

    def timing_summary(self) -> dict[str, Any]:
        total_wall = time.perf_counter() - self.started_at
        llm_latency = sum(float(call.get("latency_s") or 0.0) for call in self.call_records)
        node_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "total_s": 0.0, "max_s": 0.0, "errors": 0}
        )
        for span in self.node_spans:
            key = f"{span['tier']}:{span['node']}"
            stats = node_stats[key]
            duration = float(span["duration_s"])
            stats["calls"] += 1
            stats["total_s"] += duration
            stats["max_s"] = max(stats["max_s"], duration)
            stats["errors"] += 1 if span.get("error") else 0

        per_node = {
            key: {**value, "mean_s": value["total_s"] / value["calls"]}
            for key, value in sorted(node_stats.items())
        }
        known_costs = [call["cost"] for call in self.call_records if call.get("cost") is not None]
        unknown_cost = any(call.get("pricing_unknown") for call in self.call_records)
        return {
            "wall_time_s": total_wall,
            "llm_latency_s": llm_latency,
            "estimated_non_llm_time_s": max(total_wall - llm_latency, 0.0),
            "llm_call_count": len(self.call_records),
            "cache_read_token_count": _sum_known(self.call_records, "cache_read_tokens"),
            "cache_write_token_count": _sum_known(self.call_records, "cache_write_tokens"),
            "repair_attempt_count": sum(int(call.get("repair_attempt_count") or 0) for call in self.call_records),
            "total_cost": None if unknown_cost else sum(known_costs),
            "pricing_unknown": unknown_cost,
            "slowest_nodes": sorted(self.node_spans, key=lambda item: item["duration_s"], reverse=True)[:10],
            "per_node": per_node,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "trace_level": self.trace_level,
            "trial_id": self.trial_id,
            "timing_summary": self.timing_summary(),
            "prefixes": self.prefixes,
            "node_spans": self.node_spans,
            "llm_calls": self.call_records,
        }

    def flush(self, output_dir: Path, *, artifacts: dict[str, Any] | None = None) -> Path | None:
        if not self.enabled():
            return None
        trial_id = self.trial_id or "trial"
        trial_dir = output_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        path = trial_dir / "trace.json"
        path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
        if self.is_full() and artifacts:
            _write_artifacts(trial_dir, artifacts)
        return path

    def _prefix_hash_from_messages(self, messages: list[dict[str, Any]]) -> str | None:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("cache_control") is not None:
                        text = block.get("text")
                        if isinstance(text, str):
                            return self.register_prefix(text)
        return None


class _NodeSpanContext:
    def __init__(self, trace: RunTrace, *, tier: str, node: str, outcome: str | None) -> None:
        self.trace = trace
        self.tier = tier
        self.node = node
        self.outcome = outcome
        self.started = 0.0
        self.token: contextvars.Token[str | None] | None = None
        self.span_id = f"span_{len(trace.node_spans) + 1}"

    def __enter__(self) -> None:
        self.started = time.perf_counter()
        self.token = _active_span.set(self.span_id)
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.token is not None:
            _active_span.reset(self.token)
        self.trace.record_node_span(
            tier=self.tier,
            node=self.node,
            outcome=self.outcome,
            duration_s=time.perf_counter() - self.started,
            error=str(exc) if exc is not None else None,
        )
        return False


def _write_artifacts(trial_dir: Path, artifacts: dict[str, Any]) -> None:
    root = trial_dir / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sum_known(records: list[dict[str, Any]], key: str) -> int | None:
    values = [record.get(key) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(int(value) for value in values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.value
    return value
