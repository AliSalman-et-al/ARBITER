"""LangGraph builders for ARBITER's trial and outcome assessment tiers."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from arbiter.arbiter_algorithm.branching import DOMAIN_SQS, get_applicable_sqs, get_na_sqs
from arbiter.arbiter_algorithm.decision_tables import (
    judge_domain_1,
    judge_domain_2,
    judge_domain_3,
    judge_domain_4,
    judge_domain_5,
)
from arbiter.arbiter_algorithm.rollup import compute_overall_judgment
from arbiter.graph.nodes.context_assembly import context_assembly_node_factory
from arbiter.graph.nodes.pre_d5 import pre_d5_node
from arbiter.graph.nodes.sq_node import sq_node
from arbiter.graph.state import AssessmentRuntime, OutcomeState, TrialState
from arbiter.models import AnswerCode, ConfidenceSignals, DomainContext, DomainJudgment, SQAnswer

NodeResult = Mapping[str, Any]
AsyncNode = Callable[[Mapping[str, Any], Runtime[AssessmentRuntime]], Awaitable[NodeResult]]
SyncNode = Callable[[Mapping[str, Any], Runtime[AssessmentRuntime]], NodeResult]


def build_trial_graph():
    """Build the trial-tier graph: D1 context, SQ fixpoint, judgment."""

    builder = StateGraph(cast(Any, TrialState), context_schema=AssessmentRuntime)
    _add_domain_nodes(builder, "D1", tier="trial")
    builder.add_edge(START, "context_D1")
    builder.add_edge("context_D1", "resolve_D1")
    builder.add_edge("sq_worker_D1", "fanin_D1")
    builder.add_conditional_edges("resolve_D1", _route_domain("D1", "sq_worker_D1", "fanin_D1"))
    builder.add_conditional_edges("fanin_D1", _after_fanin("D1", "resolve_D1", "judgment_D1"))
    builder.add_edge("judgment_D1", END)
    return builder.compile()


def build_outcome_graph():
    """Build the outcome-tier graph: D2-D5 in parallel, then overall rollup."""

    builder = StateGraph(cast(Any, OutcomeState), context_schema=AssessmentRuntime)
    for domain in ("D2", "D3", "D4", "D5"):
        _add_domain_nodes(builder, domain, tier="outcome")

    builder.add_node("pre_d5", cast(Any, _wrap_sync("outcome", "pre_d5", lambda state, runtime: pre_d5_node(state))))
    builder.add_node("overall_judgment", cast(Any, _wrap_sync("outcome", "overall_judgment", _overall_judgment_node)))

    builder.add_edge(START, "context_D2")
    builder.add_edge(START, "context_D3")
    builder.add_edge(START, "context_D4")
    builder.add_edge(START, "pre_d5")
    builder.add_edge("pre_d5", "context_D5")

    for domain in ("D2", "D3", "D4", "D5"):
        builder.add_edge(f"context_{domain}", f"resolve_{domain}")
        builder.add_edge(f"sq_worker_{domain}", f"fanin_{domain}")
        builder.add_conditional_edges(
            f"resolve_{domain}",
            _route_domain(domain, f"sq_worker_{domain}", f"fanin_{domain}"),
        )
        builder.add_conditional_edges(
            f"fanin_{domain}",
            _after_fanin(domain, f"resolve_{domain}", f"judgment_{domain}"),
        )
    builder.add_edge(["judgment_D2", "judgment_D3", "judgment_D4", "judgment_D5"], "overall_judgment")

    builder.add_edge("overall_judgment", END)
    return builder.compile()


def _add_domain_nodes(builder: Any, domain: str, *, tier: str) -> None:
    context_node = context_assembly_node_factory(domain)

    def context_adapter(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
        result = context_node(_state_with_runtime_handles(state, runtime))
        context = result["domain_context"]
        return {"domain_contexts": {domain: context}}

    builder.add_node(f"context_{domain}", cast(Any, _wrap_sync(tier, f"context_{domain}", context_adapter)))
    builder.add_node(f"resolve_{domain}", cast(Any, _wrap_sync(tier, f"resolve_{domain}", _resolve_node(domain))))
    builder.add_node(f"sq_worker_{domain}", cast(Any, _wrap_async(tier, f"sq_worker_{domain}", _sq_worker_adapter)))
    builder.add_node(f"fanin_{domain}", cast(Any, _wrap_sync(tier, f"fanin_{domain}", _fanin_node(domain))))
    builder.add_node(f"judgment_{domain}", cast(Any, _wrap_sync(tier, f"judgment_{domain}", _judgment_node(domain, tier))))


def _resolve_node(domain: str) -> SyncNode:
    def resolve(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
        answers = _domain_answers(state, domain)
        na_answers = {
            sq_id: _na_answer(sq_id)
            for sq_id in get_na_sqs(domain, _effect(state), answers)
            if sq_id not in answers
        }
        _record_branching_trace(state, runtime, domain, answers, na_answers)
        return {"sq_answers": na_answers} if na_answers else {}

    return resolve


def _route_domain(domain: str, worker_node: str, fanin_node: str) -> Callable[[Mapping[str, Any]], list[Send] | str]:
    def route(state: Mapping[str, Any]) -> list[Send] | str:
        answers = _domain_answers(state, domain)
        applicable = get_applicable_sqs(domain, _effect(state), answers)
        if not applicable:
            return fanin_node
        context = _domain_context(state, domain)
        return [
            Send(
                worker_node,
                {
                    **state,
                    "sq_id": sq_id,
                    "domain_context": context,
                },
            )
            for sq_id in applicable
        ]

    return route


def _after_fanin(domain: str, resolve_node: str, judgment_node: str) -> Callable[[Mapping[str, Any]], str]:
    def route(state: Mapping[str, Any]) -> str:
        answers = _domain_answers(state, domain)
        missing_na = [sq_id for sq_id in get_na_sqs(domain, _effect(state), answers) if sq_id not in answers]
        if get_applicable_sqs(domain, _effect(state), answers) or missing_na:
            return resolve_node
        return judgment_node

    return route


async def _sq_worker_adapter(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
    return await sq_node(_state_with_runtime_handles(state, runtime))


def _fanin_node(domain: str) -> SyncNode:
    def fanin(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
        answers = _domain_answers(state, domain)
        missing = [sq_id for sq_id in DOMAIN_SQS[domain] if sq_id not in answers]
        if not missing:
            return {}
        return {"errors": [f"{domain} missing expected SQ answer(s): {', '.join(missing)}"]}

    return fanin


def _judgment_node(domain: str, tier: str) -> SyncNode:
    def judgment_node(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
        answers = _domain_answers(state, domain)
        judgment, rationale = _judge_domain(domain, answers, _effect(state))
        _record_domain_judgment_trace(state, runtime, domain, answers, judgment, rationale)
        return {
            "domain_judgments": [
                DomainJudgment(
                    domain=domain,
                    scope="trial" if tier == "trial" else "outcome",
                    judgment=judgment,
                    algorithm_rationale=rationale,
                    sq_answers=[answers[sq_id] for sq_id in DOMAIN_SQS[domain] if sq_id in answers],
                )
            ]
        }

    return judgment_node


def _overall_judgment_node(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
    judgments = [
        *_sort_domain_judgments(cast(list[DomainJudgment], state.get("trial_domain_judgments", []))),
        *_sort_domain_judgments(cast(list[DomainJudgment], state.get("domain_judgments", []))),
    ]
    overall, rationale, requires_review = compute_overall_judgment(_sort_domain_judgments(judgments))
    _record_overall_judgment_trace(state, runtime, judgments, overall, rationale, requires_review)
    return {
        "overall_judgment": overall,
        "overall_rationale": rationale,
        "requires_human_review": requires_review,
    }


def _judge_domain(domain: str, answers: Mapping[str, SQAnswer], effect: str):
    if domain == "D1":
        return judge_domain_1(answers)
    if domain == "D2":
        return judge_domain_2(answers, effect)
    if domain == "D3":
        return judge_domain_3(answers)
    if domain == "D4":
        return judge_domain_4(answers)
    if domain == "D5":
        return judge_domain_5(answers)
    raise ValueError(f"Unknown domain: {domain}")


def _domain_answers(state: Mapping[str, Any], domain: str) -> dict[str, SQAnswer]:
    answers = state.get("sq_answers", {})
    if not isinstance(answers, Mapping):
        return {}
    return {
        sq_id: value if isinstance(value, SQAnswer) else SQAnswer.model_validate(value)
        for sq_id, value in answers.items()
        if str(sq_id).startswith(domain[1:] + ".")
    }


def _domain_context(state: Mapping[str, Any], domain: str) -> DomainContext:
    contexts = state.get("domain_contexts", {})
    if not isinstance(contexts, Mapping) or domain not in contexts:
        raise KeyError(f"Missing DomainContext for {domain}")
    value = contexts[domain]
    return value if isinstance(value, DomainContext) else DomainContext.model_validate(value)


def _effect(state: Mapping[str, Any]) -> str:
    effect = state.get("effect_of_interest", "assignment")
    return str(getattr(effect, "value", effect))


def _na_answer(sq_id: str) -> SQAnswer:
    return SQAnswer(
        sq_id=sq_id,
        answer=AnswerCode.NA,
        quote="",
        page=None,
        justification="Structurally not applicable under the RoB 2 branching rules.",
        confidence=ConfidenceSignals(),
    )


def _state_with_runtime_handles(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> dict[str, Any]:
    context = runtime.context
    return {
        **state,
        "sq_model": _runtime_value(context, "llm_client_sq"),
        "aux_model": _runtime_value(context, "llm_client_aux"),
        "supplement_index": _runtime_value(context, "supplement_index"),
        "trace": _runtime_value(context, "trace"),
    }


def _runtime_value(context: AssessmentRuntime | Mapping[str, Any], key: str) -> Any:
    if isinstance(context, Mapping):
        return cast(Mapping[str, Any], context)[key]
    return getattr(context, key)


def _wrap_sync(tier: str, name: str, node: SyncNode) -> SyncNode:
    def wrapped(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> NodeResult:
        trace = _trace(runtime)
        outcome = str(state.get("outcome")) if tier == "outcome" and state.get("outcome") is not None else None
        if trace is None:
            return node(state, runtime)
        with _trace_span(trace, tier=tier, name=name, outcome=outcome):
            return node(state, runtime)

    return wrapped


def _wrap_async(tier: str, name: str, node: AsyncNode) -> AsyncNode:
    async def wrapped(state: Mapping[str, Any], runtime: Runtime[AssessmentRuntime]) -> NodeResult:
        trace = _trace(runtime)
        outcome = str(state.get("outcome")) if tier == "outcome" and state.get("outcome") is not None else None
        if trace is None:
            return await node(state, runtime)
        with _trace_span(trace, tier=tier, name=name, outcome=outcome):
            return await node(state, runtime)

    return wrapped


def _trace(runtime: Runtime[AssessmentRuntime]) -> Any | None:
    context = runtime.context
    if isinstance(context, Mapping):
        return context.get("trace")
    return getattr(context, "trace", None)


class _trace_span:
    def __init__(self, trace: Any, *, tier: str, name: str, outcome: str | None) -> None:
        self.trace = trace
        self.tier = tier
        self.name = name
        self.outcome = outcome
        self.started = 0.0
        self.delegate: Any | None = None

    def __enter__(self):
        if hasattr(self.trace, "node_span"):
            self.delegate = self.trace.node_span(tier=self.tier, node=self.name, outcome=self.outcome)
            return self.delegate.__enter__()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.delegate is not None:
            return bool(self.delegate.__exit__(exc_type, exc, tb))
        if hasattr(self.trace, "record_node_span"):
            self.trace.record_node_span(
                tier=self.tier,
                node=self.name,
                outcome=self.outcome,
                duration_s=time.perf_counter() - self.started,
                error=str(exc) if exc is not None else None,
            )
        return False


def _sort_domain_judgments(judgments: list[DomainJudgment]) -> list[DomainJudgment]:
    coerced = [item if isinstance(item, DomainJudgment) else DomainJudgment.model_validate(item) for item in judgments]
    return sorted(coerced, key=lambda item: item.domain)


def _record_branching_trace(
    state: Mapping[str, Any],
    runtime: Runtime[AssessmentRuntime],
    domain: str,
    answers: Mapping[str, SQAnswer],
    na_answers: Mapping[str, SQAnswer],
) -> None:
    qa_trace = _qa_trace(runtime)
    if qa_trace is None:
        return
    asked_sqs = [sq_id for sq_id in DOMAIN_SQS[domain] if sq_id in answers and answers[sq_id].answer != AnswerCode.NA]
    for sq_id, answer in na_answers.items():
        qa_trace.record_event(
            event_type="branching.resolved",
            status="completed",
            trial_id=_trial_id(state),
            outcome=_outcome(state),
            domain=domain,
            sq_id=sq_id,
            payload={
                "effect_of_interest": _effect(state),
                "asked_sqs": asked_sqs,
                "structurally_na": True,
                "answer": answer.answer.value,
                "basis": answer.justification,
            },
        )


def _record_domain_judgment_trace(
    state: Mapping[str, Any],
    runtime: Runtime[AssessmentRuntime],
    domain: str,
    answers: Mapping[str, SQAnswer],
    judgment: Any,
    rationale: str,
) -> None:
    qa_trace = _qa_trace(runtime)
    if qa_trace is None:
        return
    qa_trace.record_event(
        event_type="judgment.domain.completed",
        status="completed",
        trial_id=_trial_id(state),
        outcome=_outcome(state),
        domain=domain,
        payload={
            "input_sq_answers": {sq_id: answers[sq_id].answer.value for sq_id in DOMAIN_SQS[domain] if sq_id in answers},
            "output_judgment": getattr(judgment, "value", judgment),
            "algorithm_rationale": rationale,
        },
    )


def _record_overall_judgment_trace(
    state: Mapping[str, Any],
    runtime: Runtime[AssessmentRuntime],
    judgments: list[DomainJudgment],
    overall: Any,
    rationale: str,
    requires_review: bool,
) -> None:
    qa_trace = _qa_trace(runtime)
    if qa_trace is None:
        return
    sorted_judgments = _sort_domain_judgments(judgments)
    qa_trace.record_event(
        event_type="judgment.overall.completed",
        status="completed",
        trial_id=_trial_id(state),
        outcome=_outcome(state),
        payload={
            "domain_judgments": {item.domain: item.judgment.value for item in sorted_judgments},
            "rollup_policy": "ADR-0001",
            "output_judgment": getattr(overall, "value", overall),
            "algorithm_rationale": rationale,
            "requires_human_review": requires_review,
            "requires_human_review_basis": rationale if requires_review else None,
        },
    )


def _qa_trace(runtime: Runtime[AssessmentRuntime]) -> Any | None:
    trace = _trace(runtime)
    return getattr(trace, "qa_trace", None) if trace is not None else None


def _trial_id(state: Mapping[str, Any]) -> str | None:
    metadata = state.get("trial_metadata")
    trial_id = getattr(metadata, "trial_id", None)
    return str(trial_id) if trial_id is not None else None


def _outcome(state: Mapping[str, Any]) -> str | None:
    return str(state["outcome"]) if state.get("outcome") is not None else None
