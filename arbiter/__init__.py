"""ARBITER public Python API."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from .config import AssessmentConfig
from .graph.builder import build_outcome_graph, build_trial_graph
from .graph.state import AssessmentRuntime, TrialContext, base_ingestion_state
from .models import Assessment, DomainJudgment, OutcomeComparison, SourcesManifest

PIPELINE_VERSION = "0.1.0"


def ingest_trial(*args, **kwargs):
    """Ingest a trial.

    Implemented in the ingestion requirements after the project setup slice.
    """
    raise NotImplementedError("ingest_trial is implemented by REQ-02 through REQ-05")


async def assess_trial(ctx: TrialContext, config: AssessmentConfig) -> list[Assessment]:
    """Assess one already-ingested, eligible trial across configured outcomes."""

    runtime = AssessmentRuntime(
        llm_client_sq=ctx.llm_client_sq,
        llm_client_aux=ctx.llm_client_aux,
        supplement_index=ctx.supplement_index,
        trace=ctx.trace,
    )
    base_state = base_ingestion_state(ctx, config)
    trial_graph = build_trial_graph()
    trial_result = await trial_graph.ainvoke(
        {
            **base_state,
            "domain_contexts": {},
            "sq_answers": {},
            "domain_judgments": [],
            "errors": [],
            "consort": None,
        },
        context=runtime,
    )
    trial_judgments = _sort_domain_judgments(trial_result.get("domain_judgments", []))
    if len(trial_judgments) != 1 or trial_judgments[0].domain != "D1":
        raise ValueError("Trial graph must produce exactly one D1 judgment")

    outcome_graph = build_outcome_graph()
    outcomes = list(config.outcomes or ctx.trial_metadata.all_outcomes or [ctx.trial_metadata.primary_outcome])
    if not outcomes:
        outcomes = [ctx.trial_metadata.primary_outcome]

    assessments: list[Assessment] = []
    for outcome in outcomes:
        outcome_result = await outcome_graph.ainvoke(
            {
                **base_state,
                "outcome": outcome,
                "trial_domain_judgments": trial_judgments,
                "domain_contexts": {},
                "sq_answers": {},
                "domain_judgments": [],
                "overall_judgment": None,
                "overall_rationale": None,
                "requires_human_review": None,
                "errors": [],
            },
            context=runtime,
        )
        outcome_judgments = _sort_domain_judgments(outcome_result.get("domain_judgments", []))
        all_judgments = _sort_domain_judgments([*trial_judgments, *outcome_judgments])
        assessments.append(
            Assessment(
                assessment_id=str(uuid4()),
                created_at=datetime.now(UTC).isoformat(),
                pipeline_version=PIPELINE_VERSION,
                model_sq=ctx.llm_client_sq.model,
                model_aux=ctx.llm_client_aux.model,
                model_vision=None,
                trial_id=ctx.trial_metadata.trial_id,
                nct_number=ctx.trial_metadata.nct_number,
                outcome=outcome,
                requires_human_review=bool(outcome_result["requires_human_review"]),
                config_summary=dict(ctx.config_summary),
                trial_metadata=ctx.trial_metadata,
                ct_gov_data=ctx.ct_gov_data,
                outcome_comparison=_outcome_comparison(outcome_result),
                domain_judgments=all_judgments,
                overall_judgment=outcome_result["overall_judgment"],
                overall_rationale=str(outcome_result["overall_rationale"]),
                sources_manifest=SourcesManifest(
                    main_paper=str(config.paper_path),
                    supplements=[str(path) for path in config.supplement_paths],
                    ct_gov_retrieved=ctx.ct_gov_data is not None,
                    parsing_quality=ctx.section_map.parsing_quality,
                ),
                errors=[*trial_result.get("errors", []), *outcome_result.get("errors", [])],
            )
        )
    return assessments


def _sort_domain_judgments(judgments: list[DomainJudgment]) -> list[DomainJudgment]:
    return sorted(judgments, key=lambda item: item.domain)


def _outcome_comparison(state: dict) -> OutcomeComparison | None:
    fields = {
        "registered_outcome": state.get("registered_outcome"),
        "published_outcome": state.get("published_outcome"),
        "outcome_similarity_score": state.get("outcome_similarity_score"),
        "outcome_change_detected": state.get("outcome_change_detected"),
        "registered_as_primary": state.get("registered_as_primary"),
    }
    if all(value is None for value in fields.values()):
        return None
    return OutcomeComparison.model_validate(fields)


__all__ = ["AssessmentConfig", "TrialContext", "assess_trial", "ingest_trial"]
