"""ARBITER public Python API."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .config import AssessmentConfig
from .graph.builder import build_outcome_graph, build_trial_graph
from .graph.state import AssessmentRuntime, TrialContext, base_ingestion_state
from .graph.nodes.context_assembly import build_shared_prefix
from .ingestion.ctgov import fetch_ctgov
from .ingestion.metadata_extractor import extract_metadata
from .ingestion.paper import ingest_paper
from .ingestion.supplements import ingest_supplements
from .llm.factory import create_llm_client
from .models import Assessment, DomainJudgment, OutcomeComparison, SourcesManifest
from .observability import RunTrace
from .output.json_writer import write_assessment_json
from .output.report_writer import write_assessment_report
from .output.sqlite_writer import write_assessment_sqlite

PIPELINE_VERSION = "0.1.0"


async def ingest_trial(config: AssessmentConfig) -> TrialContext:
    """Run Phase 1 exactly once for a trial and return its reusable context."""

    trace = RunTrace(trace_level=config.trace_level, qa_trace=config.qa_trace)
    sq_client = create_llm_client(config.sq_model, trace=trace, settings=config.env)
    aux_client = create_llm_client(config.aux_model, trace=trace, settings=config.env)

    section_map, raw_char_stream = ingest_paper(config.paper_path)
    _record_main_paper_source(config.qa_trace, section_map, raw_char_stream)
    supplement_index = await ingest_supplements(config.supplement_paths, aux_client)
    _record_supplement_sources(config.qa_trace, supplement_index)
    nct_hint = config.nct_number or section_map.nct_number
    ct_gov_data = await fetch_ctgov(nct_hint) if nct_hint else None
    _record_ctgov_source(config.qa_trace, nct_hint, ct_gov_data)
    trial_metadata = await extract_metadata(section_map, config, aux_client, nct_hint=nct_hint)
    _record_metadata_source(config.qa_trace, trial_metadata)
    shared_prefix_text, ct_gov_block = build_shared_prefix(
        trial_metadata=trial_metadata,
        section_map=section_map,
        ctgov_record=ct_gov_data,
        settings=config.env,
    )

    trace.trial_id = trial_metadata.trial_id
    trace.register_prefix(shared_prefix_text)

    return TrialContext(
        config_summary=_config_summary(config, inputs_hash=_inputs_hash(config, raw_char_stream)),
        trial_metadata=trial_metadata,
        section_map=section_map,
        raw_char_stream=raw_char_stream,
        supplement_index=supplement_index,
        ct_gov_data=ct_gov_data,
        shared_prefix_text=shared_prefix_text,
        ct_gov_block=ct_gov_block,
        llm_client_sq=sq_client,
        llm_client_aux=aux_client,
        trace=trace,
    )


async def assess_trial(ctx: TrialContext, config: AssessmentConfig) -> list[Assessment]:
    """Assess one already-ingested, eligible trial across configured outcomes."""

    if ctx.trace is not None:
        if hasattr(ctx.trace, "trial_id"):
            ctx.trace.trial_id = ctx.trial_metadata.trial_id
        if hasattr(ctx.trace, "register_prefix"):
            ctx.trace.register_prefix(ctx.shared_prefix_text)
        ctx.llm_client_sq.trace = ctx.trace
        ctx.llm_client_aux.trace = ctx.trace

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
                pipeline_version=config.pipeline_version,
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
        json_path = write_assessment_json(assessments[-1], config.output_dir)
        _record_assessment_output(config.qa_trace, assessments[-1], json_path)
        write_assessment_sqlite(assessments[-1], config.db_path, json_path=json_path)
        if config.report_enabled:
            report_path = write_assessment_report(
                assessments[-1],
                config.output_dir,
                timing_summary=ctx.trace.timing_summary() if ctx.trace is not None else None,
            )
            _record_report_output(config.qa_trace, assessments[-1], report_path)
    if ctx.trace is not None and hasattr(ctx.trace, "flush"):
        ctx.trace.flush(
            config.output_dir,
            artifacts={
                "section_map": ctx.section_map,
                "ctgov": ctx.ct_gov_data,
                "trial_metadata": ctx.trial_metadata,
                "shared_prefix": {"text": ctx.shared_prefix_text, "ct_gov_block": ctx.ct_gov_block},
            },
        )
    return assessments


def _config_summary(config: AssessmentConfig, *, inputs_hash: str) -> dict:
    return {
        "paper_path": str(config.paper_path),
        "supplement_paths": [str(path) for path in config.supplement_paths],
        "nct_number": config.nct_number,
        "trial_label": config.trial_label,
        "outcomes": config.outcomes,
        "effect_of_interest": config.effect_of_interest,
        "sq_model": config.sq_model,
        "aux_model": config.aux_model,
        "vision_model": config.vision_model,
        "pipeline_version": config.pipeline_version,
        "inputs_hash": inputs_hash,
    }


def _inputs_hash(config: AssessmentConfig, raw_char_stream: str) -> str:
    digest = hashlib.sha256()
    digest.update(raw_char_stream.encode("utf-8", errors="replace"))
    for path in config.supplement_paths:
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            try:
                digest.update(path.read_bytes())
            except OSError:
                pass
    digest.update(str(config.nct_number or "").encode("utf-8"))
    digest.update(str(config.trial_label or "").encode("utf-8"))
    return digest.hexdigest()


def _record_main_paper_source(qa_trace, section_map, raw_char_stream: str) -> None:
    if qa_trace is None:
        return
    source_id = _source_artifact_id(section_map.source_path)
    qa_trace.write_source_artifact(
        f"sources/main_paper/{source_id}.json",
        {
            **section_map.model_dump(mode="json"),
            "raw_char_stream": raw_char_stream,
        },
        event_type="ingestion.main_paper.completed",
        event_payload={
            "source_path": section_map.source_path,
            "parsing_quality": section_map.parsing_quality,
            "section_count": len(section_map.sections),
            "page_box_count": len(section_map.page_boxes),
        },
    )


def _record_supplement_sources(qa_trace, supplement_index) -> None:
    if qa_trace is None:
        return
    segments = list(getattr(supplement_index, "segments", []) or [])
    source_files = sorted({str(segment.source_file) for segment in segments})
    source_id = _source_artifact_id("|".join(source_files) or "no-supplements")
    artifact_ref = qa_trace.write_source_artifact(
        f"sources/supplements/{source_id}.json",
        {"segments": segments},
        event_type="ingestion.supplements.completed",
        event_payload={
            "segment_count": len(segments),
            "source_files": source_files,
        },
    )
    setattr(supplement_index, "source_artifact_refs", [artifact_ref])


def _record_ctgov_source(qa_trace, nct_hint: str | None, ct_gov_data: dict | None) -> None:
    if qa_trace is None or ct_gov_data is None:
        return
    nct_id = _normalize_source_nct(nct_hint) or "unknown"
    qa_trace.write_source_artifact(
        f"sources/ctgov/{nct_id}.json",
        ct_gov_data,
        event_type="ingestion.ctgov.completed",
        trial_id=nct_id if nct_id != "unknown" else None,
        event_payload={"nct_id": nct_id},
    )


def _record_metadata_source(qa_trace, trial_metadata) -> None:
    if qa_trace is None:
        return
    qa_trace.write_source_artifact(
        f"sources/metadata/{_safe_artifact_name(trial_metadata.trial_id)}.json",
        trial_metadata,
        event_type="ingestion.metadata.completed",
        trial_id=trial_metadata.trial_id,
        event_payload={
            "trial_id": trial_metadata.trial_id,
            "nct_number": trial_metadata.nct_number,
            "outcome_count": len(trial_metadata.all_outcomes),
        },
    )


def _record_assessment_output(qa_trace, assessment: Assessment, json_path: Path) -> None:
    if qa_trace is None:
        return
    artifact_ref = f"outputs/{_safe_artifact_name(assessment.trial_id)}/{_safe_artifact_name(assessment.outcome)}.json"
    qa_trace.write_json_artifact(
        artifact_ref,
        {
            "assessment_id": assessment.assessment_id,
            "trial_id": assessment.trial_id,
            "outcome": assessment.outcome,
            "effect_of_interest": assessment.trial_metadata.effect_of_interest,
            "json_path": json_path,
        },
    )
    qa_trace.record_event(
        event_type="output.assessment_json.written",
        status="completed",
        trial_id=assessment.trial_id,
        outcome=assessment.outcome,
        artifact_refs=[artifact_ref],
        payload={"json_path": str(json_path)},
    )


def _record_report_output(qa_trace, assessment: Assessment, report_path: Path) -> None:
    if qa_trace is None:
        return
    artifact_ref = f"outputs/{_safe_artifact_name(assessment.trial_id)}/{_safe_artifact_name(assessment.outcome)}.report.json"
    qa_trace.write_json_artifact(
        artifact_ref,
        {
            "assessment_id": assessment.assessment_id,
            "trial_id": assessment.trial_id,
            "outcome": assessment.outcome,
            "report_path": report_path,
        },
    )
    qa_trace.record_event(
        event_type="output.report.written",
        status="completed",
        trial_id=assessment.trial_id,
        outcome=assessment.outcome,
        artifact_refs=[artifact_ref],
        payload={"report_path": str(report_path)},
    )


def _normalize_source_nct(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\bNCT\d{8}\b", value, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _source_artifact_id(value: str) -> str:
    name = _safe_artifact_name(Path(value).stem if value else "source")
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{name}-{digest}"


def _safe_artifact_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "source"


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
