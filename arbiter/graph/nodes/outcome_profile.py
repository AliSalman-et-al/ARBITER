"""Outcome measurement-profile classification for outcome-tier prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rapidfuzz import fuzz

from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMClient
from arbiter.models import (
    OUTCOME_MEASUREMENT_PROFILE_DEFINITIONS,
    OutcomeMeasurementProfile,
    OutcomeMeasurementProfileType,
)

MAX_REGISTRY_CANDIDATES = 5


async def outcome_profile_node(state: Mapping[str, Any]) -> dict[str, Any]:
    outcome = str(state.get("outcome", ""))
    try:
        profile = await classify_outcome_measurement_profile(
            assessed_outcome=outcome,
            ctgov_record=_ctgov_record_from_state(state),
            aux_model=_aux_model_from_state(state),
            config=_config_from_state(state),
        )
    except Exception as exc:
        profile = OutcomeMeasurementProfile(
            profile=OutcomeMeasurementProfileType.UNCLEAR,
            basis=(
                "outcome measurement-profile classification failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        _record_degradation(state, outcome=outcome, exc=exc)
    return {"outcome_measurement_profile": profile.model_dump(mode="json")}


async def classify_outcome_measurement_profile(
    *,
    assessed_outcome: str,
    ctgov_record: Mapping[str, Any] | None,
    aux_model: LLMClient,
    config: AssessmentConfig | object,
) -> OutcomeMeasurementProfile:
    candidates = _ranked_registered_outcome_candidates(
        assessed_outcome=assessed_outcome,
        ctgov_record=ctgov_record,
    )
    result = await aux_model.complete_structured(
        build_outcome_profile_messages(
            assessed_outcome=assessed_outcome,
            registered_outcome_candidates=candidates,
        ),
        OutcomeMeasurementProfile,
        temperature=0.0,
        max_tokens=_max_tokens(config),
        call_label=f"outcome_profile|{assessed_outcome}",
    )
    return OutcomeMeasurementProfile.model_validate(result)


def build_outcome_profile_messages(
    *,
    assessed_outcome: str,
    registered_outcome_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    definitions = "\n".join(
        f"- {profile.value}: {definition}"
        for profile, definition in OUTCOME_MEASUREMENT_PROFILE_DEFINITIONS.items()
    )
    candidate_lines = "\n".join(
        (
            f"- measure: {candidate['measure']}\n"
            f"  description: {candidate['description'] or '(none)'}\n"
            f"  timeFrame: {candidate['timeFrame'] or '(none)'}\n"
            f"  match_score: {candidate['match_score']}"
        )
        for candidate in registered_outcome_candidates
    )
    if not candidate_lines:
        candidate_lines = "(no ClinicalTrials.gov outcome candidates available)"

    return [
        {
            "role": "system",
            "content": (
                "Classify one randomized-trial outcome by measurement characteristics. "
                "Use only the provided taxonomy. This classification is advisory "
                "prompt context, not a risk-of-bias judgment."
            ),
        },
        {
            "role": "user",
            "content": (
                "[Assessed outcome]\n"
                f"{assessed_outcome}\n\n"
                "[Taxonomy]\n"
                f"{definitions}\n\n"
                "[ClinicalTrials.gov outcome candidates]\n"
                f"{candidate_lines}\n\n"
                "[Task]\n"
                "Return the best outcome measurement profile. Classify from "
                "measurement characteristics, not from outcome names alone. Use "
                "vital-status only when mortality/death is the single event that "
                "counts; composites that include death are clinician-composite. "
                "Use unclear when the available outcome definition text is "
                "insufficient. Keep basis to one concise sentence and, when a "
                "candidate was used, copy its measure exactly as "
                "matched_registered_outcome."
            ),
        },
    ]


def _ranked_registered_outcome_candidates(
    *,
    assessed_outcome: str,
    ctgov_record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    records = _registered_outcome_records(ctgov_record)
    ranked: list[dict[str, Any]] = []
    for record in records:
        measure = _string_field(record, "measure")
        if not measure:
            continue
        ranked.append(
            {
                "measure": measure,
                "description": _string_field(record, "description"),
                "timeFrame": _string_field(record, "timeFrame"),
                "match_score": round(fuzz.ratio(assessed_outcome, measure) / 100, 3),
            }
        )
    return sorted(ranked, key=lambda item: item["match_score"], reverse=True)[
        :MAX_REGISTRY_CANDIDATES
    ]


def _registered_outcome_records(
    ctgov_record: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if ctgov_record is None:
        return []
    protocol = ctgov_record.get("protocolSection")
    if not isinstance(protocol, Mapping):
        return []
    outcomes_module = protocol.get("outcomesModule")
    if not isinstance(outcomes_module, Mapping):
        return []

    records: list[Mapping[str, Any]] = []
    for key in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        values = outcomes_module.get(key)
        if not isinstance(values, list):
            continue
        records.extend(value for value in values if isinstance(value, Mapping))
    return records


def _string_field(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _aux_model_from_state(state: Mapping[str, Any]) -> LLMClient:
    model = state.get("aux_model")
    if isinstance(model, LLMClient):
        return model
    runtime = state.get("runtime")
    runtime_model = getattr(runtime, "aux_model", None)
    if isinstance(runtime_model, LLMClient):
        return runtime_model
    raise TypeError("outcome_profile_node requires an aux_model LLMClient")


def _config_from_state(state: Mapping[str, Any]) -> AssessmentConfig | object:
    config = state.get("config")
    return config if config is not None else object()


def _max_tokens(config: AssessmentConfig | object) -> int:
    env = getattr(config, "env", None)
    configured = getattr(env, "outcome_profile_max_tokens", None)
    return int(configured or 1024)


def _ctgov_record_from_state(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    record = state.get("ctgov_record", state.get("ct_gov_data"))
    return record if isinstance(record, Mapping) else None


def _record_degradation(
    state: Mapping[str, Any], *, outcome: str, exc: Exception
) -> None:
    trace = state.get("trace")
    if trace is None or not hasattr(trace, "record_degradation"):
        return
    cast(Any, trace).record_degradation(
        category="outcome_profile_classification_failed",
        reason=f"{type(exc).__name__}: {exc}",
        severity="warning",
        trial_id=_trial_id(state),
        outcome=outcome,
        payload={"fallback_profile": OutcomeMeasurementProfileType.UNCLEAR.value},
    )


def _trial_id(state: Mapping[str, Any]) -> str | None:
    metadata = state.get("trial_metadata")
    trial_id = getattr(metadata, "trial_id", None)
    return str(trial_id) if trial_id is not None else None
