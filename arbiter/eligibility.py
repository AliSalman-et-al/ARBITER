"""Deterministic RoB 2 eligibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from arbiter.models import SectionMap, StudyDesign, TrialMetadata


@dataclass(frozen=True)
class EligibilityDecision:
    """Decision used by the manifest/CLI gate."""

    eligible: bool
    basis: str
    study_design: StudyDesign
    requires_human_review: bool = False


def decide_eligibility(
    trial_metadata: TrialMetadata,
    *,
    ct_gov_data: Mapping[str, Any] | None = None,
    section_map: SectionMap | None = None,
    raw_char_stream: str | None = None,
) -> EligibilityDecision:
    """Decide whether a trial is inside v0.1 individually-randomised parallel RCT scope."""

    registry = _registry_design(ct_gov_data)
    if registry["out_of_scope"]:
        return EligibilityDecision(
            eligible=False,
            basis=str(registry["basis"]),
            study_design=_metadata_or_registry_design(trial_metadata, registry),
        )
    if registry["parallel_rct"]:
        return EligibilityDecision(
            eligible=True,
            basis=str(registry["basis"]),
            study_design=StudyDesign.PARALLEL_RCT,
        )

    if trial_metadata.study_design == StudyDesign.PARALLEL_RCT:
        return EligibilityDecision(
            eligible=True,
            basis=trial_metadata.study_design_basis or "LLM metadata classified the study as a parallel-group RCT.",
            study_design=StudyDesign.PARALLEL_RCT,
        )

    if trial_metadata.study_design in {
        StudyDesign.CLUSTER_RCT,
        StudyDesign.CROSSOVER_RCT,
        StudyDesign.SINGLE_ARM,
        StudyDesign.NON_RCT,
    }:
        return EligibilityDecision(
            eligible=False,
            basis=trial_metadata.study_design_basis or "positive metadata evidence indicates an out-of-scope design.",
            study_design=trial_metadata.study_design,
        )

    return EligibilityDecision(
        eligible=True,
        basis="eligibility ambiguous; proceeding with human-review flag instead of skipping on uncertainty",
        study_design=trial_metadata.study_design,
        requires_human_review=True,
    )


def reconcile_trial_metadata(
    trial_metadata: TrialMetadata,
    *,
    ct_gov_data: Mapping[str, Any] | None = None,
    section_map: SectionMap | None = None,
    raw_char_stream: str | None = None,
) -> TrialMetadata:
    """Replace uncertain LLM study design with deterministic eligibility evidence when available."""

    decision = decide_eligibility(
        trial_metadata,
        ct_gov_data=ct_gov_data,
        section_map=section_map,
        raw_char_stream=raw_char_stream,
    )
    if decision.study_design == trial_metadata.study_design and decision.basis == trial_metadata.study_design_basis:
        return trial_metadata
    return trial_metadata.model_copy(
        update={
            "study_design": decision.study_design,
            "study_design_basis": decision.basis,
        }
    )


def _registry_design(ct_gov_data: Mapping[str, Any] | None) -> dict[str, object]:
    if ct_gov_data is None:
        return {"parallel_rct": False, "out_of_scope": False, "basis": None, "study_design": None}
    protocol = _mapping(ct_gov_data.get("protocolSection"))
    design_module = _mapping(protocol.get("designModule"))
    design_info = _mapping(design_module.get("designInfo"))
    study_type = _norm(design_module.get("studyType") or protocol.get("studyType"))
    allocation = _norm(design_info.get("allocation"))
    intervention_model = _norm(design_info.get("interventionModel"))

    if "OBSERVATIONAL" in study_type:
        return {
            "parallel_rct": False,
            "out_of_scope": True,
            "basis": "ClinicalTrials.gov studyType is OBSERVATIONAL.",
            "study_design": StudyDesign.NON_RCT,
        }
    if "SINGLE_GROUP" in intervention_model:
        return {
            "parallel_rct": False,
            "out_of_scope": True,
            "basis": "ClinicalTrials.gov interventionModel is SINGLE_GROUP.",
            "study_design": StudyDesign.SINGLE_ARM,
        }
    if "CROSSOVER" in intervention_model:
        return {
            "parallel_rct": False,
            "out_of_scope": True,
            "basis": "ClinicalTrials.gov interventionModel is CROSSOVER.",
            "study_design": StudyDesign.CROSSOVER_RCT,
        }
    if "INTERVENTIONAL" in study_type and "RANDOM" in allocation and "PARALLEL" in intervention_model:
        return {
            "parallel_rct": True,
            "out_of_scope": False,
            "basis": "ClinicalTrials.gov reports studyType=INTERVENTIONAL, allocation=RANDOMIZED, interventionModel=PARALLEL.",
            "study_design": StudyDesign.PARALLEL_RCT,
        }
    return {"parallel_rct": False, "out_of_scope": False, "basis": None, "study_design": None}


def _metadata_or_registry_design(trial_metadata: TrialMetadata, registry: Mapping[str, object]) -> StudyDesign:
    design = registry.get("study_design")
    return design if isinstance(design, StudyDesign) else trial_metadata.study_design


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _norm(value: object) -> str:
    return "_".join(str(value or "").upper().replace("-", " ").split())
