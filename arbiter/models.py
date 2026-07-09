"""Core data models shared by ARBITER pipeline slices."""

from __future__ import annotations

from enum import Enum
from typing import Literal, cast

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _join_stringish(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )
    if value is None:
        return ""
    return str(value)


class AnswerCode(str, Enum):
    Y = "Y"
    PY = "PY"
    PN = "PN"
    N = "N"
    NI = "NI"
    NA = "NA"


LLMAnswerCode = Literal["Y", "PY", "PN", "N", "NI"]


class Judgment(str, Enum):
    LOW = "Low"
    SOME_CONCERNS = "Some concerns"
    HIGH = "High"
    UNRESOLVED = "Unresolved"


class ConfidenceFlag(str, Enum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    FLAGGED = "FLAGGED"


class SQFallbackKind(str, Enum):
    SQ_CALL_FAILED = "sq_call_failed"


class ReliabilityStatus(str, Enum):
    OK = "OK"
    FAILURE_FALLBACK_EXCESSIVE = "FAILURE_FALLBACK_EXCESSIVE"


class BlindingStatus(str, Enum):
    OPEN_LABEL = "open_label"
    SINGLE_BLIND = "single_blind"
    DOUBLE_BLIND = "double_blind"
    UNCLEAR = "unclear"


class ParsingQuality(str, Enum):
    STANDARD = "STANDARD"
    DEGRADED = "DEGRADED"


class DocType(str, Enum):
    SAP = "sap"
    PROTOCOL = "protocol"
    APPENDIX = "appendix"
    DISCLOSURE = "disclosure"
    ADMINISTRATIVE = "administrative"
    UNKNOWN = "unknown"


SQ_QUOTE_HARD_LIMIT = 4000
SQ_JUSTIFICATION_HARD_LIMIT = 1000


class EffectOfInterest(str, Enum):
    ASSIGNMENT = "assignment"
    ADHERING = "adhering"


class StudyDesign(str, Enum):
    PARALLEL_RCT = "parallel_rct"
    CLUSTER_RCT = "cluster_rct"
    CROSSOVER_RCT = "crossover_rct"
    SINGLE_ARM = "single_arm"
    NON_RCT = "non_rct"
    UNCLEAR = "unclear"


class OutcomeMeasurementProfileType(str, Enum):
    VITAL_STATUS = "vital-status"
    BIOMARKER = "biomarker"
    CLINICIAN_COMPOSITE = "clinician-composite"
    CLINICIAN_GRADED = "clinician-graded"
    PATIENT_REPORTED = "patient-reported"
    UNCLEAR = "unclear"


OUTCOME_MEASUREMENT_PROFILE_DEFINITIONS = {
    OutcomeMeasurementProfileType.VITAL_STATUS: (
        "All-cause or disease-specific mortality assessed as a single criterion; "
        "death is the only event that counts, excluding composites that merely "
        "include death."
    ),
    OutcomeMeasurementProfileType.BIOMARKER: (
        "Laboratory or imaging measurement with a pre-defined numerical threshold."
    ),
    OutcomeMeasurementProfileType.CLINICIAN_COMPOSITE: (
        "Composite or time-to-event outcome requiring clinical or radiological "
        "judgment."
    ),
    OutcomeMeasurementProfileType.CLINICIAN_GRADED: (
        "Standardized clinical grading scale that still requires judgment."
    ),
    OutcomeMeasurementProfileType.PATIENT_REPORTED: "Self-report instrument.",
    OutcomeMeasurementProfileType.UNCLEAR: (
        "Measurement characteristics could not be assigned confidently from the "
        "available outcome definition text."
    ),
}


class PageBox(BaseModel):
    boxclass: str
    text: str
    bbox: tuple[float, float, float, float]
    page: int


class DocumentSection(BaseModel):
    label: str
    pages: list[int]
    char_start: int
    char_end: int
    text: str
    domain_tags: list[str] = Field(default_factory=list)


class SectionMap(BaseModel):
    source_path: str
    full_text: str
    sections: list[DocumentSection]
    page_boxes: list[PageBox]
    parsing_quality: ParsingQuality = ParsingQuality.STANDARD
    nct_number: str | None = None


class SupplementSegment(BaseModel):
    segment_id: str
    source_file: str
    doc_type: DocType
    heading: str
    pages: list[int]
    raw_text: str
    domain_tags: list[str] = Field(default_factory=list)
    doc_item_labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    char_count: int


class TrialMetadata(BaseModel):
    trial_id: str
    title: str
    intervention: str
    comparator: str
    primary_outcome: str
    all_outcomes: list[str]
    effect_of_interest: EffectOfInterest
    blinding: BlindingStatus
    nct_number: str | None = None
    study_design: StudyDesign = StudyDesign.UNCLEAR
    study_design_basis: str | None = None


class ConfidenceSignals(BaseModel):
    supplement_segments_retrieved: int = 0
    supplement_segments_available: int = 0
    retrieval_top_score: float | None = None
    quote_verified: bool = True
    quote_source_type: Literal["main_paper", "supplement", "registry"] | None = None
    context_sufficient: bool | None = None
    context_sufficiency_reason: str | None = None
    entailment_score: float | None = None
    faithfulness_score: float | None = None
    grounding_method: Literal[
        "quote_verification", "lexical_overlap", "not_applicable"
    ] = "not_applicable"
    flag: ConfidenceFlag = ConfidenceFlag.CONFIDENT
    flag_reason: str | None = None
    fallback_kind: SQFallbackKind | None = None


class AssessmentReliability(BaseModel):
    status: ReliabilityStatus = ReliabilityStatus.OK
    sq_answer_count: int = 0
    failure_fallback_sq_count: int = 0
    failure_fallback_fraction: float = 0.0
    failure_fallback_threshold: float = 0.25
    failure_fallback_min_count: int = 2
    basis: str | None = None


class SQRawAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: LLMAnswerCode = "NI"
    quote: str = ""
    justification: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_shape_drift(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "answer" not in normalized and "answer_code" in normalized:
            normalized["answer"] = normalized["answer_code"]
            del normalized["answer_code"]
        if "quote" not in normalized:
            for alias in ("quotes", "quoted_text", "source"):
                if alias in normalized:
                    normalized["quote"] = normalized.pop(alias)
                    break
        if "justification" not in normalized:
            for alias in ("reasoning", "rationale", "explanation"):
                if alias in normalized:
                    normalized["justification"] = normalized.pop(alias)
                    break
        for key in ("quote", "justification"):
            if key in normalized:
                normalized[key] = _join_stringish(normalized[key])
        return normalized

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_answer(cls, value: Any) -> LLMAnswerCode:
        normalized = str(value or "").strip().upper()
        if normalized in {"Y", "PY", "PN", "N", "NI"}:
            return cast(LLMAnswerCode, normalized)
        return "NI"

    @field_validator("quote")
    @classmethod
    def truncate_quote(cls, value: str) -> str:
        return value[:SQ_QUOTE_HARD_LIMIT]

    @field_validator("justification")
    @classmethod
    def truncate_justification(cls, value: str) -> str:
        return value[:SQ_JUSTIFICATION_HARD_LIMIT]

    @model_validator(mode="after")
    def require_justification(self) -> "SQRawAnswer":
        if not self.justification.strip():
            raise ValueError("SQRawAnswer requires a non-empty justification")
        return self


class OutcomeComparison(BaseModel):
    registered_outcome: str | None = None
    published_outcome: str | None = None
    outcome_similarity_score: float | None = None
    outcome_change_detected: bool | None = None
    registered_as_primary: bool | None = None


class OutcomeMeasurementProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: OutcomeMeasurementProfileType = OutcomeMeasurementProfileType.UNCLEAR
    definition: str = ""
    basis: str = ""
    matched_registered_outcome: str | None = None
    match_score: float | None = None

    @model_validator(mode="after")
    def fill_default_definition(self) -> "OutcomeMeasurementProfile":
        if not self.definition:
            self.definition = OUTCOME_MEASUREMENT_PROFILE_DEFINITIONS[self.profile]
        return self


class DomainContext(BaseModel):
    domain: str
    domain_specific_text: str = ""
    supplement_block: str = ""
    retrieval_top_score: float | None = None
    segments_retrieved: int = 0
    segments_available: int = 0


class SQAnswer(BaseModel):
    sq_id: str
    answer: AnswerCode
    quote: str = ""
    page: int | None = None
    justification: str = ""
    confidence: ConfidenceSignals = Field(default_factory=ConfidenceSignals)


class DomainJudgment(BaseModel):
    domain: str
    scope: Literal["trial", "outcome"]
    judgment: Judgment
    algorithm_rationale: str
    sq_answers: list[SQAnswer] = Field(default_factory=list)


class SourcesManifest(BaseModel):
    main_paper: str
    supplements: list[str] = Field(default_factory=list)
    ct_gov_retrieved: bool = False
    parsing_quality: ParsingQuality = ParsingQuality.STANDARD


class Assessment(BaseModel):
    assessment_id: str
    created_at: str
    pipeline_version: str
    model_sq: str
    model_aux: str
    model_vision: str | None = None
    trial_id: str
    nct_number: str | None = None
    outcome: str
    requires_human_review: bool
    config_summary: dict
    trial_metadata: TrialMetadata
    ct_gov_data: dict | None = None
    outcome_comparison: OutcomeComparison | None = None
    domain_judgments: list[DomainJudgment]
    overall_judgment: Judgment
    overall_rationale: str
    reliability: AssessmentReliability = Field(default_factory=AssessmentReliability)
    sources_manifest: SourcesManifest
    errors: list[str] = Field(default_factory=list)


class SkipRecord(BaseModel):
    assessment_id: str
    created_at: str
    trial_id: str
    nct_number: str | None = None
    study_design: StudyDesign
    study_design_basis: str | None = None
    requires_human_review: bool = True
    model_sq: str
    model_aux: str | None = None
    pipeline_version: str
    inputs_hash: str | None = None
    errors: list[str] = Field(default_factory=list)
