"""Core data models shared by ARBITER pipeline slices."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class ConfidenceFlag(str, Enum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    FLAGGED = "FLAGGED"


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
    UNKNOWN = "unknown"


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
    annotation: str
    domain_tags: list[str] = Field(default_factory=list)
    char_count: int

    @property
    def annotated_text(self) -> str:
        return f"{self.annotation}\n\n{self.raw_text}".strip()


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
    flag: ConfidenceFlag = ConfidenceFlag.CONFIDENT
    flag_reason: str | None = None


class SQRawAnswer(BaseModel):
    answer: LLMAnswerCode
    quote: str = Field(max_length=4000)
    justification: str = Field(max_length=1000)


class OutcomeComparison(BaseModel):
    registered_outcome: str | None = None
    published_outcome: str | None = None
    outcome_similarity_score: float | None = None
    outcome_change_detected: bool | None = None
    registered_as_primary: bool | None = None


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
