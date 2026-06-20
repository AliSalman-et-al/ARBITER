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


class Judgment(str, Enum):
    LOW = "Low"
    SOME_CONCERNS = "Some concerns"
    HIGH = "High"


class ConfidenceFlag(str, Enum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    FLAGGED = "FLAGGED"


class EffectOfInterest(str, Enum):
    ASSIGNMENT = "assignment"
    ADHERING = "adhering"


class PageBox(BaseModel):
    boxclass: str
    text: str
    bbox: tuple[float, float, float, float]
    page: int


class ConfidenceSignals(BaseModel):
    supplement_segments_retrieved: int = 0
    supplement_segments_available: int = 0
    retrieval_top_score: float | None = None
    quote_verified: bool = True
    flag: ConfidenceFlag = ConfidenceFlag.CONFIDENT
    flag_reason: str | None = None


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
