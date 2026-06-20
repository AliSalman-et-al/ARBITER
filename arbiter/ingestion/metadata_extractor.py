"""Trial metadata extraction from the main paper."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMClient
from arbiter.models import BlindingStatus, EffectOfInterest, SectionMap, StudyDesign, TrialMetadata

NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
SECTION_LABELS = {
    "abstract": ("ABSTRACT", "SUMMARY"),
    "methods": (
        "METHOD",
        "METHODS",
        "MATERIALS AND METHODS",
        "PATIENTS AND METHODS",
        "PARTICIPANTS AND METHODS",
    ),
}


class MetadataExtractionResult(BaseModel):
    """Schema returned by the aux model before deterministic post-processing."""

    title: Annotated[str, Field(min_length=1)]
    intervention: Annotated[str, Field(min_length=1)]
    comparator: Annotated[str, Field(min_length=1)]
    primary_outcome: Annotated[str, Field(min_length=1)]
    all_outcomes: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    blinding: BlindingStatus
    nct_number: str | None = None
    study_design: StudyDesign = StudyDesign.UNCLEAR
    study_design_basis: str | None = None

    @field_validator("nct_number")
    @classmethod
    def normalize_nct_number(cls, value: str | None) -> str | None:
        return normalize_nct(value)


async def extract_metadata(
    section_map: SectionMap,
    config: AssessmentConfig,
    aux_client: LLMClient,
    nct_hint: str | None,
) -> TrialMetadata:
    """Extract and normalize trial metadata with one aux-model call."""

    source_text = build_metadata_source_text(section_map, config.env.metadata_token_budget)
    messages = [
        {
            "role": "system",
            "content": (
                "Extract structured metadata for a randomized trial paper. "
                "Return only fields in the provided schema. Do not infer an effect of interest."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the following main-paper text to identify trial metadata.\n\n"
                f"NCT hint: {normalize_nct(config.nct_number) or normalize_nct(nct_hint) or 'none'}\n\n"
                f"{source_text}"
            ),
        },
    ]
    extracted = await aux_client.complete_structured(
        messages,
        MetadataExtractionResult,
        temperature=0.0,
        max_tokens=1024,
        call_label="metadata",
    )
    result = MetadataExtractionResult.model_validate(extracted)

    nct_number = choose_nct_number(config.nct_number, nct_hint, result.nct_number)
    primary_outcome = result.primary_outcome.strip()
    all_outcomes = normalize_outcomes(primary_outcome, result.all_outcomes, config.env.max_outcomes)

    return TrialMetadata(
        trial_id=build_trial_id(
            nct_number=nct_number,
            trial_label=config.trial_label,
            paper_path=config.paper_path,
            fallback_text=section_map.full_text,
        ),
        title=result.title.strip(),
        intervention=result.intervention.strip(),
        comparator=result.comparator.strip(),
        primary_outcome=primary_outcome,
        all_outcomes=all_outcomes,
        effect_of_interest=EffectOfInterest(config.effect_of_interest),
        blinding=result.blinding,
        nct_number=nct_number,
        study_design=result.study_design,
        study_design_basis=_clean_optional_sentence(result.study_design_basis),
    )


def build_metadata_source_text(section_map: SectionMap, token_budget: int) -> str:
    """Return abstract + methods text, falling back to full text when absent."""

    chunks: list[str] = []
    for label_group in ("abstract", "methods"):
        labels = SECTION_LABELS[label_group]
        for section in section_map.sections:
            if section.label.upper() in labels:
                chunks.append(f"{section.label}\n{section.text}".strip())

    source = "\n\n".join(chunks).strip() or section_map.full_text
    return truncate_to_token_budget(source, token_budget)


def truncate_to_token_budget(text: str, token_budget: int) -> str:
    """Conservatively cap text using whitespace tokens to avoid extra deps."""

    if token_budget <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= token_budget:
        return text.strip()
    return " ".join(tokens[:token_budget]).strip()


def choose_nct_number(
    config_nct_number: str | None,
    nct_hint: str | None,
    extracted_nct_number: str | None,
) -> str | None:
    """Apply REQ-05 NCT precedence."""

    return normalize_nct(config_nct_number) or normalize_nct(nct_hint) or normalize_nct(extracted_nct_number)


def normalize_nct(value: str | None) -> str | None:
    if value is None:
        return None
    match = NCT_PATTERN.search(value)
    return match.group(0).upper() if match else None


def normalize_outcomes(primary_outcome: str, all_outcomes: list[str], max_outcomes: int) -> list[str]:
    """Place the primary outcome first, dedupe, and enforce the configured cap."""

    outcomes: list[str] = []
    for outcome in [primary_outcome, *all_outcomes]:
        cleaned = " ".join(outcome.split())
        if cleaned and cleaned.lower() not in {existing.lower() for existing in outcomes}:
            outcomes.append(cleaned)
        if len(outcomes) >= max(1, max_outcomes):
            break
    return outcomes


def build_trial_id(
    *,
    nct_number: str | None,
    trial_label: str | None,
    paper_path: Path,
    fallback_text: str,
) -> str:
    """Build the stable trial id: NCT, slug label, then paper content hash."""

    if nct_number:
        return nct_number
    if trial_label and (slug := slugify(trial_label)):
        return slug
    return hash_paper(paper_path, fallback_text)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def hash_paper(path: Path, fallback_text: str) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        payload = fallback_text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _clean_optional_sentence(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None
