"""Trial metadata extraction from the main paper."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from arbiter.ingestion.paper import TOP_LEVEL_SECTION_LABELS, normalize_heading
from arbiter.config import AssessmentConfig
from arbiter.llm.base import LLMClient
from arbiter.models import (
    BlindingStatus,
    EffectOfInterest,
    SectionMap,
    StudyDesign,
    TrialMetadata,
)
from arbiter.token_budgeting import cap_text_to_tokens

NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
NULL_OUTCOME_SENTINELS = {"null", "none", "n/a", "na"}
UNKNOWN_INTERVENTION = "Intervention not extracted"
UNKNOWN_COMPARATOR = "Comparator not extracted"
UNKNOWN_OUTCOME = "Outcome not extracted"
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
MIN_CANONICAL_SECTION_CHARS = 500


class MetadataExtractionResult(BaseModel):
    """Schema returned by the aux model before deterministic post-processing."""

    title: str = ""
    intervention: str = ""
    comparator: str = ""
    primary_outcome: str = ""
    all_outcomes: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    blinding: BlindingStatus
    nct_number: str | None = None
    study_design: StudyDesign = StudyDesign.UNCLEAR
    study_design_basis: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_shape_drift(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        source = _unwrap_payload(value)
        if not isinstance(source, dict):
            return value
        normalized = dict(source)
        _copy_alias(normalized, "all_outcomes", ("outcomes", "secondary_outcomes"))
        _copy_alias(normalized, "blinding", ("masking", "blind", "blinding_status"))
        _copy_alias(
            normalized,
            "nct_number",
            ("nct", "nct_id", "registry_id", "registration_number"),
        )
        _copy_alias(normalized, "study_design", ("design", "trial_design"))
        _copy_alias(
            normalized, "study_design_basis", ("design_basis", "study_type_basis")
        )
        for key in (
            "title",
            "intervention",
            "comparator",
            "primary_outcome",
            "study_design_basis",
            "nct_number",
        ):
            if key in normalized:
                normalized[key] = _coerce_string(normalized[key])
        if "all_outcomes" in normalized and isinstance(normalized["all_outcomes"], str):
            normalized["all_outcomes"] = [normalized["all_outcomes"]]
        return normalized

    @field_validator(
        "title",
        "intervention",
        "comparator",
        "primary_outcome",
        "study_design_basis",
        mode="before",
    )
    @classmethod
    def normalize_text_field(cls, value: Any) -> str:
        return _clean_text(_coerce_string(value))

    @field_validator("nct_number")
    @classmethod
    def normalize_nct_number(cls, value: str | None) -> str | None:
        return normalize_nct(value)

    @field_validator("all_outcomes", mode="before")
    @classmethod
    def clean_all_outcomes(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value

        outcomes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = clean_outcome_name(item)
            if cleaned and cleaned.casefold() not in {
                existing.casefold() for existing in outcomes
            }:
                outcomes.append(cleaned)
        return outcomes


def _unwrap_payload(value: dict[str, Any]) -> Any:
    for key in ("metadata", "result", "response", "data"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return nested
    return value


def _copy_alias(
    payload: dict[str, Any], canonical: str, aliases: tuple[str, ...]
) -> None:
    if canonical in payload:
        return
    for alias in aliases:
        if alias in payload:
            payload[canonical] = payload[alias]
            return


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )
    return str(value)


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


async def extract_metadata(
    section_map: SectionMap,
    config: AssessmentConfig,
    aux_client: LLMClient,
    nct_hint: str | None,
) -> TrialMetadata:
    """Extract and normalize trial metadata with one aux-model call."""

    source_text = build_metadata_source_text(
        section_map, config.env.metadata_token_budget
    )
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
        max_tokens=config.env.metadata_extraction_max_tokens,
        call_label="metadata",
    )
    result = MetadataExtractionResult.model_validate(extracted)

    nct_number = choose_nct_number(config.nct_number, nct_hint, result.nct_number)
    primary_outcome = choose_primary_outcome(result, config)
    all_outcomes = normalize_outcomes(
        primary_outcome,
        [*(config.outcomes or []), *result.all_outcomes],
        config.env.max_outcomes,
    )

    return TrialMetadata(
        trial_id=build_trial_id(
            nct_number=nct_number,
            trial_label=config.trial_label,
            paper_path=config.paper_path,
            fallback_text=section_map.full_text,
        ),
        title=choose_title(result.title, config, nct_number),
        intervention=result.intervention or UNKNOWN_INTERVENTION,
        comparator=result.comparator or UNKNOWN_COMPARATOR,
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
        selected = [section for section in section_map.sections if section.label.upper() in labels]
        text_length = sum(len(section.text.strip()) for section in selected)
        if selected and text_length >= MIN_CANONICAL_SECTION_CHARS:
            chunks.extend(f"{section.label}\n{section.text}".strip() for section in selected)
            continue
        fallback = _slice_full_text_section(section_map, labels)
        if fallback:
            chunks.append(f"{selected[0].label if selected else labels[0]}\n{fallback}".strip())
        else:
            chunks.extend(f"{section.label}\n{section.text}".strip() for section in selected)

    source = "\n\n".join(chunks).strip() or section_map.full_text
    return truncate_to_token_budget(source, token_budget)


def _slice_full_text_section(section_map: SectionMap, labels: tuple[str, ...]) -> str:
    starts = [
        section
        for section in section_map.sections
        if section.label.upper() in labels and 0 <= section.char_start < len(section_map.full_text)
    ]
    if not starts:
        return ""
    start_section = min(starts, key=lambda section: section.char_start)
    later_top_level = [
        section.char_start
        for section in section_map.sections
        if section.char_start > start_section.char_start
        and normalize_heading(section.label) in TOP_LEVEL_SECTION_LABELS
    ]
    end = min(later_top_level, default=len(section_map.full_text))
    return section_map.full_text[start_section.char_start : end].strip()


def truncate_to_token_budget(text: str, token_budget: int) -> str:
    """Cap text using ARBITER's configured tokenizer."""

    return cap_text_to_tokens(text, token_budget, "metadata").text


def choose_nct_number(
    config_nct_number: str | None,
    nct_hint: str | None,
    extracted_nct_number: str | None,
) -> str | None:
    """Apply REQ-05 NCT precedence."""

    return (
        normalize_nct(config_nct_number)
        or normalize_nct(nct_hint)
        or normalize_nct(extracted_nct_number)
    )


def normalize_nct(value: str | None) -> str | None:
    if value is None:
        return None
    match = NCT_PATTERN.search(value)
    return match.group(0).upper() if match else None


def normalize_outcomes(
    primary_outcome: str, all_outcomes: list[str], max_outcomes: int
) -> list[str]:
    """Place the primary outcome first, dedupe, and enforce the configured cap."""

    outcomes: list[str] = []
    for outcome in [primary_outcome, *all_outcomes]:
        cleaned = clean_outcome_name(outcome)
        if cleaned and cleaned.casefold() not in {
            existing.casefold() for existing in outcomes
        }:
            outcomes.append(cleaned)
        if len(outcomes) >= max(1, max_outcomes):
            break
    return outcomes


def choose_primary_outcome(
    result: MetadataExtractionResult, config: AssessmentConfig
) -> str:
    """Pick the best usable primary outcome without trusting one model field."""

    for candidate in [
        result.primary_outcome,
        *(config.outcomes or []),
        *result.all_outcomes,
    ]:
        if cleaned := clean_outcome_name(candidate):
            return cleaned
    return UNKNOWN_OUTCOME


def choose_title(
    extracted_title: str, config: AssessmentConfig, nct_number: str | None
) -> str:
    """Keep metadata construction nonfatal when title extraction degrades."""

    return (
        extracted_title
        or _clean_text(config.trial_label)
        or nct_number
        or config.paper_path.stem
        or "Untitled trial"
    )


def clean_outcome_name(value: str) -> str | None:
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if cleaned.casefold() in NULL_OUTCOME_SENTINELS:
        return None
    if re.fullmatch(r"[\W_]+", cleaned):
        return None
    return cleaned


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
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
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
