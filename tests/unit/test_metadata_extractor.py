from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from arbiter.config import AssessmentConfig
from arbiter.ingestion.metadata_extractor import (
    build_metadata_source_text,
    extract_metadata,
    slugify,
)
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import (
    BlindingStatus,
    DocumentSection,
    ParsingQuality,
    SectionMap,
    StudyDesign,
)


class RecordingMockLLMClient(MockLLMClient):
    def __init__(self, *, responses: dict[str, Any]) -> None:
        super().__init__(responses=responses)
        self.messages: list[list[dict[str, Any]]] = []

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        call_label: str | None = None,
    ) -> BaseModel:
        self.messages.append(messages)
        return await super().complete_structured(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            call_label=call_label,
        )


def _section_map(path: Path) -> SectionMap:
    return SectionMap(
        source_path=str(path),
        full_text=(
            "ABSTRACT\nAbstract allocation text.\n\n"
            "METHODS\nMethods randomisation text.\n\n"
            "RESULTS\nResults should not be sent to metadata extraction."
        ),
        sections=[
            DocumentSection(
                label="ABSTRACT",
                pages=[0],
                char_start=0,
                char_end=34,
                text="ABSTRACT\nAbstract allocation text.",
                domain_tags=[],
            ),
            DocumentSection(
                label="METHODS",
                pages=[0],
                char_start=36,
                char_end=72,
                text="METHODS\nMethods randomisation text.",
                domain_tags=[],
            ),
            DocumentSection(
                label="RESULTS",
                pages=[0],
                char_start=74,
                char_end=127,
                text="RESULTS\nResults should not be sent to metadata extraction.",
                domain_tags=[],
            ),
        ],
        page_boxes=[],
        parsing_quality=ParsingQuality.STANDARD,
        nct_number="NCT22222222",
    )


def _metadata_response(**overrides: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "title": "Trial of Intervention A",
        "intervention": "Intervention A",
        "comparator": "Placebo",
        "primary_outcome": "Overall survival",
        "all_outcomes": ["Overall survival", "Progression-free survival", "Overall Survival"],
        "blinding": BlindingStatus.DOUBLE_BLIND.value,
        "nct_number": "NCT33333333",
        "study_design": StudyDesign.PARALLEL_RCT.value,
        "study_design_basis": "Participants were individually randomized to parallel treatment groups.",
    }
    response.update(overrides)
    return response


def test_build_metadata_source_text_prefers_abstract_and_methods(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"paper")

    source_text = build_metadata_source_text(_section_map(paper_path), token_budget=100)

    assert "Abstract allocation text" in source_text
    assert "Methods randomisation text" in source_text
    assert "Results should not be sent" not in source_text


def test_build_metadata_source_text_falls_back_to_full_text_and_caps_tokens(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    section_map = SectionMap(
        source_path=str(paper_path),
        full_text="one two three four five",
        sections=[],
        page_boxes=[],
        parsing_quality=ParsingQuality.DEGRADED,
    )

    assert build_metadata_source_text(section_map, token_budget=3) == "one two three"


@pytest.mark.asyncio
async def test_extract_metadata_applies_nct_precedence_and_normalizes_outcomes(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"stable paper bytes")
    config = AssessmentConfig(paper_path=paper_path, nct_number="NCT11111111")
    config.env.max_outcomes = 2
    client = RecordingMockLLMClient(responses={"metadata": _metadata_response()})

    metadata = await extract_metadata(_section_map(paper_path), config, client, nct_hint="NCT22222222")

    assert client.calls == ["metadata"]
    assert metadata.trial_id == "NCT11111111"
    assert metadata.nct_number == "NCT11111111"
    assert metadata.effect_of_interest.value == "assignment"
    assert metadata.all_outcomes == ["Overall survival", "Progression-free survival"]
    assert metadata.study_design == StudyDesign.PARALLEL_RCT
    assert metadata.study_design_basis == "Participants were individually randomized to parallel treatment groups."


@pytest.mark.asyncio
async def test_extract_metadata_regex_nct_overrides_llm_nct(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"stable paper bytes")
    config = AssessmentConfig(paper_path=paper_path)
    client = RecordingMockLLMClient(responses={"metadata": _metadata_response()})

    metadata = await extract_metadata(_section_map(paper_path), config, client, nct_hint="NCT22222222")

    assert metadata.nct_number == "NCT22222222"
    assert metadata.trial_id == "NCT22222222"


@pytest.mark.asyncio
async def test_extract_metadata_uses_slugged_label_then_content_hash_without_nct(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    payload = b"stable paper bytes"
    paper_path.write_bytes(payload)
    no_nct_response = _metadata_response(nct_number=None)

    labelled = await extract_metadata(
        _section_map(paper_path),
        AssessmentConfig(paper_path=paper_path, trial_label="ARASENS Trial!"),
        MockLLMClient(responses={"metadata": no_nct_response}),
        nct_hint=None,
    )
    hashed = await extract_metadata(
        _section_map(paper_path),
        AssessmentConfig(paper_path=paper_path),
        MockLLMClient(responses={"metadata": no_nct_response}),
        nct_hint=None,
    )

    assert labelled.trial_id == "arasens-trial"
    assert hashed.trial_id == hashlib.sha256(payload).hexdigest()[:12]


def test_slugify_is_ascii_and_stable() -> None:
    assert slugify("  Étude Phase III: A/B  ") == "etude-phase-iii-a-b"
