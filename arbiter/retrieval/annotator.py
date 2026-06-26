"""LLM enrichment for supplement segments."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from arbiter.config import EnvSettings
from arbiter.llm.base import LLMClient
from arbiter.models import SupplementSegment


class SegmentAnnotation(BaseModel):
    annotation: str = Field(min_length=1)


async def annotate_segment(
    segment: SupplementSegment,
    *,
    document_preamble: str,
    aux_client: LLMClient,
    settings: EnvSettings | None = None,
) -> str:
    settings = settings or EnvSettings()
    messages = [
        {
            "role": "system",
            "content": (
                "You annotate clinical-trial supplementary material for Cochrane RoB 2 assessment. "
                "Return 2-3 concise sentences naming methods, populations, procedures, or analyses "
                "relevant to randomisation, blinding, missing data, outcome assessment, or selective reporting. "
                'Always return the required structured object with an "annotation" field. '
                'If there is no risk-of-bias relevant content, set "annotation" to '
                '"No risk-of-bias relevant content."'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document preamble:\n{document_preamble}\n\n"
                f"Segment heading: {segment.heading}\n"
                f"Domain tags: {', '.join(segment.domain_tags) or 'none'}\n\n"
                f"Segment text:\n{segment.raw_text}"
            ),
        },
    ]
    response = cast(
        SegmentAnnotation,
        await aux_client.complete_structured(
            messages,
            SegmentAnnotation,
            temperature=0.0,
            max_tokens=256,
            call_label=f"supplement_annotation:{segment.segment_id}",
        ),
    )
    annotation = response.annotation.strip()
    return annotation or "No risk-of-bias relevant content."


def choose_segments_for_annotation(
    segments: list[SupplementSegment],
    *,
    settings: EnvSettings | None = None,
) -> set[str]:
    settings = settings or EnvSettings()
    tagged = [segment for segment in segments if segment.domain_tags]
    tagged.sort(key=lambda segment: (-len(segment.domain_tags), segment.segment_id))
    return {segment.segment_id for segment in tagged[: settings.max_annotations_per_doc]}


def document_preamble(text: str, *, settings: EnvSettings | None = None) -> str:
    settings = settings or EnvSettings()
    # Token budget is approximate; supplement annotation only needs a stable prefix.
    words = text.split()
    return " ".join(words[: settings.annotation_preamble_tokens])
