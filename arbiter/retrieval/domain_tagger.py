"""Semantic RoB 2 domain tagging for supplementary material."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from arbiter.config import EnvSettings
from arbiter.ingestion.paper import DOMAIN_TAGS


DOMAIN_PROTOTYPES: dict[str, str] = {
    "D1": (
        "Risk of bias domain 1: randomisation process, random allocation sequence, "
        "allocation concealment, assignment generation, baseline imbalances."
    ),
    "D2": (
        "Risk of bias domain 2: deviations from intended interventions, blinding or "
        "masking of participants and personnel, adherence, compliance, protocol "
        "deviations, non-protocol interventions, intention-to-treat analysis."
    ),
    "D3": (
        "Risk of bias domain 3: missing outcome data, incomplete follow-up, attrition, "
        "withdrawal, dropout, censoring, imputation, sensitivity analyses for missingness."
    ),
    "D4": (
        "Risk of bias domain 4: measurement of the outcome, outcome assessment method, "
        "assessor awareness, adjudication, endpoint review, measurement timing, thresholds, "
        "participant-reported or clinician-assessed outcomes."
    ),
    "D5": (
        "Risk of bias domain 5: selection of the reported result, prespecified outcomes, "
        "protocol or registry outcomes, statistical analysis plan, multiple measurements, "
        "multiple time points, multiple analysis methods."
    ),
}


class DomainTagEmbeddingBackend(Protocol):
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class SemanticDomainTagger:
    """Assign supplement domain tags by segment/prototype embedding similarity."""

    backend: DomainTagEmbeddingBackend
    threshold: float = 0.20
    margin: float = 0.04
    max_tags: int = 2

    def tag(self, heading: str, text: str) -> list[str]:
        return self.tag_many([(heading, text)])[0]

    def tag_many(self, segments: Sequence[tuple[str, str]]) -> list[list[str]]:
        texts_to_score = [
            f"{heading}\n{text}".strip()
            for heading, text in segments
        ]
        if not texts_to_score:
            return []

        prototype_vectors = self.backend.encode_queries(
            [DOMAIN_PROTOTYPES[domain] for domain in DOMAIN_TAGS]
        )
        document_vectors = self.backend.encode_documents(texts_to_score)
        if len(prototype_vectors) != len(DOMAIN_TAGS) or len(document_vectors) != len(texts_to_score):
            return [[] for _ in texts_to_score]

        tags: list[list[str]] = []
        for document_vector in document_vectors:
            scores = {
                domain: _cosine(document_vector, prototype_vector)
                for domain, prototype_vector in zip(DOMAIN_TAGS, prototype_vectors, strict=True)
            }
            tags.append(
                _select_domain_tags(
                    scores,
                    threshold=self.threshold,
                    margin=self.margin,
                    max_tags=self.max_tags,
                )
            )
        return tags

def tag_segments_semantically(
    segments: Sequence,
    *,
    backend: DomainTagEmbeddingBackend | None,
    settings: EnvSettings,
) -> list:
    """Return copies of segments with semantic domain tags.

    If no dense backend is configured or semantic scoring fails, all RoB 2 tags are
    assigned as a neutral soft-boost signal. Retrieval no longer hard-filters by
    these tags, so this preserves recall without falling back to keyword lexicons.
    """

    if not segments:
        return []
    if backend is None:
        return [
            segment.model_copy(update={"domain_tags": list(DOMAIN_TAGS)})
            for segment in segments
        ]

    tagger = SemanticDomainTagger(
        backend=backend,
        threshold=settings.domain_tag_similarity_threshold,
        margin=settings.domain_tag_similarity_margin,
        max_tags=settings.domain_tag_max_tags,
    )
    tagged = []
    try:
        tags_by_segment = tagger.tag_many(
            [(segment.heading, segment.raw_text) for segment in segments]
        )
        for segment, tags in zip(segments, tags_by_segment, strict=True):
            tagged.append(segment.model_copy(update={"domain_tags": tags or list(DOMAIN_TAGS)}))
    except Exception:
        return [
            segment.model_copy(update={"domain_tags": list(DOMAIN_TAGS)})
            for segment in segments
        ]
    return tagged


def _select_domain_tags(
    scores: dict[str, float],
    *,
    threshold: float,
    margin: float,
    max_tags: int,
) -> list[str]:
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_score = ordered[0][1]
    cutoff = max(threshold, top_score - margin)
    selected = [
        domain
        for domain, score in ordered
        if score >= cutoff
    ][: max(1, max_tags)]
    return selected or [ordered[0][0]]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
