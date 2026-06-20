"""In-memory hybrid retrieval index for supplementary material."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from arbiter.config import EnvSettings
from arbiter.models import SupplementSegment

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class SupplementIndex:
    """In-memory sparse+dense index over supplement segments."""

    def __init__(
        self,
        segments: Sequence[SupplementSegment] | None = None,
        *,
        dense_encoder: Callable[[list[str]], list[list[float]]] | None = None,
        settings: EnvSettings | None = None,
    ) -> None:
        self.segments = list(segments or [])
        self.settings = settings or EnvSettings()
        self._tokens = [_tokenize(segment.annotated_text) for segment in self.segments]
        self._doc_freqs = _document_frequencies(self._tokens)
        self._avg_doc_len = sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 0.0
        self._dense_encoder = dense_encoder
        self._dense_vectors: list[list[float]] | None = None
        if self.segments:
            self._dense_vectors = self._encode_dense([segment.annotated_text for segment in self.segments])

    @classmethod
    def empty(cls) -> "SupplementIndex":
        return cls([])

    def retrieve(
        self,
        query_terms: list[str],
        domain: str,
        top_k: int = 5,
    ) -> tuple[list[SupplementSegment], float | None]:
        if not self.segments or top_k <= 0:
            return [], None

        candidate_indices = [idx for idx, segment in enumerate(self.segments) if domain in segment.domain_tags]
        if len(candidate_indices) < 2:
            candidate_indices = list(range(len(self.segments)))

        query = " ".join(query_terms)
        bm25_scores = self._bm25_scores(query, candidate_indices)
        dense_scores = self._dense_scores(query, candidate_indices)
        fused_indices = _rrf_rank(candidate_indices, bm25_scores, dense_scores)
        selected_indices = fused_indices[:top_k]
        if not selected_indices:
            return [], None

        normalised_bm25 = _normalise_scores(bm25_scores)
        normalised_dense = _normalise_scores(dense_scores)
        top_idx = selected_indices[0]
        top_score = max(normalised_bm25.get(top_idx, 0.0), normalised_dense.get(top_idx, 0.0))
        return [self.segments[idx] for idx in selected_indices], top_score

    def _bm25_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {idx: 0.0 for idx in candidate_indices}

        scores: dict[int, float] = {}
        doc_count = len(self._tokens)
        k1 = 1.5
        b = 0.75
        for idx in candidate_indices:
            tokens = self._tokens[idx]
            counts = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for token in query_tokens:
                freq = counts[token]
                if freq == 0:
                    continue
                doc_freq = self._doc_freqs.get(token, 0)
                idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                denominator = freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1.0))
                score += idf * (freq * (k1 + 1)) / denominator
            scores[idx] = score
        return scores

    def _dense_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        if not query.strip() or self._dense_vectors is None:
            return {idx: 0.0 for idx in candidate_indices}
        query_vector = self._encode_dense([query])[0]
        return {idx: _cosine(query_vector, self._dense_vectors[idx]) for idx in candidate_indices}

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        if self._dense_encoder is not None:
            return self._dense_encoder(texts)
        return [_hash_embedding(text) for text in texts]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _document_frequencies(documents: list[list[str]]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _rrf_rank(
    candidate_indices: list[int],
    bm25_scores: dict[int, float],
    dense_scores: dict[int, float],
    *,
    k: int = 60,
) -> list[int]:
    rrf_scores = {idx: 0.0 for idx in candidate_indices}
    for scores in (bm25_scores, dense_scores):
        ranked = sorted(candidate_indices, key=lambda idx: (-scores.get(idx, 0.0), idx))
        for rank, idx in enumerate(ranked, start=1):
            rrf_scores[idx] += 1 / (k + rank)
    return sorted(candidate_indices, key=lambda idx: (-rrf_scores[idx], idx))


def _normalise_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {idx: 1.0 if value > 0 else 0.0 for idx, value in scores.items()}
    return {idx: (value - low) / (high - low) for idx, value in scores.items()}


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _hash_embedding(text: str, *, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[idx] += sign
    return vector
