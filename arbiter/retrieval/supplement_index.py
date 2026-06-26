"""In-memory hybrid retrieval index for supplementary material."""

from __future__ import annotations

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
            dense_vectors = self._encode_dense([segment.annotated_text for segment in self.segments])
            self._dense_vectors = dense_vectors or None

    @classmethod
    def empty(cls) -> "SupplementIndex":
        return cls([])

    def retrieve(
        self,
        query_terms: list[str],
        domain: str,
        top_k: int = 5,
    ) -> tuple[list[SupplementSegment], float | None]:
        result = self.retrieve_with_metadata(query_terms, domain, top_k=top_k)
        return result["segments"], result["top_score"]

    def retrieve_with_metadata(
        self,
        query_terms: list[str],
        domain: str,
        top_k: int = 5,
    ) -> dict:
        if not self.segments or top_k <= 0:
            return {
                "segments": [],
                "top_score": None,
                "candidate_indices": [],
                "selected_indices": [],
                "bm25_scores": {},
                "dense_scores": {},
                "rrf_scores": {},
            }

        candidate_indices = [idx for idx, segment in enumerate(self.segments) if domain in segment.domain_tags]
        if len(candidate_indices) < 2:
            candidate_indices = list(range(len(self.segments)))

        query = " ".join(query_terms)
        bm25_scores = self._bm25_scores(query, candidate_indices)
        dense_scores = self._dense_scores(query, candidate_indices)
        rrf_scores = _rrf_scores(candidate_indices, bm25_scores, dense_scores)
        fused_indices = sorted(candidate_indices, key=lambda idx: (-rrf_scores[idx], idx))
        selected_indices = fused_indices[:top_k]
        if not selected_indices:
            top_score = None
        else:
            top_idx = selected_indices[0]
            top_score = self._top_relevance(top_idx, dense_scores)

        return {
            "segments": [self.segments[idx] for idx in selected_indices],
            "top_score": top_score,
            "candidate_indices": candidate_indices,
            "selected_indices": selected_indices,
            "bm25_scores": bm25_scores,
            "dense_scores": dense_scores,
            "rrf_scores": rrf_scores,
        }

    def _top_relevance(self, top_idx: int, dense_scores: dict[int, float]) -> float | None:
        """Absolute relevance magnitude of the top passage for REQ-11.

        The dense cosine similarity of the RRF-top passage to the query, clamped
        to [0, 1]. This is an ABSOLUTE scale comparable across queries, unlike a
        min-max value over the candidate set, which pins the top passage to ~1.0
        and makes the REQ-11 UNCERTAIN/FLAGGED score thresholds dead. Returns None
        when no dense signal is available (BM25-only arm); the REQ-11 score-based
        clauses then correctly do not fire. RRF stays the ranking mechanism; only
        this surfaced confidence score is the absolute magnitude.
        """
        if self._dense_vectors is None:
            return None
        cosine = dense_scores.get(top_idx)
        if cosine is None:
            return None
        return max(0.0, min(1.0, cosine))

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
        query_vectors = self._encode_dense([query])
        if not query_vectors:
            return {idx: 0.0 for idx in candidate_indices}
        query_vector = query_vectors[0]
        return {idx: _cosine(query_vector, self._dense_vectors[idx]) for idx in candidate_indices}

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        if self._dense_encoder is not None:
            return self._dense_encoder(texts)
        if self.settings.dense_embedding_model is None:
            return []
        try:
            self._dense_encoder = _sentence_transformer_encoder(self.settings.dense_embedding_model)
        except Exception:
            return []
        return self._dense_encoder(texts)


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
    rrf_scores = _rrf_scores(candidate_indices, bm25_scores, dense_scores, k=k)
    return sorted(candidate_indices, key=lambda idx: (-rrf_scores[idx], idx))


def _rrf_scores(
    candidate_indices: list[int],
    bm25_scores: dict[int, float],
    dense_scores: dict[int, float],
    *,
    k: int = 60,
) -> dict[int, float]:
    rrf_scores = {idx: 0.0 for idx in candidate_indices}
    for scores in (bm25_scores, dense_scores):
        ranked = sorted(
            [idx for idx in candidate_indices if scores.get(idx, 0.0) > 0.0],
            key=lambda idx: (-scores.get(idx, 0.0), idx),
        )
        for rank, idx in enumerate(ranked, start=1):
            rrf_scores[idx] += 1 / (k + rank)
    return rrf_scores


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _sentence_transformer_encoder(model_name: str) -> Callable[[list[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(texts)
        return [list(map(float, embedding)) for embedding in embeddings]

    return encode
