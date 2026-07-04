"""Docling-metadata-backed in-memory hybrid retrieval for supplements."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import bm25s

from arbiter.config import EnvSettings
from arbiter.models import DocType, SupplementSegment

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
LOW_YIELD_DOC_TYPES = {DocType.DISCLOSURE, DocType.ADMINISTRATIVE}
TABLE_RETRIEVAL_DOMAINS = {"D3", "D5"}
TABLE_QUERY_TERMS = {
    "adverse",
    "ae",
    "attrition",
    "censor",
    "missing",
    "subgroup",
    "table",
    "withdraw",
    "withdrawal",
}
DOMAIN_METADATA_TERMS: dict[str, tuple[str, ...]] = {
    "D1": ("random", "allocation", "conceal", "baseline", "sequence"),
    "D2": ("blind", "mask", "adherence", "deviation", "intervention"),
    "D3": ("missing", "withdraw", "lost", "censor", "participant flow", "attrition"),
    "D4": ("outcome", "endpoint", "assessment", "adjudication", "measurement"),
    "D5": ("protocol", "statistical analysis", "sap", "outcome", "subgroup", "registry"),
}


class SupplementIndex:
    """In-memory sparse+dense index over Docling supplement chunks."""

    def __init__(
        self,
        segments: Sequence[SupplementSegment] | None = None,
        *,
        dense_encoder: Callable[[list[str]], list[list[float]]] | None = None,
        dense_backend: DenseEmbeddingBackend | None = None,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        settings: EnvSettings | None = None,
    ) -> None:
        self.segments = list(segments or [])
        self.settings = settings or EnvSettings()
        self._tokens = [_tokenize(segment.raw_text) for segment in self.segments]
        self._bm25 = bm25s.BM25(k1=1.5, b=0.75)
        if self.segments:
            self._bm25.index(self._tokens, show_progress=False)
        self._dense_encoder = dense_encoder
        self._dense_backend = dense_backend
        self._reranker = reranker
        self._dense_vectors: list[list[float]] | None = None
        if self.segments:
            dense_vectors = self._encode_dense_documents([_embedding_text(segment) for segment in self.segments])
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
            return _empty_result()

        candidate_indices = list(range(len(self.segments)))
        selectable_indices = self._selectable_candidate_indices(candidate_indices)
        query = " ".join(query_terms)
        bm25_scores = self._bm25_scores(query, candidate_indices)
        dense_scores = self._dense_scores(query, candidate_indices)
        metadata_scores = self._metadata_scores(query_terms, domain, candidate_indices)
        hybrid_scores = _hybrid_scores(candidate_indices, bm25_scores, dense_scores, metadata_scores)

        fused_indices = sorted(selectable_indices, key=lambda idx: (-hybrid_scores[idx], idx))
        reranker_scores = self._reranker_scores(query, fused_indices, top_k=top_k)
        if reranker_scores:
            selected_pool = list(reranker_scores)
            selected_indices = sorted(selected_pool, key=lambda idx: (-reranker_scores[idx], idx))[:top_k]
        else:
            selected_indices = fused_indices[:top_k]

        top_score = self._best_selected_relevance(selected_indices, dense_scores) if selected_indices else None
        return {
            "segments": [self.segments[idx] for idx in selected_indices],
            "top_score": top_score,
            "candidate_indices": candidate_indices,
            "selected_indices": selected_indices,
            "bm25_scores": bm25_scores,
            "dense_scores": dense_scores,
            "metadata_scores": metadata_scores,
            "hybrid_scores": hybrid_scores,
            "rrf_scores": {},
            "reranker_scores": reranker_scores,
            "suppressed_low_yield_indices": [
                idx for idx in candidate_indices if idx not in selectable_indices
            ],
        }

    def _selectable_candidate_indices(self, candidate_indices: list[int]) -> list[int]:
        high_yield = [
            idx
            for idx in candidate_indices
            if self.segments[idx].doc_type not in LOW_YIELD_DOC_TYPES
        ]
        return high_yield

    def _bm25_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.segments:
            return {idx: 0.0 for idx in candidate_indices}
        raw_scores = self._bm25.get_scores(query_tokens)
        return {idx: float(raw_scores[idx]) for idx in candidate_indices}

    def _dense_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        if not query.strip() or self._dense_vectors is None:
            return {idx: 0.0 for idx in candidate_indices}
        query_vector = self._encode_dense_query(query)
        if not query_vector:
            return {idx: 0.0 for idx in candidate_indices}
        return {idx: _cosine(query_vector, self._dense_vectors[idx]) for idx in candidate_indices}

    def _metadata_scores(
        self,
        query_terms: Sequence[str],
        domain: str,
        candidate_indices: list[int],
    ) -> dict[int, float]:
        query = " ".join(query_terms).lower()
        wants_table = domain in TABLE_RETRIEVAL_DOMAINS and any(term in query for term in TABLE_QUERY_TERMS)
        domain_terms = DOMAIN_METADATA_TERMS.get(domain, ())
        scores: dict[int, float] = {}
        for idx in candidate_indices:
            segment = self.segments[idx]
            labels = set(segment.doc_item_labels)
            metadata_text = f"{segment.heading}\n{' '.join(segment.doc_item_labels)}".lower()
            score = 0.0
            if wants_table and "table" in labels:
                score += 1.0
            if "table" in labels and domain in TABLE_RETRIEVAL_DOMAINS:
                score += 0.25
            score += 0.15 * sum(1 for term in domain_terms if term in metadata_text)
            scores[idx] = score
        return scores

    def _best_selected_relevance(
        self,
        selected_indices: Sequence[int],
        dense_scores: dict[int, float],
    ) -> float | None:
        """Best absolute dense relevance among selected passages for REQ-11."""

        if self._dense_vectors is None:
            return None
        relevances = [
            max(0.0, min(1.0, cosine))
            for idx in selected_indices
            if (cosine := dense_scores.get(idx)) is not None
        ]
        return max(relevances) if relevances else None

    def _encode_dense_documents(self, texts: list[str]) -> list[list[float]]:
        if self._dense_encoder is not None:
            return self._dense_encoder(texts)
        if self._dense_backend is not None:
            return self._dense_backend.encode_documents(texts)
        if self.settings.dense_embedding_model is None:
            return []
        try:
            self._dense_backend = sentence_transformer_backend(
                self.settings.dense_embedding_model,
                self.settings.dense_embedding_cache_path,
            )
        except Exception:
            return []
        return self._dense_backend.encode_documents(texts)

    def _encode_dense_query(self, query: str) -> list[float] | None:
        if self._dense_encoder is not None:
            vectors = self._dense_encoder([query])
            return vectors[0] if vectors else None
        if self._dense_backend is None:
            if self.settings.dense_embedding_model is None:
                return None
            try:
                self._dense_backend = sentence_transformer_backend(
                    self.settings.dense_embedding_model,
                    self.settings.dense_embedding_cache_path,
                )
            except Exception:
                return None
        vectors = self._dense_backend.encode_queries([query])
        return vectors[0] if vectors else None

    def _reranker_scores(self, query: str, fused_indices: list[int], *, top_k: int) -> dict[int, float]:
        if not query.strip() or not fused_indices:
            return {}
        pool_size = max(top_k, min(len(fused_indices), self.settings.dense_rerank_pool_size))
        pool_indices = fused_indices[:pool_size]
        try:
            reranker = self._reranker
            if reranker is None and self.settings.dense_reranker_model is not None:
                reranker = _cross_encoder_reranker(self.settings.dense_reranker_model)
                self._reranker = reranker
            if reranker is None:
                return {}
            scores = reranker(query, [self.segments[idx].raw_text for idx in pool_indices])
        except Exception:
            return {}
        return {
            idx: float(score)
            for idx, score in zip(pool_indices, scores, strict=False)
        }


def _empty_result() -> dict[str, Any]:
    return {
        "segments": [],
        "top_score": None,
        "candidate_indices": [],
        "selected_indices": [],
        "bm25_scores": {},
        "dense_scores": {},
        "metadata_scores": {},
        "rrf_scores": {},
        "reranker_scores": {},
        "suppressed_low_yield_indices": [],
    }


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _hybrid_scores(
    candidate_indices: list[int],
    bm25_scores: dict[int, float],
    dense_scores: dict[int, float],
    metadata_scores: dict[int, float],
) -> dict[int, float]:
    bm25_norm = _normalize_scores(bm25_scores, candidate_indices)
    dense_norm = _normalize_scores(
        {idx: max(0.0, dense_scores.get(idx, 0.0)) for idx in candidate_indices},
        candidate_indices,
    )
    return {
        idx: bm25_norm[idx] + dense_norm[idx] + metadata_scores.get(idx, 0.0)
        for idx in candidate_indices
    }


def _normalize_scores(scores: dict[int, float], candidate_indices: list[int]) -> dict[int, float]:
    values = [max(0.0, scores.get(idx, 0.0)) for idx in candidate_indices]
    maximum = max(values, default=0.0)
    if maximum <= 0.0:
        return {idx: 0.0 for idx in candidate_indices}
    return {idx: max(0.0, scores.get(idx, 0.0)) / maximum for idx in candidate_indices}


def _embedding_text(segment: SupplementSegment) -> str:
    value = segment.metadata.get("embedding_text")
    return str(value) if value else segment.raw_text


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class DenseEmbeddingBackend(Protocol):
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        ...


class _PersistentEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, list[float]] | None = None

    def get(self, key: str) -> list[float] | None:
        return self._load().get(key)

    def put_many(self, entries: dict[str, list[float]]) -> None:
        if not entries:
            return
        loaded = self._load()
        loaded.update(entries)
        self._write(loaded)

    def _load(self) -> dict[str, list[float]]:
        if self._entries is not None:
            return self._entries
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raw = {}
        self._entries = {
            str(key): [float(value) for value in vector]
            for key, vector in raw.items()
            if isinstance(vector, list)
        }
        return self._entries

    def _write(self, entries: dict[str, list[float]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temp_path.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")
            temp_path.replace(self.path)
        except OSError:
            return


class _SentenceTransformerBackend:
    def __init__(self, model_name: str, cache_path: Path) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = _PersistentEmbeddingCache(cache_path)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode("document", texts, self._model_encode_documents)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode("query", texts, self._model_encode_queries)

    def _encode(
        self,
        role: str,
        texts: list[str],
        encoder: Callable[[list[str]], Any],
    ) -> list[list[float]]:
        keys = [_embedding_cache_key(self.model_name, role, text) for text in texts]
        cached = [self.cache.get(key) for key in keys]
        missing_positions = [idx for idx, vector in enumerate(cached) if vector is None]
        if missing_positions:
            missing_texts = [texts[idx] for idx in missing_positions]
            encoded = _as_float_vectors(encoder(missing_texts))
            self.cache.put_many(
                {
                    keys[idx]: vector
                    for idx, vector in zip(missing_positions, encoded, strict=False)
                }
            )
            for idx, vector in zip(missing_positions, encoded, strict=False):
                cached[idx] = vector
        return [vector for vector in cached if vector is not None]

    def _model_encode_documents(self, texts: list[str]) -> Any:
        encode_document = getattr(self.model, "encode_document", None)
        if encode_document is not None:
            return encode_document(texts)
        return self.model.encode(texts)

    def _model_encode_queries(self, texts: list[str]) -> Any:
        encode_query = getattr(self.model, "encode_query", None)
        if encode_query is not None:
            return encode_query(texts)
        return self.model.encode(texts)


def _embedding_cache_key(model_name: str, role: str, text: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{role}:{content_hash}"


def _as_float_vectors(embeddings: Any) -> list[list[float]]:
    return [list(map(float, embedding)) for embedding in embeddings]


def sentence_transformer_backend(model_name: str, cache_path: Path) -> DenseEmbeddingBackend:
    return _SentenceTransformerBackend(model_name, cache_path)


def _cross_encoder_reranker(model_name: str) -> Callable[[str, list[str]], list[float]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def rerank(query: str, passages: list[str]) -> list[float]:
        ranks = getattr(model, "rank", None)
        if ranks is not None:
            ranked = ranks(query, passages)
            scores = [0.0 for _ in passages]
            for rank in ranked:
                corpus_id = int(rank["corpus_id"])
                scores[corpus_id] = float(rank["score"])
            return scores
        return [float(score) for score in model.predict([(query, passage) for passage in passages])]

    return rerank
