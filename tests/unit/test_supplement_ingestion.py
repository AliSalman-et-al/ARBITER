from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

import pytest
from langchain_core.documents import Document

from arbiter.config import EnvSettings
from arbiter.ingestion.supplements import detect_document_type, ingest_supplements
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import DocType, SupplementSegment
from arbiter.retrieval.supplement_index import SupplementIndex


def _segment(
    segment_id: str,
    text: str,
    *,
    doc_type: DocType = DocType.PROTOCOL,
    heading: str = "Methods",
    labels: list[str] | None = None,
    pages: list[int] | None = None,
) -> SupplementSegment:
    return SupplementSegment(
        segment_id=segment_id,
        source_file=f"{segment_id}.pdf",
        doc_type=doc_type,
        heading=heading,
        pages=pages or [0],
        raw_text=text,
        domain_tags=[],
        doc_item_labels=labels or ["text"],
        char_count=len(text),
    )


def _doc(
    text: str,
    *,
    headings: list[str] | None = None,
    labels: list[str] | None = None,
    page_no: int = 1,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "supplement.pdf",
            "dl_meta": {
                "headings": headings or [],
                "doc_items": [
                    {
                        "label": label,
                        "prov": [{"page_no": page_no, "bbox": {"l": 0, "t": 0, "r": 1, "b": 1}}],
                    }
                    for label in (labels or ["text"])
                ],
            },
        },
    )


def test_detect_document_type_uses_filename_and_docling_chunk_text(tmp_path: Path) -> None:
    detected = detect_document_type(
        tmp_path / "nejmoa1503747_disclosures.pdf",
        "Authors report institutional grants and conflict of interest disclosures.",
    )

    assert detected is DocType.DISCLOSURE


@pytest.mark.asyncio
async def test_ingest_supplements_empty_paths_returns_empty_index() -> None:
    client = MockLLMClient()

    index = await ingest_supplements([], client)

    assert isinstance(index, SupplementIndex)
    assert index.retrieve(["concealment"], "D1") == ([], None)


@pytest.mark.asyncio
async def test_ingest_supplements_maps_langchain_docling_chunks_to_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    supplement_dir = tmp_path / "supplements"
    supplement_dir.mkdir()
    supplement_path = supplement_dir / "protocol.pdf"
    supplement_path.write_bytes(b"fake")
    docs = [
        _doc(
            "Randomisation\nThe allocation concealment method used an IWRS system.",
            headings=["Randomisation"],
            labels=["section_header", "text"],
            page_no=3,
        ),
        _doc(
            "Missing Data\nMissing survival data were handled with sensitivity analyses.",
            headings=["Missing Data"],
            labels=["table"],
            page_no=7,
        ),
    ]
    monkeypatch.setattr("arbiter.ingestion.supplements.load_docling_chunks", lambda _path, _settings: docs)
    client = MockLLMClient()

    index = await ingest_supplements([supplement_dir], client)

    assert client.calls == []
    assert len(index.segments) == 2
    assert index.segments[0].heading == "Randomisation"
    assert index.segments[0].pages == [2]
    assert index.segments[1].doc_item_labels == ["table"]
    assert index.segments[1].metadata["docling"]["headings"] == ["Missing Data"]


@pytest.mark.asyncio
async def test_ingest_supplements_skips_low_yield_disclosure_at_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "coi.pdf"
    path.write_bytes(b"fake")
    docs = [
        _doc(
            "Conflict of Interest Disclosure Statement\nThe authors disclose consulting fees. Randomisation appears only in the article title.",
            headings=["Conflict of Interest Disclosure Statement"],
        )
    ]
    monkeypatch.setattr("arbiter.ingestion.supplements.load_docling_chunks", lambda _path, _settings: docs)

    index = await ingest_supplements([path], MockLLMClient())
    retrieval = index.retrieve_with_metadata(["randomisation"], "D1", top_k=5)

    assert {segment.doc_type for segment in index.segments} == {DocType.DISCLOSURE}
    assert retrieval["candidate_indices"] == [0]
    assert retrieval["segments"] == []
    assert retrieval["suppressed_low_yield_indices"] == [0]


def test_docling_table_metadata_boosts_table_chunks_for_d3_queries() -> None:
    narrative = _segment(
        "appendix-narrative",
        "The appendix says missing data are summarized elsewhere.",
        heading="Missing Data",
        labels=["text"],
    )
    table = _segment(
        "appendix-table",
        "Arm | Missing outcome data\nADT | 3\nADT+D | 0",
        heading="Table S2 / Missing Data",
        labels=["table"],
    )

    index = SupplementIndex([narrative, table])
    result = index.retrieve_with_metadata(["missing outcome table"], "D3", top_k=1)

    assert result["segments"] == [table]
    assert result["metadata_scores"][1] > result["metadata_scores"][0]
    assert result["rrf_scores"] == {}


def test_default_dense_arm_uses_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, list[str]]] = []
    module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts: list[str]) -> list[list[float]]:
            calls.append((self.model_name, texts))
            return [
                [1.0, 0.0] if "allocation concealment" in text.lower() else [0.0, 1.0]
                for text in texts
            ]

    setattr(module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    central_randomisation = _segment(
        "sap-1",
        "Participants used allocation concealment through a central web service.",
        doc_type=DocType.SAP,
    )
    unrelated = _segment(
        "sap-2",
        "Overall survival was summarized with Kaplan-Meier curves.",
        doc_type=DocType.SAP,
    )

    settings = EnvSettings()
    settings.dense_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    settings.dense_embedding_cache_path = tmp_path / "embeddings.json"
    index = SupplementIndex([unrelated, central_randomisation], settings=settings)
    result = index.retrieve_with_metadata(["allocation concealment"], "D1", top_k=1)

    assert result["segments"] == [central_randomisation]
    assert result["top_score"] == pytest.approx(1.0)
    assert calls[0][1] == [unrelated.raw_text, central_randomisation.raw_text]


def test_dense_backend_uses_asymmetric_document_and_query_encoding() -> None:
    class FakeDenseBackend:
        def __init__(self) -> None:
            self.document_calls: list[list[str]] = []
            self.query_calls: list[list[str]] = []

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls.append(texts)
            return [
                [1.0, 0.0] if "allocation sequence" in text.lower() else [0.0, 1.0]
                for text in texts
            ]

        def encode_queries(self, texts: list[str]) -> list[list[float]]:
            self.query_calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

    backend = FakeDenseBackend()
    allocation_sequence = _segment("protocol-allocation", "The allocation sequence was concealed until assignment.")
    unrelated = _segment("protocol-analysis", "Overall survival analyses used Cox proportional hazards models.")

    index = SupplementIndex([unrelated, allocation_sequence], dense_backend=backend)
    result = index.retrieve_with_metadata(["How was allocation concealed?"], "D1", top_k=1)

    assert backend.document_calls == [[unrelated.raw_text, allocation_sequence.raw_text]]
    assert backend.query_calls == [["How was allocation concealed?"]]
    assert result["segments"] == [allocation_sequence]
    assert result["top_score"] == pytest.approx(1.0)


def test_reranker_reorders_hybrid_candidate_pool() -> None:
    first_stage_lexical_match = _segment(
        "appendix-caption",
        "Allocation allocation allocation caption for an unrelated figure.",
        doc_type=DocType.APPENDIX,
    )
    best_passage = _segment(
        "protocol-methods",
        "Randomisation used a central allocation sequence with concealment.",
    )

    def reranker(_query: str, passages: list[str]) -> list[float]:
        return [10.0 if "central allocation sequence" in passage else 0.1 for passage in passages]

    index = SupplementIndex(
        [first_stage_lexical_match, best_passage],
        reranker=reranker,
        settings=EnvSettings(),
    )

    result = index.retrieve_with_metadata(["allocation"], "D1", top_k=1)

    assert result["selected_indices"] == [1]
    assert result["segments"] == [best_passage]
    assert result["reranker_scores"][1] == pytest.approx(10.0)


def test_sentence_transformer_backend_caches_by_role_and_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, list[str]]] = []
    module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode_document(self, texts: list[str]) -> list[list[float]]:
            calls.append(("document", texts))
            return [[float(len(text)), 0.0] for text in texts]

        def encode_query(self, texts: list[str]) -> list[list[float]]:
            calls.append(("query", texts))
            return [[0.0, float(len(text))] for text in texts]

    setattr(module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    segment = _segment("protocol-methods", "Randomisation used concealed allocation.")
    settings = EnvSettings()
    settings.dense_embedding_model = "test/asymmetric-model"
    settings.dense_embedding_cache_path = tmp_path / "embeddings.json"

    first = SupplementIndex([segment], settings=settings)
    first.retrieve_with_metadata(["allocation concealment"], "D1", top_k=1)
    second = SupplementIndex([segment], settings=settings)
    second.retrieve_with_metadata(["allocation concealment"], "D1", top_k=1)

    assert calls == [
        ("document", [segment.raw_text]),
        ("query", ["allocation concealment"]),
    ]
    assert settings.dense_embedding_cache_path.exists()


def test_retrieval_top_score_uses_best_selected_dense_relevance() -> None:
    def dense_encoder(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "randomisation procedures" in lowered:
                vectors.append([0.6, 0.8])
            elif lowered == "query":
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.3, (1 - 0.3**2) ** 0.5])
        return vectors

    noisy_bm25_match = _segment("appendix-figure", "Query query query caption about a Kaplan-Meier curve.")
    randomisation_procedures = _segment(
        "protocol-randomisation",
        "Randomisation procedures used a central allocation service.",
    )
    index = SupplementIndex(
        [noisy_bm25_match, randomisation_procedures],
        dense_encoder=dense_encoder,
    )

    result = index.retrieve_with_metadata(["query"], "D1", top_k=2)

    assert result["selected_indices"] == [0, 1]
    assert result["dense_scores"][0] == pytest.approx(0.3)
    assert result["dense_scores"][1] == pytest.approx(0.6)
    assert result["top_score"] == pytest.approx(0.6)
