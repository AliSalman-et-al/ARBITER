from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
import sys

import pymupdf
import pytest

from arbiter.ingestion.supplements import (
    _page_text,
    _parse_pdf_window,
    _parse_pdf_windows,
    ingest_supplements,
)
from arbiter.llm.mock_client import MockLLMClient
from arbiter.config import EnvSettings
from arbiter.models import DocType, PageBox, SupplementSegment
from arbiter.retrieval.segmenter import ParsedSupplementWindow, detect_document_type, segment_document
from arbiter.retrieval.supplement_index import SupplementIndex


def _write_supplement_pdf(path: Path, sections: list[tuple[str, str]]) -> None:
    doc = pymupdf.open()
    for heading, body in sections:
        page = doc.new_page()
        page.insert_text((72, 72), heading, fontsize=16)
        page.insert_text((72, 120), body, fontsize=11)
    doc.save(path)
    doc.close()


def _semantic_test_encoder(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        if "central web-based randomisation" in lowered or "allocation concealment" in lowered:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


class _FakeDenseBackend:
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


def _box(text: str, page: int, boxclass: str = "section-header") -> PageBox:
    return PageBox(
        boxclass=boxclass,
        text=text,
        bbox=(0.0, 0.0, 0.0, 0.0),
        page=page,
    )


class _FakeStructuredPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, *_args: Any) -> str:
        return self._text

    def find_tables(self, *_args: Any, **_kwargs: Any) -> Any:
        class _NoTables:
            tables: list[Any] = []

        return _NoTables()


def test_sparse_structured_page_appends_generic_spatial_fallback() -> None:
    page = _FakeStructuredPage(
        "\n".join(
            [
                "Figure 2: Participant Flow",
                "Group Alpha randomized n=120",
                "Lost to follow-up n=4",
                "Withdrew consent n=2",
                "Group Beta randomized n=118",
                "Lost to follow-up n=1",
                "Withdrew consent n=5",
            ]
        )
    )
    markdown_chunks = [{"text": "**Figure 2: Participant Flow**\n\n"}]

    text = _page_text(page, 0, markdown_chunks)

    assert "Spatial text fallback:" in text
    assert "Group Alpha randomized n=120" in text
    assert "Group Beta randomized n=118" in text
    assert "Lost to follow-up n=4" in text
    assert "Withdrew consent n=5" in text


@pytest.mark.asyncio
async def test_ingest_supplements_empty_paths_returns_empty_index() -> None:
    client = MockLLMClient()

    index = await ingest_supplements([], client)

    assert isinstance(index, SupplementIndex)
    assert index.retrieve(["concealment"], "D1") == ([], None)


@pytest.mark.asyncio
async def test_ingest_supplements_expands_directory_and_retrieves_top_k(
    tmp_path: Path,
) -> None:
    supplement_dir = tmp_path / "supplements"
    supplement_dir.mkdir()
    _write_supplement_pdf(
        supplement_dir / "sap.pdf",
        [
            (
                "Statistical Analysis Plan",
                "The allocation concealment method used an IWRS system.",
            ),
            (
                "Missing Data",
                "Missing overall survival data were handled with sensitivity analyses.",
            ),
            (
                "Outcome Assessment",
                "The endpoint committee was blinded to treatment assignment.",
            ),
        ],
    )
    client = MockLLMClient()

    index = await ingest_supplements([supplement_dir], client)
    segments, score = index.retrieve(["concealment", "allocation"], "D1", top_k=5)

    assert client.calls == []
    assert len(segments) <= 5
    assert segments
    assert score is None or 0.0 <= score <= 1.0
    assert all(segment.raw_text.strip() for segment in index.segments)
    assert "allocation concealment" in segments[0].raw_text.lower()


@pytest.mark.asyncio
async def test_ingest_supplements_skips_low_yield_disclosure_annotation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coi.pdf"
    _write_supplement_pdf(
        path,
        [
            (
                "Conflict of Interest Disclosure Statement",
                "The authors disclose consulting fees and institutional grants. "
                "The form mentions randomisation only in the article title.",
            ),
            (
                "Copyright and Licence",
                "This administrative page describes reuse permissions and publisher licence terms.",
            ),
        ],
    )
    client = MockLLMClient(responses={})

    index = await ingest_supplements([path], client)

    assert client.calls == []
    assert index.segments
    assert {segment.doc_type for segment in index.segments} == {DocType.DISCLOSURE}
    retrieval = index.retrieve_with_metadata(["randomisation"], "D1", top_k=5)
    assert retrieval["candidate_indices"] == [0]
    assert retrieval["segments"] == []
    assert retrieval["suppressed_low_yield_indices"] == [0]


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
            return _semantic_test_encoder(texts)

    setattr(module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    central_randomisation = SupplementSegment(
        segment_id="sap-1",
        source_file="sap.pdf",
        doc_type=DocType.SAP,
        heading="Randomisation",
        pages=[1],
        raw_text="Participants used a central web-based randomisation service.",
        domain_tags=["D1"],
        char_count=62,
    )
    unrelated = SupplementSegment(
        segment_id="sap-2",
        source_file="sap.pdf",
        doc_type=DocType.SAP,
        heading="Analysis",
        pages=[2],
        raw_text="Overall survival was summarized with Kaplan-Meier curves.",
        domain_tags=["D1"],
        char_count=57,
    )

    settings = EnvSettings()
    settings.dense_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    settings.dense_embedding_cache_path = tmp_path / "embeddings.json"
    index = SupplementIndex([unrelated, central_randomisation], settings=settings)
    result = index.retrieve_with_metadata(["allocation concealment"], "D1", top_k=1)

    assert result["segments"] == [central_randomisation]
    assert result["top_score"] == pytest.approx(1.0)
    assert calls[0][0] == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls[0][1] == [unrelated.raw_text, central_randomisation.raw_text]


def test_dense_backend_uses_asymmetric_document_and_query_encoding() -> None:
    backend = _FakeDenseBackend()
    allocation_sequence = SupplementSegment(
        segment_id="protocol-allocation",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Randomisation",
        pages=[4],
        raw_text="The allocation sequence was concealed until assignment.",
        domain_tags=["D1"],
        char_count=58,
    )
    unrelated = SupplementSegment(
        segment_id="protocol-analysis",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Analysis",
        pages=[12],
        raw_text="Overall survival analyses used Cox proportional hazards models.",
        domain_tags=["D1"],
        char_count=62,
    )

    index = SupplementIndex([unrelated, allocation_sequence], dense_backend=backend)
    result = index.retrieve_with_metadata(["How was allocation concealed?"], "D1", top_k=1)

    assert backend.document_calls == [[unrelated.raw_text, allocation_sequence.raw_text]]
    assert backend.query_calls == [["How was allocation concealed?"]]
    assert result["segments"] == [allocation_sequence]
    assert result["top_score"] == pytest.approx(1.0)


def test_reranker_reorders_hybrid_candidate_pool() -> None:
    first_stage_lexical_match = SupplementSegment(
        segment_id="appendix-caption",
        source_file="appendix.pdf",
        doc_type=DocType.APPENDIX,
        heading="Figure Caption",
        pages=[8],
        raw_text="Allocation allocation allocation caption for an unrelated figure.",
        domain_tags=["D1"],
        char_count=61,
    )
    best_passage = SupplementSegment(
        segment_id="protocol-methods",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Methods",
        pages=[5],
        raw_text="Randomisation used a central allocation sequence with concealment.",
        domain_tags=["D1"],
        char_count=65,
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


def test_domain_tag_miss_does_not_exclude_relevant_segment() -> None:
    tagged_irrelevant = [
        SupplementSegment(
            segment_id="protocol-visit-schedule",
            source_file="protocol.pdf",
            doc_type=DocType.PROTOCOL,
            heading="Visit Schedule",
            pages=[5],
            raw_text="Clinic visits were scheduled every twelve weeks during treatment.",
            domain_tags=["D4"],
            char_count=64,
        ),
        SupplementSegment(
            segment_id="protocol-safety",
            source_file="protocol.pdf",
            doc_type=DocType.PROTOCOL,
            heading="Safety Monitoring",
            pages=[6],
            raw_text="Adverse events were summarized by arm and severity.",
            domain_tags=["D4"],
            char_count=60,
        ),
    ]
    paraphrased_relevant = SupplementSegment(
        segment_id="appendix-adjudication",
        source_file="appendix.pdf",
        doc_type=DocType.APPENDIX,
        heading="Endpoint Review",
        pages=[9],
        raw_text="An independent endpoint committee blinded to treatment assignment reviewed outcomes.",
        domain_tags=[],
        char_count=78,
    )

    index = SupplementIndex([*tagged_irrelevant, paraphrased_relevant])

    result = index.retrieve_with_metadata(
        ["endpoint committee blinded treatment assignment"],
        "D4",
        top_k=1,
    )

    assert result["candidate_indices"] == [0, 1, 2]
    assert result["segments"] == [paraphrased_relevant]


def test_domain_tag_is_soft_boost_for_relevant_segments() -> None:
    untagged_match = SupplementSegment(
        segment_id="appendix-unclassified",
        source_file="appendix.pdf",
        doc_type=DocType.APPENDIX,
        heading="Outcome Review",
        pages=[9],
        raw_text="The endpoint committee reviewed outcomes.",
        domain_tags=[],
        char_count=40,
    )
    tagged_match = SupplementSegment(
        segment_id="protocol-tagged",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Outcome Assessment",
        pages=[7],
        raw_text="The endpoint committee reviewed outcomes.",
        domain_tags=["D4"],
        char_count=40,
    )

    index = SupplementIndex([untagged_match, tagged_match])

    result = index.retrieve_with_metadata(
        ["endpoint committee reviewed outcomes"],
        "D4",
        top_k=1,
    )

    assert result["segments"] == [tagged_match]
    assert result["rrf_scores"][1] > result["rrf_scores"][0]


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

    segment = SupplementSegment(
        segment_id="protocol-methods",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Methods",
        pages=[5],
        raw_text="Randomisation used concealed allocation.",
        domain_tags=["D1"],
        char_count=39,
    )
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

    noisy_bm25_match = SupplementSegment(
        segment_id="appendix-figure",
        source_file="appendix.pdf",
        doc_type=DocType.APPENDIX,
        heading="Figure S2B",
        pages=[7],
        raw_text="Query query query caption about a Kaplan-Meier curve.",
        domain_tags=["D1"],
        char_count=52,
    )
    randomisation_procedures = SupplementSegment(
        segment_id="protocol-randomisation",
        source_file="protocol.pdf",
        doc_type=DocType.PROTOCOL,
        heading="Randomisation Procedures",
        pages=[3],
        raw_text="Randomisation procedures used a central allocation service.",
        domain_tags=["D1"],
        char_count=61,
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


@pytest.mark.asyncio
async def test_document_with_no_section_headers_yields_one_full_document_segment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plain.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This supplement reports central randomisation and allocation concealment without explicit headings.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    client = MockLLMClient()

    index = await ingest_supplements([path], client)

    assert client.calls == []
    assert len(index.segments) == 1
    assert index.segments[0].heading == "FULL_DOCUMENT"
    assert index.segments[0].domain_tags == ["D1", "D2", "D3", "D4", "D5"]


def test_segment_document_collapses_too_few_segments_to_full_document(
    tmp_path: Path,
) -> None:
    window = ParsedSupplementWindow(
        full_text="No obvious heading text.",
        page_starts=[0],
        page_boxes=[],
        page_offset=0,
    )

    segments = segment_document(
        tmp_path / "appendix.pdf", [window], doc_type=DocType.APPENDIX
    )

    assert len(segments) == 1
    assert segments[0].heading == "FULL_DOCUMENT"


def test_detect_document_type_uses_body_text_when_headers_are_unreliable(
    tmp_path: Path,
) -> None:
    boxes = [
        _box("Form generated by journal submission system", 0, boxclass="text"),
        _box(
            "Disclosure Statement: authors report institutional grants and conflict of interest disclosures.",
            0,
            boxclass="text",
        ),
    ]

    detected = detect_document_type(
        boxes,
        source_file=tmp_path / "nejmoa1503747_disclosures.pdf",
        settings=EnvSettings(),
    )

    assert detected.doc_type is DocType.DISCLOSURE


def test_segment_document_merges_short_form_like_fragments(tmp_path: Path) -> None:
    settings = EnvSettings()
    settings.min_segments = 1
    settings.min_supplement_segment_chars = 120
    window = ParsedSupplementWindow(
        full_text=(
            "Study Procedures\n"
            "Participants were assigned centrally with concealed allocation. "
            "The protocol specified follow-up visits and outcome capture.\n"
            "FAX\n"
            "FROM\n"
            "NOTE\n"
            "Missing-data procedures remained in the main study procedure section.\n"
            "Statistical Analysis\n"
            "The analysis population and censoring rules were prespecified. "
            "Sensitivity analyses were planned for incomplete outcome data."
        ),
        page_starts=[0],
        page_boxes=[
            _box("Study Procedures", 0),
            _box("FAX", 0),
            _box("FROM", 0),
            _box("NOTE", 0),
            _box("Statistical Analysis", 0),
        ],
    )

    segments = segment_document(
        tmp_path / "protocol.pdf",
        [window],
        doc_type=DocType.PROTOCOL,
        settings=settings,
    )

    assert [segment.heading for segment in segments] == [
        "STUDY PROCEDURES",
        "STATISTICAL ANALYSIS",
    ]
    assert "Missing-data procedures" in segments[0].raw_text


def test_segment_document_collapses_when_heading_count_exceeds_cap(tmp_path: Path) -> None:
    settings = EnvSettings()
    settings.min_segments = 1
    settings.max_supplement_segments_per_doc = 5
    settings.min_supplement_segment_chars = 0
    full_text = "\n".join(
        f"Section {idx}\nThis section contains enough body text to stand alone."
        for idx in range(8)
    )
    window = ParsedSupplementWindow(
        full_text=full_text,
        page_starts=[0],
        page_boxes=[_box(f"Section {idx}", 0) for idx in range(8)],
    )

    segments = segment_document(
        tmp_path / "overfragmented.pdf",
        [window],
        doc_type=DocType.UNKNOWN,
        settings=settings,
    )

    assert len(segments) == 1
    assert segments[0].heading == "FULL_DOCUMENT"


def test_chaarted_supplements_do_not_overfragment() -> None:
    settings = EnvSettings()
    supplement_dir = Path("eval/reference/pdfs/supplement/CHAARTED")
    total_segments = 0
    headings: list[str] = []
    for path in sorted(supplement_dir.glob("*.pdf")):
        windows = _parse_pdf_windows(path, settings)
        page_boxes = [box for window in windows for box in window.page_boxes]
        doc_type = detect_document_type(page_boxes, settings=settings).doc_type
        segments = segment_document(path, windows, doc_type=doc_type, settings=settings)
        total_segments += len(segments)
        headings.extend(segment.heading for segment in segments)

    assert total_segments < 200
    assert {"FAX", "FROM", "OR", "PAGE_1_OF_1"}.isdisjoint(headings)


def test_chaarted_consort_diagram_keeps_arm_resolved_fallback_text() -> None:
    settings = EnvSettings()
    path = Path("eval/reference/pdfs/supplement/CHAARTED/nejmoa1503747_appendix.pdf")

    text = "\n".join(window.full_text for window in _parse_pdf_windows(path, settings))

    assert "Figure S1: CONSORT Diagram" in text
    assert "Randomized to ADT alone" in text
    assert "Randomized to ADT+D" in text
    assert "Documented lost to follow-up (n=3)" in text
    assert "Documented lost to follow-up (n=0)" in text


def test_parse_window_keeps_other_pages_when_one_page_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, *_args: Any) -> str:
            return self._text

    class FakeDocument:
        def load_page(self, page_index: int) -> FakePage:
            if page_index == 1:
                raise RuntimeError("bad page")
            return FakePage(f"page {page_index} allocation concealment")

    monkeypatch.setattr(
        "arbiter.ingestion.supplements._extract_lines", lambda _page, _page_index: []
    )

    window = _parse_pdf_window(FakeDocument(), 0, 3)  # type: ignore[arg-type]

    assert "page 0 allocation concealment" in window.full_text
    assert "page 2 allocation concealment" in window.full_text
    assert any(
        box.boxclass == "degraded-page" and box.page == 1 for box in window.page_boxes
    )
