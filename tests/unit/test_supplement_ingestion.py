from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
import sys

import pymupdf
import pytest

from arbiter.ingestion.supplements import _parse_pdf_window, _parse_pdf_windows, ingest_supplements
from arbiter.llm.base import LLMRequestTimeoutError
from arbiter.llm.mock_client import MockLLMClient
from arbiter.config import EnvSettings
from arbiter.models import AnnotationStatus, DocType, PageBox, SupplementSegment
from arbiter.retrieval.annotator import annotate_segment
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


def _box(text: str, page: int) -> PageBox:
    return PageBox(
        boxclass="section-header",
        text=text,
        bbox=(0.0, 0.0, 0.0, 0.0),
        page=page,
    )


@pytest.mark.asyncio
async def test_annotation_prompt_requires_schema_wrapped_no_content_response() -> None:
    segment = SupplementSegment(
        segment_id="appendix.pdf__FULL_DOCUMENT__0",
        source_file="appendix.pdf",
        doc_type=DocType.APPENDIX,
        heading="FULL_DOCUMENT",
        pages=[0],
        raw_text="Administrative supplement content with no trial methods.",
        annotation="No risk-of-bias relevant content.",
        domain_tags=["D1"],
        char_count=58,
    )
    client = MockLLMClient(
        responses={
            "supplement_annotation:appendix.pdf__FULL_DOCUMENT__0": {
                "annotation": "No risk-of-bias relevant content."
            }
        }
    )
    settings = EnvSettings()

    annotation = await annotate_segment(
        segment,
        document_preamble="Administrative supplement.",
        aux_client=client,
        settings=settings,
    )

    system_prompt = client.trace_messages[0][0]["content"]
    assert annotation == "No risk-of-bias relevant content."
    assert client.max_tokens == [settings.supplement_annotation_max_tokens]
    assert 'set "annotation" to "No risk-of-bias relevant content."' in system_prompt
    assert 'return exactly "No risk-of-bias relevant content."' not in system_prompt


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
    client = MockLLMClient(
        responses={
            "supplement_annotation:sap.pdf__STATISTICAL_ANALYSIS_PLAN__0_0": {
                "annotation": "SAP describes allocation concealment and the analysis population."
            },
            "supplement_annotation:sap.pdf__MISSING_DATA__0_1": {
                "annotation": "SAP describes missing outcome data handling."
            },
            "supplement_annotation:sap.pdf__OUTCOME_ASSESSMENT__0_2": {
                "annotation": "SAP describes blinded outcome assessment."
            },
        }
    )

    index = await ingest_supplements([supplement_dir], client)
    segments, score = index.retrieve(["concealment", "allocation"], "D1", top_k=5)

    assert len(segments) <= 5
    assert segments
    assert score is None or 0.0 <= score <= 1.0
    assert all(segment.raw_text.strip() for segment in index.segments)
    assert all(segment.annotation.strip() for segment in index.segments)
    assert {
        segment.annotation_status for segment in index.segments
    } == {AnnotationStatus.SUCCEEDED_SUBSTANTIVE}


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
    assert {
        segment.annotation_status for segment in index.segments
    } == {AnnotationStatus.NOT_RUN}
    segments, _score = index.retrieve(["randomisation"], "D1", top_k=5)
    assert segments


@pytest.mark.asyncio
async def test_ingest_supplements_records_annotation_failure_without_aborting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "appendix.pdf"
    _write_supplement_pdf(
        path,
        [("Randomisation", "Allocation concealment used a central IWRS system.")],
    )
    client = MockLLMClient(
        responses={
            "supplement_annotation:appendix.pdf__FULL_DOCUMENT__0": LLMRequestTimeoutError(
                "mock timed out"
            )
        }
    )

    index = await ingest_supplements([path], client)

    assert len(index.segments) == 1
    segment = index.segments[0]
    assert segment.annotation_status is AnnotationStatus.FAILED
    assert segment.annotation_error == "mock timed out"
    assert "Allocation concealment" in segment.annotated_text


def test_default_dense_arm_uses_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
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
        annotation="Participants used a central web-based randomisation service.",
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
        annotation="Overall survival was summarized with Kaplan-Meier curves.",
        domain_tags=["D1"],
        char_count=57,
    )

    settings = EnvSettings()
    settings.dense_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    index = SupplementIndex([unrelated, central_randomisation], settings=settings)
    result = index.retrieve_with_metadata(["allocation concealment"], "D1", top_k=1)

    assert result["segments"] == [central_randomisation]
    assert result["top_score"] == pytest.approx(1.0)
    assert calls[0][0] == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls[0][1] == [unrelated.annotated_text, central_randomisation.annotated_text]


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
    client = MockLLMClient(
        responses={
            "supplement_annotation:plain.pdf__FULL_DOCUMENT__0": {
                "annotation": "Supplement reports randomisation and allocation concealment."
            }
        }
    )

    index = await ingest_supplements([path], client)

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
