from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import pytest

from arbiter.ingestion.supplements import _parse_pdf_window, ingest_supplements
from arbiter.llm.mock_client import MockLLMClient
from arbiter.models import DocType, SupplementSegment
from arbiter.retrieval.annotator import annotate_segment
from arbiter.retrieval.segmenter import ParsedSupplementWindow, segment_document
from arbiter.retrieval.supplement_index import SupplementIndex


def _write_supplement_pdf(path: Path, sections: list[tuple[str, str]]) -> None:
    doc = pymupdf.open()
    for heading, body in sections:
        page = doc.new_page()
        page.insert_text((72, 72), heading, fontsize=16)
        page.insert_text((72, 120), body, fontsize=11)
    doc.save(path)
    doc.close()


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

    annotation = await annotate_segment(
        segment,
        document_preamble="Administrative supplement.",
        aux_client=client,
    )

    system_prompt = client.trace_messages[0][0]["content"]
    assert annotation == "No risk-of-bias relevant content."
    assert 'set "annotation" to "No risk-of-bias relevant content."' in system_prompt
    assert 'return exactly "No risk-of-bias relevant content."' not in system_prompt


@pytest.mark.asyncio
async def test_ingest_supplements_empty_paths_returns_empty_index() -> None:
    client = MockLLMClient()

    index = await ingest_supplements([], client)

    assert isinstance(index, SupplementIndex)
    assert index.retrieve(["concealment"], "D1") == ([], None)


@pytest.mark.asyncio
async def test_ingest_supplements_expands_directory_and_retrieves_top_k(tmp_path: Path) -> None:
    supplement_dir = tmp_path / "supplements"
    supplement_dir.mkdir()
    _write_supplement_pdf(
        supplement_dir / "sap.pdf",
        [
            ("Statistical Analysis Plan", "The allocation concealment method used an IWRS system."),
            ("Missing Data", "Missing overall survival data were handled with sensitivity analyses."),
            ("Outcome Assessment", "The endpoint committee was blinded to treatment assignment."),
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
    assert score is not None
    assert all(segment.raw_text.strip() for segment in index.segments)
    assert all(segment.annotation.strip() for segment in index.segments)


@pytest.mark.asyncio
async def test_document_with_no_section_headers_yields_one_full_document_segment(tmp_path: Path) -> None:
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


def test_segment_document_collapses_too_few_segments_to_full_document(tmp_path: Path) -> None:
    window = ParsedSupplementWindow(
        full_text="No obvious heading text.",
        page_starts=[0],
        page_boxes=[],
        page_offset=0,
    )

    segments = segment_document(tmp_path / "appendix.pdf", [window], doc_type=DocType.APPENDIX)

    assert len(segments) == 1
    assert segments[0].heading == "FULL_DOCUMENT"


def test_parse_window_keeps_other_pages_when_one_page_raises(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("arbiter.ingestion.supplements._extract_lines", lambda _page, _page_index: [])

    window = _parse_pdf_window(FakeDocument(), 0, 3)  # type: ignore[arg-type]

    assert "page 0 allocation concealment" in window.full_text
    assert "page 2 allocation concealment" in window.full_text
    assert any(box.boxclass == "degraded-page" and box.page == 1 for box in window.page_boxes)
