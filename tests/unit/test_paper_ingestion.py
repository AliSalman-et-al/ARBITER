from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from arbiter.ingestion.paper import ingest_paper, normalize_heading
from arbiter.models import ParsingQuality


class _FakeDoclingDocument:
    def __init__(self, pages: list[str], items: list[Any]) -> None:
        self._pages = pages
        self._items = items

    def num_pages(self) -> int:
        return len(self._pages)

    def export_to_markdown(self, *, page_no: int, **_kwargs: Any) -> str:
        return self._pages[page_no - 1]

    def iterate_items(self) -> Any:
        for item in self._items:
            yield item, 0


def _prov(page_no: int) -> SimpleNamespace:
    return SimpleNamespace(
        page_no=page_no,
        bbox=SimpleNamespace(l=0.0, t=0.0, r=100.0, b=20.0),
    )


def _item(text: str, label: str, page_no: int) -> SimpleNamespace:
    return SimpleNamespace(
        text=text, label=SimpleNamespace(value=label), prov=[_prov(page_no)]
    )


def test_ingest_paper_uses_single_docling_representation_for_sections_and_raw_stream(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paper_path = tmp_path / "trial.pdf"
    paper_path.write_bytes(b"fake")
    document = _FakeDoclingDocument(
        [
            "# METHODS\nThe allocation sequence was random and centrally concealed. NCT12345678",
            "# RESULTS\nBaseline characteristics and outcome results were reported.",
        ],
        [
            _item("METHODS", "section_header", 1),
            _item(
                "The allocation sequence was random and centrally concealed.", "text", 1
            ),
            _item("RESULTS", "section_header", 2),
            _item(
                "Baseline characteristics and outcome results were reported.", "text", 2
            ),
        ],
    )
    monkeypatch.setattr(
        "arbiter.ingestion.paper.convert_pdf",
        lambda _path, _settings, **_kwargs: document,
    )

    section_map, raw_stream = ingest_paper(paper_path)

    assert section_map.parsing_quality == ParsingQuality.STANDARD
    assert section_map.source_path == str(paper_path)
    assert raw_stream == section_map.full_text
    assert section_map.nct_number == "NCT12345678"
    assert [section.label for section in section_map.sections] == ["METHODS", "RESULTS"]
    assert section_map.sections[0].pages == [0]
    assert "D1" in section_map.sections[0].domain_tags
    assert section_map.page_boxes[0].page == 0


def test_ingest_paper_keeps_subsections_inside_parent_canonical_section(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paper_path = tmp_path / "nested.pdf"
    paper_path.write_bytes(b"fake")
    document = _FakeDoclingDocument(
        [
            "\n".join(
                [
                    "# Methods",
                    "Study Oversight",
                    "Oversight body text remained under methods.",
                    "Statistical Analysis",
                    "The analysis used the intention-to-treat population.",
                    "# Results",
                    "Results body text starts here.",
                ]
            )
        ],
        [
            _item("Methods", "section_header", 1),
            _item("Statistical Analysis", "section_header", 1),
            _item("Results", "section_header", 1),
        ],
    )
    monkeypatch.setattr(
        "arbiter.ingestion.paper.convert_pdf",
        lambda _path, _settings, **_kwargs: document,
    )

    section_map, _ = ingest_paper(paper_path)
    methods = next(
        section for section in section_map.sections if section.label == "METHODS"
    )

    assert "Study Oversight" in methods.text
    assert "Statistical Analysis" in methods.text
    assert "intention-to-treat" in methods.text
    assert "Results body text starts here" not in methods.text


def test_ingest_paper_degrades_for_unreadable_pdf(monkeypatch, tmp_path: Path) -> None:
    paper_path = tmp_path / "broken.pdf"
    paper_path.write_bytes(b"not a pdf")
    monkeypatch.setattr(
        "arbiter.ingestion.paper.convert_pdf",
        lambda _path, _settings, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("bad pdf")
        ),
    )

    section_map, raw_stream = ingest_paper(paper_path)

    assert raw_stream == ""
    assert section_map.parsing_quality == ParsingQuality.DEGRADED
    assert section_map.sections[0].label == "FULL_TEXT"
    assert section_map.sections[0].domain_tags == ["D1", "D2", "D3", "D4", "D5"]


def test_normalize_heading_strips_surrounding_punctuation() -> None:
    assert normalize_heading("  1. Methods: ") == "1. METHODS"
