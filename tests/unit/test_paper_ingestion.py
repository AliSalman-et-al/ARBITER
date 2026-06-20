from __future__ import annotations

from pathlib import Path

import pymupdf

from arbiter.ingestion.paper import ingest_paper, normalize_heading
from arbiter.models import ParsingQuality


def _write_rct_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Randomised Trial NCT12345678", fontsize=18)
    page.insert_text((72, 120), "METHODS", fontsize=16)
    page.insert_text(
        (72, 150),
        "The allocation sequence was random and centrally concealed.",
        fontsize=11,
    )
    page.insert_text((72, 200), "RESULTS", fontsize=16)
    page.insert_text(
        (72, 230),
        "Baseline characteristics and outcome results were reported.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()


def test_ingest_paper_returns_section_map_and_raw_stream(tmp_path: Path) -> None:
    paper_path = tmp_path / "trial.pdf"
    _write_rct_pdf(paper_path)

    section_map, raw_stream = ingest_paper(paper_path)

    assert section_map.parsing_quality == ParsingQuality.STANDARD
    assert section_map.source_path == str(paper_path)
    assert section_map.full_text.strip()
    assert raw_stream.strip()
    assert "allocation sequence" in raw_stream
    assert section_map.nct_number == "NCT12345678"
    assert len(section_map.sections) >= 1


def test_ingest_paper_detects_uppercase_methods_and_results_sections(tmp_path: Path) -> None:
    paper_path = tmp_path / "trial.pdf"
    _write_rct_pdf(paper_path)

    section_map, _ = ingest_paper(paper_path)

    labels = [section.label for section in section_map.sections]
    assert "METHODS" in labels
    assert "RESULTS" in labels
    assert all(label == label.upper() for label in labels)


def test_ingest_paper_tracks_section_offsets_and_pages(tmp_path: Path) -> None:
    paper_path = tmp_path / "trial.pdf"
    _write_rct_pdf(paper_path)

    section_map, _ = ingest_paper(paper_path)
    methods = next(section for section in section_map.sections if section.label == "METHODS")

    assert section_map.full_text[methods.char_start : methods.char_end].strip() == methods.text
    assert methods.pages == [0]
    assert "D1" in methods.domain_tags


def test_ingest_paper_degrades_for_unreadable_pdf(tmp_path: Path) -> None:
    paper_path = tmp_path / "broken.pdf"
    paper_path.write_bytes(b"not a pdf")

    section_map, raw_stream = ingest_paper(paper_path)

    assert raw_stream == ""
    assert section_map.parsing_quality == ParsingQuality.DEGRADED
    assert section_map.sections[0].label == "FULL_TEXT"
    assert section_map.sections[0].domain_tags == ["D1", "D2", "D3", "D4", "D5"]


def test_normalize_heading_strips_surrounding_punctuation() -> None:
    assert normalize_heading("  1. Methods: ") == "1. METHODS"
