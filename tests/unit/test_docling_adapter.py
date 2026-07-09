from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat

from arbiter.config import EnvSettings
from arbiter.ingestion import docling_adapter


def test_docling_num_threads_defaults_to_host_cpu_count(monkeypatch) -> None:
    monkeypatch.delenv("ARBITER_DOCLING_NUM_THREADS", raising=False)
    monkeypatch.setattr("arbiter.config.os.cpu_count", lambda: 16)

    settings = EnvSettings()

    assert settings.docling_num_threads == 16


def test_docling_num_threads_can_be_configured_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_DOCLING_NUM_THREADS", "12")

    settings = EnvSettings()

    assert settings.docling_num_threads == 12


def test_docling_supplement_tables_default_off(monkeypatch) -> None:
    monkeypatch.delenv("ARBITER_DOCLING_SUPPLEMENT_TABLES", raising=False)

    settings = EnvSettings()

    assert settings.docling_supplement_tables is False


def test_docling_supplement_tables_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_DOCLING_SUPPLEMENT_TABLES", "true")

    settings = EnvSettings()

    assert settings.docling_supplement_tables is True


def test_docling_backend_defaults_to_docling_parse_v2(monkeypatch) -> None:
    monkeypatch.delenv("ARBITER_DOCLING_BACKEND", raising=False)

    settings = EnvSettings()

    assert settings.docling_backend == "docling-parse-v2"


def test_docling_backend_can_be_configured_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_DOCLING_BACKEND", "pypdfium2")

    settings = EnvSettings()

    assert settings.docling_backend == "pypdfium2"


def test_docling_parse_cache_path_defaults_under_arbiter_cache(monkeypatch) -> None:
    monkeypatch.delenv("ARBITER_DOCLING_PARSE_CACHE_PATH", raising=False)

    settings = EnvSettings()

    assert settings.docling_parse_cache_path == Path(".arbiter/cache/docling")


def test_docling_parse_cache_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ARBITER_DOCLING_PARSE_CACHE_ENABLED", "false")

    settings = EnvSettings()

    assert settings.docling_parse_cache_enabled is False


def test_build_docling_converter_configures_cpu_accelerator_options(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeDocumentConverter:
        def __init__(self, *, format_options: dict[Any, Any]) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeDocumentConverter)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    settings = EnvSettings()
    settings.docling_num_threads = 7

    converter = docling_adapter.build_docling_converter(settings)

    assert isinstance(converter, FakeDocumentConverter)
    pdf_option = captured["format_options"][InputFormat.PDF]
    accelerator_options = pdf_option.pipeline_options.accelerator_options
    assert accelerator_options.num_threads == 7
    assert accelerator_options.device is AcceleratorDevice.AUTO
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_build_docling_converter_defaults_to_v2_backend(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeDocumentConverter:
        def __init__(self, *, format_options: dict[Any, Any]) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeDocumentConverter)
    settings = EnvSettings()

    docling_adapter.build_docling_converter(settings)

    pdf_option = captured["format_options"][InputFormat.PDF]
    assert pdf_option.backend is DoclingParseV2DocumentBackend


def test_build_docling_converter_can_select_pypdfium2_backend(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeDocumentConverter:
        def __init__(self, *, format_options: dict[Any, Any]) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeDocumentConverter)
    settings = EnvSettings()
    settings.docling_backend = "pypdfium2"

    docling_adapter.build_docling_converter(settings)

    pdf_option = captured["format_options"][InputFormat.PDF]
    assert pdf_option.backend is PyPdfiumDocumentBackend


def test_build_docling_converter_rejects_unknown_backend(monkeypatch) -> None:
    class FakeDocumentConverter:
        def __init__(self, *, format_options: dict[Any, Any]) -> None:
            raise AssertionError("invalid backend should fail before converter creation")

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeDocumentConverter)
    settings = EnvSettings()
    settings.docling_backend = cast(Any, "unknown")

    with pytest.raises(ValueError, match="Unsupported ARBITER_DOCLING_BACKEND"):
        docling_adapter.build_docling_converter(settings)


def test_build_docling_converter_can_disable_table_structure(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeDocumentConverter:
        def __init__(self, *, format_options: dict[Any, Any]) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeDocumentConverter)
    settings = EnvSettings()
    settings.docling_do_ocr = False

    docling_adapter.build_docling_converter(settings, do_table_structure=False)

    pdf_option = captured["format_options"][InputFormat.PDF]
    assert pdf_option.pipeline_options.do_ocr is False
    assert pdf_option.pipeline_options.do_table_structure is False


def test_convert_pdf_reuses_cached_docling_document_by_content_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "trial.pdf"
    pdf_path.write_bytes(b"stable pdf bytes")
    settings = EnvSettings()
    settings.docling_parse_cache_path = tmp_path / "docling-cache"
    calls = 0

    class FakeDoclingDocument:
        def __init__(self, text: str) -> None:
            self.text = text

        def export_to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeConverter:
        def convert(self, path: Path, *, raises_on_error: bool) -> object:
            nonlocal calls
            calls += 1
            assert path == pdf_path
            assert raises_on_error is True
            return SimpleNamespace(document=FakeDoclingDocument(f"parsed-{calls}"))

    monkeypatch.setattr(
        docling_adapter,
        "_document_from_cache_payload",
        lambda payload: FakeDoclingDocument(payload["text"]),
    )

    first = docling_adapter.convert_pdf(
        pdf_path, settings, converter=cast(Any, FakeConverter())
    )
    second = docling_adapter.convert_pdf(
        pdf_path, settings, converter=cast(Any, FakeConverter())
    )

    assert first.text == "parsed-1"
    assert second.text == "parsed-1"
    assert calls == 1
    assert len(list(settings.docling_parse_cache_path.glob("*.json"))) == 1


def test_convert_pdf_force_refresh_bypasses_docling_parse_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "trial.pdf"
    pdf_path.write_bytes(b"stable pdf bytes")
    settings = EnvSettings()
    settings.docling_parse_cache_path = tmp_path / "docling-cache"
    calls = 0

    class FakeDoclingDocument:
        def __init__(self, text: str) -> None:
            self.text = text

        def export_to_dict(self) -> dict[str, str]:
            return {"text": self.text}

    class FakeConverter:
        def convert(self, _path: Path, *, raises_on_error: bool) -> object:
            nonlocal calls
            calls += 1
            assert raises_on_error is True
            return SimpleNamespace(document=FakeDoclingDocument(f"parsed-{calls}"))

    monkeypatch.setattr(
        docling_adapter,
        "_document_from_cache_payload",
        lambda payload: FakeDoclingDocument(payload["text"]),
    )

    cached = docling_adapter.convert_pdf(
        pdf_path, settings, converter=cast(Any, FakeConverter())
    )
    refreshed = docling_adapter.convert_pdf(
        pdf_path,
        settings,
        converter=cast(Any, FakeConverter()),
        force_refresh_cache=True,
    )
    reused = docling_adapter.convert_pdf(
        pdf_path, settings, converter=cast(Any, FakeConverter())
    )

    assert cached.text == "parsed-1"
    assert refreshed.text == "parsed-2"
    assert reused.text == "parsed-1"
    assert calls == 2
