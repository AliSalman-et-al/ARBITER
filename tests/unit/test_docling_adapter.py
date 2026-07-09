from __future__ import annotations

import os
from typing import Any

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
