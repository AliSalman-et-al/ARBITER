from __future__ import annotations

import httpx
import pytest

from arbiter.ingestion.ctgov import fetch_ctgov


@pytest.mark.asyncio
async def test_fetch_ctgov_returns_verbatim_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567"},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Overall Survival", "timeFrame": "36 months"}],
                "secondaryOutcomes": [{"measure": "Progression-Free Survival"}],
            },
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://clinicaltrials.gov/api/v2/studies/NCT01234567"
        return httpx.Response(200, json=payload)

    monkeypatch.setattr("arbiter.ingestion.ctgov._make_transport", lambda: httpx.MockTransport(handler))

    result = await fetch_ctgov(" nct01234567 ")

    assert result == payload


@pytest.mark.asyncio
async def test_fetch_ctgov_returns_none_for_invalid_nct_without_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid NCT should not call ClinicalTrials.gov")

    monkeypatch.setattr("arbiter.ingestion.ctgov._make_transport", lambda: httpx.MockTransport(handler))

    assert await fetch_ctgov("NCT123") is None


@pytest.mark.asyncio
async def test_fetch_ctgov_returns_none_for_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    monkeypatch.setattr("arbiter.ingestion.ctgov._make_transport", lambda: httpx.MockTransport(handler))

    assert await fetch_ctgov("NCT01234567") is None


@pytest.mark.asyncio
async def test_fetch_ctgov_returns_none_for_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    monkeypatch.setattr("arbiter.ingestion.ctgov._make_transport", lambda: httpx.MockTransport(handler))

    assert await fetch_ctgov("NCT01234567") is None


@pytest.mark.asyncio
async def test_fetch_ctgov_returns_none_for_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    monkeypatch.setattr("arbiter.ingestion.ctgov._make_transport", lambda: httpx.MockTransport(handler))

    assert await fetch_ctgov("NCT01234567") is None
