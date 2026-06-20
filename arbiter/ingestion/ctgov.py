"""ClinicalTrials.gov ingestion."""

from __future__ import annotations

import logging
import re

import httpx

LOGGER = logging.getLogger(__name__)

CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_number}"
NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
REQUEST_TIMEOUT_SECONDS = 20.0


def _normalize_nct(nct_number: str) -> str | None:
    normalized = nct_number.strip().upper()
    if not NCT_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _make_transport() -> httpx.AsyncBaseTransport | None:
    return None


async def fetch_ctgov(nct_number: str) -> dict | None:
    """Fetch and return the verbatim ClinicalTrials.gov v2 study JSON."""

    normalized_nct = _normalize_nct(nct_number)
    if normalized_nct is None:
        LOGGER.warning("Invalid ClinicalTrials.gov NCT number: %r", nct_number)
        return None

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=_make_transport(),
        ) as client:
            response = await client.get(CTGOV_STUDY_URL.format(nct_number=normalized_nct))
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        LOGGER.warning("ClinicalTrials.gov fetch failed for %s: %s", normalized_nct, exc)
        return None

    if not isinstance(payload, dict):
        LOGGER.warning("ClinicalTrials.gov returned non-object JSON for %s", normalized_nct)
        return None

    return payload
