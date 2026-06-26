"""Fuzzy quote verification against parsed trial text."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from arbiter.models import PageBox

DEFAULT_QUOTE_VERIFY_THRESHOLD = 85
DEFAULT_QUOTE_MIN_VERIFY_CHARS = 15


@dataclass(frozen=True)
class QuoteSource:
    source_document: str | None
    raw_char_stream: str
    page_boxes: list[PageBox]


def verify_quote(
    quote: str,
    raw_char_stream: str,
    threshold: int | None = None,
) -> bool:
    """Return whether a quote can be located in the raw PDF character stream."""
    normalized_quote = _normalize_text(quote)
    if len(normalized_quote) < _quote_min_verify_chars():
        return False
    if not _normalize_text(raw_char_stream):
        return False

    return _partial_ratio(normalized_quote, _normalize_text(raw_char_stream)) >= (
        _quote_verify_threshold() if threshold is None else threshold
    )


def locate_quote_page(quote: str, page_boxes: list[PageBox]) -> int | None:
    """Resolve a verified quote to the earliest 0-based source page.

    Matches the quote against PER-PAGE concatenated text rather than individual
    layout boxes. A quote that spans several boxes on one page still localises
    this way; per-box matching would miss it and wrongly return None even though
    the quote verifies against the raw character stream (REQ-10/REQ-15). Page-break
    straddles are handled by also scoring each page joined with the next and
    attributing the match to the earlier page. Returns None for an empty/short
    quote or when no page clears the verify threshold; the verified-but-unlocalised
    best-page fallback that keeps `page is None iff no verifiable quote` true
    end-to-end lives in the resolve_quote facade (REQ-15), which knows the
    verification result this function does not.
    """
    normalized_quote = _normalize_text(quote)
    if len(normalized_quote) < _quote_min_verify_chars():
        return None

    page_texts = _page_texts(page_boxes)
    if not page_texts:
        return None

    threshold = _quote_verify_threshold()
    best_score = -1.0
    best_page: int | None = None
    for index, (page, text) in enumerate(page_texts):
        candidates = [text]
        if index + 1 < len(page_texts) and _quote_starts_in_text(normalized_quote, text):
            candidates.append(_normalize_text(f"{text} {page_texts[index + 1][1]}"))

        score = max((_partial_ratio(normalized_quote, candidate) for candidate in candidates if candidate), default=0.0)
        if score >= threshold and (score > best_score or (score == best_score and _is_earlier(page, best_page))):
            best_score = score
            best_page = page

    return best_page


def resolve_quote(quote: str, raw_char_stream: str, page_boxes: list[PageBox]) -> tuple[bool, int | None]:
    """Verify a quote and resolve its page through one deterministic facade."""
    verified, page, _source_document = resolve_quote_source(
        quote,
        [QuoteSource(source_document=None, raw_char_stream=raw_char_stream, page_boxes=page_boxes)],
    )
    return verified, page


def resolve_quote_source(quote: str, sources: list[QuoteSource]) -> tuple[bool, int | None, str | None]:
    """Verify a quote against all source text the SQ answer was allowed to quote."""
    if len(_normalize_text(quote)) < _quote_min_verify_chars():
        return False, None, None

    for source in sources:
        verified = verify_quote(quote, source.raw_char_stream)
        if not verified:
            continue

        page = locate_quote_page(quote, source.page_boxes)
        if page is not None:
            return True, page, source.source_document

        best_page = _best_quote_page(quote, source.page_boxes)
        if best_page is not None:
            return True, best_page, source.source_document

    return False, None, None


def describe_quote_verification(
    quote: str,
    raw_char_stream: str,
    page_boxes: list[PageBox],
    *,
    source_document: str | None = None,
    threshold: int | None = None,
) -> dict[str, object]:
    """Return trace-ready deterministic quote verification details."""
    return describe_quote_verification_sources(
        quote,
        [QuoteSource(source_document=source_document, raw_char_stream=raw_char_stream, page_boxes=page_boxes)],
        threshold=threshold,
    )


def describe_quote_verification_sources(
    quote: str,
    sources: list[QuoteSource],
    *,
    threshold: int | None = None,
) -> dict[str, object]:
    """Return trace-ready quote verification details across allowed quote sources."""

    normalized_quote = _normalize_text(quote)
    effective_threshold = _quote_verify_threshold() if threshold is None else threshold
    short_quote = len(normalized_quote) < _quote_min_verify_chars()

    best_score = 0.0
    best_source: QuoteSource | None = None
    source_text_seen = False
    for source in sources:
        normalized_source = _normalize_text(source.raw_char_stream)
        if normalized_source:
            source_text_seen = True
        score = _partial_ratio(normalized_quote, normalized_source) if normalized_quote and normalized_source else 0.0
        if score > best_score:
            best_score = score
            best_source = source

    verified = False
    page = None
    matched_source_document = None
    if not short_quote and best_source is not None and best_score >= effective_threshold:
        page = locate_quote_page(quote, best_source.page_boxes)
        if page is None:
            page = _best_quote_page(quote, best_source.page_boxes)
        verified = page is not None
        matched_source_document = best_source.source_document if verified else None

    failure_reason = None
    if short_quote:
        failure_reason = "quote shorter than minimum verification length"
    elif not source_text_seen:
        failure_reason = "source text is empty"
    elif not verified:
        failure_reason = "quote did not meet verification threshold"

    return {
        "normalized_quote": normalized_quote,
        "verified": verified,
        "matched_source_document": matched_source_document,
        "matched_page": page if verified else None,
        "matched_span": None,
        "match_strategy": "partial_ratio",
        "match_score": best_score,
        "verification_threshold": effective_threshold,
        "failure_reason": failure_reason if not verified else None,
    }


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    normalized = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _partial_ratio(quote: str, source: str) -> float:
    if not quote or not source:
        return 0.0
    return float(fuzz.partial_ratio(quote, source))


def _quote_verify_threshold() -> int:
    return _env_int("ARBITER_QUOTE_VERIFY_THRESHOLD", DEFAULT_QUOTE_VERIFY_THRESHOLD)


def _quote_min_verify_chars() -> int:
    return _env_int("ARBITER_QUOTE_MIN_VERIFY_CHARS", DEFAULT_QUOTE_MIN_VERIFY_CHARS)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _page_texts(page_boxes: list[PageBox]) -> list[tuple[int, str]]:
    """Concatenate each page's boxes in reading order → [(page, normalized_text)]."""
    ordered = sorted(page_boxes, key=lambda box: (box.page, box.bbox[1], box.bbox[0]))
    pages: dict[int, list[str]] = {}
    for box in ordered:
        pages.setdefault(box.page, []).append(box.text)
    return [(page, _normalize_text(" ".join(parts))) for page, parts in sorted(pages.items())]


def _best_quote_page(quote: str, page_boxes: list[PageBox]) -> int | None:
    normalized_quote = _normalize_text(quote)
    if len(normalized_quote) < _quote_min_verify_chars():
        return None

    page_texts = _page_texts(page_boxes)
    best_score = -1.0
    best_page: int | None = None
    for index, (page, text) in enumerate(page_texts):
        candidates = [text]
        if index + 1 < len(page_texts) and _quote_starts_in_text(normalized_quote, text):
            candidates.append(_normalize_text(f"{text} {page_texts[index + 1][1]}"))
        score = max((_partial_ratio(normalized_quote, candidate) for candidate in candidates if candidate), default=0.0)
        if score > best_score or (score == best_score and _is_earlier(page, best_page)):
            best_score = score
            best_page = page
    return best_page


def _quote_starts_in_text(quote: str, text: str) -> bool:
    words = quote.split()
    if not words:
        return False
    prefix = " ".join(words[: min(2, len(words))])
    return prefix in text


def _is_earlier(page: int, current: int | None) -> bool:
    return current is None or page < current
