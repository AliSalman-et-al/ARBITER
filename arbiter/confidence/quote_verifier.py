"""Fuzzy quote verification against parsed trial text."""

from __future__ import annotations

import os
import re
import unicodedata

from rapidfuzz import fuzz

from arbiter.models import PageBox

DEFAULT_QUOTE_VERIFY_THRESHOLD = 85
DEFAULT_QUOTE_MIN_VERIFY_CHARS = 15


def verify_quote(
    quote: str,
    raw_char_stream: str,
    threshold: int = DEFAULT_QUOTE_VERIFY_THRESHOLD,
) -> bool:
    """Return whether a quote can be located in the raw PDF character stream."""
    normalized_quote = _normalize_text(quote)
    if len(normalized_quote) < _quote_min_verify_chars():
        return True
    if not _normalize_text(raw_char_stream):
        return False

    return _partial_ratio(normalized_quote, _normalize_text(raw_char_stream)) >= threshold


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


def _quote_starts_in_text(quote: str, text: str) -> bool:
    words = quote.split()
    if not words:
        return False
    prefix = " ".join(words[: min(2, len(words))])
    return prefix in text


def _is_earlier(page: int, current: int | None) -> bool:
    return current is None or page < current
