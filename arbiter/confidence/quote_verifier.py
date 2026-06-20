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
    """Resolve a verified quote to the earliest 0-based source page."""
    normalized_quote = _normalize_text(quote)
    if len(normalized_quote) < _quote_min_verify_chars():
        return None

    threshold = _quote_verify_threshold()
    best_score = -1.0
    best_page: int | None = None
    sorted_boxes = sorted(page_boxes, key=lambda box: (box.page, box.bbox[1], box.bbox[0]))

    for index, box in enumerate(sorted_boxes):
        normalized_box_text = _normalize_text(box.text)
        candidates = [normalized_box_text]
        next_box = sorted_boxes[index + 1] if index + 1 < len(sorted_boxes) else None
        if next_box is not None and next_box.page != box.page and _quote_starts_in_text(normalized_quote, normalized_box_text):
            candidates.append(_normalize_text(f"{box.text} {next_box.text}"))

        box_score = max((_partial_ratio(normalized_quote, candidate) for candidate in candidates if candidate), default=0.0)
        if box_score >= threshold and (box_score > best_score or (box_score == best_score and _is_earlier(box.page, best_page))):
            best_score = box_score
            best_page = box.page

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


def _quote_starts_in_text(quote: str, text: str) -> bool:
    words = quote.split()
    if not words:
        return False
    prefix = " ".join(words[: min(2, len(words))])
    return prefix in text


def _is_earlier(page: int, current: int | None) -> bool:
    return current is None or page < current
