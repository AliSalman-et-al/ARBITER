"""Deterministic confidence helpers."""

from .signals import compute_confidence
from .quote_verifier import locate_quote_page, verify_quote

__all__ = ["compute_confidence", "locate_quote_page", "verify_quote"]
