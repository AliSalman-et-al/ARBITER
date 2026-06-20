"""Document ingestion utilities."""

from .ctgov import fetch_ctgov
from .paper import ingest_paper
from .supplements import ingest_supplements

__all__ = ["fetch_ctgov", "ingest_paper", "ingest_supplements"]
