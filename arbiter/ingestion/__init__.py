"""Document ingestion utilities."""

from .paper import ingest_paper
from .supplements import ingest_supplements

__all__ = ["ingest_paper", "ingest_supplements"]
