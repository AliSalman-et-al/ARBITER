"""Document ingestion utilities."""

from .ctgov import fetch_ctgov
from .metadata_extractor import extract_metadata
from .paper import ingest_paper
from .supplements import ingest_supplements

__all__ = ["extract_metadata", "fetch_ctgov", "ingest_paper", "ingest_supplements"]
