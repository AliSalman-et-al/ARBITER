"""ARBITER public Python API."""

from .config import AssessmentConfig


def ingest_trial(*args, **kwargs):
    """Ingest a trial.

    Implemented in the ingestion requirements after the project setup slice.
    """
    raise NotImplementedError("ingest_trial is implemented by REQ-02 through REQ-05")


def assess_trial(*args, **kwargs):
    """Assess a trial.

    Implemented in the assessment orchestration requirements after setup.
    """
    raise NotImplementedError("assess_trial is implemented by REQ-12")


__all__ = ["AssessmentConfig", "assess_trial", "ingest_trial"]
