# Fail closed when configured dense retrieval is unavailable

ARBITER treats a configured dense supplement retriever as part of the evidence contract, not as an opportunistic optimization. If a dense embedding model is configured and supplement segments exist, ingestion must initialize the backend successfully or fail the run after recording an error-grade degradation event; continuing sparse-only would make the trace look like hybrid retrieval was used when it was not.

This keeps the default `google/embeddinggemma-300m` model in place while making access, download, and initialization failures explicit. Runs without a configured dense model remain valid sparse retrieval runs, and tests that need sparse-only behavior disable dense retrieval explicitly.
