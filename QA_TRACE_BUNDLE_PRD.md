# ARBITER QA Trace Bundle: Product Requirements Document

**Project name:** ARBITER QA Trace Bundle

**Version:** 0.1.0

**Audience:** Engineers and coding agents implementing live QA observability for ARBITER

---

## 1. Purpose

ARBITER needs live QA observability so agents and engineers can inspect what happens inside a real pipeline run while it is executing. The QA Trace Bundle is a read-only, per-run artifact that exposes pipeline inputs, outputs, intermediate data, and full LLM call payloads in real time.

The bundle exists to answer one question:

> Why did this assessment produce this result?

It is not part of the deterministic assessment record. It is an observability side-channel for live testing, debugging, and improvement work.

---

## 2. Scope

### In Scope

- Real-time trace writes for `arbiter assess` and `arbiter batch`.
- A generated `run_id` for each traced execution.
- A top-level run manifest with environment, command, model, provider, git, and config metadata.
- A tail-safe append-only event log.
- Step-scoped JSON artifacts for easier inspection.
- Full raw LLM prompt and response bodies.
- Normalized LLM call records across providers.
- Failed, repaired, and final LLM attempts.
- Complete parsed source artifacts for papers, supplements, and ClinicalTrials.gov records.
- Retrieval, context assembly, quote verification, SQ finalization, deterministic branching, judgment, and output-writing trace data.
- Fail-closed behavior when full QA tracing is explicitly enabled.

### Out of Scope

- Redaction or summarization mode.
- A live web UI or trace server.
- Secrets capture.
- Making trace data part of the deterministic `Assessment`, SQLite assessment row, or LangGraph state.
- Tracing every low-level helper, parser internals, or implementation detail.

---

## 3. Users

### QA Agents

Agents need tail-safe access to live trace events so they can inspect runs while the pipeline is still executing. They need stable event schemas and direct artifact references.

### Engineers

Engineers need durable, step-scoped artifacts for debugging retrieval, prompt construction, provider behavior, schema repair, quote verification, deterministic branching, and output writing.

### Evidence-Synthesis Reviewers

Reviewers are not the primary user of this bundle. Reviewer-facing audit remains the Markdown report. The QA Trace Bundle can support deeper investigation when engineers or agents need to diagnose a report or assessment.

---

## 4. Core Requirements

### REQ-QA-01: Generated Run Identity

Each traced execution must receive a generated run ID.

Recommended format:

```text
YYYYMMDD-HHMMSS-<shortid>
```

The `run_id` identifies a run instance, not a deterministic assessment identity. Deterministic identity remains `trial_id` plus the existing assessment resume key.

Acceptance criteria:

- Each `--trace full` run creates exactly one `run_id`.
- Re-running the same manifest/config creates a new `run_id`.
- Every trace event and artifact can be associated with the `run_id`.

### REQ-QA-02: Trace Root Layout

Full QA traces must be written under a separate run root.

Recommended layout:

```text
runs/<run_id>/qa_trace/
  run_manifest.json
  events.jsonl
  sources/
  llm_calls/
  retrieval/
  context/
  quote_verification/
  sq_answers/
  judgments/
  outputs/
  artifacts/
```

Assessment outputs may remain in their existing output locations. The trace root links to those outputs instead of replacing them.

Acceptance criteria:

- Trace data is not mixed into reviewer-facing output directories by default.
- `run_manifest.json` records all output paths.
- Artifact references in events are relative to the QA trace root or absolute paths when necessary.

### REQ-QA-03: Real-Time Event Log

The trace bundle must include a tail-safe append-only event log at:

```text
runs/<run_id>/qa_trace/events.jsonl
```

Each event is written as one complete JSON line and flushed immediately.

Acceptance criteria:

- Agents can tail `events.jsonl` during execution.
- Readers never need to parse an incomplete JSON object.
- Event writes are line-buffered or explicitly flushed.
- A failed run still leaves all successfully written events available.

### REQ-QA-04: Atomic Artifact Writes

Step-scoped artifacts must be written atomically.

Acceptance criteria:

- Large JSON artifacts are written to a temporary file and then renamed into place.
- Readers never see partially written artifact files.
- Event records reference artifacts only after the artifact is durable.

### REQ-QA-05: Stable Event Schema

Every event in `events.jsonl` must use a stable schema.

Required fields:

```json
{
  "schema_version": "1",
  "run_id": "20260626-153012-a1b2c3",
  "event_id": "evt_...",
  "parent_event_id": null,
  "timestamp": "2026-06-26T15:30:12.000Z",
  "event_type": "llm_call.completed",
  "status": "completed",
  "trial_id": "NCT...",
  "outcome": "overall survival",
  "domain": "D1",
  "sq_id": "1.1",
  "artifact_refs": [],
  "payload": {}
}
```

Fields that do not apply must be present as `null` or omitted only where the schema explicitly allows it.

Acceptance criteria:

- Event consumers can filter by `event_type`, `trial_id`, `outcome`, `domain`, and `sq_id`.
- Schema version is present on every event.
- Event IDs are unique within a run.
- Parent IDs support grouping attempts under higher-level operations.

### REQ-QA-06: Run Manifest

The bundle must include:

```text
runs/<run_id>/qa_trace/run_manifest.json
```

The manifest records execution metadata.

Required content:

- `run_id`
- command and CLI args
- started timestamp
- ARBITER package version
- pipeline version hash
- git commit
- dirty worktree flag
- input manifest path and hash, when applicable
- paper/supplement paths for single assessment runs
- provider/model configuration
- temperature and retry/repair settings
- trace mode
- output paths

Secrets and API keys must not be recorded.

Acceptance criteria:

- A reader can identify what code/config produced the trace.
- Secret-looking environment variables are excluded.
- Dirty worktree status is captured without dumping diffs.

### REQ-QA-07: Full Source Capture

The trace bundle must store complete parsed source artifacts once per run or trial.

Required source artifacts:

- parsed main paper text and page metadata
- parsed supplement text and segment metadata
- raw ClinicalTrials.gov v2 JSON responses
- derived trial metadata

Acceptance criteria:

- LLM and retrieval events can link to exact source chunks.
- Engineers can inspect both what the LLM saw and what retrieval/context assembly could have selected.
- ClinicalTrials.gov raw responses are stored under `sources/ctgov/<nct_id>.json`.

### REQ-QA-08: Quality-Debugging Event Coverage

The trace must cover the pipeline path needed to debug assessment quality.

Required event families:

- ingestion
- ClinicalTrials.gov fetch
- metadata extraction
- retrieval
- context assembly
- LLM calls
- LLM validation and repair attempts
- quote verification
- SQ finalization
- deterministic branching
- domain and overall judgment
- output writing
- errors

Acceptance criteria:

- A reader can reconstruct how each SQ answer became a finalized answer.
- A reader can reconstruct how finalized SQ answers became domain and overall judgments.
- Low-level helper calls are not traced unless they produce quality-relevant artifacts.

### REQ-QA-09: LLM Call Trace Records

Each LLM call must have a normalized trace record plus the full raw prompt and response bodies.

Required normalized fields:

- `call_id`
- `trial_id`
- `outcome`
- `domain`
- `sq_id`
- `effect_of_interest`
- `model`
- `provider`
- `temperature`
- provider routing metadata, when available
- cache-control metadata, when available
- token usage, when available
- cost metadata, when available
- prompt messages or request payload
- raw provider response
- parsed response
- validation result
- repair attempts
- final `SignalingQuestionRawAnswer`, when produced

Acceptance criteria:

- OpenAI, OpenRouter, Anthropic, and mock calls can be compared through the normalized fields.
- Raw provider request and response bodies are preserved exactly as sent/received, excluding secrets.
- Failed attempts and repaired attempts are retained, not overwritten.

### REQ-QA-10: Repair Ladder Trace

When schema validation or parsing fails, every repair attempt must be traced.

Acceptance criteria:

- Original invalid response is retained.
- Validation error is retained.
- Repair prompt/request is retained.
- Repair response is retained.
- Final parsed success or terminal failure is retained.

### REQ-QA-11: Retrieval and Context Trace

Retrieval and context assembly must be traceable per SQ.

Required content:

- query terms or retrieval request
- candidate passages
- selected passages
- ranking scores and fusion metadata, when available
- source document/page/segment references
- final assembled context passed to the LLM

Acceptance criteria:

- Engineers can distinguish retrieval failure from LLM reasoning failure.
- Engineers can inspect why a passage was or was not included in the prompt.

### REQ-QA-12: Quote Verification Trace

Quote verification trace must include matching details, not only the final boolean.

Required content:

- normalized quote text
- source document/page/span matched
- match score or strategy, when available
- failure reason when unverified
- resulting confidence flag

Acceptance criteria:

- Engineers can determine whether an unverified quote is a verifier bug, extraction bug, source parsing issue, or unsupported LLM quote.

### REQ-QA-13: Deterministic Core Trace

Branching, domain judgments, and overall rollup must be traced as deterministic transformations.

Acceptance criteria:

- Branching events identify which SQs were asked and which were structurally `NA`.
- Judgment events identify input SQ answers and output judgment.
- Overall rollup events identify domain judgments and rollup policy.
- The trace does not introduce new LLM reasoning into deterministic judgment steps.

### REQ-QA-14: Fail-Closed Full Trace Mode

When full QA tracing is explicitly enabled, trace write failures must fail the run.

Acceptance criteria:

- `--trace full` fails if the trace root cannot be created.
- `--trace full` fails if required events or artifacts cannot be written.
- `--trace summary` and `--trace off` keep their lighter behavior and do not become fail-closed QA modes.

### REQ-QA-15: CLI Integration

The existing full trace mode should become the QA Trace Bundle mode.

Required behavior:

```text
arbiter assess ... --trace full
arbiter batch ... --trace full
```

Acceptance criteria:

- No separate QA tracing flag is required.
- `--trace full` writes the real-time QA Trace Bundle.
- `--trace summary` remains summary-only.
- `--trace off` writes no trace bundle.

---

## 5. Non-Functional Requirements

### Read Safety

Multiple readers must be able to inspect the trace while the pipeline writes it.

### Reproducibility

The trace should capture enough code/config/source metadata to compare runs, but it must not claim deterministic replay of provider behavior.

### Observability Isolation

Trace data must stay outside:

- LangGraph state
- deterministic assessment models
- reviewer-facing reports
- SQLite assessment rows, except for optional trace path references if explicitly added later

### Storage Pragmatism

The bundle is expected to be large. v0.1 prioritizes debuggability over compactness.

---

## 6. Acceptance Checklist

- [ ] `--trace full` creates `runs/<run_id>/qa_trace/`.
- [ ] `run_manifest.json` captures command, code, config, model/provider, input, output, and git metadata without secrets.
- [ ] `events.jsonl` is tail-safe and flushed event by event.
- [ ] Large artifacts are written atomically.
- [ ] Stable event schema is used for every event.
- [ ] Full parsed paper, supplement, and CT.gov sources are available under `sources/`.
- [ ] Every LLM call includes normalized metadata plus full raw prompt and response bodies.
- [ ] Failed validation and repair attempts are retained.
- [ ] Retrieval/context artifacts show candidates, selected passages, scores, source refs, and final LLM context.
- [ ] Quote verification artifacts include matching details and failure reasons.
- [ ] Branching and judgment events expose deterministic inputs and outputs.
- [ ] `arbiter assess --trace full` and `arbiter batch --trace full` both produce QA Trace Bundles.
- [ ] Full trace write failure fails the run.
- [ ] Trace data remains outside deterministic assessment state and records.

---

## 7. Open Implementation Questions

These are implementation choices, not unresolved product requirements:

- Exact event type vocabulary.
- Exact artifact file naming conventions.
- Whether to store raw provider envelopes inline in LLM call records or as sibling files when large.
- Whether output records should include an optional pointer to the trace root.
- Whether old trace bundles need cleanup tooling.

