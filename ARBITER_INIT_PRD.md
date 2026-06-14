# ARBITER: Product Requirements Document

**Project name:** ARBITER (Automated Risk of Bias Inference Tool for Evidence Reviews)

**Version:** 0.1.0
**Audience:** Engineers and coding agents implementing this system

---

## How to Read This Document

This is a **product requirements document**, not an implementation spec. It describes the _shape of the end goal_: what each component must do, why it exists, and the acceptance criteria it must satisfy. It deliberately does **not** prescribe implementation bodies — function signatures, data models, and behavioural contracts are given, but how you fulfil them is your choice, provided the acceptance criteria pass.

Each requirement has:

- A clear statement of **what** must be built
- The **reason** it exists (so you can make sensible tradeoffs at edge cases)
- **Acceptance criteria** you can test against
- Explicit **do-not** rules where common mistakes would break the design

Read sections 1–5 fully before writing any code. They establish the vocabulary, the users, the data flow, and the constraints that every later decision depends on.

> **The single most important constraint:** ARBITER never lets an LLM make a risk-of-bias _judgment_. The LLM only answers individual signaling questions (find a quote, pick an answer code). All Low / Some concerns / High judgments are computed deterministically from the **official Cochrane RoB 2 algorithm**, which this repo vendors as a pinned reference (see [REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)).

---

## Table of Contents

1. [Domain Background: RoB 2](#1-domain-background-rob-2)
2. [System Overview](#2-system-overview)
3. [Users and Workflow](#3-users-and-workflow)
4. [Repository Structure](#4-repository-structure)
5. [Data Models](#5-data-models)
6. [REQ-01: Dependency and Project Setup](#req-01-dependency-and-project-setup)
7. [REQ-02: Main Paper Ingestor](#req-02-main-paper-ingestor)
8. [REQ-03: Supplementary Material Ingestor](#req-03-supplementary-material-ingestor)
9. [REQ-04: ClinicalTrials.gov Fetcher](#req-04-clinicaltrialsgov-fetcher)
10. [REQ-05: Trial Metadata Extractor](#req-05-trial-metadata-extractor)
11. [REQ-06: LLM Abstraction Layer](#req-06-llm-abstraction-layer)
12. [REQ-07: RoB 2 Algorithm — Deterministic Judgments](#req-07-rob-2-algorithm-deterministic-judgments)
13. [REQ-08: Conditional Branching](#req-08-conditional-branching)
14. [REQ-09: Signaling-Question Prompt Templates](#req-09-signaling-question-prompt-templates)
15. [REQ-10: Quote Verifier](#req-10-quote-verifier)
16. [REQ-11: Confidence Signal System](#req-11-confidence-signal-system)
17. [REQ-12: Assessment Orchestration (Two-Tier Graph)](#req-12-assessment-orchestration-two-tier-graph)
18. [REQ-13: Pre-D5 Outcome Comparison](#req-13-pre-d5-outcome-comparison)
19. [REQ-14: Context Assembly](#req-14-context-assembly)
20. [REQ-15: SQ Worker](#req-15-sq-worker)
21. [REQ-16: Output — JSON and SQLite Writers](#req-16-output-json-and-sqlite-writers)
22. [REQ-17: Batch Runner and Manifest](#req-17-batch-runner-and-manifest)
23. [REQ-18: CLI](#req-18-cli)
24. [REQ-19: Python API](#req-19-python-api)
25. [REQ-20: Error Handling and Retry](#req-20-error-handling-and-retry)
26. [REQ-21: Evaluation Harness](#req-21-evaluation-harness)
27. [REQ-22: Testing](#req-22-testing)
28. [REQ-23: Run Trace & Timing Instrumentation](#req-23-run-trace--timing-instrumentation)
29. [REQ-24: Reviewer-Facing Markdown Report](#req-24-reviewer-facing-markdown-report)
30. [Implementation Order](#implementation-order)
31. [Acceptance Checklist](#acceptance-checklist)
32. [Appendix A: The 22 Signaling Questions](#appendix-a-the-22-signaling-questions)

---

## 1. Domain Background: RoB 2

### What RoB 2 Is

Cochrane Risk of Bias 2 (RoB 2) is the standard methodology for assessing whether a randomised controlled trial (RCT) is at risk of producing a biased result. It is applied **per outcome**: the same trial assessed for two different outcomes (e.g., Overall Survival and Progression-Free Survival) can receive different RoB 2 scores.

### The Five Domains

| Domain | What it assesses              | Key question                                                              | Scope             |
| ------ | ----------------------------- | ------------------------------------------------------------------------- | ----------------- |
| D1     | Randomisation process         | Was the sequence truly random and properly concealed?                     | **Trial-level**   |
| D2     | Deviations from interventions | Were participants/clinicians unblinded, or did protocol deviations occur? | **Outcome-level** |
| D3     | Missing outcome data          | Were there dropouts or exclusions that could bias _this outcome_?         | **Outcome-level** |
| D4     | Outcome measurement           | Was _this outcome_ measured without knowledge of allocation?              | **Outcome-level** |
| D5     | Selection of reported result  | Does _this reported result_ match what was pre-registered?                | **Outcome-level** |

The trial-level / outcome-level split is a core architectural fact for ARBITER — see [§2](#2-system-overview) and [REQ-12](#req-12-assessment-orchestration-two-tier-graph). Only **D1** is genuinely outcome-invariant (computed once per trial). D2's _inputs_ (blinding) are trial-wide, but its judgment is **outcome-scoped** — its SQs reference "the outcome" — so ARBITER judges D2 per outcome alongside D3–D5.

### Signaling Questions

Each domain is evaluated by answering between 3 and 7 signaling questions (SQs) — 22 in total across all five domains. Some SQs are conditional: they are answered only if a preceding SQ was answered in a particular way. Unanswered conditional SQs receive the code `NA` (Not Applicable). The complete set is in [Appendix A](#appendix-a-the-22-signaling-questions).

### Answer Codes

Every SQ receives exactly one of:

| Code | Meaning                                                   |
| ---- | --------------------------------------------------------- |
| `Y`  | Yes                                                       |
| `PY` | Probably Yes                                              |
| `PN` | Probably No                                               |
| `N`  | No                                                        |
| `NI` | No Information — the text provides no basis for answering |
| `NA` | Not Applicable — the SQ's trigger condition was not met   |

`NI` is not a safe default. Use it only when the relevant text genuinely contains no information on the question. Overuse of `NI` degrades the quality of the assessment.

### Judgments Are Deterministic, Not Generated

Once all applicable SQs in a domain are answered, the domain receives **Low**, **Some concerns**, or **High** via the published RoB 2 decision algorithm. The overall judgment is then derived from the five domain judgments. **The LLM never produces a judgment** — this is a hard constraint. The authoritative algorithm is the official Cochrane RoB 2 tool, which this repository vendors (see [REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)). This PRD does **not** restate the decision tables; an implementer reads them from the vendored source so that ARBITER's logic is traceable to a fixed, citable version.

### Effect of Interest

RoB 2 distinguishes two effects, which change how **D2** is assessed:

- **Assignment (ITT):** the effect of being _assigned_ to the intervention, regardless of adherence. The common case.
- **Adhering (per-protocol):** the effect of actually _adhering_ to the intervention.

The two effects use different SQ subsets within D2 (see [REQ-08](#req-08-conditional-branching) and [Appendix A](#appendix-a-the-22-signaling-questions)). The effect of interest is a per-assessment input; it does not change D1, D3, D4, or D5.

---

## 2. System Overview

### What This System Does

ARBITER is a Python pipeline that automates Cochrane RoB 2 assessments **at batch scale, unattended**. Given a manifest of trials — each with a main paper PDF and optional supplements, NCT number, and outcome list — it produces a complete, auditable RoB 2 assessment for every trial-outcome pair.

v0.1 is built as the assessment engine for a **Living Evidence System** ([lisr.org](https://lisr.org/)) — its eventual home is a continuously-updated synthesis platform, not a one-off batch. That north star drives the design toward auditability, full pipeline observability, and production runway (checkpoint/resume, re-assessment on changed inputs), even where a minimal v0.1 could skip them. The complementary research goal that scopes what v0.1 _measures_ is in [§3](#3-users-and-workflow).

### Core Invariants

Four properties are load-bearing for the whole system. They are stated **here, once**; later sections reference this list rather than restating it.

1. **Deterministic judgment core.** Given a fixed set of SQ answers, branching → the RoB 2 decision tables → the overall rollup always produce the same judgment ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments), [REQ-08](#req-08-conditional-branching)). This is the auditability guarantee. Everything that _produces_ the SQ answers — retrieval and the LLM calls — is best-effort and sits strictly upstream of this core.
2. **The LLM never judges.** Its only job is, per signaling question, to find and quote the supporting sentence(s) and select an answer code. Domain judgments, the overall judgment, section selection, and retrieval are all deterministic.
3. **Observability is side-channel.** Trace, timing, and cost data never enter the deterministic core, the LangGraph state, or the `Assessment` record; they are collected via a runtime handle into a separate artifact ([REQ-23](#req-23-run-trace--timing-instrumentation)). The assessment record stays reproducible; the trace is "what happened this run."
4. **`trial_id` is deterministic.** `NCT → slugified trial_label → sha256(paper)[:12]` ([REQ-05](#req-05-trial-metadata-extractor)) — never random. Batch resume is backed by this id plus the DB unique key, **not** by the LLM reproducing bytes.

### Determinism, precisely

- **Deterministic core** (invariant 1): context assembly, quote verification, branching, the decision tables, and the rollup — bit-reproducible given fixed SQ answers.
- **Best-effort LLM layer:** every SQ call runs at **temperature 0** with **no resampling** to minimise variance, but temperature 0 is most-likely-token sampling, **not** bit-identical output, and is more variable under OpenRouter multi-provider routing (per-route hardware/quantisation). Pin a single `provider` — in prod and in the eval — to tighten reproducibility.
- **Retrieval** (hybrid BM25 + pinned-weight BGE-M3 dense, RRF-fused, [REQ-03](#req-03-supplementary-material-ingestor)) is itself deterministic but irrelevant to invariant 1: it changes only _what context an SQ call sees_, never the judgment given fixed answers.
- **Batch idempotency** follows from invariant 4, not LLM determinism: the DB key `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` skips completed pairs regardless of byte-level replay ([REQ-16](#req-16-output-json-and-sqlite-writers)/[REQ-17](#req-17-batch-runner-and-manifest)).

### Orchestration

Assessment runs as two LangGraph graphs (trial tier and outcome tier; [REQ-12](#req-12-assessment-orchestration-two-tier-graph)). LangGraph is chosen for its **checkpoint/resume runway** toward the living-evidence deployment: v0.1 runs in-memory with **no checkpointer**, but the node/state structure is the seam a later resume layer plugs into. Runtime handles (LLM clients, the retrieval index, the trace collector) are passed as **ephemeral objects, never durable state** — a single rule that follows from invariant 3 and the no-checkpoint choice.

### The Two-Tier Structure

Because D1 (randomisation) is a property of the _trial_ while D2–D5 vary by _outcome_, ARBITER assesses in two tiers:

1. **Trial tier (once per trial):** ingest paper + supplements, fetch CT.gov, extract metadata, assess **D1**. (The trial tier is also where the deferred-to-v0.2 CONSORT vision node plugs in — see [§3](#3-users-and-workflow); v0.1 runs no vision here.)
2. **Outcome tier (once per outcome):** assess **D2, D3, D4, D5**, then roll up the overall judgment over the shared D1.

The principal saving is **ingestion reuse** (parse, supplement index, CT.gov, metadata), which dominates per-trial cost; sharing D1 saves only its three SQ calls. D1 is shared because randomisation happens once. D2 is **not** shared: its SQs (e.g. 2.4/2.5) reference "the outcome," and a deviation can affect one outcome and not another, so D2 is judged per outcome alongside D3–D5.

### What the LLM Does and Does Not Do

Per invariant 2, the LLM's one job is: for each signaling question, **find and quote the specific sentence(s)** that most directly address it, and **select an answer code**. It does **not** make any domain or overall judgment, decide which sections are relevant, decide which supplement segments to retrieve, or run the decision algorithm — all of those are deterministic.

### The Context Engineering Principle

LLM accuracy on RoB 2 SQs degrades as context broadens. ARBITER counters this with:

1. **Per-SQ context assembly** — each call receives only its domain's paper sections and supplement passages, never the full paper.
2. **A cached static prefix** — trial metadata, Methods/Results, and CT.gov data are marked cacheable and shared across a trial's SQ calls. The client **forwards** `cache_control` to OpenRouter and strips it for vanilla OpenAI ([REQ-06](#req-06-llm-abstraction-layer)); where a route cannot cache, the prefix structure is harmless.
3. **One question per call.**

### Three Phases (per trial)

- **Phase 1 — Ingestion:** parse the main paper and supplements, fetch CT.gov, extract metadata.
- **Phase 2 — Assessment:** trial tier (D1) then outcome tier (D2–D5 + rollup) per outcome.
- **Phase 3 — Output:** one JSON file and one SQLite row per trial-outcome pair.

---

## 3. Users and Workflow

### What v0.1 is for — build target vs measurement scope

Two goals interlock, and keeping them distinct prevents most scope confusion:

- **Build target (the product):** the assessment engine for a Living Evidence System (§2). The thesis is architectural — _build the harness well enough that a small/open model slotted into it rivals frontier direct-judgment._ Features that improve the pipeline are kept on that basis even before the eval can fully score them.
- **Measurement scope (what v0.1 proves):** the research result — the open-weight-vs-frontier head-to-head within the guaranteed-deterministic-judgment architecture ([REQ-21](#req-21-evaluation-harness)).

The interlock: **eval measurability scopes what v0.1 _claims_, not what it _builds_.** A pipeline-improving component the powered benchmark cannot yet isolate is reported as a _measured-but-conditionally-reported_ lever, not a headline claim — e.g. the hybrid retriever, which **ships on by default** while its recall@k is reported only as a per-domain characterisation ([REQ-03](#req-03-supplementary-material-ingestor)/[REQ-21](#req-21-evaluation-harness)). The same logic also runs the other way on cost: a high-complexity, vision-dependent lever the eval can only call `DIRECTIONAL` — **CONSORT vision** ([REQ-14](#req-14-context-assembly)) — keeps its _seam_ in v0.1 but **defers its build to v0.2** (see the CONSORT scope note below). Unattended production use is the build target, validated incrementally by the eval; it is **not** a v0.1 acceptance gate.

### Primary user

A **systematic-review / evidence-synthesis team** running ARBITER as an **automated batch tool** over a set of trials (e.g., a tumour-type literature base). Output is consumed **without a mandatory human-review gate** in v0.1 — but every human-facing artifact (the Markdown report [REQ-24](#req-24-reviewer-facing-markdown-report), `requires_human_review`, the confidence flags) is **advisory metadata for optional audit**, not a workflow gate. ARBITER is built so a reviewer _can_ audit any judgment via its verbatim quotes; it does not _require_ them to.

### Workflow

1. The team assembles a **manifest** (one row per trial: main paper, optional supplements/NCT/outcomes).
2. They run `arbiter batch <manifest>`.
3. ARBITER produces, per trial-outcome, a JSON record (with verbatim supporting quotes + page numbers + confidence flags) and a SQLite row.
4. Downstream tooling consumes the SQLite table (e.g., to build evidence maps or RoB summary figures). Flagged items _may_ be spot-checked but are not required to be.

### Honest limitation (v0.1)

Because output is consumed without a mandatory review gate, accuracy matters — and **v0.1 is a first-pass-draft tool, not an oracle**: confidence flags are advisory and uncalibrated, and every judgment is auditable via its verbatim quotes. The dev smoke-test (mHSPC-28) only proves the pipeline runs end-to-end; it is **never a published number**. Powered claims come from the paper eval ([REQ-21](#req-21-evaluation-harness)).

**CONSORT vision (scope — deferred to v0.2; v0.1 D3 is text-only).** CONSORT participant-flow _vision_ extraction is ARBITER's most complex, least-powered, vision-dependent component, and the eval can only ever characterise it as `DIRECTIONAL ONLY` ([REQ-21](#req-21-evaluation-harness)). So its **build is deferred to v0.2**: v0.1 ships D3 **text-only** — Results-text flow numbers / flow-diagram captions plus the CT.gov `enrollmentInfo.count` denominator (mechanics in [REQ-14](#req-14-context-assembly)), which is the default behaviour either way. What v0.1 **keeps is the seam** — the `ConsortFlow`/`ConsortExtraction` data models ([§5.3](#53-trial--outcome-models)), the `complete_vision` ABC stub + `supports_vision` flag ([REQ-06](#req-06-llm-abstraction-layer)), and the optional `model_vision` field — so the v0.2 vision vertical (deterministic figure detector, `consort_extract` node, `consort_vision.py` prompt, `ChatOpenRouter.complete_vision` impl, the conservative wrong-figure-worse-than-missed gate, and the A/B contribution metric) plugs in without rework. This is the "measurability scopes what v0.1 _claims/builds_" principle (above) applied from the cost side: a high-complexity lever the powered eval can't isolate keeps its seam now and defers its build.

---

## 4. Repository Structure

Implement the following layout. Module paths referenced throughout correspond to it.

```
arbiter/
├── arbiter/
│   ├── __init__.py               # Exports: ingest_trial(), assess_trial(), AssessmentConfig
│   ├── cli.py                    # Click CLI (assess, batch)
│   ├── config.py                 # AssessmentConfig, MODEL_REGISTRY, env vars
│   ├── models.py                 # All Pydantic models and enums
│   ├── manifest.py               # Manifest parsing (CSV/JSON) → list[TrialManifestEntry]
│   ├── ingestion/
│   │   ├── paper.py              # ingest_paper() → (SectionMap, raw_char_stream)
│   │   ├── supplements.py        # ingest_supplements() → SupplementIndex
│   │   ├── ctgov.py              # fetch_ctgov() → dict | None
│   │   └── metadata_extractor.py # extract_metadata() → TrialMetadata
│   ├── retrieval/
│   │   ├── segmenter.py          # segment_document() → list[SupplementSegment]
│   │   ├── annotator.py          # annotate_segment() (aux LLM call per segment)
│   │   └── supplement_index.py   # SupplementIndex class (hybrid BM25 + dense, RRF-fused)
│   ├── graph/
│   │   ├── state.py              # TrialState / OutcomeState (TypedDict) + reducers
│   │   ├── builder.py            # build_trial_graph(), build_outcome_graph()
│   │   └── nodes/                # context_assembly, sq_node, fanin, judgment, pre_d5, overall  (consort_extract: v0.2)
│   ├── arbiter_algorithm/        # rules transcribed ONCE from docs/rob2/ into pure Python (no runtime binary dependency)
│   │   ├── decision_tables.py    # judge_domain_1..5() — IRPG cell logic hand-extracted from the vendored .xlsm
│   │   ├── rollup.py             # compute_overall_judgment()
│   │   └── branching.py          # get_applicable_sqs(), get_na_sqs()
│   ├── llm/
│   │   ├── base.py               # LLMClient ABC + structured-output enforcement
│   │   ├── openai_client.py      # wraps ChatOpenAI (vanilla OpenAI only)
│   │   ├── openrouter_client.py  # wraps ChatOpenRouter (langchain-openrouter)
│   │   ├── anthropic_client.py   # wraps ChatAnthropic
│   │   └── mock_client.py        # MockLLMClient for tests
│   ├── prompts/
│   │   ├── system.py             # build_system_prompt()
│   │   ├── metadata_extraction.py
│   │   ├── supplement_annotation.py
│   │   ├── consort_vision.py     # CONSORT-figure vision prompt → ConsortFlow  (v0.2; seam only in v0.1)
│   │   └── sq_prompts.py         # SQ_PROMPTS keyed by (sq_id, effect), SQPromptTemplate
│   ├── confidence/
│   │   ├── quote_verifier.py     # verify_quote() → bool
│   │   └── signals.py            # compute_confidence() → ConfidenceSignals
│   ├── observability/            # SIDE-CHANNEL trace/timing — never touches the deterministic core (REQ-23)
│   │   ├── trace.py              # RunTrace collector + CallRecord / NodeSpan models + contextvar span ctx
│   │   └── cost.py               # token→cost from MODEL_REGISTRY prices; null-not-zero discipline
│   └── output/
│       ├── json_writer.py        # write_assessment_json()
│       ├── report_writer.py      # write_assessment_report() → reviewer-facing Markdown (REQ-24)
│       └── sqlite_writer.py      # write_assessment_sqlite() + idempotency helpers
├── docs/
│   └── rob2/                     # VENDORED official RoB 2 reference (IRPG; pinned; see REQ-07)
│       ├── README.md             # version + retrieval date + URLs + licence note (committed)
│       ├── rob2_irpg_algorithm.xlsm  # ROB2_IRPG_beta_v9 — decision tables + SQ routing (GIT-IGNORED)
│       ├── rob2_guidance.pdf     # Higgins et al., 22 Aug 2019 — verbatim SQ wording (GIT-IGNORED)
│       ├── rob2_cribsheet.pdf    # condensed reviewer guidance (GIT-IGNORED)
│       └── rob2_template.pdf     # blank assessment form (GIT-IGNORED)
│                                 # binaries are non-commercial/no-derivatives — not committed to this public repo
├── eval/
│   ├── reference/                # DEV smoke-test (mHSPC-28): overall_survival.csv, progression_free_survival.csv, adverse_events.csv
│   ├── benchmarks/               # PAPER eval: cochrane_mined/ (primary, domain+overall+quotes, traceable) + depth/ (own-built, 15–25, SQ-level, internal)
│   ├── enrichment/               # NCT links + open-access supplement URLs/DOIs (copyright-safe; PDFs git-ignored)
│   └── run_eval.py               # Dev smoke-test + paper-eval harness: per-SQ/domain/overall, quote-span, model roster, ablations (REQ-21)
├── tests/
│   ├── unit/  integration/  fixtures/
├── pyproject.toml
├── .env.example
└── README.md
```

> **In v0.1 (observability):** the side-channel run trace + timing/cost ([REQ-23](#req-23-run-trace--timing-instrumentation)) and the reviewer-facing Markdown report ([REQ-24](#req-24-reviewer-facing-markdown-report)) — strictly observability per invariant 3 (§2).
>
> **Not in v0.1 (planned; interfaces kept extension-ready):** the **CONSORT vision vertical** — `prompts/consort_vision.py`, the trial-tier `graph/nodes/consort_extract.py`, the figure detector, and `ChatOpenRouter.complete_vision` — deferred to v0.2 (§3); v0.1 keeps only the seam (inventory in [§3](#3-users-and-workflow)). Also: graph-level checkpoint/resume; refresh-on-change re-assessment for the living-evidence deployment ([REQ-17](#req-17-batch-runner-and-manifest)); a cross-encoder **reranker** over the hybrid-fused candidates ([REQ-03](#req-03-supplementary-material-ingestor)); a prompt-hash LLM response cache. **Artifact-reuse caching is deliberately rejected** — its stale-artifact bug class fights the "see the real data" goal; intermediates are dumped for inspection, not reused ([REQ-23](#req-23-run-trace--timing-instrumentation)).

---

## 5. Data Models

All models live in `arbiter/models.py`. Use **Pydantic v2**, except the LangGraph state TypedDicts in `graph/state.py`.

### 5.1 Enums

```python
class AnswerCode(str, Enum):
    Y = "Y"; PY = "PY"; PN = "PN"; N = "N"; NI = "NI"; NA = "NA"

class Judgment(str, Enum):
    LOW = "Low"; SOME_CONCERNS = "Some concerns"; HIGH = "High"

class ConfidenceFlag(str, Enum):
    CONFIDENT = "CONFIDENT"; UNCERTAIN = "UNCERTAIN"; FLAGGED = "FLAGGED"

class BlindingStatus(str, Enum):
    OPEN_LABEL = "open_label"; SINGLE_BLIND = "single_blind"
    DOUBLE_BLIND = "double_blind"; UNCLEAR = "unclear"

class EffectOfInterest(str, Enum):
    ASSIGNMENT = "assignment"; ADHERING = "adhering"

class DocType(str, Enum):
    SAP = "sap"; PROTOCOL = "protocol"; APPENDIX = "appendix"; UNKNOWN = "unknown"

class ParsingQuality(str, Enum):
    STANDARD = "standard"; DEGRADED = "degraded"

class StudyDesign(str, Enum):       # eligibility gate (REQ-05/REQ-17); only PARALLEL_RCT is in IRPG scope
    PARALLEL_RCT = "parallel_rct"; CLUSTER_RCT = "cluster_rct"
    CROSSOVER_RCT = "crossover_rct"; SINGLE_ARM = "single_arm"
    NON_RCT = "non_rct"; UNCLEAR = "unclear"
```

### 5.2 Ingestion & Retrieval Models

```python
class PageBox(BaseModel):
    boxclass: str          # "section-header" | "text" | "table" | "picture" | ...
    text: str
    bbox: tuple[float, float, float, float]
    page: int              # 0-based

class DocumentSection(BaseModel):
    label: str             # normalised heading, e.g. "RANDOMIZATION"
    pages: list[int]
    char_start: int
    char_end: int
    text: str
    domain_tags: list[str] = Field(default_factory=list)   # ["D1","D3"] etc.

class SectionMap(BaseModel):
    source_path: str
    full_text: str
    sections: list[DocumentSection]
    page_boxes: list[PageBox]
    parsing_quality: ParsingQuality = ParsingQuality.STANDARD

class SupplementSegment(BaseModel):
    segment_id: str        # "{filename}__{heading}__{idx}"
    source_file: str
    doc_type: DocType
    heading: str
    pages: list[int]
    raw_text: str
    annotation: str
    domain_tags: list[str]
    char_count: int

    @property
    def annotated_text(self) -> str:
        return self.annotation + "\n\n" + self.raw_text
```

### 5.3 Trial & Outcome Models

```python
class TrialMetadata(BaseModel):
    trial_id: str                  # deterministic: NCT → slugified trial_label → sha256(paper)[:12] (REQ-05)
    title: str
    intervention: str
    comparator: str
    all_outcomes: list[str]        # primary first, then secondary; capped at ARBITER_MAX_OUTCOMES (default 10)
    effect_of_interest: EffectOfInterest
    blinding: BlindingStatus
    nct_number: str | None = None
    study_design: StudyDesign = StudyDesign.UNCLEAR   # IRPG eligibility (REQ-05/REQ-17)
    study_design_basis: str | None = None             # one-sentence LLM basis for the classification

class ConfidenceSignals(BaseModel):
    supplement_segments_retrieved: int
    supplement_segments_available: int
    retrieval_top_score: float | None
    quote_verified: bool
    flag: ConfidenceFlag
    flag_reason: str | None = None

class OutcomeComparison(BaseModel):     # REQ-13 pre-D5 result; every field None when CT.gov data is absent
    registered_outcome: str | None = None      # best-matching registered measure
    published_outcome: str | None = None       # the assessed outcome string
    outcome_similarity_score: float | None = None   # best match, normalised 0–1 (REQ-13)
    outcome_change_detected: bool | None = None     # best_score < ARBITER_OUTCOME_MATCH_THRESHOLD
    registered_as_primary: bool | None = None       # whether the best match came from the primary list

class SQAnswer(BaseModel):
    sq_id: str                     # "1.1", "2.3", "5.2" ...
    answer: AnswerCode
    quote: str                     # verbatim; empty for NA/NI
    page: int | None               # 0-based; first page of the quote span (quotes may cross a page break); None for NA/NI. DERIVED DETERMINISTICALLY post-verification from page_boxes, never LLM-supplied (REQ-10/REQ-15)
    justification: str             # exactly one sentence
    confidence: ConfidenceSignals

class SQRawAnswer(BaseModel):
    """Schema the LLM SQ call must satisfy (post-validation). `page` is NOT
    requested from the model — it is derived deterministically from the verified
    quote's location in `page_boxes` (REQ-10/REQ-15), so the LLM's only outputs
    are the quote, the answer code, and the justification."""
    answer: AnswerCode
    quote: str = Field(max_length=400)
    justification: str = Field(max_length=200)

class DomainJudgment(BaseModel):
    domain: str                    # "D1" ... "D5"
    scope: Literal["trial", "outcome"]
    judgment: Judgment
    algorithm_rationale: str
    sq_answers: list[SQAnswer]

class ConsortFlow(BaseModel):          # the extracted participant-flow counts (REQ-14). KEPT SEAM — only populated by the v0.2 vision vertical; unused in v0.1.
    randomised: int | None = None
    allocated_intervention: int | None = None
    allocated_control: int | None = None
    lost_intervention: int | None = None
    lost_control: int | None = None
    analysed_intervention: int | None = None
    analysed_control: int | None = None

class ConsortExtraction(BaseModel):    # one nested object localises the whole CONSORT feature's state (REQ-12/14). KEPT SEAM — in v0.1 always None (no vision); the v0.2 vertical populates it.
    detected: bool = False             # CONSORT-flow figure scored ≥ threshold → vision fired (audit)
    detection_score: float | None = None  # detector's best score (audit); None when CONSORT disabled
    flow: ConsortFlow | None = None    # extracted counts; None if vision didn't fire (text-only fallback)
```

### 5.4 The Assessment Record (flat: one per trial-outcome)

```python
class SourcesManifest(BaseModel):
    main_paper: str
    supplements: list[str]
    ct_gov_retrieved: bool
    parsing_quality: ParsingQuality

class Assessment(BaseModel):
    assessment_id: str             # uuid4
    created_at: str                # ISO 8601
    pipeline_version: str
    model_sq: str                  # model used for SQ judgments
    model_aux: str                 # model used for auxiliary calls
    model_vision: str | None       # model used for CONSORT vision; KEPT SEAM — always None in v0.1 (vision deferred to v0.2)

    trial_id: str
    nct_number: str | None
    outcome: str                   # THE outcome this record assesses

    requires_human_review: bool    # advisory
    config_summary: dict
    trial_metadata: TrialMetadata
    ct_gov_data: dict | None               # verbatim CT.gov v2 JSON — stays an untyped dict by design (REQ-04)
    outcome_comparison: OutcomeComparison | None

    # D1 is trial-level (reused identically across this trial's outcomes);
    # D2 + D3 + D4 + D5 are specific to `outcome`.
    domain_judgments: list[DomainJudgment]   # exactly D1..D5
    overall_judgment: Judgment
    overall_rationale: str

    sources_manifest: SourcesManifest
    errors: list[str]
```

`domain_judgments` always contains five entries; the D1 entry carries `scope="trial"` and is byte-identical across a trial's outcome records by construction. D2–D5 carry `scope="outcome"` and may differ across a trial's outcomes.

#### The skip record (ineligible trials)

An `Assessment` requires a full D1–D5 + overall judgment, which a trial rejected by the eligibility gate ([REQ-17](#req-17-batch-runner-and-manifest)) never produces — the gate fires _before_ any domain is assessed. Ineligible trials therefore emit a **`SkipRecord`, not an `Assessment`**, keeping the `Assessment` happy-path free of nullable judgment fields:

```python
class SkipRecord(BaseModel):
    assessment_id: str             # uuid4 (per-write surrogate)
    created_at: str                # ISO 8601
    pipeline_version: str
    trial_id: str                  # deterministic (REQ-05)
    nct_number: str | None
    study_design: StudyDesign      # the reason it was skipped
    study_design_basis: str | None # one-sentence LLM basis
    requires_human_review: bool = True
    errors: list[str]              # e.g. ["ineligible study_design=single_arm: <basis>"]
    sources_manifest: SourcesManifest
```

A `SkipRecord` is written as **one SQLite row per trial** with a sentinel `outcome = "__TRIAL__"` (the trial has no eligible outcome to scope) and `NULL` in every judgment column (see [REQ-16](#req-16-output-json-and-sqlite-writers)), plus a `skip.json` at `{output_dir}/{trial_id}/skip.json` for audit parity. No per-outcome rows, no `data.json`.

### 5.5 Manifest Models

```python
class TrialManifestEntry(BaseModel):
    main_paper: Path                       # REQUIRED
    supplements: Path | None = None        # optional; a file OR a directory of PDFs
    nct_number: str | None = None          # optional; else derived from the paper
    outcomes: list[str] | None = None      # optional; else defaults to primary outcome
    trial_label: str | None = None         # optional human label (e.g. "ARASENS")

class BatchManifest(BaseModel):
    entries: list[TrialManifestEntry]
```

### 5.6 LangGraph State

Two TypedDicts — one per tier. Per §2 (Orchestration + invariant 3): no LLM client, retrieval index, or `RunTrace` collector is stored as durable state — they are ephemeral runtime handles (no checkpointing in v0.1) — and no trace/timing data appears in the reducers or the `Assessment` record.

```python
# The trial-static ingestion fields are shared by both tiers. They live in ONE base
# TypedDict that both tiers inherit, so "all TrialState ingestion fields" is a real
# structural seam, not a comment to keep in sync by hand (add an ingestion artifact
# once, both tiers get it). The per-tier fields below differ in semantics (D1-only vs
# D2–D5 accumulation), so they are declared per tier.
class _IngestionState(TypedDict):
    config_summary: dict
    trial_metadata: TrialMetadata
    section_map: SectionMap
    raw_char_stream: str
    supplement_index: SupplementIndex      # runtime handle
    ct_gov_data: dict | None
    shared_prefix_text: str                # trial-static cacheable prefix (trial metadata + Methods/Results + ct_gov_block), byte-identical across every domain/SQ call; built ONCE in Phase 1 (REQ-19) per the REQ-14 spec, read (never rebuilt) by the per-domain context nodes, hashed once for caching + trace (REQ-06/23)
    ct_gov_block: str | None               # trial-static rendered CT.gov text, folded into shared_prefix_text
    llm_client_sq: LLMClient               # runtime handle
    llm_client_aux: LLMClient              # runtime handle

# CONSORT vision is a TRIAL-TIER operation producing a trial-static ConsortFlow (REQ-14),
# so its seam fields live on TrialState ONLY — not the shared base. This keeps the outcome
# tier from carrying an inert vision client handle + an always-None `consort` slot in v0.1.
# In v0.2 the resulting ConsortFlow is seeded into each OutcomeState as a plain value, exactly
# the way the reused D1 (`trial_domain_judgments`) is — so the outcome tier reads it without
# inheriting a live field.
class TrialState(_IngestionState):     # trial tier: D1 (+ v0.2 CONSORT extraction)
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]   # D1 only
    errors: Annotated[list[str], operator.add]
    llm_client_vision: LLMClient | None    # runtime handle (CONSORT extraction); KEPT SEAM — None in v0.1 (no vision)
    consort: ConsortExtraction | None      # detection + extracted flow as one object (once per trial); KEPT SEAM — always None in v0.1, populated by the v0.2 vision vertical

class OutcomeState(_IngestionState):   # outcome tier: D2, D3, D4, D5 + rollup
    # inherits every _IngestionState field, plus (v0.2: the trial-tier ConsortFlow is seeded
    # here as a plain value, like the reused D1 below — not inherited from the base):
    outcome: str
    trial_domain_judgments: list[DomainJudgment]   # the reused D1
    outcome_change_detected: bool | None
    registered_outcome: str | None
    published_outcome: str | None
    outcome_similarity_score: float | None
    registered_as_primary: bool | None
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]   # D2, D3, D4, D5
    overall_judgment: Judgment | None
    overall_rationale: str | None
    requires_human_review: bool | None
    errors: Annotated[list[str], operator.add]

class DomainContext(BaseModel):
    domain: str
    # NOTE: the cacheable shared prefix (trial metadata + Methods/Results + ct_gov_block)
    # is NOT held here — it is trial-static and byte-identical across all five domains, so it
    # lives once on the state (TrialState.shared_prefix_text / ct_gov_block) and is composed
    # with this domain's suffix at call-build time (REQ-14/REQ-15). Holding it per-domain would
    # re-embed the ~4k-token prefix 5× in the full-trace domain_contexts.json dump — the exact
    # duplication REQ-23's prefix_hash dedup exists to avoid.
    domain_specific_text: str             # only THIS domain's extra sections (e.g. RANDOMIZATION); part of the dynamic suffix
    supplement_block: str | None = None
    outcome_comparison_block: str | None = None
    retrieval_top_score: float | None = None  # NORMALISED [0,1] relevance of the RRF-top passage (NOT the raw RRF score); None when no retrieval ran. Full definition: REQ-03/REQ-11.
    segments_retrieved: int = 0
    segments_available: int = 0

# Plain runtime bundle (NOT LangGraph state, NOT checkpointed) returned by `ingest_trial`
# (REQ-19) and consumed by `check_eligibility` (REQ-17) then `assess_trial`. It exists so
# Phase 1 runs exactly once and the eligibility gate can read `trial_metadata` BEFORE any
# domain is judged. The LLM clients, `supplement_index`, and `trace` are runtime handles;
# none of this is durable state. The clients + trace are created once per trial inside
# `ingest_trial` (which makes the first aux calls) and reused by `assess_trial`.
@dataclass
class TrialContext:
    config_summary: dict
    trial_metadata: TrialMetadata
    section_map: SectionMap
    raw_char_stream: str
    supplement_index: SupplementIndex      # runtime handle
    ct_gov_data: dict | None
    shared_prefix_text: str                # built once (REQ-14), seeded onto every state by assess_trial
    ct_gov_block: str | None
    llm_client_sq: LLMClient               # runtime handle (trace attached)
    llm_client_aux: LLMClient              # runtime handle (trace attached)
    trace: RunTrace | None                 # per-trial collector; assess_trial flushes it (REQ-23)
```

---

## REQ-01: Dependency and Project Setup

### What to Build

A `pyproject.toml` declaring dependencies, optional provider extras, and the CLI entrypoint; and a `.env.example` with all environment variables. **The project is managed with `uv`** (`uv add` for dependencies, `uv sync --extra <name>` to install extras) — not pip.

### Dependencies (intent, not a lockfile)

- **Orchestration:** `langgraph`, `langchain-core` _(no `langgraph-checkpoint-sqlite` — v0.1 does not checkpoint)_
- **PDF parsing:** `pymupdf`, `pymupdf4llm`
- **Retrieval:** `bm25s`, `nltk`; **hybrid dense** via `FlagEmbedding` (BGE-M3) or `sentence-transformers` — local, CPU-runnable, pinned weights (BM25 stays the deterministic baseline arm; see [REQ-03](#req-03-supplementary-material-ingestor))
- **Fuzzy matching:** `rapidfuzz`
- **Validation:** `pydantic >= 2`
- **HTTP:** `httpx`
- **CLI:** `click`
- **Env:** `python-dotenv`
- **LLM providers (optional extras):** `langchain-anthropic` (extra `anthropic`), `langchain-openai` (extra `openai`), `langchain-openrouter` (extra `openrouter`). The concrete `LLMClient` classes wrap these LangChain chat integrations — one per provider, OpenRouter dedicated ([REQ-06](#req-06-llm-abstraction-layer)); `all` installs all three.
- **Dev:** `pytest`, `pytest-asyncio`

The CLI entrypoint is `arbiter = "arbiter.cli:cli"`.

### config.py

```python
@dataclass
class AssessmentConfig:
    paper_path: Path
    supplement_paths: list[Path] = field(default_factory=list)
    nct_number: str | None = None
    outcomes: list[str] | None = None            # None → [primary outcome]
    effect_of_interest: str = "assignment"       # "assignment" | "adhering"
    sq_model: str = "gpt-oss-120b"               # accuracy-critical SQ judgments (HEADLINE open-weight SUT; see note)
    aux_model: str = "gpt-oss-120b"              # metadata + supplement annotation (open-weight)
    vision_model: str | None = None              # CONSORT-figure extraction model. KEPT SEAM — inert in v0.1 (vision deferred to v0.2); v0.2 resolves a vision-capable slug
    consort_vision_enabled: bool = False         # KEPT SEAM — always False/inert in v0.1 (D3 is text-only); the v0.2 vision vertical activates it
    sq_max_tokens: int = 2048                    # completion budget per SQ call; MUST cover reasoning-model CoT (gpt-oss is Harmony/reasoning) before the JSON, or the answer truncates. Output itself is tiny; the headroom is the point.
    output_dir: Path = Path("./output")
    db_path: Path = Path("./arbiter.db")
    force: bool = False                          # re-run even if already in DB
    trace_level: str = "full"                    # "off" | "summary" | "full" (REQ-23); CLI default full for assess, summary for batch
    report_enabled: bool = True                  # write reviewer-facing Markdown report (REQ-24); --no-report disables
```

> **`pipeline_version` is the umbrella identifier for the pipeline _configuration_, not just the code.** The batch idempotency key ([REQ-16](#req-16-output-json-and-sqlite-writers)/[REQ-17](#req-17-batch-runner-and-manifest)) keys on `sq_model` (the headline variable the eval sweeps) + `pipeline_version` + `(trial_id, outcome, effect_of_interest)`. It deliberately does **not** include `aux_model`, `vision_model`, `consort_vision_enabled`, or the retriever knobs — so changing any of those and re-running would otherwise **silently skip** the existing (now-stale) row. The rule: **bump `pipeline_version` whenever `aux_model`, `vision_model`, `consort_vision_enabled`, or the retriever config changes** (any change to those exact dims — there is no "material vs immaterial" judgment call). This keeps the resume key small and human-readable while making config changes an explicit, versioned act. **On the eval path the bump is _automatic_:** the [REQ-21](#req-21-evaluation-harness) harness derives a distinct `pipeline_version` per roster arm by hashing these non-keyed dims, so an unattended tens-of-thousands-of-calls sweep — where `aux_model` legitimately varies per arm — **cannot silently skip or clobber** a stale row. Manual bumping applies only to hand-run `assess`/`batch`, where a human is choosing the config anyway.

> **The open-weight model is the system-under-test, not a dev convenience.** ARBITER's headline research question is **how open-weight models (e.g. `gpt-oss-120b`) perform on RoB 2 relative to frontier APIs and the human inter-rater ceiling, within a guaranteed-deterministic-judgment architecture** — reported **descriptively** (see [REQ-21](#req-21-evaluation-harness); ARBITER asserts **no** pre-registered "competitive" pass/fail claim) — and that, because open weights are downloadable, the same pipeline _can_ be deployed fully on-premise (an affordance establishing the feasibility of zero-API-cost, privacy-preserving RoB 2 automation; we evaluate via hosted APIs for accessibility and do **not** claim an air-gapped run). So `gpt-oss-120b` being the default `sq_model` is **deliberate** — it is the headline model — and the frontier models (a current frontier Claude (Sonnet-class), a GPT-5-class model) are the **control arm / ceiling**, not "production." The eval roster (REQ-21) runs open and frontier models through the same pipeline **code** as _whole-pipeline_ arms — each arm uses its own model for both `sq_model` and `aux_model` (see the whole-pipeline note in [REQ-21](#req-21-evaluation-harness)). Pin **dated snapshots** and, for OpenRouter-routed open models, a single `provider`, so reported numbers are reproducible.

`MODEL_REGISTRY` maps model name → `{provider, base_url?, supports_cache, supports_native_schema, supports_vision, price_per_mtok_in?, price_per_mtok_out?, price_per_mtok_cache_read?, price_per_mtok_cache_write?}`. The optional price fields back the cost instrumentation in [REQ-23](#req-23-run-trace--timing-instrumentation): when a price is **absent**, cost is reported as `null` ("pricing unknown"), **never `0`** — only genuinely free models (gpt-oss, Gemma free tier) carry explicit `0`. v0.1 entries:

Eval roster — **open-weight (system-under-test)** run alongside a **frontier control arm** through the same pipeline code as whole-pipeline arms (see [REQ-21](#req-21-evaluation-harness)):

| model                          | provider   | role                         | supports_cache     | supports_native_schema | supports_vision |
| ------------------------------ | ---------- | ---------------------------- | ------------------ | ---------------------- | --------------- |
| `gpt-oss-120b`                 | openrouter | **headline (open)**          | routing-dependent¹ | False                  | False           |
| `gpt-oss-20b`                  | openrouter | floor / intra-family         | routing-dependent¹ | False                  | False           |
| `qwen3-32b`                    | openrouter | cross-family peer (open)     | routing-dependent¹ | True                   | False           |
| `llama-3.3-70b-instruct`       | openrouter | prior-work continuity (open) | routing-dependent¹ | partial                | False           |
| `qwen3-235b-a22b`              | openrouter | open "ceiling" (cluster)     | routing-dependent¹ | True                   | False           |
| frontier Claude (Sonnet-class) | anthropic  | **frontier control**         | True               | True                   | True            |
| GPT-5-class model              | openai     | **frontier control**         | True               | True                   | True            |
| `google/gemma-4-31b-it:free`   | openrouter | CONSORT vision **(v0.2)**    | routing-dependent¹ | False                  | **True**        |

¹ OpenRouter caches via `cache_control` breakpoints **when routed to a caching-capable provider**; pin one via OpenRouter's `provider` routing arg. The client **forwards** `cache_control` rather than stripping it (see [REQ-06](#req-06-llm-abstraction-layer)).

> **Model names denote roles — resolve to real, dated snapshots at implementation time.** The frontier rows ("frontier Claude", "GPT-5-class model") are _roles_, not release IDs; the `google/gemma-4-31b-it:free` vision default is a **v0.2** named slug to confirm against OpenRouter (vision-capable; free-tier rate limits apply under batch use) — unused in v0.1. The open slugs (`gpt-oss-120b`/`20b`, `qwen3-32b`/`235b`, `llama-3.3-70b`) are real models but should still be pinned to dated snapshots. Before any run, resolve each role and slug to a concrete provider snapshot. `MODEL_REGISTRY` is the single place these are pinned.

> The open-weight slugs are the artifacts a hospital could self-host — the table doubles as the local-deployability evidence. The 235B "open ceiling" deliberately exceeds single-GPU footprint: it separates the _openness_ question from the _scale_ question and is **excluded from the local-deployability claim**. Native-schema is `False` for the gpt-oss family (Harmony format), so REQ-06's hybrid structured-output path is **headline-critical**, not a fallback; Qwen3's native JSON support lets the eval **report per-model schema-repair rates** as a reproducibility signal. Vision is `False` for every open SUT model — in v0.1 **all** models are assessed **text-only on D3** (CONSORT vision is the v0.2 vertical, run on a separate vision model regardless of `sq_model`; the gemma row is the kept seam, inert in v0.1). Pin dated snapshots for the frontier APIs (they get deprecated; open weights are re-downloadable forever — a reproducibility asymmetry that favours the open arm).

`provider == "openrouter"` uses the dedicated `ChatOpenRouter` wrapper (`langchain-openrouter`) with `OPENROUTER_API_KEY` — **not** the OpenAI client.

### .env.example (variables)

**Every operational threshold/cap/budget below is an env knob with a conservative default — no behaviour-governing magic numbers are hard-coded.** The defaults are tuned for accuracy-first, free-tier-cheap operation; they are exposed so the [REQ-21](#req-21-evaluation-harness) ablations can sweep them rather than a code edit. (Two numeric constants are deliberately **not** env knobs because they are _policy/contract_, not operational tuning: `OVERALL_HIGH_SC_THRESHOLD` — owned by an ADR + bumps `pipeline_version`, [REQ-07](#req-07-rob-2-algorithm-deterministic-judgments) — and the `SQRawAnswer` `quote`/`justification` `max_length`s, which are part of the data contract, [§5.3](#53-trial--outcome-models).)

- **Keys / models / paths:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`; `ARBITER_SQ_MODEL`, `ARBITER_AUX_MODEL`, `ARBITER_VISION_MODEL` (v0.2 seam); `ARBITER_OUTPUT_DIR`, `ARBITER_DB_PATH`.
- **Context / extraction budgets (tokens):** `ARBITER_PREFIX_TOKEN_BUDGET` (4000; cacheable shared prefix, [REQ-14](#req-14-context-assembly)), `ARBITER_SUPPLEMENT_TOKEN_BUDGET` (2000; retrieved-supplement block, REQ-14), `ARBITER_METADATA_TOKEN_BUDGET` (3000; metadata-extraction input, [REQ-05](#req-05-trial-metadata-extractor)), `ARBITER_SQ_MAX_TOKENS` (2048; per-SQ completion budget — must cover reasoning-model CoT, [REQ-01](#req-01-dependency-and-project-setup)/[REQ-06](#req-06-llm-abstraction-layer)), `ARBITER_MAX_OUTCOMES` (10; cap on `all_outcomes`, REQ-05), `ARBITER_DOMAIN_TEXT_MIN_CHARS` (500; **chars** — below this the abstract is prepended to a domain's `domain_specific_text`, [REQ-14](#req-14-context-assembly)).
- **Retrieval:** `ARBITER_RETRIEVAL_TOP_K` (5), `ARBITER_SMALL_SEGMENT_TOKEN_THRESHOLD` (1500; below this a segment is included verbatim), `ARBITER_LARGE_SEGMENT_CHAR_THRESHOLD` (6000; above this a segment is sentence-sub-ranked, REQ-14), `ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD` (0.35; **normalised [0,1]** relevance floor — strongest single-retriever component of the RRF-top passage, **not** the raw RRF score — for the `UNCERTAIN`/`FLAGGED` rules, [REQ-11](#req-11-confidence-signal-system)).
- **Parsing heuristics ([REQ-02](#req-02-main-paper-ingestor)/[REQ-03](#req-03-supplementary-material-ingestor)):** `ARBITER_SUPPLEMENT_PARSE_WINDOW` (20; page-window size for per-window degraded fallback), `ARBITER_DOCTYPE_SCAN_PAGES` (10; doc-type lexicon scan window), `ARBITER_MIN_SEGMENTS` (3; below this a document is treated as one whole-doc segment), `ARBITER_DOMAIN_TAG_SCAN_CHARS` (300; body chars scanned for domain-tag keywords).
- **Verification / matching:** `ARBITER_QUOTE_VERIFY_THRESHOLD` (85; `partial_ratio` cutoff, 0–100, [REQ-10](#req-10-quote-verifier)), `ARBITER_QUOTE_MIN_VERIFY_CHARS` (15; quotes shorter than this auto-verify), `ARBITER_OUTCOME_MATCH_THRESHOLD` (0.85; normalised 0–1 registered-vs-published cutoff, [REQ-13](#req-13-pre-d5-outcome-comparison)).
- **Retries / annotation:** `ARBITER_SCHEMA_REPAIR_MAX_RETRIES` (2; non-native schema repair ladder, REQ-06), `ARBITER_NETWORK_MAX_RETRIES` (3; transport backoff, [REQ-20](#req-20-error-handling-and-retry)), `ARBITER_MAX_ANNOTATIONS_PER_DOC` (40; per-document annotation-call cap, REQ-03), `ARBITER_ANNOTATION_PREAMBLE_TOKENS` (500; tokens of static document-preamble prefix per annotation call, REQ-03).
- **CONSORT (v0.2 — defined but inert in v0.1; D3 text-only, [§3](#3-users-and-workflow)):** `ARBITER_CONSORT_DETECT_THRESHOLD`, `ARBITER_CONSORT_ENABLED`.
- **Concurrency / observability:** `ARBITER_MAX_CONCURRENCY` (**max concurrent in-flight LLM calls** — a single global async semaphore in the LLM client, REQ-06/[REQ-17](#req-17-batch-runner-and-manifest); sized for OpenRouter free-tier rate limits), `ARBITER_TRACE_LEVEL` (`off` | `summary` | `full`; [REQ-23](#req-23-run-trace--timing-instrumentation) — CLI default `full` for `assess`, `summary` for `batch`), `ARBITER_REPORT_ENABLED` (`true`; write the reviewer-facing Markdown report, [REQ-24](#req-24-reviewer-facing-markdown-report)).

### Acceptance Criteria

- `uv sync --extra anthropic`, `--extra openai`, and `--extra openrouter` complete without error.
- `arbiter --help` prints usage.
- All `.env.example` variables are read in `config.py` with sensible defaults.
- A run with `sq_model = aux_model = gpt-oss-120b` works end-to-end against OpenRouter.
- **Do not** depend on any checkpoint package; there is no resume in v0.1.

---

## REQ-02: Main Paper Ingestor

**Module:** `arbiter/ingestion/paper.py`
**Signature:** `def ingest_paper(path: Path) -> tuple[SectionMap, str]`

### What to Build

Parse the main RCT paper into a `SectionMap` (layout-aware, section-labelled, reading-order-corrected text — what the LLM sees) **plus** a raw character stream (unprocessed `pymupdf` text — used only for fuzzy quote verification, because it best matches what a human finds when searching the PDF).

### Behavioural Requirements

- Produce `SectionMap.full_text` by concatenating page texts in order; track running character offsets so each `DocumentSection` has correct `char_start`/`char_end`.
- Detect sections from layout `section-header` boxes; normalise headings to uppercase, stripped of surrounding whitespace/punctuation.
- **Domain-tag** each section using `SECTION_KEYWORDS` (one keyword list per domain plus METHODS/RESULTS). A section is tagged for domain D if any of D's keywords appears in its normalised label or first `ARBITER_DOMAIN_TAG_SCAN_CHARS` (default 300) characters of body. Multiple tags allowed.
- **Degraded fallback:** if layout parsing raises, fall back to raw `pymupdf` text as a single `FULL_TEXT` section tagged for all domains, set `parsing_quality = DEGRADED`, and return — never raise.
- Extract the NCT number opportunistically here too (regex `NCT\d{8}`), exposed for the metadata extractor.

### Acceptance Criteria

- Returns a `SectionMap` with ≥1 section and non-empty `full_text` for any valid PDF.
- The raw char stream contains the same textual content (formatting may differ).
- A corrupt/unreadable PDF returns the degraded fallback, not an exception.
- Section labels are uppercase. METHODS and RESULTS sections exist for a standard RCT fixture.

---

## REQ-03: Supplementary Material Ingestor

**Module:** `arbiter/ingestion/supplements.py`
**Signature:** `async def ingest_supplements(paths: list[Path], aux_client: LLMClient) -> SupplementIndex`

The most complex ingestion step. Read fully before implementing.

### What to Build

For each supplementary PDF, produce a `SupplementIndex` of domain-tagged, contextually annotated segments backed by an in-memory hybrid index (BM25S + dense, RRF-fused). Supplements can be hundreds of pages; injecting them wholesale into every SQ call is too expensive and harms accuracy (lost-in-the-middle). Instead, segment → tag → annotate → index, then retrieve only relevant passages at query time.

### Input Handling

`paths` may contain **files or directories**. For any directory, glob all `*.pdf` within it (non-recursive is sufficient; document the choice). **The caller never categorises PDFs** — ARBITER infers each document's type itself (below).

### Pipeline (per document)

- **Parse** via the same layout extraction as `ingest_paper`, but in **bounded page windows** (`ARBITER_SUPPLEMENT_PARSE_WINDOW`, default 20 pages) so a single bad page does not lose the whole document. Degraded fallback is **per-window, not per-document**: a window that fails layout parsing becomes a `DEGRADED` segment carrying its raw `pymupdf` text, tagged for all domains, while the remaining windows parse normally. (Supplements are the highest-value documents to protect — concealment (D1), analysis population (D2), and pre-specified plan (D5) evidence lives in fat SAPs/protocols that are exactly where whole-document fallback would lose the most. The main paper (REQ-02) keeps whole-document fallback — it is short enough that the blast radius is small.)
- **Detect document type (rule-based, no LLM):** score the text of `section-header` boxes on the first `ARBITER_DOCTYPE_SCAN_PAGES` (default 10) pages against three lexicons (`sap`, `protocol`, `appendix`); highest score wins; tie → `sap`; no headers → `unknown` (default SAP lexicon).
- **Segment** at each `section-header` boundary (heading, page range, body until next header). Domain-tag each segment with `SECTION_KEYWORDS` (heading + first `ARBITER_DOMAIN_TAG_SCAN_CHARS` chars). If a document yields fewer than `ARBITER_MIN_SEGMENTS` (default 3) segments, treat the whole document as one segment tagged for all domains.
- **Annotate (one aux LLM call per _domain-tagged_ segment — bounded):** annotation is an **enrichment, not a gate**. To avoid a cost/latency explosion on fat protocols (a 200-page SAP can segment into 100+ chunks), annotate **only segments carrying ≥1 `domain_tag`**, and cap the number of annotation calls per document at `ARBITER_MAX_ANNOTATIONS_PER_DOC` (default 40), prioritising segments by domain-tag count. Untagged or over-cap segments get an empty annotation and are **still indexed on raw text**. For each annotated segment: with a static document-preamble prefix (title page + first `ARBITER_ANNOTATION_PREAMBLE_TOKENS`, default 500, tokens) and a per-segment suffix, ask for 2–3 sentences naming the methods/populations/procedures relevant to randomisation, blinding, missing data, outcome assessment, or selective reporting. If a segment has no RoB-relevant content, the model returns exactly `"No risk-of-bias relevant content."`. Store as `annotation`.
- **Index:** build a `SupplementIndex` over each segment's `annotated_text` (annotation + raw text). Because **every** segment is indexed on raw text regardless of annotation, BM25 recall is unaffected by the annotation cap.

> **Retrieval is local hybrid (BM25 + dense) in v0.1, with BM25 as the deterministic baseline arm.** Sparse (BM25) and dense embeddings capture _orthogonal_ relevance signals; RRF fusion of the two consistently improves recall (+15–30% across 2026 benchmarks), while BM25 alone still beats dense alone on jargon-dense technical text — which describes SAPs/protocols exactly (rare-IDF terms: allocation concealment, IWRS, MNAR, per-protocol). The dense side uses **BGE-M3** (MIT-licensed, ~560M, local/CPU-runnable, 8192-token window) which emits dense + lexical-sparse + multi-vector from **one** model, so "hybrid" is a single dependency; the curated `key_terms` plus the annotation step further bridge vocabulary gaps. **Retrieval lives in the best-effort layer _upstream_ of the LLM** — it changes only what the SQ call sees, never the deterministic judgment (which depends solely on SQ answers → decision tables), and pinned BGE-M3 weights are themselves reproducible, so adopting dense does not weaken the deterministic-core claim ([§2](#2-system-overview)). **Hybrid BM25 + dense + RRF is the committed v0.1 default retriever — it is _not_ gated on the eval.** The [REQ-21](#req-21-evaluation-harness) recall@k ablation (BM25 vs dense vs hybrid, against **ARBITER-Depth annotated passages only** — we annotate these directly, so no support-quote localization is needed) is reported per domain as a **characterisation** of the retriever's behaviour (n=15–25, not powered), and informs the parked reranker decision; it does **not** decide the shipped retriever. A cross-encoder **reranker** (e.g. `bge-reranker-v2`) — the single biggest recall gain — is the planned next rung, held back for v0.1 on dependency/latency grounds. _`SupplementIndex` keeps its interface; the sparse/dense fusion is internal._

### SupplementIndex

`SupplementIndex.retrieve(query_terms: list[str], domain: str, top_k: int = 5) -> tuple[list[SupplementSegment], float | None]` returns the top-k segments (ranked by **RRF fusion** of the BM25 and dense lists) and a **normalised [0,1] relevance score** for the top segment — the strongest single-retriever component (min-max-normalised BM25 score or dense cosine), **not** the raw RRF value, which is a rank artefact unsuitable as a confidence magnitude ([REQ-11](#req-11-confidence-signal-system)). Returns `None` for that score when the index is empty (no passage to score). It filters to segments tagged with `domain` first; if fewer than 2 such segments exist, it falls back to the full set. RRF stays the **ranking** mechanism; only the surfaced confidence score is normalised. The index is in-memory only; **never serialised to disk**.

### Acceptance Criteria

- Returns a `SupplementIndex` for any list of paths, including `[]` → empty index, and a directory of PDFs.
- Each segment has non-empty `annotation` and `raw_text` (`domain_tags` may be empty).
- `retrieve(["concealment","allocation"], "D1", top_k=5)` returns ≤5 segments.
- A document with no section headers yields one full-document segment.
- A supplement with **one unparseable page still yields usable segments from the rest of the document** (per-window degraded fallback, not whole-document loss).

---

## REQ-04: ClinicalTrials.gov Fetcher

**Module:** `arbiter/ingestion/ctgov.py`
**Signature:** `async def fetch_ctgov(nct_number: str) -> dict | None`

### What to Build

Fetch a study from the ClinicalTrials.gov v2 REST API (`GET https://clinicaltrials.gov/api/v2/studies/{nct}`) and return the full JSON dict. On any HTTP or network error, **log a warning and return `None`** — never abort the assessment. Store the response verbatim; downstream nodes read fields defensively with `.get()`.

### Fields Used Downstream

`protocolSection.outcomesModule.primaryOutcomes[*].measure` / `.timeFrame` **and** `secondaryOutcomes[*]` (pre-D5 matches the assessed outcome against the **full** registered set — primaries ∪ secondaries — not just `primaryOutcomes[0]`; see [REQ-13](#req-13-pre-d5-outcome-comparison)); `designModule.designInfo.maskingInfo` (D2/D4); `designModule.designInfo.allocation` (D1); `designModule.enrollmentInfo.count` (**randomised-N denominator hint injected into D3 context**, see [REQ-14](#req-14-context-assembly)); `armsInterventionsModule.armGroups` (metadata).

### Acceptance Criteria

- Returns a dict for a valid NCT (e.g., `NCT01234567`); `None` for an invalid NCT or network error — does not raise.
- Response is stored verbatim; no field extraction at this stage.

---

## REQ-05: Trial Metadata Extractor

**Module:** `arbiter/ingestion/metadata_extractor.py`
**Signature:** `async def extract_metadata(section_map: SectionMap, config: AssessmentConfig, aux_client: LLMClient, nct_hint: str | None) -> TrialMetadata`

### What to Build

Extract structured trial metadata from the main paper using one **aux** LLM call over the abstract + methods text (capped at `ARBITER_METADATA_TOKEN_BUDGET`, default 3,000, tokens; fall back to the first `ARBITER_METADATA_TOKEN_BUDGET` tokens of `full_text` if those sections are absent). The model returns: `title`, `intervention`, `comparator`, `primary_outcome`, `all_outcomes` (primary first, capped at `ARBITER_MAX_OUTCOMES`, default 10), `blinding`, `nct_number`, and **`study_design`** (`parallel_rct` | `cluster_rct` | `crossover_rct` | `single_arm` | `non_rct` | `unclear`) **plus a one-sentence `study_design_basis`**. It does **not** return an effect-of-interest hint: `effect_of_interest` comes from config (default `assignment`, `--effect` override), so an extracted hint could never win and is omitted.

### Eligibility (IRPG scope enforcement)

`study_design` rides this **existing** aux call — it adds **no new LLM call**. It exists to enforce the IRPG scope ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)): RoB 2-IRPG is valid only for individually-randomised parallel-group RCTs, so a non-`parallel_rct` input must **not** be silently given a fabricated five-domain assessment. The classification is advisory metadata here; the **deterministic gate** that acts on it is a shared pre-assessment precondition ([REQ-17](#req-17-batch-runner-and-manifest)) applied by **both** `arbiter batch` and `arbiter assess` before any domain is assessed, so that judgment stays deterministic and no LLM ever decides eligibility on its own. `study_design` and `study_design_basis` are added to `TrialMetadata` ([§5.3](#53-trial--outcome-models)) as a `StudyDesign` enum ([§5.1](#51-enums)).

### NCT Derivation (precedence)

1. `config.nct_number` (from manifest) if set.
2. Else `nct_hint` from the deterministic `NCT\d{8}` regex scan ([REQ-02](#req-02-main-paper-ingestor)).
3. Else the LLM-extracted `nct_number`.

### Post-Processing

- Use `config.effect_of_interest` directly (default `assignment`, overridable via `--effect`). The extractor does **not** emit an effect hint.
- **`trial_id` is deterministic — never random** (a random component in an idempotency key defeats [REQ-17](#req-17-batch-runner-and-manifest) for NCT-less trials). Precedence:
  1. `nct_number` if known (per the precedence above).
  2. Else the manifest `trial_label`, slugified.
  3. Else `sha256(paper_bytes)[:12]` — a content hash of the main paper, stable across runs and unique per paper.
     `uuid4()` is used only for `assessment_id` (a per-write surrogate), **not** `trial_id`. Because `trial_id` is stable, the JSON filename ([REQ-16](#req-16-output-json-and-sqlite-writers)) is stable too, so re-runs overwrite rather than accumulate.

### Acceptance Criteria

- Returns `TrialMetadata` for any parseable paper; `intervention`, `comparator`, and at least the primary entry of `all_outcomes` are non-empty.
- Manifest `nct_number` and regex-derived NCT both override the LLM value per the precedence above.

---

## REQ-06: LLM Abstraction Layer

**Module:** `arbiter/llm/`

### What to Build

An abstract `LLMClient` plus **three dedicated concrete clients — one per provider** (`ChatAnthropic`, `ChatOpenAI`, `ChatOpenRouter`) — and a mock for tests. The layer **guarantees a validated Pydantic instance** regardless of whether the provider natively enforces a schema.

> **Transport is delegated to LangChain; the contract is ARBITER's.** Each concrete client **wraps its provider's LangChain chat model** — `ChatAnthropic`, `ChatOpenAI` (vanilla OpenAI), and `ChatOpenRouter` (the dedicated `langchain-openrouter` integration) — rather than hand-rolling `httpx`. **OpenRouter does _not_ reuse the OpenAI client**: `ChatOpenRouter` handles OpenRouter's `cache_control` breakpoints, `provider` routing, and usage reporting natively, so the previous `ChatOpenAI`+`base_url` hack is dropped. ARBITER already depends on `langchain-core` (LangGraph), so transport, auth, message formatting, and vision content-block encoding come from dependencies already in the tree. **Network retry is the exception: ARBITER owns it ([REQ-20](#req-20-error-handling-and-retry)) and constructs each LangChain client with `max_retries=0`** — delegating to the dependency's internal retry would both nest under ARBITER's backoff loop (up to 3×3 attempts) and hide the per-attempt transient-error history the [REQ-23](#req-23-run-trace--timing-instrumentation) `CallRecord` records. The `LLMClient` ABC stays because the pieces that make this layer _ARBITER's_ must live here regardless: the schema-repair ladder, the `cache_control` strip policy (vanilla OpenAI only), `provider` pinning, the `CallRecord` trace injection, and the headline-flip to provider-enforced schema. The ABC is also the seam the rest of the codebase imports and the contract `MockLLMClient` satisfies for network-free tests.

### Base Class (`llm/base.py`)

```python
class LLMClient(ABC):
    def __init__(self, model: str): self.model = model

    @abstractmethod
    async def complete_structured(
        self, messages: list[dict], schema: type[BaseModel],
        temperature: float = 0.0, max_tokens: int = 2048,   # callers pass config.sq_max_tokens; budget must cover reasoning-model CoT (REQ-01)
        *, call_label: str | None = None,                   # contract-neutral call identity for fixture keying + trace attribution; NOT sent to the provider
    ) -> BaseModel: ...

    @abstractmethod
    def supports_prompt_caching(self) -> bool: ...
    @abstractmethod
    def supports_native_schema(self) -> bool: ...
    @abstractmethod
    def supports_vision(self) -> bool: ...

    # KEPT SEAM. v0.1 ships only this stub (CONSORT vision is deferred to v0.2, REQ-14/§3),
    # so it raises NotImplementedError on every client; the v0.2 vertical implements it on
    # vision-capable clients. The signature + supports_vision flag exist now so v0.2 plugs in.
    async def complete_vision(self, image_bytes: bytes, prompt: str,
                              schema: type[BaseModel]) -> BaseModel:
        raise NotImplementedError("Vision (CONSORT extraction) lands in v0.2.")
```

`complete_vision` is a **stub in v0.1** (raises `NotImplementedError` on every client) and is **implemented in v0.2** on the `ChatOpenRouter` client for vision-capable OpenRouter models (e.g. the free Gemma vision model): the image will be sent as a base64 **data-URL** content block, and the same hybrid-structured-output contract applies (return a validated `schema` instance — here `ConsortFlow`), used by the trial-tier CONSORT-extraction node ([REQ-14](#req-14-context-assembly)). v0.1 keeps the signature + `supports_vision` flag as the seam so v0.2 plugs in without touching the ABC.

### Hybrid Structured Output (the critical contract)

`complete_structured` must return a validated `schema` instance, never a dict or string. Both paths build on LangChain's `with_structured_output(schema, include_raw=True)`, which returns `{"parsed", "raw", "parsing_error"}` — the parsed instance, the raw `AIMessage` (carrying `usage_metadata`, incl. cache-read tokens where the route reports them), and any parse exception:

1. **If `supports_native_schema()`:** call `with_structured_output` with a native `method` (`function_calling`/`json_schema` for Anthropic and OpenAI). `parsed` is authoritative.
2. **Else (e.g. gpt-oss via OpenRouter):** call `with_structured_output` in `json_mode` (or inject the schema into the prompt). On a non-null `parsing_error`, **retry up to `ARBITER_SCHEMA_REPAIR_MAX_RETRIES`** (bounded; default 2) feeding the error and the `raw` content back into the prompt. This is the ARBITER-owned repair ladder; LangChain supplies `{parsed, raw, parsing_error}`, ARBITER supplies the loop. These **schema-validation retries are separate** from the network-error retries in [REQ-20](#req-20-error-handling-and-retry).
3. If still invalid after retries, raise `ValueError` with a descriptive message.

> **Reasoning-model token budget (don't mistake truncation for a format failure).** For reasoning-family open models (the gpt-oss SUT emits Harmony-format CoT before the JSON), the completion budget must cover the reasoning **plus** the answer — hence the `2048` default ([REQ-01](#req-01-dependency-and-project-setup) `sq_max_tokens`), not `512`. Where the provider exposes it, pin a **low reasoning-effort** so SQ calls stay bounded and reproducible. When a response comes back with `finish_reason == "length"`, record that **distinctly** in the trace (it is a truncation, not a substantive answer) so a budget-exhaustion `NI`/`FLAGGED` is never conflated with a real "No Information" — this feeds REQ-21's substantive-vs-format-failure `NI` split.

The `{raw, parsing_error}` shape is also what the trace records (the full repair ladder, [REQ-23](#req-23-run-trace--timing-instrumentation)) and what backs the `null`-not-`0` cost discipline (cache tokens are surfaced via `usage_metadata` only when the underlying provider reports them).

> **Headline-time escape hatch — provider-enforced schema (parked, one-line flip).** `supports_native_schema` is a **provider+model** property, not a fixed model property. On OpenRouter, an open model such as `gpt-oss-120b` _can_ get decode-time JSON-schema enforcement when routed to a structured-output-capable provider (e.g. Fireworks): send `response_format={"type":"json_schema","strict":true}` with `provider: {"require_parameters": true, "only": [<provider>], "quantizations": [...]}`. **In dev (free tier) the path-2 repair ladder above is the real path** — that's fine and intended. **For the headline paid run** ([REQ-21](#req-21-evaluation-harness) execution modes), flipping this on makes path 1 apply to the open models too, demoting the repair ladder to a rare fallback and removing the format-failure confound from the headline numbers. Build the path-2 ladder now; keep this flip as a config knob, not new machinery.

### Caching

Messages may contain content blocks with `"cache_control": {"type":"ephemeral"}`. **`ChatAnthropic` and `ChatOpenRouter` pass these through natively** — OpenRouter honours breakpoint caching, including for gpt-oss when routed to a caching-capable provider. The **vanilla-OpenAI client strips `cache_control`** (OpenAI caches automatically and rejects the unknown key). Where a route genuinely cannot cache, the prefix structure is preserved at no benefit and no harm.

> **Two impl-time verifications (not assumptions):** (i) confirm `ChatOpenRouter` forwards the `provider` routing arg and `cache_control` (via its routing/`extra_body` surface) without stripping them; (ii) confirm `usage_metadata` carries cache-read/write counts on the pinned OpenRouter route — the docs surface these only "if provided by the underlying provider," which maps cleanly to the `null`-not-`0` rule ([REQ-23](#req-23-run-trace--timing-instrumentation)) when a route omits them.

### Trace Integration (side-channel — does not change the return contract)

Each client may be given an optional `RunTrace` handle (constructor kwarg `trace: RunTrace | None = None`, attachable post-construction). When present, `complete_structured` / `complete_vision` record one `CallRecord` per call into the **currently-active span** (a `contextvars`-scoped span entered by the graph node wrapper, [REQ-23](#req-23-run-trace--timing-instrumentation)) as a side effect. **The return type is unchanged** — callers still receive only a validated `schema` instance; no metadata leaks into the deterministic path. The `CallRecord` captures the full picture for debugging:

- **Call identity:** the `call_label` the caller passed (`"{sq_id}|{effect}"`, `"metadata"`, `"annotate:{segment_id}"`, …) so each record is attributable to its SQ/call-type without parsing prompts.
- **Prompt I/O, prefix-deduped:** the cacheable static prefix is registered once per trial (keyed by its hash); the per-call record stores `{prefix_hash, dynamic_suffix, ...}` rather than re-logging the ~4k-token prefix on every SQ call.
- **The repair ladder:** for the non-native path ([REQ-06](#req-06-llm-abstraction-layer) hybrid output), **every** attempt's raw response + the parse/validation error fed back, and which attempt finally validated (or that it raised). For the network layer ([REQ-20](#req-20-error-handling-and-retry)), the attempt count and transient errors.
- **Usage:** in/out tokens (always, from provider `usage`), cache read/write tokens **where the provider reports them** (`null` on a non-caching route — not `false`/`0`), and the call latency.

When no `trace` is attached (or `--trace-level off`), clients run exactly as before — recording is a no-op.

### Factory

`create_llm_client(model: str, trace: RunTrace | None = None) -> LLMClient` looks up `MODEL_REGISTRY`, constructs the right client by `provider` (`anthropic` → `ChatAnthropic`, `openai` → `ChatOpenAI`, `openrouter` → `ChatOpenRouter`), attaches the optional `trace` handle, and raises `ValueError` for unknown models.

### Acceptance Criteria

- `complete_structured` returns a validated model instance for **both** a native-schema provider and gpt-oss/OpenRouter (non-native path exercised by a test).
- Validation failures after bounded retries raise `ValueError`.
- `MockLLMClient` returns deterministic fixture responses **keyed on `call_label`** (the SQ worker passes `"{sq_id}|{effect}"`, aux callers `"metadata"` / `"annotate:{segment_id}"`; vision responses keyed separately) with no network calls — no prompt-text parsing.
- `complete_vision` is a stub in v0.1: it raises `NotImplementedError` on every client (the validated-`ConsortFlow` impl + the mock-vision test are v0.2, [REQ-14](#req-14-context-assembly)/[REQ-22](#req-22-testing)).

---

## REQ-07: RoB 2 Algorithm — Deterministic Judgments

**Module:** `arbiter/arbiter_algorithm/decision_tables.py`, `rollup.py`

### Source of Truth (vendored, pinned)

The decision logic is **not** restated in this PRD. It is implemented **directly from the vendored official Cochrane RoB 2 tool** under `docs/rob2/`, scoped to the **IRPG variant** (Individually Randomized, Parallel-Group — cluster/crossover are out of scope for v0.1):

- `rob2_guidance.pdf` — the **RoB 2 guidance document** (Higgins et al., 22 August 2019): verbatim SQ wording + answer definitions.
- `rob2_irpg_algorithm.xlsm` — the **official RoB 2 IRPG Excel tool** (`ROB2_IRPG_beta_v9`): the explicit decision-table cell logic and SQ routing/gating. Key sheets: `Print_format (ITT)`, `Print_format (PP)`, `Function Tab`.

`docs/rob2/README.md` (committed) records the **version label, source URLs, retrieval date, and the licence/redistribution note**. The binaries themselves are **git-ignored** (non-commercial/no-derivatives licence on a public repo — fetch per the README). This pinning makes every ARBITER judgment traceable to a fixed algorithm version; updating the algorithm is a deliberate, reviewed change to the vendored files.

### What to Build

Five pure functions, one per domain, plus the overall rollup:

```python
def judge_domain_1(answers: dict[str, SQAnswer]) -> tuple[Judgment, str]
def judge_domain_2(answers: dict[str, SQAnswer], effect: EffectOfInterest) -> tuple[Judgment, str]
def judge_domain_3(answers: dict[str, SQAnswer]) -> tuple[Judgment, str]
def judge_domain_4(answers: dict[str, SQAnswer]) -> tuple[Judgment, str]
def judge_domain_5(answers: dict[str, SQAnswer]) -> tuple[Judgment, str]
def compute_overall_judgment(domain_judgments: list[DomainJudgment]) -> tuple[Judgment, str, bool]
```

Each domain function returns `(Judgment, rationale)`; `compute_overall_judgment` returns `(judgment, rationale, requires_human_review)`.

### Hard Correctness Rules

- The mapping from SQ answers to judgments must reproduce the vendored Excel logic **exactly**. In particular, the direction of each SQ matters (e.g., for D1.3 _"baseline differences suggest a problem"_, `Y/PY` pushes toward **High** and `N/PN` is required for **Low** — do not invert it). Every SQ ID used by these functions must mean the same thing it means in [Appendix A](#appendix-a-the-22-signaling-questions) and in the SQ prompt templates.
- **Two different sources of truth in the rollup — be honest about which is which:**
  - The **domain tables** (`judge_domain_*`) reproduce the vendored Excel logic **verbatim** — fully traceable.
  - The **overall rollup's "multiple domains" boundary is an ARBITER policy decision**, _not_ a verbatim rule. The official guidance defines overall **High** as "High in ≥1 domain **or** some concerns for _multiple domains in a way that substantially lowers confidence_" — and explicitly leaves "multiple domains / substantially lowers confidence" to **reviewer judgment**. Because ARBITER runs unattended, it **operationalises** that judgment deterministically using a named constant `OVERALL_HIGH_SC_THRESHOLD` (default **3**): **Low** if all five Low; **High** if any domain High **or** `≥ OVERALL_HIGH_SC_THRESHOLD` domains Some concerns; otherwise **Some concerns**. `requires_human_review = True` is set on **exactly the two policy-driven paths** — the cases where ARBITER's rule, not the verbatim tables, decided the outcome, and **both boundaries are derived from `OVERALL_HIGH_SC_THRESHOLD`, never a hardcoded literal**: (a) **sub-threshold multi-SC** — `2 ≤ #SC < OVERALL_HIGH_SC_THRESHOLD` with **no** domain High → Some concerns (the SC/High boundary; at the default of 3 this is exactly the 2-SC case), and (b) `#SC ≥ OVERALL_HIGH_SC_THRESHOLD` with **no** domain High → High (High-by-accumulation — the "multiple domains substantially lower confidence" case the guidance leaves to reviewer judgment). Deriving (a) from the constant means tuning the threshold can never leave a sub-threshold multi-SC outcome silently unflagged. The table-clean outcomes (all-five-Low → Low; any-domain-High → High; a lone `1 SC` → Some concerns) are **not** flagged, so `requires_human_review` is a precise "policy decided this" signal, not a one-sided boundary detector. Document this operationalisation — the `OVERALL_HIGH_SC_THRESHOLD` default and **both** constant-derived review-flag paths — in an **ADR** (`docs/adr/`) so the policy is reviewable and the threshold is owned by ARBITER, not misattributed to Cochrane. The eval's rollup-normalisation ([REQ-21](#req-21-evaluation-harness)) reads the **same constant**.
  - SQ-direction still matters and must not invert (e.g. D1.3 _"baseline differences suggest a problem"_: `Y/PY` → toward **High**, `N/PN` required for **Low**). Every SQ ID used here means what it means in [Appendix A](#appendix-a-the-22-signaling-questions) and the prompt templates.
- **No function makes an LLM or network call.** All are pure (no side effects, no global state).

### Acceptance Criteria

- Each domain function is tested **exhaustively** against the vendored RoB 2 Excel logic; every reachable path is covered (a **synthetic conformance test** enumerating the SQ-answer combinations the Excel maps to each judgment — including the `High` paths the human eval set cannot reach). The enumeration is **transcribed once** from the `.xlsm` into a committed Python truth-table (cell logic, not Cochrane prose — copyright-safe), so the test is **hermetic**: it needs no runtime access to the git-ignored binary, in CI or a fresh clone.
- A regression test asserts D1.3 direction is **not** inverted (a known prior bug).
- `compute_overall_judgment` matches the **documented ARBITER rollup policy** (above) on all 5-domain combinations, including `requires_human_review = True` on **both** policy-driven paths (sub-threshold multi-SC `2 ≤ #SC < OVERALL_HIGH_SC_THRESHOLD`, **and** `#SC ≥ OVERALL_HIGH_SC_THRESHOLD` with no domain High) and `False` on the table-clean outcomes (incl. a lone 1-SC); **both** flag boundaries are read from `OVERALL_HIGH_SC_THRESHOLD`, not hardcoded (a test bumps the constant and asserts the sub-threshold flag band moves with it); the ADR is referenced in its docstring.

---

## REQ-08: Conditional Branching

**Module:** `arbiter/arbiter_algorithm/branching.py`

### What to Build

```python
def get_applicable_sqs(domain: str, effect: EffectOfInterest,
                       current_answers: dict[str, SQAnswer]) -> list[str]
def get_na_sqs(domain: str, effect: EffectOfInterest,
               current_answers: dict[str, SQAnswer]) -> list[str]
```

`get_applicable_sqs` returns the SQ IDs that should be asked **given the answers so far**; `get_na_sqs` returns the IDs that are Not Applicable for this domain+effect (including the _other_ D2 effect's unique SQs). Together they must account for every SQ ID in the domain.

### Structure (aligned to the official tool — see [Appendix A](#appendix-a-the-22-signaling-questions))

- **D1** — always `1.1, 1.2, 1.3` (no branching).
- **D2 (assignment effect)** — `2.1, 2.2` always; `2.3` gated on `Y/PY/NI to 2.1 or 2.2`; `2.4` gated on `Y/PY to 2.3`; `2.5` gated on `Y/PY/NI to 2.4`; `2.6` always; `2.7` gated on `N/PN/NI to 2.6`. `2.3`–`2.7` carry their **assignment** wording.
- **D2 (adhering effect)** — `2.1, 2.2` always; `2.3` gated on `Y/PY/NI to 2.1 or 2.2`; **`2.4`, `2.5` always** — the official tool (`Function Tab` H4/H5) tags these "[If applicable]" with **no logical gate**, so there is no machine-evaluable predicate; a deterministic pipeline asks them whenever the adhering effect is in scope; `2.6` gated on the **compound** condition `N/PN/NI to 2.3` **or** `Y/PY/NI to 2.4 or 2.5`. `2.3`–`2.6` carry their **adhering** wording. **`2.7` does not exist for adhering → it is the _only_ D2 SQ in `get_na_sqs` under this effect.** (`2.3`–`2.6` are answered under _both_ effects with different wording — they are **not** NA; see [Appendix A](#appendix-a-the-22-signaling-questions).)
  > **Compound-gate NA rule:** when a gate references an SQ that resolved to `NA` (e.g. 2.6's `Y/PY/NI to 2.4 or 2.5` if 2.4/2.5 are ever NA), the `NA` operand evaluates as **not-satisfied** — `NA` never satisfies a gate clause. This keeps `NA` from leaking into branching decisions.
- **D3** — `3.1` always; `3.2` gated on `N/PN/NI to 3.1`; `3.3` gated on `N/PN to 3.2`; `3.4` gated on `Y/PY/NI to 3.3`.
- **D4** — `4.1` **and** `4.2` always; `4.3` gated on `N/PN/NI to 4.1 **and** 4.2`; `4.4` gated on `Y/PY/NI to 4.3`; `4.5` gated on `Y/PY/NI to 4.4`. **Note the official meaning of each D4 SQ ID** (4.1 = appropriateness of measurement method; 4.3 = assessor awareness; etc.) — branching must use those meanings, not a relabelled set. (4.2 is **always asked**, not gated.)
- **D5** — `5.1, 5.2, 5.3` per the official gating.

> Because the exact conditional wording and gating are defined by the vendored tool, implement branching **from `docs/rob2/`** and keep the SQ-ID meanings consistent with Appendix A and the prompt templates.

### Branch-Resolver Loop (one uniform pattern, all domains)

The conditional chains run **deeper than two rounds** — `2.1/2.2 → 2.3 → 2.4 → 2.5` (D2 assignment), `3.1 → 3.2 → 3.3 → 3.4` (D3), and `4.1/4.2 → 4.3 → 4.4 → 4.5` (D4) each have an applicability depth of **four**: you cannot know whether 2.5/3.4/4.5 applies until its predecessor is answered. A fixed two-round fan-out therefore cannot evaluate them, and a flat parallel fan-out would ask ungated SQs.

So every domain uses **one** pattern — a **fixpoint loop**: call `get_applicable_sqs(domain, effect, answers_so_far)`; `Send`-fan-out the **newly-applicable** SQs in parallel; fan in; **repeat** until `get_applicable_sqs` yields nothing new. This handles arbitrary chain depth uniformly: D1 (no branching) resolves in a single wave; D5 in one or two; D2/D3/D4 in up to four. The loop is guaranteed to terminate (each wave only **adds** answers, so the applicable set is monotonic), but it carries a **defensive max-wave cap** (`= max chain depth + 1`, i.e. 5) that aborts the domain with an `errors` entry rather than spinning, so a future non-monotonic bug in `get_applicable_sqs` surfaces as a logged failure instead of a hang. SQs determined to be NA are recorded with `answer = NA`, empty quote, `page = None`. `get_applicable_sqs`/`get_na_sqs` are the gate oracle the loop calls — they are unchanged by this structure.

### Acceptance Criteria

- For D2 assignment with `2.1 = Y, 2.2 = N` and nothing else answered, `get_applicable_sqs` returns the correct next-round SQ(s) and **not** SQs whose trigger hasn't been answered yet.
- For D2 **adhering**, `get_na_sqs({})` returns **exactly `{2.7}`** (the only effect-exclusive ID); for D2 **assignment**, `get_na_sqs({})` returns **`{}`** (no adhering-only IDs exist — `2.3`–`2.6` are reused with effect-specific wording, not NA'd).
- For D4, `4.1` and `4.2` are always applicable; `4.3` is **not** applicable when either `4.1` or `4.2` is `Y/PY` (the assessor-awareness chain is skipped because measurement is already **inappropriate** (`4.1 = Y/PY`) or **could differ between groups** (`4.2 = Y/PY`) — a problem is established without needing the chain). Note the direction: `4.1` asks whether the method was _inappropriate_, so `Y/PY` is the _problem_ state, not the clean one.
- All SQ IDs in a domain are partitioned across applicable ∪ NA for any effect (D2 partitioned over the _effect-appropriate_ ID set).

---

## REQ-09: Signaling-Question Prompt Templates

**Module:** `arbiter/prompts/sq_prompts.py`

### What to Build

A `SQ_PROMPTS` dict with one `SQPromptTemplate` per SQ ID. These are the highest-leverage prompts in the system.

```python
@dataclass
class SQPromptTemplate:
    sq_id: str
    effect: Literal["assignment", "adhering", "both"]  # part of the key for D2 2.3–2.6
    question_text: str       # verbatim RoB 2 question text
    answer_definitions: str  # Y/PY/PN/N/NI criteria specific to THIS question
    key_terms: list[str]     # query terms for supplement retrieval (BM25 arm + dense)
```

> **No `applies_to` field.** Scope — which SQs are in play for a given domain+effect — is owned **solely** by branching ([REQ-08](#req-08-conditional-branching)). The template carries only `effect`, the wording discriminator for the `(sq_id, effect)` lookup. (An earlier draft duplicated scope in an `applies_to` list; it was dropped to remove a two-sources-of-truth bug, since branching already NAs `2.7` under adhering without consulting the template.)

**Keying.** `SQ_PROMPTS` is keyed by `(sq_id, effect)`, **not** `sq_id` alone. This is mandatory: for D2, IDs `2.3`–`2.6` are **different questions** under the assignment vs adhering effect (see [Appendix A](#appendix-a-the-22-signaling-questions)) and cannot share one entry. IDs whose wording does not vary by effect (`1.x`, `2.1`, `2.2`, `3.x`, `4.x`, `5.x`, plus assignment-only `2.7`) use `effect="both"`; `2.7` is simply never looked up under adhering, where it is `NA`. The SQ worker looks up `(sq_id, config.effect_of_interest)` with a `"both"` fallback.

The authoritative `question_text` and `answer_definitions` wording is the vendored RoB 2 guidance (`docs/rob2/`); the full reference set is reproduced in [Appendix A](#appendix-a-the-22-signaling-questions) and **must be reconciled verbatim against the vendored source** at implementation time. Each SQ ID and its meaning must match the algorithm ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)) and branching ([REQ-08](#req-08-conditional-branching)).

### The PY/PN bridge (empirically-motivated NI countermeasure)

Two independent studies found LLMs **over-select the "can't tell" codes** — Huang et al. (JMIR 2025) had to remove `NA` instructions because overuse degraded judgments, and ROBoto2 found every model over-selected "No Information" (it flagged 101 high-risk trials where humans flagged 47), a conservative bias that propagates up through the rollup. ARBITER neutralises the **NA** half _by construction_ (NA is set deterministically by branching — [REQ-08](#req-08-conditional-branching) — the LLM cannot emit it), which is a stated architectural advantage. The **NI** half is addressed here: every `answer_definitions` block must **explicitly route reasonable inferences to `PY`/`PN`** and reserve `NI` for genuine textual silence — i.e. _"if the source supports a reasonable inference even without stating it outright, answer `PY`/`PN`; use `NI` only when the text provides no basis at all."_ This is not a hack — it is how human RoB 2 assessors are trained to use the probable-yes/probable-no codes, and it directly targets the documented failure mode. The NI rate is then **measured against the human NI rate** as an eval calibration metric ([REQ-21](#req-21-evaluation-harness)); a persistently elevated NI rate is the trigger to enable the (eval-gated) targeted re-retrieval in [REQ-11](#req-11-confidence-signal-system).

### Acceptance Criteria

- `SQ_PROMPTS` covers the **22 SQ positions** (`1.1–1.3`, `2.1–2.7`, `3.1–3.4`, `4.1–4.5`, `5.1–5.3`) across **26 templates** — D2 `2.3`–`2.6` each appear twice (assignment + adhering), all other IDs once (`effect="both"`).
- Looking up `(sq_id="2.4", effect="adhering")` returns the **per-protocol** wording (failures in implementation), distinct from `(sq_id="2.4", effect="assignment")` (deviations affecting the outcome).
- Every entry has non-empty `question_text`, `answer_definitions`, and `key_terms`, with explicit criteria for each answer code applicable to that question.
- Every `(sq_id, effect)` referenced by the algorithm and branching modules exists in `SQ_PROMPTS` with a consistent meaning.

---

## REQ-10: Quote Verifier

**Module:** `arbiter/confidence/quote_verifier.py`
**Signatures:**

- `def verify_quote(quote: str, raw_char_stream: str, threshold: int = 85) -> bool`
- `def locate_quote_page(quote: str, page_boxes: list[PageBox]) -> int | None`

### What to Build

**`verify_quote`** — fuzzy-match the LLM-returned quote against the raw PDF character stream (via `rapidfuzz`). Normalise both (collapse whitespace, lowercase), slide a window over the stream, and return whether the best `partial_ratio` ≥ `threshold`. Quotes shorter than `ARBITER_QUOTE_MIN_VERIFY_CHARS` (default 15) characters (typical of NA/NI answers) are trivially verified as `True`.

**`locate_quote_page`** — the deterministic page resolver that replaces the LLM-supplied page ([REQ-15](#req-15-sq-worker)/[§5.3](#53-trial--outcome-models)). For a verified quote, normalise it the same way and `partial_ratio`-match it against each `PageBox.text`; return the `page` of the best-scoring box (the **first** page when the best match spans a break, by taking the earliest box above threshold). Returns `None` for an empty/sub-`ARBITER_QUOTE_MIN_VERIFY_CHARS` quote (NA/NI). Matching against `page_boxes` — which carry per-page text + `page` — recovers the page from the layout parse without threading page offsets through the flat `raw_char_stream`.

### Design Notes

- Threshold 85 absorbs OCR artefacts, hyphenation, and ligature differences.
- Must complete in well under a second for a ~50,000-character stream.
- `locate_quote_page` is **deterministic** and makes the record's `page` a verified field, not an LLM guess — the only LLM outputs that survive into `SQAnswer` are `quote`, `answer`, and `justification`.

### Acceptance Criteria

- `verify_quote`: `True` for an exact quote and for quotes differing only in whitespace/case; `False` for a quote absent from the stream; `True` for empty or <15-char quotes; completes in <500 ms for a 50,000-char stream.
- `locate_quote_page`: returns the correct 0-based page for a quote on a known page of a fixture; returns the **earlier** page for a quote straddling a page break; returns `None` for an empty/NA/NI quote — with no LLM or network call.

---

## REQ-11: Confidence Signal System

**Module:** `arbiter/confidence/signals.py`
**Signature:** `def compute_confidence(answer, quote_verified, segments_retrieved, segments_available, retrieval_top_score) -> ConfidenceSignals`

### What to Build

A **deterministic** flag derived from retrieval and verification signals. The pipeline is **single-shot** per SQ, for two _separate_ reasons that should not be conflated:

- **Plain resampling is rejected on its own merits:** re-issuing the _same prompt_ at temperature 0 yields a near-identical answer and carries no information, so self-consistency / second-sample voting is pointless here.
- **Targeted re-retrieval is deferred on its own merits:** re-asking a `FLAGGED`/`NI` SQ with a _widened retrieval context_ is a genuinely different call that _could_ recover a better answer (this is the defensible core of auto-rob2's "pivotality" idea). It is **not** ruled out by the resampling argument above — it is **deferred from v0.1** to keep the per-trial call budget and latency bounded, and logged as a planned enhancement gated on eval evidence that flagged SQs are losing real passages. If adopted, it would only ever change the **SQ answer** the LLM produces, never the judgment (the deterministic decision tables still decide) — auto-rob2's label-adjudication step is **permanently rejected** as a violation of the no-LLM-judgment constraint ([§2](#2-system-overview)).

There is therefore **no self-consistency / second-sample mechanism** in v0.1.

### Flag Rules (priority order)

> **Score scale (read first).** `retrieval_top_score` is a **normalised [0,1]** relevance value (full definition in [REQ-03](#req-03-supplementary-material-ingestor)/[§5.6](#56-langgraph-state)) — **not** the raw RRF fusion score — so `ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD` is on the **0–1 scale** (default **0.35**). It can be `None` when **no retrieval ran** (empty index / no domain key-terms); the two score-based clauses below treat `None` as **"no score signal" — they do not fire** (a missing score never triggers `UNCERTAIN`/`FLAGGED` on its own), so the comparison can never throw.

1. `answer == NA` → always `CONFIDENT` (set by the algorithm, not the LLM).
2. `FLAGGED` if: the quote could not be verified on a non-NI/NA answer; **or** the answer is `NI` while domain-relevant supplements existed (`segments_available > 0`) yet the best retrieved passage was weak (`retrieval_top_score` is not `None` **and** `< ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD`) — a model that claims "no information" when supplements for the domain were present but poorly-matched is the suspicious case. _(This replaces an earlier "0 segments retrieved" condition that was unreachable: [REQ-03](#req-03-supplementary-material-ingestor)'s `retrieve` returns the top-k of whatever is indexed and falls back to the full set, so `segments_retrieved` is 0 only when the whole index is empty — in which case `segments_available` is 0 too.)_
3. `UNCERTAIN` if: `retrieval_top_score` is not `None` and below `ARBITER_RETRIEVAL_UNCERTAIN_THRESHOLD` (default **0.35**, on the 0–1 scale); **or** the answer is `NI` with no supplementary materials for the domain.
4. Otherwise `CONFIDENT`.

`flag_reason` is a human-readable string for any non-CONFIDENT flag. All flags are **advisory** (see [§3](#3-users-and-workflow)).

> **NI over-selection is a measured concern, not just a flagged one.** The `FLAGGED`/`UNCERTAIN` rules above _detect_ a suspicious `NI`; the _prevention_ is the PY/PN-bridge wording in [REQ-09](#req-09-signaling-question-prompt-templates). The eval ([REQ-21](#req-21-evaluation-harness)) reports ARBITER's per-domain **NI rate vs the human NI rate**; if ARBITER runs persistently NI-heavy, that is the eval-gated trigger to enable **targeted re-retrieval** (widened context for `FLAGGED`/`NI` SQs — deferred below). `retrieval_top_score` is **retriever-agnostic by design**: it is the normalised relevance of the top passage (above), so the rules read the same whether the underlying retriever is BM25-only or the BM25 + dense RRF hybrid.

### Acceptance Criteria

- `NA` → `CONFIDENT` always.
- Unverified quote on a non-NI/NA answer → `FLAGGED` with reason.
- `NI` + supplements available (`segments_available > 0`) + weak best passage (`retrieval_top_score < threshold`) → `FLAGGED`.
- Low retrieval score → `UNCERTAIN`.
- Verified quote with adequate retrieval → `CONFIDENT`.
- `ConfidenceSignals` has **no** `answer_consistency` field.

---

## REQ-12: Assessment Orchestration (Two-Tier Graph)

**Modules:** `arbiter/graph/state.py`, `arbiter/graph/builder.py`, `arbiter/graph/nodes/`

### What to Build

Two LangGraph graphs reflecting the trial/outcome tiers. **No checkpointer** — graphs run in-memory; resilience is at the batch layer ([REQ-17](#req-17-batch-runner-and-manifest)).

**Trial graph** (`build_trial_graph`) — runs once per trial:

```
START → context_D1 → resolve_loop(D1) → judgment_D1 → END
```

(D1 has no branching, so its `resolve_loop` is a single wave of `1.1, 1.2, 1.3`.)

Output: `domain_judgments = [D1]` with `scope="trial"`. (The trial tier is also where the **v0.2** CONSORT extraction node will run — see [REQ-14](#req-14-context-assembly); v0.1 has no such node.)

**Outcome graph** (`build_outcome_graph`) — runs once per outcome, seeded with the reused D1. **D2, D3, D4, and D5 have no inter-domain data dependency** (each domain's SQ answers depend only on that domain's context; `pre_d5` is deterministic and depends only on CT.gov + the outcome string, not on D2–D4), so the four domains run as **parallel branches** fanning into `overall_judgment`:

```
                  ┌─ context_D2 → resolve_loop(D2) → judgment_D2 ─┐
START ──┤         ├─ context_D3 → resolve_loop(D3) → judgment_D3 ─┤
                  ├─ context_D4 → resolve_loop(D4) → judgment_D4 ─┼─→ overall_judgment → END
                  └─ pre_d5 → context_D5 → resolve_loop(D5) → judgment_D5 ─┘

where  resolve_loop(domain) =
   [Send] sq_workers(newly-applicable SQs) → fanin → (get_applicable_sqs yields more? ↺ : ↓)
```

`resolve_loop` is the single [REQ-08](#req-08-conditional-branching) fixpoint motif: fan out the newly-applicable SQs, fan in, and repeat until no new SQ becomes applicable (one wave for D5's leading SQ, up to four for the D2/D3/D4 chains). `pre_d5` is deterministic and feeds **only** the D5 context ([REQ-14](#req-14-context-assembly)), so it sits at the head of the D5 branch, not in a shared pre-D5 stage.

**Why parallel, not serial.** The conditional chains serialise _within_ a domain (e.g. D2 assignment `2.1 → 2.3 → 2.4 → 2.5`, ~4 sequential LLM round-trips), so four domains in series would be ~16 sequential round-trips per outcome; as parallel branches it is ≈ max(4,4,4,5) ≈ 5. The rate-limit guard is already in place: `ARBITER_MAX_CONCURRENCY` is a **single global semaphore over LLM calls** ([REQ-17](#req-17-batch-runner-and-manifest)), so overlapping the domain branches cannot cause a 429 storm — it only lets calls that would happen anyway run concurrently. (The win is single-outcome latency on the `assess`/dev path; at batch scale the global semaphore is already saturated by cross-trial parallelism, so throughput is roughly unchanged.)

Output: `domain_judgments = [D2, D3, D4, D5]` (scope `"outcome"`) + overall judgment/rationale + `requires_human_review`, rolled up over the reused D1 plus D2/D3/D4/D5. The outcome graph is seeded with `effect_of_interest`, which already threads into `OutcomeState` and selects the per-effect D2 wording/gates.

### State Reducers

`sq_answers` and `domain_contexts` use a dict-merge reducer (never overwrite). `domain_judgments` and `errors` use `operator.add`. All other fields are last-write-wins. Runtime handles (LLM clients, retrieval index) are passed through state but are **not** durable/checkpointed.

> **Parallel fan-in ⇒ `domain_judgments` arrives unordered.** Because D2–D5 now run as parallel branches, the `operator.add` reducer appends their `DomainJudgment`s in **non-deterministic completion order**. The `Assessment` assembly ([REQ-19](#req-19-python-api)/[REQ-16](#req-16-output-json-and-sqlite-writers)) therefore **sorts `domain_judgments` by domain id (D1…D5)** before writing, so the record's domain order is stable and the side-channel invariant (byte-identical `data.json` across runs) holds despite scheduling jitter.

### Fan-In and Judgment Nodes

- **Fan-in** validates that all expected SQ answers for the domain are present; appends a descriptive string to `errors` for any missing SQ (does **not** abort).
- **Judgment** calls the matching `judge_domain_*` and emits a `DomainJudgment` with the correct `scope`.

### Trace Spans (build-time node wrapping)

`build_trial_graph` / `build_outcome_graph` accept an optional `trace: RunTrace | None`. When present, **each node is wrapped at build time** in a span decorator that opens a `contextvars`-scoped `NodeSpan` (`tier`, node name, and — outcome tier — `outcome`) around the node body and records its wall-clock duration and any raised error. Node bodies are **not edited** — the wrapping happens in the builder. Because the span is a contextvar, it propagates across LangGraph parallel `Send` fan-out (each SQ worker task copies the context and times itself), and the LLM `CallRecord`s ([REQ-06](#req-06-llm-abstraction-layer)) attach to whichever span is active. The collector is an ephemeral runtime handle (invariant 3, §2).

### Acceptance Criteria

- Both graphs compile and run; a full trial produces D1–D5 and an overall judgment per outcome.
- D1 in every outcome record for a trial is byte-identical (reused, not re-judged); D2–D5 are judged per outcome and may legitimately differ across a trial's outcomes.
- D2 branching: with `effect=assignment` and `2.1=Y`, the gated next-round SQ is answered, `2.6` is asked, and `2.7` is gated on `2.6`; with `effect=adhering`, `2.3`–`2.6` use their per-protocol wording and **`2.7` is the only `NA`** (see [REQ-08](#req-08-conditional-branching) — `2.3`–`2.6` are _not_ NA'd across effects, they are re-worded).
- **No CONSORT node in v0.1 — D3 is text-only.** The trial-tier CONSORT extraction node (locate + conservatively vision-extract `ConsortFlow` once per trial, thread into each `OutcomeState`) is the **v0.2** vertical ([§3](#3-users-and-workflow)/[REQ-14](#req-14-context-assembly)). In v0.1 no such node exists, `consort` stays `None`, and D3 reads text + the CT.gov denominator only.

---

## REQ-13: Pre-D5 Outcome Comparison

**Module:** `arbiter/graph/nodes/pre_d5.py`

### What to Build

A deterministic node (no LLM) that matches the **assessed** outcome (the current outcome string) against the **full set of registered outcomes** from CT.gov — `primaryOutcomes[*] ∪ secondaryOutcomes[*]` — via `rapidfuzz.ratio`, keeping the **best** match. This avoids the false positive of comparing a secondary outcome (PFS, AE) against `primaryOutcomes[0]` (OS) and spuriously flagging a switch on every non-primary outcome. Returns:

- `registered_outcome` — the best-matching registered measure,
- `published_outcome` — the assessed outcome string,
- `outcome_similarity_score` (0–1, rounded) — that best score. **`rapidfuzz.ratio` returns 0–100, so normalise to 0–1 (`ratio / 100`) before storing/comparing.** (Note this is a different scale from [REQ-10](#req-10-quote-verifier)'s `partial_ratio` threshold, which stays on the 0–100 scale — don't mix them.)
- `outcome_change_detected = best_score < ARBITER_OUTCOME_MATCH_THRESHOLD` (default 0.85) — i.e. the assessed outcome is **not found in the registry at all** (possible selective reporting / unregistered outcome); the threshold is on the normalised 0–1 scale,
- `registered_as_primary: bool | None` — whether the best match came from the **primary** list (lets D5 distinguish "a registered secondary was promoted to the headline result" from "absent from registry").

If CT.gov data is absent or the outcomes module is missing, return all fields as `None`.

The node writes these as flat `OutcomeState` fields (the reducers stay simple); the `Assessment` assembly ([REQ-19](#req-19-python-api)) packs them into the typed `OutcomeComparison` ([§5.3](#53-trial--outcome-models)) for `Assessment.outcome_comparison`.

### Acceptance Criteria

- Returns all fields whether or not CT.gov data is present.
- A secondary outcome that **is** registered (e.g. PFS registered as a secondary) yields a **high** best-match score and `outcome_change_detected = False` — it is **not** flagged merely for not being the primary.
- `outcome_change_detected = True` only when the assessed outcome matches **no** registered outcome well; `None` when no CT.gov data.
- `registered_as_primary` reflects which list the best match came from; `None` when no CT.gov data.
- No LLM calls.

---

## REQ-14: Context Assembly

**Module:** `arbiter/graph/nodes/context_assembly.py`
**Factory:** `def context_assembly_node_factory(domain: str) -> Callable`

### What to Build

For each domain, assemble the `DomainContext` shared by that domain's SQ workers. Runs once per domain before fan-out. **Context assembly builds only the per-domain dynamic suffix and the retrieval signals — it does _not_ build the shared prefix.** The trial-static prefix is assembled once in Phase 1 ([REQ-19](#req-19-python-api)) and seeded onto every state, so the per-domain node has one uniform job with no "build once" conditional and no trial/outcome asymmetry.

- **Shared prefix (built once per trial in Phase 1 — [REQ-19](#req-19-python-api) — onto the _state_, byte-identical across domains → the cacheable prefix):** a dedicated `build_shared_prefix(...)` step (called once after ingestion, **not** the per-domain context node) assembles trial metadata + the **METHODS and RESULTS** sections + the CT.gov block, **jointly capped at `ARBITER_PREFIX_TOKEN_BUDGET` (default 4,000) tokens** → `state.shared_prefix_text` (+ `state.ct_gov_block`), seeded into `TrialState` and every `OutcomeState`. The per-domain context node **reads** this field and never rebuilds it; the SQ worker composes `state.shared_prefix_text` with the domain's suffix at call-build time, so the ~4k-token prefix is stored once per trial (matching the single `cache_control` breakpoint and the single trace `prefix_hash`), never re-embedded per domain ([§5.6](#56-langgraph-state)). The CT.gov block is a **rendered compact text block** of the REQ-04 fields (outcomes, masking, allocation, enrolment count, arms) — **not** the raw JSON, which stays in `ct_gov_data` — and it counts **inside** the prefix budget. When the budget binds, **prioritise within it**: metadata + the CT.gov block first (small, always cross-domain relevant), then METHODS, then RESULTS, trimming the RESULTS _tail_ last, so a long results narrative never crowds out the structured allocation/masking/enrolment signal D1–D4 lean on. Because METHODS/RESULTS are trial-static and always present, building the prefix **once in Phase 1** (rather than re-selecting them per domain under a per-domain cap, which would truncate them differently each time) makes the prefix stable so a single `cache_control` breakpoint and a single per-trial trace `prefix_hash` actually apply across **all** of a trial's SQ calls ([REQ-06](#req-06-llm-abstraction-layer)/[REQ-23](#req-23-run-trace--timing-instrumentation)). A bounded prefix matters most on the **non-caching path** (free-tier / open-model routes that can't cache), where every prefix token is re-billed on each of the trial's ~22 SQ calls — so the cap is kept tight rather than raised.
- **Domain-specific text:** collect `DocumentSection`s whose label partially matches the domain's heading list (`DOMAIN_SECTIONS`), case-insensitive — the **domain's _extra_ sections only** (METHODS/RESULTS already live in the shared prefix); if the selection is shorter than `ARBITER_DOMAIN_TEXT_MIN_CHARS` (default 500) characters, prepend the abstract → `domain_specific_text`. This is part of the **dynamic suffix**, not the cached prefix.
- **Supplement selection:** query `SupplementIndex.retrieve(union_of_domain_key_terms, domain, top_k)`. Include small segments verbatim; for large segments (`char_count ≥ ARBITER_LARGE_SEGMENT_CHAR_THRESHOLD`, default 6,000), sub-rank the segment's sentences by BM25 against the domain key terms and include the top-ranked sentences in rank order until the supplement block reaches its cap. Cap supplement context at `ARBITER_SUPPLEMENT_TOKEN_BUDGET` (default 2,000) tokens (this cap — not an unstated "top few" — is what bounds the included sentences). Record `retrieval_top_score` (the **normalised [0,1]** relevance of the top passage from `retrieve`, not the raw RRF score; `None` if no retrieval ran), `segments_retrieved`, `segments_available`.
- **D3 — participant flow (v0.1: text-only):** inject (a) a **flow block built by a dedicated deterministic extractor** and (b) the **CT.gov `enrollmentInfo.count`** as a randomised-N denominator hint (an _enrolled_ count, which is ≥ randomised-N — a hint, not the denominator; cf. the trial-level caveat below). The extractor (no LLM) scans `section_map` RESULTS + flow-diagram `page_boxes` caption text for participant-flow sentences by keyword/regex (`randomi[sz]ed`, `assessed for eligibility`, `discontinued`, `lost to follow-up`, `analy[sz]ed`, `withdrew`, `N=…`) and writes the matched sentences **into D3's own dynamic suffix**. This is deliberately **independent of the shared prefix**: the prefix is jointly capped and trims its RESULTS tail first when the budget binds (above), so relying on the prefix to carry D3's flow numbers would silently drop them on long papers exactly when D3 needs them. Putting the flow block in D3's suffix makes D3 **trim-robust** while keeping the prefix builder domain-agnostic; the small duplication when RESULTS also survives the prefix is cheap and harmless. It is also the v0.2 seam: (v0.2 adds (c) the trial-tier **`consort.flow`** vision counts when CONSORT fires, coexisting with this text block, with any text-vs-vision disagreement surfaced as a confidence signal — see the deferred CONSORT note below. In v0.1 `consort.flow` is always absent; when text flow numbers are also missing, D3's SQ answers simply fall through to the standard [REQ-11](#req-11-confidence-signal-system) flag rules — a missing-data `NI` resolves to `UNCERTAIN` or `FLAGGED` per D3 supplement availability, with no D3-specific flag path.)
- **D5 only:** inject the outcome-comparison block when `outcome_change_detected is not None`.
- **Budget:** the cacheable prefix (`shared_prefix_text` **incl.** `ct_gov_block`) ≤ `ARBITER_PREFIX_TOKEN_BUDGET` (default 4,000) tokens; the supplement block ≤ `ARBITER_SUPPLEMENT_TOKEN_BUDGET` (default 2,000) tokens, with `domain_specific_text` and the SQ's question/definitions adding a few hundred more on top. At the defaults the prefix is ~4k and the total ~6k — ≈6k is the **total**, never the prefix. All three budgets are conservative defaults, exposed as knobs precisely so the [REQ-21](#req-21-evaluation-harness) recall@k / per-SQ ablation can move them rather than a code edit (the discipline stays "narrow context per SQ"; the eval, not intuition, decides any widening).

### CONSORT extraction (trial tier) — DEFERRED TO v0.2

> CONSORT participant-flow _vision_ extraction is **not built in v0.1** ([§3](#3-users-and-workflow)); D3 is text-only (above). v0.1 keeps the seam (inventory in [§3](#3-users-and-workflow)); the **node, detector, prompt, and vision impl land in v0.2** so this design is recorded but not implemented now:

- **Deterministic detector (no LLM):** score candidate pages/images using `PageBox` (`boxclass == "picture"` + `bbox` + `page`) plus nearby text (caption / within ~300 chars) for CONSORT-flow vocabulary (`CONSORT`, `flow`, `enrol(l)`, `randomi[sz]ed`, `allocated`, `discontinued`, `lost to follow-up`, `analy[sz]ed`, `assessed for eligibility`), biased toward Methods/early-Results, low text density.
- **Conservative gate:** only when the best score ≥ `ARBITER_CONSORT_DETECT_THRESHOLD` render that page to an image and call `vision_model.complete_vision(..., schema=ConsortFlow)`. **Asymmetric risk justifies conservatism:** a _missed_ figure falls back to text-only + an `UNCERTAIN` flag (safe), but a _wrong_ image injects bogus counts and can make D3 confidently wrong.
- **No confident candidate → text-only fallback** (no vision call); record `consort.detected=False` and `consort.detection_score` for auditability.
- **Caveat to document:** the CONSORT flow is the trial-level/primary-analysis flow; per-outcome missingness can differ — treat it as a denominator _hint_, with outcome-specific analysed-N still taken from the Results text.

### Acceptance Criteria

- The Phase 1 `build_shared_prefix` step produces a non-empty `state.shared_prefix_text` once per trial; context assembly **reads** it (never rebuilds it) and returns a `DomainContext` for every domain (with non-empty `domain_specific_text` when the domain has matching sections) for any parseable paper.
- D3 context contains the dedicated-extractor flow block (in D3's **suffix**, not relying on prefix-retained RESULTS) and the CT.gov enrolment denominator when available (v0.1 is text-only; no `ConsortFlow` block — that is v0.2). A paper whose RESULTS tail is trimmed from the shared prefix still gets its flow numbers in D3.
- When D3 text flow numbers are absent, D3's SQ answers fall through to the standard [REQ-11](#req-11-confidence-signal-system) flag rules (no D3-specific flag path).
- D5 context includes the outcome-comparison block when CT.gov data is present.
- Combined main+supplement context does not exceed ~6,000 tokens.

---

## REQ-15: SQ Worker

**Module:** `arbiter/graph/nodes/sq_node.py`

### What to Build

The core LLM node: processes **one** SQ per invocation. Ordering is owned entirely by the `resolve_loop` ([REQ-08](#req-08-conditional-branching)/[REQ-12](#req-12-assessment-orchestration-two-tier-graph)): each wave fans out its **newly-applicable** SQs in parallel and serialises **only across waves** (so within any wave — including the ungated round-1 triggers `{2.1,2.2,2.6}` / `{4.1,4.2}` — all SQs run concurrently). This node makes no scheduling decision of its own.

- Build messages with a **cacheable static prefix** = `state.shared_prefix_text` (trial-static — trial metadata + Methods/Results + CT.gov; held once on the state per [REQ-14](#req-14-context-assembly), marked `cache_control`, hashed once per trial) and a **dynamic suffix** = the `DomainContext`'s `domain_specific_text` + supplement block + the SQ's `question_text` + `answer_definitions` + the task instructions. (The prefix is read from the state, not from the `DomainContext`, which no longer carries it — see [§5.6](#56-langgraph-state).) The task: find the most relevant verbatim sentence(s) in the SOURCE TEXT, copy them exactly, choose the answer code, and write exactly one justification sentence. The model is **not** asked for a page number. **Apply the PY/PN bridge ([REQ-09](#req-09-signaling-question-prompt-templates))** — route reasonable inferences to `PY`/`PN`, reserving `NI` for genuine textual silence. Only if no relevant text exists in any provided source, answer `NI` with an empty quote.
- Call `sq_model`'s `complete_structured(..., schema=SQRawAnswer, call_label=f"{sq_id}|{config.effect_of_interest}")` (temperature 0).
- **Finalize the raw answer through one deep function** — `finalize_sq_answer(raw: SQRawAnswer, sq_id: str, context: DomainContext, *, raw_char_stream: str, page_boxes: list[PageBox]) -> SQAnswer` — which hides the three coupled deterministic post-LLM steps and **owns the invariants that bind them**, so the SQ node never re-derives their ordering or short-circuits by hand:
  1. **Verify** the quote against `raw_char_stream` ([REQ-10](#req-10-quote-verifier)).
  2. **Derive the page deterministically** from the verified quote's location via `locate_quote_page` over `page_boxes` ([REQ-10](#req-10-quote-verifier)) — the LLM never supplies the page.
  3. **Compute** confidence ([REQ-11](#req-11-confidence-signal-system)) from the verification result + the `context` retrieval signals. **No second sample.**
     The invariants enforced **once, here**: an NA/NI answer short-circuits to empty quote + `page = None` + (for NA) `CONFIDENT`; `page is None` **iff** there is no verifiable quote; `compute_confidence` consumes step 1's boolean. `verify_quote`, `locate_quote_page`, and `compute_confidence` remain **separately unit-tested pure functions** ([REQ-22](#req-22-testing)) — `finalize_sq_answer` is a thin facade over them, not a merge, and is the seam the v0.2 CONSORT/vision path reuses to emit an `SQAnswer` without copying this logic.
- Emit `{"sq_answers": {sq_id: finalize_sq_answer(...)}}`.

LLM API errors are handled by the retry wrapper ([REQ-20](#req-20-error-handling-and-retry)), not here.

### Acceptance Criteria

- Returns `{"sq_answers": {sq_id: SQAnswer(...)}}` every invocation, with `confidence` always populated.
- Exactly one LLM call per SQ (no resampling).
- Uses `sq_model`, not `aux_model`.
- `SQAnswer.page` is set by the deterministic `locate_quote_page` over `page_boxes` ([REQ-10](#req-10-quote-verifier)), **not** taken from the LLM (which is not asked for it); `None` for NA/NI.
- `finalize_sq_answer` is independently testable on a raw `SQRawAnswer` + fixture context (no graph node), and enforces the verify→page→confidence invariants in one place; the three underlying functions still pass their own unit tests.

---

## REQ-16: Output — JSON and SQLite Writers

**Modules:** `arbiter/output/json_writer.py`, `arbiter/output/sqlite_writer.py`

### JSON Writer

`write_assessment_json(assessment: Assessment, output_dir: Path) -> Path` writes one file per trial-outcome into a **per-trial nested layout** (creating directories as needed):

```
{output_dir}/{trial_id}/
  trace.json                              # per-trial trace/timing (REQ-23)
  artifacts/                              # trial-tier intermediate dumps (REQ-23, trace-level=full)
  {outcome_slug}__{effect}/
    data.json                             # ← this writer
    report.md                             # reviewer-facing report (REQ-24)
    artifacts/                            # outcome-tier intermediate dumps (REQ-23, trace-level=full)
```

The `{effect}` segment keeps the two effect-of-interest assessments of one trial-outcome from colliding (mirrors the unique key below). Co-locating `data.json`, `report.md`, the per-trial `trace.json`, and the intermediate `artifacts/` under one `{trial_id}/` directory keeps everything for a trial in one place.

> **The on-disk layout is keyed by `trial_id`, not `model_sq`/`pipeline_version` — so a multi-model run MUST shard `output_dir`.** The SQLite unique key disambiguates rows by `model_sq` + `pipeline_version`, but the path `{trial_id}/{outcome_slug}__{effect}/data.json` does not. Running the [REQ-21](#req-21-evaluation-harness) roster (7 models in v0.1) into one `output_dir` would have every model overwrite the previous model's `data.json`/`report.md`/`trace.json`, and 7/8 SQLite rows would carry a `json_path` that lies. The rule: **each `(sq_model, pipeline_version)` gets its own `output_dir` shard** (`<base>/{sq_model}__{pipeline_version}/…`), which the eval harness sets automatically. Inside a shard the per-trial layout is collision-free; the **SQLite `db_path` stays shared** (rows are already model-keyed) and each row's `json_path` points into its own shard. The JSON contains: identifiers, models used, `trial_metadata`, `outcome_comparison`, a `domains` object (`D1`–`D5`, each with judgment, rationale, and per-SQ `{answer, quote, page, justification, confidence}`), the `overall` block, `sources_manifest`, and `errors`. **It contains no trace/timing data** — that is a side-channel artifact ([REQ-23](#req-23-run-trace--timing-instrumentation)). The D1 block is present and identical across the trial's outcome files; D2–D5 are outcome-specific.

### SQLite Writer

`write_assessment_sqlite(assessment: Assessment, db_path: Path) -> None` upserts one row per trial-outcome into `arbiter_assessments` (created on first write). Columns include: `assessment_id` (PK), `created_at`, `trial_id`, `nct_number`, `title`, `outcome`, `effect_of_interest`, `overall_judgment`, `d1..d5_judgment`, `flagged_sq_count`, `uncertain_sq_count`, `requires_human_review`, `study_design` (nullable; populated on `SkipRecord` rows for the ineligibility reason), `model_sq`, `model_aux`, `pipeline_version`, `inputs_hash` (non-keyed; the living-evidence refresh runway, [REQ-17](#req-17-batch-runner-and-manifest)), `json_path`, `errors` (JSON array). A **unique key** on `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` backs batch idempotency ([REQ-17](#req-17-batch-runner-and-manifest)). `effect_of_interest` is in the key because it changes D2 (and therefore potentially the overall judgment): without it, assessing one trial-outcome under `assignment` then `adhering` would clobber or skip the second. `pipeline_version` is the umbrella identifier for the rest of the pipeline **configuration** (`aux_model`, `vision_model`, `consort_vision_enabled`, retriever knobs), deliberately **not** in the key — so it must be bumped when any of those changes ([REQ-01](#req-01-dependency-and-project-setup) owns the rule + rationale, including the eval harness's automatic per-arm derivation that makes the unattended sweep collision-proof). Use `INSERT … ON CONFLICT … DO UPDATE` (or `INSERT OR REPLACE`) so re-writes don't raise. The row carries **no trace/timing/cost columns** (invariant 3, §2). `json_path` points at the nested `…/{outcome_slug}__{effect}/data.json`.

The judgment columns (`overall_judgment`, `d1..d5_judgment`, `flagged_sq_count`, `uncertain_sq_count`) are **nullable**, because an ineligible-trial **`SkipRecord`** ([§5.4](#54-the-assessment-record-flat-one-per-trial-outcome)/[REQ-17](#req-17-batch-runner-and-manifest)) writes **one row per trial** with sentinel `outcome = "__TRIAL__"`, `study_design` set, judgment columns `NULL`, `requires_human_review = 1`, and the reason in `errors` — no `data.json`, no per-outcome rows. On a skip row `title` is `NULL` (the minimal `SkipRecord` does not carry it) and `json_path` points at the trial's `skip.json`, not a `data.json`.

`write_skip_record(skip: SkipRecord, output_dir: Path, db_path: Path) -> Path` is the **dedicated** persister for the skip path — it writes the sentinel SQLite row (into the same `arbiter_assessments` table, using the nullable columns above) **and** the `{output_dir}/{trial_id}/skip.json`. It is deliberately **separate** from `write_assessment_json`/`write_assessment_sqlite`, for the same reason `SkipRecord` is separate from `Assessment` ([§5.4](#54-the-assessment-record-flat-one-per-trial-outcome)): overloading the assessment writers to `Assessment | SkipRecord` would re-introduce the nullable-judgment branching into the happy-path writers that `SkipRecord` exists to keep clean. The eligibility gate ([REQ-17](#req-17-batch-runner-and-manifest)) calls it; `assess_trial` never does.

### Acceptance Criteria

- One JSON file and one SQLite row per trial-outcome; JSON is valid and contains all 22 SQ answers under their domain keys.
- Re-writing the same `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` does not raise.
- `flagged_sq_count` / `uncertain_sq_count` match the per-SQ flags; `errors` is a JSON array; `requires_human_review` reflects the rollup.

---

## REQ-17: Batch Runner and Manifest

**Modules:** `arbiter/manifest.py`, batch logic invoked from `arbiter/cli.py`

### What to Build

The **primary interface**: run ARBITER unattended over a manifest of trials.

### Manifest

A CSV or JSON file parsed into `BatchManifest`. Per entry: `main_paper` (**required**); `supplements` (optional — a file or a directory of PDFs); `nct_number` (optional — else derived from the paper); `outcomes` (optional — else `[primary outcome]`); `trial_label` (optional). CSV columns mirror these names; multiple outcomes are a delimited list.

### Runner Behaviour

- For each entry: **`ingest_trial` once** ([REQ-19](#req-19-python-api)) → **`check_eligibility(ctx.trial_metadata)`** (below); on pass, **`assess_trial(ctx, …)`** — D1 once, then loop the entry's outcomes (outcome tier — D2/D3/D4/D5 per outcome), writing one record per trial-outcome. Ingestion happens exactly once, before the gate, so an ineligible trial is never ingested twice.
- **Resume-on-interrupt (not refresh-on-change):** before assessing a trial-outcome, check the DB unique key `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)`; **skip** if already present unless `--force`. Because `trial_id` is deterministic ([REQ-05](#req-05-trial-metadata-extractor)) — including the NCT-less content-hash fallback — re-running a batch resumes by doing only missing/failed pairs even for trials with no registry metadata. **This is resume semantics, not refresh:** if a trial's _inputs_ change (a new supplement is published, the registry record updates) but the key is unchanged, the pair is **skipped, not re-assessed**. Refresh-on-change is a living-evidence concern deferred to v0.2 (see the limitation below); v0.1 ships only the runway for it: a non-keyed `inputs_hash` column on each row ([REQ-16](#req-16-output-json-and-sqlite-writers)) = a hash of `(paper_bytes, sorted supplement_bytes, ctgov_snapshot)`, where `ctgov_snapshot` is the CT.gov v2 `protocolSection.statusModule.lastUpdatePostDateStruct.date` (the registry record's last-update date — its stable per-fetch snapshot key; `None` when CT.gov was absent). It does not affect skip/resume behaviour today, but lets a future refresh layer detect stale rows with `WHERE inputs_hash != ?` and no schema migration.
- **Continue-on-error:** any exception assessing one trial or outcome is caught, recorded (error row / `errors` field), and the batch proceeds to the next. One bad PDF never halts the batch.
- **Bounded concurrency:** the runner dispatches entries via `asyncio`, and `ARBITER_MAX_CONCURRENCY` (default low) is a **single global async semaphore over LLM calls**, acquired inside the client's `complete_structured`/`complete_vision` ([REQ-06](#req-06-llm-abstraction-layer)). It throttles the quantity providers actually rate-limit — total in-flight calls — directly, regardless of how trials/domains/SQ-waves nest above it. This is deliberately **not** a trial-level cap: bounding trials while leaving each trial's intra-domain fan-out (the [REQ-08](#req-08-conditional-branching) waves, ~2–3 SQs each after the conditional chains serialise) uncapped would let in-flight calls = trials × fan-out exceed the free-tier limit, leaving REQ-20 backoff to absorb a storm of 429s reactively. One global call semaphore prevents the storm; the [REQ-20](#req-20-error-handling-and-retry) backoff stays as the safety net for bursts that still slip through.
- **Eligibility gate (deterministic; enforces IRPG scope) — a shared precondition, not a batch-only feature.** Factor it into a small helper `check_eligibility(trial_metadata) -> SkipRecord | None`, called on `ctx.trial_metadata` **after `ingest_trial`** ([REQ-19](#req-19-python-api), which performs the [REQ-05](#req-05-trial-metadata-extractor) metadata extraction) and **before** `assess_trial`, by **both** entry points: the batch runner (per entry) **and** `arbiter assess` ([REQ-18](#req-18-cli)) — so the single-trial path cannot bypass it and fabricate an out-of-scope assessment. Because ingestion already ran in `ingest_trial`, the gate reads metadata that exists, with no second ingestion. (`assess_trial` itself stays gate-free and "always computes" — [REQ-19](#req-19-python-api) — with the gate as a documented caller precondition; this keeps its `list[Assessment]` return type clean of skip handling.) The helper checks `trial_metadata.study_design`: if it is **not** `parallel_rct`, do **not** fabricate a five-domain assessment — emit a **`SkipRecord`** ([§5.4](#54-the-assessment-record-flat-one-per-trial-outcome)) and persist it via **`write_skip_record`** ([REQ-16](#req-16-output-json-and-sqlite-writers)): **one SQLite row per trial** (sentinel `outcome = "__TRIAL__"`, judgment columns `NULL`) carrying the design, the `study_design_basis`, `requires_human_review=True`, and a clear `errors` entry (`"ineligible study_design=<x>: <basis>"`), plus a `skip.json` for audit; then proceed (batch: next entry; assess: exit). No `Assessment` and no `data.json` are written for an ineligible trial. `unclear` is treated as **ineligible-but-flagged** (skipped, not assessed) so neither path ever emits a confident-looking RoB 2 score for an input ARBITER could not confirm is a parallel-group RCT. This is the analogue of auto-rob2's RCT-screening step, implemented as a deterministic gate over the existing aux call (no new LLM call, no LLM judgment).

### Acceptance Criteria

- `arbiter batch manifest.csv` processes every entry; a manifest with only `main_paper` columns runs (NCT derived, primary-outcome default).
- A directory in `supplements` ingests all PDFs within it without manual categorisation.
- Re-running the same batch skips already-completed trial-outcome pairs; `--force` re-runs them.
- A deliberately corrupt entry produces an error record and does not stop the batch.
- An input classified `study_design != parallel_rct` (e.g. a single-arm or observational PDF) produces a **`SkipRecord`** (one trial row, sentinel `outcome`, `requires_human_review=True`, `NULL` judgments) and **no** fabricated D1–D5 assessment; the batch continues. `unclear` is likewise skipped-and-flagged.

---

## REQ-18: CLI

**Module:** `arbiter/cli.py`

### Commands

```
arbiter assess   — run a single trial (all its outcomes)
arbiter batch    — run a manifest of trials (primary interface)
```

_(There is no `resume` command — resilience is batch idempotency, REQ-17.)_

### `arbiter assess` options

`--paper PATH` (required); `--supplement PATH` (repeatable; file or dir); `--nct TEXT`; `--outcome TEXT` (repeatable; default = extracted primary); `--effect [assignment|adhering]` (default assignment); `--sq-model TEXT` / `--aux-model TEXT` (default from env); `--trace-level [off|summary|full]` (**default `full`** for `assess` — single-trial dev inspection; emits LLM I/O + intermediate artifacts, [REQ-23](#req-23-run-trace--timing-instrumentation)); `--no-report` (suppress the Markdown report, [REQ-24](#req-24-reviewer-facing-markdown-report)); `--output-dir PATH`; `--db PATH`; `--force`. (`--vision-model` / `--consort` are **v0.2** — vision is deferred, D3 is text-only.)

### `arbiter batch` options

`--manifest PATH` (required); `--sq-model` / `--aux-model`; `--trace-level [off|summary|full]` (**default `summary`** for `batch` — timings/counts only, no prompt bodies or artifact dumps, to stay lean at volume); `--no-report`; `--output-dir`; `--db`; `--max-concurrency INT`; `--force`. (`--vision-model` / `--consort` are **v0.2**.)

### Stdout

`assess` prints the trial id and, per outcome on completion, the outcome and overall judgment plus the JSON path. `batch` prints a per-trial progress line and a final summary: counts (completed / skipped / errored) **plus timing/cost totals** read from each trial's trace summary — total wall time, total LLM latency, total LLM calls, total tokens, total cost (`null` where any model's pricing is unknown), and the slowest trials ([REQ-23](#req-23-run-trace--timing-instrumentation)).

### Acceptance Criteria

- `arbiter assess --paper paper.pdf` runs end-to-end (primary outcome).
- `arbiter assess` applies the **shared eligibility gate** (`check_eligibility`, [REQ-17](#req-17-batch-runner-and-manifest)) before assessing: a non-`parallel_rct` (incl. `unclear`) input writes a `SkipRecord` (`skip.json` + one sentinel SQLite row, no fabricated D1–D5) and exits — identical to the `batch` path, so `assess` cannot bypass the gate.
- `arbiter assess --paper paper.pdf --supplement sap.pdf --nct NCT01234567 --outcome "Overall Survival" --outcome "Progression-Free Survival"` produces two records.
- `arbiter batch --manifest m.csv` runs and is idempotent on re-run.
- Commands exit 0 on success, non-zero on unrecoverable error (e.g., auth failure).

---

## REQ-19: Python API

**Module:** `arbiter/__init__.py`

### What to Build

```python
async def ingest_trial(config: AssessmentConfig) -> TrialContext:
    """Phase 1 ONCE per trial: create the per-trial RunTrace + LLM clients
    (trace attached), parse paper + supplements, fetch CT.gov, extract
    metadata, and build the trial-static `shared_prefix_text`. Returns a
    TrialContext bundling these (clients + trace included) so the caller can
    run the eligibility gate on `ctx.trial_metadata` BEFORE any domain is
    judged, and so ingestion provably happens exactly once."""

async def assess_trial(ctx: TrialContext, config: AssessmentConfig) -> list[Assessment]:
    """Assess one ALREADY-INGESTED, ALREADY-ELIGIBLE trial across
    config.outcomes (or its primary outcome). Judges D1 ONCE, reusing it
    across outcomes; D2-D5 are judged per outcome. Does NOT re-ingest and
    does NOT gate — both are the caller's job (below), which is what keeps
    the return type a clean `list[Assessment]`, never a `SkipRecord`."""
```

**The Phase 1 / gate / Phase 2 seam is explicit.** `ingest_trial` and `assess_trial` are split so the **eligibility gate sits cleanly between them, at the caller, with no double-ingestion**: `trial_metadata` is produced by `ingest_trial` and consumed by `check_eligibility` _before_ `assess_trial` ever runs ([REQ-17](#req-17-batch-runner-and-manifest)). `TrialContext` bundles the Phase 1 outputs (`section_map`, `raw_char_stream`, `supplement_index`, `ct_gov_data`, `trial_metadata`, and the built `shared_prefix_text` + `ct_gov_block`) so `assess_trial` consumes them rather than rebuilding them; the v0.2 living-evidence refresh layer ([REQ-17](#req-17-batch-runner-and-manifest)) also gets a natural place to diff `inputs_hash` before deciding whether to re-ingest.

Caller flow (the batch runner per entry, or `arbiter assess` — [REQ-18](#req-18-cli)): **`ingest_trial(config)`** — which **creates the per-trial `RunTrace` collector** ([REQ-23](#req-23-run-trace--timing-instrumentation)), constructs the `sq_model`/`aux_model` clients with the trace attached, runs Phase 1 ingestion, and calls **`build_shared_prefix(...)` once** to assemble the trial-static `shared_prefix_text` + `ct_gov_block` ([REQ-14](#req-14-context-assembly)) — returning a `TrialContext` that bundles all of these (clients + trace included) → **`check_eligibility(ctx.trial_metadata)`** ([REQ-17](#req-17-batch-runner-and-manifest)); on a non-`parallel_rct` (incl. `unclear`) write a `SkipRecord` and stop, **no `assess_trial`** → otherwise **`assess_trial(ctx, config)`**, reusing `ctx`'s clients/trace: seed `shared_prefix_text` into the trial state and every outcome state → Phase 2 trial graph (**D1**) once → for each outcome, outcome graph (**D2, D3, D4, D5** + rollup) seeded with the reused D1 (and the shared prefix) → Phase 3 build `Assessment` (**sorting `domain_judgments` D1…D5** — the parallel outcome branches append them in non-deterministic order, [REQ-12](#req-12-assessment-orchestration-two-tier-graph)), write `data.json` + SQLite + the Markdown report ([REQ-24](#req-24-reviewer-facing-markdown-report), unless `--no-report`) → **flush `ctx.trace` to `trace.json` (+ intermediate `artifacts/` when `trace-level=full`)**. Idempotency and concurrency are likewise enforced by the caller; `assess_trial` **assumes an eligible `parallel_rct`** and always computes. The trace is collected as a side effect and never affects the returned `Assessment` list.

### Acceptance Criteria

- `assess_trial` returns one `Assessment` per requested outcome; D1 identical across them (D2–D5 judged per outcome).
- `ingest_trial` runs ingestion (PDF parsing, supplement indexing, CT.gov, metadata, shared-prefix build) **once per trial**, returns a `TrialContext`, and is the **only** ingestion call — `assess_trial` consumes the context and never re-ingests.
- The eligibility gate runs on `ctx.trial_metadata` between `ingest_trial` and `assess_trial`; `assess_trial` is never reached for an ineligible trial.

---

## REQ-20: Error Handling and Retry

**Module:** `arbiter/llm/base.py` (applied in all clients) and the batch runner.

### Two Independent Retry Layers

1. **Network/transient retries (LLM transport):** wrap `complete_structured` calls with bounded exponential backoff + jitter (`ARBITER_NETWORK_MAX_RETRIES`, default 3 attempts) for rate-limit / timeout / connection errors. This is the **sole** network-retry layer — each LangChain client is constructed with **`max_retries=0`** so the dependency does not retry underneath this loop ([REQ-06](#req-06-llm-abstraction-layer)); ARBITER owning the loop is also what lets the [REQ-23](#req-23-run-trace--timing-instrumentation) `CallRecord` capture each attempt's transient error. **Authentication / invalid-request errors raise immediately** (no retry) and abort the run with a clear message.
2. **Schema-validation retries (non-native providers):** the bounded re-prompt loop in [REQ-06](#req-06-llm-abstraction-layer). Distinct from layer 1.

### Per-SQ Final Failure

If an SQ's LLM call still fails after network retries, the SQ node **does not crash the graph**: it records the SQ as `answer = NI`, empty quote, `confidence.flag = FLAGGED` (reason names the failure), and appends a string to `errors`. The domain judgment then runs on whatever answers are available.

### Batch-Level

Per [REQ-17](#req-17-batch-runner-and-manifest): a failure assessing a trial/outcome is caught, recorded, and the batch continues.

### Acceptance Criteria

- A mocked rate-limit error retried 3× then succeeds is transparent to the caller; 3× failure yields an `NI`/`FLAGGED` SQ and an `errors` entry, with the graph completing.
- An auth error aborts immediately with a clear message.

---

## REQ-21: Evaluation Harness

**Module:** `eval/run_eval.py`; reference data in `eval/reference/`

### Purpose

The harness serves **two distinct goals that must not be conflated** — they have different inputs, configs, and outputs:

1. **Dev smoke-test (internal):** confirm the pipeline runs end-to-end and produces sane output. Fast, run during development/CI. **It never produces a published number.**
2. **Paper eval (traceable mined set + own depth set):** the powered, reproducible comparison that backs the research claims — the **whole-pipeline open-vs-frontier head-to-head** (REQ-01 roster; each arm runs its own model as both `sq_model` and `aux_model` — see the whole-pipeline note under Model comparison) run through the same pipeline code, plus the architecture ablations below.

### Execution modes — free-tier (dev) vs pinned paid (headline)

The two goals run on **different provider policies**, and a published number must come from the second:

- **Dev / smoke-test / iteration → OpenRouter free tier.** Cheap and good enough to shake out the pipeline. Accepts the non-native schema path (REQ-06 repair ladder is the real path here) and the multi-provider routing variance.
- **Headline paper run → a pinned _paid_ provider + dated snapshot, per model** — including paid `gpt-oss-120b`. On a paid structured-output-capable provider, even the open models can use decode-time JSON-schema enforcement (`response_format=json_schema` + `provider.require_parameters=true`, [REQ-06](#req-06-llm-abstraction-layer)/[REQ-01](#req-01-dependency-and-project-setup)), so the schema-repair confound largely **evaporates for the headline numbers** and routing variance is pinned out. The free-tier run, if reported, is a **secondary "also runs at \$0, higher variance" footnote**, never the headline.

**Size the run before launching it.** A full roster sweep is large, but cost splits between two tiers — don't attribute the per-trial calls to each outcome. **Per outcome:** only the D2–D5 SQ calls (≤19, fewer after branching NAs them). **Per trial (once, amortised across its outcomes):** the metadata aux call + the 3 D1 SQ calls + the per-document supplement-annotation calls (up to `ARBITER_MAX_ANNOTATIONS_PER_DOC`=40/doc). At a ~100–200 trial-outcome mined set — which skews **one-outcome-per-trial** (so per-trial ≈ per-outcome here) — that lands on the order of ≈2.5k–5k calls/model; the 7 v0.1 roster models (the 8th registry row is the v0.2 Gemma vision model) + the ARBITER-Depth ablations + optional `--repeats K` ⇒ **tens of thousands of calls**. This is slow on the free tier — another reason the headline run uses pinned paid providers.

### Datasets

| Set                                                                                             | n                                      | Granularity                                         | Role                                         | Notes                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **mHSPC-28** (`eval/reference/{overall_survival,progression_free_survival,adverse_events}.csv`) | 28 trial-outcome                       | domain + overall                                    | **dev smoke-test only**                      | single tumour type; L/S only (no High); non-standard rollup (below). **Used for end-to-end/sanity only — accuracy is never tuned against it and it is not a tuning signal.** Produced by a **different team in our lab** (label-independent from the pipeline developers), so reusing a slice in ARBITER-Depth introduces no develop/report leak. NCTs + SAPs present.                                      |
| **Cochrane-RoB2-mined** (provisional name; `eval/benchmarks/cochrane_mined/`)                   | curation target ~100–200 trial-outcome | **domain + overall + support-for-judgement quotes** | **PRIMARY powered set**                      | derived from **published Cochrane reviews that used RoB 2** (the ~86-review / 1,399-RCT frame Huang et al. sampled, plus other traceable published RoB 2). **Traceable** (each included study → PMID/DOI/NCT), **dual-reviewer-adjudicated, peer-reviewed**, multi-condition, High labels present. **No published per-SQ codes** — domain-level truth only. Release: derived layer + pointers only (below). |
| **ARBITER-Depth** (own-built, **internal only**; `eval/benchmarks/depth/`)                      | 15–25 trial-outcome                    | **SQ + domain + overall + evidence passages**       | **differentiator validation + per-SQ depth** | multi-condition, selected for **input richness** (NCT + registry + full text + ≥1 supplement + CONSORT), **High / outcome-switching deliberately included**, **dual-annotated + adjudicated** via the official RoB 2 Excel workbook (captures SQ codes). May reuse an mHSPC slice — **safe**, because mHSPC-28 is smoke-only and label-independent (no develop/report leak).                                |

> **Why this shape.** Published Cochrane RoB 2 tables give domain/overall judgments + a support-for-judgement quote + a traceable trial citation, _for free and already adjudicated_ — but **not** the per-SQ answer codes (those live in the assessor's Excel workbook, not the published review). So the **powered headline is domain/overall agreement on the mined set** (with quote **faithfulness** — the REQ-10 fuzzy locate-rate in the trial text — reported as an integrity property, **not** matched against the reviewer's support-for-judgement quote), and **per-SQ accuracy is a depth metric on ARBITER-Depth only** (where we control the workbooks). This is _why_ the prior plan's ROBoto2/RoBuster sets were dropped: their released data exposes no trial identifiers, so neither the trials nor your registry/supplement differentiators can be located — disqualifying for this architecture. They survive only as **cited related work** (NI over-selection motivation; human-IRR ceiling).

> **Rollup conventions.** mHSPC-28 used a NON-STANDARD overall rollup (overall Low = "low in all domains **or some concerns in one domain**"); since it is dev-only this no longer contaminates a published claim, but the dev harness still reports overall **rollup-normalised**. The mined-set reviews and ARBITER-Depth use the **official RoB 2 flowchart rollup**; overall agreement is reported **rollup-normalised to ARBITER's policy** plus as-published.

> **ROBUST-RCT is a different instrument (6-item), not RoB 2** — cite it as related work, never as RoB 2 ground truth.

### Building the primary set — mining published RoB 2 reviews

The mined set is built by **curation, not de-novo annotation** (which is why it scales under low assessor capacity):

- **Sampling frame:** published systematic reviews that used **RoB 2** (not RoB 1 — different domains, not comparable) and publish a per-included-study **"Risk of bias" table**. Cochrane reviews from ~2020 on are the spine (Huang et al. found **86 eligible RoB 2 reviews covering 1,399 RCTs**); add any other traceable published RoB 2 sets.
- **Extraction, per included study × review outcome:** the **domain judgments**, the **overall judgment**, the **support-for-judgement quote** (recorded for **provenance only** — it is **not** used for any scoring metric now that quote-vs-reviewer concordance is dropped), and the **trial citation** → resolve to **PMID/DOI/NCT**. Confirm each trial resolves to a fetchable full text; de-duplicate studies appearing in multiple reviews.
- **Copyright-safe release (mirrors `docs/rob2/` vendoring):** publish **derived assessments + trial pointers (PMID/DOI/NCT) + the extraction code** — **never** the Wiley/Cochrane review prose, and trial full-text/supplement PDFs are git-ignored with fetch instructions committed. The derived layer may itself be a **secondary dataset contribution** ("a traceable RoB 2 evaluation set from published reviews").
- **Known limitations (record):** **domain-level only** (no SQ codes → per-SQ is depth-set-only); **between-review heterogeneity** (different teams, RoB 2 IRR is only fair). ARBITER assesses the mined trials with its **enriched ship-default config (registry-enriched throughout via CT.gov/NCT; supplements best-effort)**, and we **assume the mined-set reviewers consulted registry/protocol comparably**; where they did not, enriched ARBITER may **correctly diverge** from the published label — a **partial-matching limitation we accept rather than stratify**. Full supplement-grounding (and the per-outcome D2 correctness) is therefore validated on **ARBITER-Depth**, not the mined headline. The mined set also skews **one-outcome-per-trial**, so per-outcome D2 divergence is exercised on mHSPC-28/ARBITER-Depth, not the headline.

### The own-built depth set (ARBITER-Depth, internal)

**15–25 trial-outcomes**, multi-condition, **dual-annotated + adjudicated** with the official RoB 2 Excel workbook (so SQ codes are captured). Selected for **input richness** — every trial has NCT + registry + accessible full text + ≥1 supplement (SAP/protocol) and ideally a CONSORT diagram — and to **deliberately include High-risk and outcome-switching cases**. It is **internal validation only** (not released). Its only jobs: (a) **per-SQ depth**, (b) **validating the D5/supplement/CONSORT differentiators** on inputs we control, (c) the **divergent-cell adjudication** below. It is **not** sized to power headline agreement — the mined set carries that.

> **Crediting grounding needs adjudication, not just agreement.** A _correct_ enriched pipeline will **disagree** with a main-text-derived label precisely when the supplement/registry changes the right answer (e.g. an SAP reveals an outcome switch the paper hid → true D5 = High, while the report-only label said Low). So "enriched agrees more" inverts exactly where grounding pays off. On **ARBITER-Depth** (inputs controlled), run a **matched main-text-only arm** vs a **supplement/registry-enriched arm**, then have an assessor **re-judge the divergent D5 (and supplement-moved D3) cells with the supplements in hand, blinded to arm** → a "grounding fixed X% of D5 errors" number. The bare enriched-vs-main-text delta is **feasibility only** until adjudicated.

### Per-SQ scoring conventions (ARBITER-Depth only)

Per-SQ accuracy is scored only where we hold SQ codes (ARBITER-Depth). Conventions:

1. **Collapsed classes.** Score on `{Y/PY, N/PN, NI}` + structural `NA` (RoB 2 gating treats Y/PY and N/PN as units); full 6-code exact-match is an **optional secondary strictness column**.
2. **Effect pinned to `assignment`** for matching, with **D2 flagged as the highest-risk domain** (independently the worst-agreeing domain in Huang et al.).
3. **Structural `NA` excluded** from the per-SQ denominator; **branching-applicability agreement** reported as a separate small metric.

### Metrics

**Agreement statistics (applied to every agreement metric below):** report the **triad — (1) raw % agreement, (2) Gwet's AC2, (3) Cohen's κ — each with bootstrap 95% CIs**, plus the confusion matrix. RoB 2 labels are severely imbalanced (mostly Low/Some concerns), which triggers the **prevalence paradox**: Cohen's κ collapses toward 0 at high % agreement (why prior work reports D5 κ≈0.10). **Gwet's AC2 is the prevalence-robust headline chance-corrected statistic**; % agreement is the interpretable floor; κ is retained only for comparability with Huang/Nagao/Eisele-Metzger (state the prevalence caveat). Always benchmark agreement against the **human–human inter-rater ceiling** — RoB 2 IRR among trained assessors is only _fair-to-moderate_ (κ≈0.40 in ROBoto2's reliability sample, Fleiss≈0.45 in prior work) — ARBITER should be judged relative to that ceiling, not to perfection.

**Core accuracy:**

- **Per-domain agreement (HEADLINE — mined set):** exact-match over judged domain cells vs the published review judgment, per domain + confusion matrix. Rollup-independent. The powered, multi-condition headline.
- **Overall-judgment agreement (HEADLINE — mined set):** **rollup-normalised to ARBITER's policy** (recompute overall from domain cells, then compare) as primary; **as-published** reported alongside.
- **Quote faithfulness (integrity guarantee — all sets):** the fraction of ARBITER's quotes **located in the trial full text by the [REQ-10](#req-10-quote-verifier) fuzzy verifier** (`partial_ratio ≥ 85`, absorbing OCR/hyphenation/ligature artefacts; quotes <15 chars auto-pass). This operationalises the central guarantee ("the LLM only finds quotes, never invents them") — but report it **honestly as a fuzzy locate-rate, not as "verbatim"**: publish the **distribution of match scores** (not just the pass-rate) so near-threshold (85–90) matches — the ones most likely to be paraphrase rather than copy — are visible, plus a **stricter near-exact column (`ratio ≥ 98`)** alongside the robust rate. Reported as a **property** of the architecture (expected near-ceiling for every model), **not** a discriminating headline result. It is deliberately **not** compared against the reviewer's chosen support-for-judgement quote: a paper contains many valid supporting sentences, so matching the reviewer's specific snippet would measure quote-selection agreement, not grounding — that quote-vs-quote concordance was dropped as low-value complexity.
- **Per-SQ accuracy (DEPTH — ARBITER-Depth only):** collapsed-class match (`{Y/PY, N/PN, NI}`, structural `NA` excluded — see conventions above) of ARBITER's answer code vs the workbook SQ codes, per SQ and per domain. The direct test of the LLM's only job — **demonstrated, not powered** (n=15–25). **Reported two ways to separate RoB 2 competence from a JSON-formatting artifact:** (i) **parsed-only** — over SQs where the model returned a schema-valid answer; and (ii) **end-to-end** — including SQs where the [REQ-06](#req-06-llm-abstraction-layer) repair ladder exhausted and [REQ-20](#req-20-error-handling-and-retry) recorded a format-failure `NI`/`FLAGGED`. On the headline paid run (schema enforced) the two converge; on the free-tier dev run they can diverge, and reporting both prevents a formatting failure from masquerading as a substantive answer.

**Model comparison (the paper's spine):**

- **Open-vs-frontier head-to-head (descriptive):** every metric above, per model in the REQ-01 roster → a **descriptive characterisation** of the open–frontier gap per metric, interpreted against the human-IRR ceiling. **No pre-registered "competitive" threshold is asserted.** Report token **cost per assessment** per model (frontier $ vs open ≈ $0); the $0-cost and on-prem-locality affordances are reported as **factual**, independent of where accuracy lands.
  > **The roster sweeps `sq_model` _and_ `aux_model` together — this is a _whole-pipeline_ comparison, not an `sq_model`-isolated one.** Each open roster entry runs **its own model as both `sq_model` and `aux_model`** (metadata extraction, supplement annotation), and the frontier control runs a frontier model throughout. This is what the **$0-cost / "a hospital could self-host"** affordance actually requires — an on-prem deployment cannot assume a frontier `aux_model` exists — so the honest headline is "**whole-pipeline open vs whole-pipeline frontier**," and the prose must say so rather than leaning on the looser "identical pipeline." Consequence: because `aux_model` varies per roster entry, **the harness assigns each roster config a distinct, auto-derived `pipeline_version`** — computed by hashing the non-keyed config dims (`aux_model`, retriever knobs, vision/CONSORT flags) onto the base code version (the [REQ-01](#req-01-dependency-and-project-setup) bump rule, automated here) — so the model-keyed rows never silently skip or clobber, even in the danger case where `sq_model` is held fixed and only `aux_model` moves (the aux-isolating ablation below). If you _also_ want to isolate the SQ-answering effect, add an **optional secondary arm holding `aux_model` fixed** — labelled explicitly as the `sq_model`-isolating ablation, never the headline. Avoid leaving this unstated: a reader otherwise can't tell whether an open model's per-SQ score credits its SQ-answering or its aux.
- **Per-model schema-repair rate:** for non-native-schema models (gpt-oss family), the rate at which the REQ-06 hybrid path needed re-prompting — a reproducibility/robustness signal that the headline path holds up.
- **Intra-model run-to-run consistency (OPT-IN, `--repeats K`, default off):** repeat each model K× and report SQ-answer agreement + judgment stability across runs — characterises the OpenRouter-routing variance the single-shot production path accepts. Off by default to conserve API budget.

**Architecture ablations (paper results):**

- **Retrieval recall@k (BM25 vs dense vs hybrid):** against gold evidence passages drawn from **ARBITER-Depth's annotated evidence passages only** — we annotate these directly, so **no support-quote localization is needed** (the mined set is not used for the retrieval ablation), per domain → **characterises** the retriever (hybrid ships by default regardless; this informs the parked reranker, it does **not** gate the shipped default — [REQ-03](#req-03-supplementary-material-ingestor)). A demonstrated architecture-tuning metric (n=15–25), not powered.
- **D5 stratified by registry/protocol availability:** split D5 agreement into a **present arm** (ARBITER-Depth, which always has registry/supplement; plus mined-set trials whose review cited a registry/protocol) vs **absent arm** — isolating registry-grounding as the lever (tests Nagao's hypothesis directly). Expect D5 to recover in the present arm.
- **Enriched vs main-text-only ARBITER (ARBITER-Depth):** the value of supplements/CT.gov, with the **divergent-cell adjudication** above converting it from feasibility to a correctness claim.
- **NI-rate vs human NI-rate:** per domain — the calibration check for the NI countermeasure ([REQ-09](#req-09-signaling-question-prompt-templates)/[REQ-11](#req-11-confidence-signal-system)); an elevated rate triggers eval-gated re-retrieval. **Split `NI` into substantive-`NI` (the model genuinely answered "No Information") and format-failure-`NI` (the [REQ-20](#req-20-error-handling-and-retry) fallback after the repair ladder exhausted)** — only the substantive rate is the calibration signal; conflating them would credit the PY/PN bridge with fixing what is actually a formatting failure.
- **CONSORT-vision contribution (D3 A/B) — v0.2, out of v0.1 eval scope.** The vision-on vs text-only D3 delta is the planned metric for the v0.2 CONSORT vertical ([§3](#3-users-and-workflow)/[REQ-14](#req-14-context-assembly)); v0.1 D3 is text-only so there is no A/B to run yet. When built it is **underpowered on any single set** (`DIRECTIONAL ONLY`), and a non-positive delta would signal raising `ARBITER_CONSORT_DETECT_THRESHOLD`, not evidence either way.
- **Confidence-flag calibration (diagnostic):** whether disagreements concentrate in `FLAGGED`/`UNCERTAIN` cells (expected in D3).

**Operational diagnostics:** the harness reads each run's `timing_summary` ([REQ-23](#req-23-run-trace--timing-instrumentation)) for wall time, LLM latency, call/cache/repair counts, slowest nodes, token/cost totals (cost `null` where pricing unknown), plus an **artifact-status table**. Diagnostics, not accuracy.

Reporting is **descriptive**: there is **no pre-registered "competitive" pass/fail threshold**. Any _descriptive_ reporting bar (e.g. the dev smoke-test sanity gate) is set after a baseline run; the open-vs-frontier result is characterised relative to the frontier control and the human-IRR ceiling using the agreement triad + bootstrap CIs, and the reader judges.

### Documented Limitations (record in the harness output and README)

- **mHSPC-28 is dev-only** (single tumour type, L/S only, non-standard rollup) — no published number rests on it.
- **Mined set is domain-level only** → per-SQ accuracy is demonstrated (ARBITER-Depth, n=15–25), not powered.
- **Mined-set ground truth carries between-review heterogeneity** (different teams; RoB 2 IRR only fair). ARBITER runs enriched; where reviewer inputs didn't match, it may **correctly diverge** (a partial-matching limitation). The mined set skews one-outcome-per-trial → per-outcome D2 divergence is validated on mHSPC-28/ARBITER-Depth, not the headline.
- **Grounding correctness rests on the small adjudicated set** (divergent-cell adjudication on ARBITER-Depth) — a demonstration, not a powered estimate.
- Overall agreement is reported **rollup-normalised and as-published**.
- **Human IRR is only fair** (κ≈0.40–0.45) → interpret relative to that ceiling; **Gwet's AC2** is the headline statistic given label imbalance (κ for comparability only).
- **High-path _algorithm_ correctness** is covered by the synthetic conformance test ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)/[REQ-22](#req-22-testing)); the human sets validate SQ-answering + tables, not the rarely-reached High paths.

### Acceptance Criteria

- Dev smoke-test (`run_eval.py`, mHSPC-28) prints per-domain + rollup-normalised overall agreement + confusion matrices, marked **dev-only**.
- Paper-eval mode runs on a **pinned paid provider/snapshot** over the REQ-01 roster through the same pipeline code as **whole-pipeline arms** (sq+aux per arm): headline on **Cochrane-mined** (per-domain + rollup-normalised overall; quote faithfulness as integrity property, not vs reviewer text); per-SQ + ablations on **ARBITER-Depth**; **descriptive** (no competitive threshold). Prints the agreement triad + bootstrap CIs, the open-vs-frontier table (cost/assessment + schema-repair rate), and the ablations.
- Per-SQ (ARBITER-Depth) follows the scoring conventions: collapsed classes, structural `NA` excluded, `effect=assignment`, branching-applicability separate, parsed-only + end-to-end, NI split.
- Grounding earns a correctness claim only via **blinded divergent-cell adjudication** ("grounding fixed X% of D5 errors"); the bare delta is feasibility until adjudicated.
- Mined set built **copyright-safe**: derived assessments + pointers (PMID/DOI/NCT) + extraction code only; trial PDFs git-ignored.
- Every report **stamps** models / dated snapshots / pinned `provider` / execution mode / dataset+arm — no number is quotable without its provenance.
- `--repeats K` enables run-to-run consistency reporting (off by default).
- The harness prints the limitations above.

---

## REQ-22: Testing

### Unit (`tests/unit/`) — every pure function

Priority: decision tables (exhaustive vs vendored RoB 2, incl. the D1.3-direction regression); rollup (all 5-domain combinations incl. **both** `requires_human_review` paths — sub-threshold multi-SC `2 ≤ #SC < OVERALL_HIGH_SC_THRESHOLD` and `#SC ≥ OVERALL_HIGH_SC_THRESHOLD` with no domain High — both boundaries read from the constant, with a test that bumps the constant and asserts the sub-threshold flag band moves); branching (all conditional triggers, both D2 effects); quote verifier (exact / fuzzy / OCR / mismatch) **+ `locate_quote_page`** (correct 0-based page; earlier page on a page-break straddle; `None` for NA/NI — [REQ-10](#req-10-quote-verifier)); confidence signals (all flag conditions; **`retrieval_top_score = None` fires neither score-based clause**; assert no `answer_consistency`); segmenter (doc-type scoring, domain tagging); SQ prompts (22 entries, required fields); **cost mapping** ([REQ-23](#req-23-run-trace--timing-instrumentation) — known price → numeric cost; missing price → `null` + `pricing_unknown`; free model → `0`; non-caching call → `cache_hit: null`); **report rendering** ([REQ-24](#req-24-reviewer-facing-markdown-report) — given a fixture `Assessment`, the Markdown contains the overall judgment, the Needs-attention list, the algorithm-rationale pattern lines, and the deterministic confidence flag — not a graded "evidence support" — with no numerical effect estimate).

### Integration (`tests/integration/`) — `MockLLMClient`, no network

1. Full trial on **Fixture A** (simple RCT, no supplements, NCT present) — overall judgment matches reference.
2. Full trial on **Fixture B** (SAP, multi-outcome, D5 concerns) — `outcome_change_detected = True`; D5 High where expected.
3. **Two-tier reuse:** a trial with 2 outcomes — **D1** judged once, identical across both outcome records; **D2–D5 judged per outcome** (and may differ); ingestion runs once.
4. **LLM failure on one SQ** — mock raises rate-limit 3× for one SQ; graph completes; that SQ is `NI`/`FLAGGED`; `errors` populated.
5. **Quote-verification failure** — mock returns an absent quote; `quote_verified = False`, `flag = FLAGGED`.
6. **D2 branching assignment** — `2.1 = Y`; gated SQ answered; adhering-only SQs `NA`.
7. **D2 branching adhering** — `effect = adhering`; assignment-only SQs `NA`.
8. **Batch idempotency** — run a 1-trial manifest twice; second run skips the completed pair; `--force` re-runs it.
9. **Non-native structured output** — a mock simulating a model that emits slightly malformed JSON; the validate-and-retry path recovers a valid `SQRawAnswer`.
10. **D3 is text-only (no vision in v0.1)** — assert no `complete_vision` call is made and `consort` stays `None` through a full trial; with D3 text flow numbers absent, the D3 SQ flags resolve via the standard REQ-11 rules (no D3-specific flag path). _(The three CONSORT-vision integration tests — detector-fires, safety-paths, default-off — are **v0.2**, alongside the vision impl.)_
11. **Trace side-channel invariant** ([REQ-23](#req-23-run-trace--timing-instrumentation)) — run a trial at `trace-level=full`; assert a per-trial `trace.json` with tier/outcome-tagged node spans and per-call records (tokens, latency, prefix referenced by hash); assert the same trial run twice yields **byte-identical** `data.json` **D1** blocks despite differing traces (this holds because `MockLLMClient` is deterministic — it exercises the deterministic _core_, not LLM reproducibility), and that the `Assessment`/SQLite row carry **no** trace fields.
12. **Trace levels** — `summary` writes timings/counts but no prompt bodies and no `artifacts/` dump; `full` additionally writes bodies + the intermediate `artifacts/` (segments, not the index); `off` writes no trace artifact and recording is a no-op. A non-caching mock route records `cache_hit: null`; a registry entry with no price records `cost: null` + `pricing_unknown`.
13. **Markdown report** ([REQ-24](#req-24-reviewer-facing-markdown-report)) — `report.md` is written next to `data.json` with the overall judgment, a Needs-attention list of flagged/uncertain SQs, the D1–D5 table, per-SQ tables with `justification` + the deterministic confidence flag, and the algorithm-rationale pattern lines; `--no-report` suppresses it; the writer makes no LLM/network call (no mock consulted).

### Fixtures

`tests/fixtures/fixture_a|b/`: open-access `paper.pdf` (+ `sap.pdf` for B), `expected_assessment.json` (human-reviewed), `mock_llm_responses.yaml` (per-SQ `SQRawAnswer`). No dedicated Fixture C. _(v0.2 extends Fixture B with a CONSORT participant-flow figure + a `mock_vision_responses.yaml`, and adds the synthetic-`PageBox` CONSORT detector unit test — none of which exist in v0.1.)_

### Acceptance Criteria

- `pytest tests/unit/` and `pytest tests/integration/` are fully green.
- No test makes a real LLM or network call.
- Suite completes in under 60 seconds.

---

## REQ-23: Run Trace & Timing Instrumentation

**Module:** `arbiter/observability/` (`trace.py`, `cost.py`)

### Why this exists

ARBITER is in active development, and the team needs to **see every input, output, and intermediate state flowing through the pipeline** — to inspect what's working, optimise, and debug. This REQ delivers two granularities from one machine: a **timing/cost summary** (always cheap) and a **full trace** (prompts, responses, repair attempts, and the deterministic intermediate data structures). It subsumes the originally-separate "full trace JSON" and "timing/cost instrumentation" ideas — timing/cost is just the summary view of the trace.

### Side-channel — see invariant 3 (§2)

This REQ implements invariant 3: trace/timing/cost data is collected via a `RunTrace` runtime handle into its own artifact and never enters the deterministic core, the LangGraph state, or the `Assessment` record. The rest of this section is the _how_.

### What to Build

> **Type shells ship in step 1, behaviour here.** The `RunTrace` / `CallRecord` / `NodeSpan` **model definitions** live in `models.py` from step 1 ([Implementation Order](#implementation-order)), because the LLM clients (step 7) and the graph builder (step 9) wire their optional `trace` seams against those types. This REQ fills in the **collection, prefix registry, node-span wrapping, and `cost.py`** behind them — no rework to steps 7/9, which already pass the handle (or `None`, a no-op).

A `RunTrace` collector, created **once per trial in `ingest_trial`** (spanning all its outcomes, carried on the `TrialContext`, and flushed by `assess_trial` — [REQ-19](#req-19-python-api)) and passed as a runtime handle:

- **`NodeSpan`** — one per graph-node invocation: `tier` (`"trial"`/`"outcome"`), node name, `outcome` (outcome tier only), wall-clock duration, error (if raised). Opened by the **build-time node wrapper** ([REQ-12](#req-12-assessment-orchestration-two-tier-graph)) via a `contextvars` span context, so it propagates across parallel `Send` fan-out. Node bodies are not edited.
- **`CallRecord`** — one per LLM call, recorded by the client ([REQ-06](#req-06-llm-abstraction-layer)) into the active span: model, `call_label` (call identity — SQ id+effect / metadata / annotation), `{prefix_hash, dynamic_suffix}` (prefix-deduped), the **full repair ladder** (every attempt's raw response + the parse/validation error fed back, and which attempt validated or that it raised), network-retry attempts + transient errors, in/out tokens, cache read/write tokens, latency.
- **Prefix registry** — the cacheable static prefix (`shared_prefix_text` + `ct_gov_block`, trial-static by construction — [REQ-14](#req-14-context-assembly)) is stored **once per trial** keyed by its hash; per-call records reference `prefix_hash`. Because the prefix is byte-identical across every SQ call in the trial, the per-trial hash is genuinely stable; this mirrors the caching architecture and avoids re-logging the ~4k-token prefix on every SQ call (which would balloon a trial's trace into megabytes of duplicated copyrighted text).

### Trace levels (one knob: `--trace-level` / `ARBITER_TRACE_LEVEL`)

| Level                           | Node spans + timings | Token/cache/repair **counts** | Prompt/response **bodies** + repair ladder | Intermediate-artifact dump |
| ------------------------------- | -------------------- | ----------------------------- | ------------------------------------------ | -------------------------- |
| `off`                           | —                    | —                             | —                                          | —                          |
| `summary` (**batch default**)   | ✓                    | ✓                             | —                                          | —                          |
| `full` (**assess/dev default**) | ✓                    | ✓                             | ✓                                          | ✓                          |

`summary` is the always-on, safe, small level (the "timing/cost instrumentation" view). `full` is the deep-debug level (the "full trace" view).

### Intermediate-artifact dump (at `full` only)

The pipeline's deterministic intermediate **data structures** are written for inspection — **dumped, never reused** (no caching, no hashing, no staleness; see [§4](#4-repository-structure)). Layout under the per-trial output dir ([REQ-16](#req-16-output-json-and-sqlite-writers)):

```
{output_dir}/{trial_id}/
  trace.json                              # the RunTrace, flushed
  artifacts/                              # trial-tier intermediates
    section_map.json  supplement_segments.json  ctgov.json
    trial_metadata.json                     # consort.json: v0.2 (no CONSORT in v0.1)
  {outcome_slug}__{effect}/artifacts/     # outcome-tier intermediates
    domain_contexts.json  sq_answers.json
```

- **Dump segments, not the index.** The `SupplementIndex` is **never serialised** ([REQ-03](#req-03-supplementary-material-ingestor)); we serialise the `list[SupplementSegment]` (pure data). Likewise no LLM client / retrieval-index handle is dumped.
- **The shared prefix is dumped once, not per domain.** Because `shared_prefix_text` now lives on the state (not on each `DomainContext`, [§5.6](#56-langgraph-state)), `domain_contexts.json` carries only each domain's suffix/retrieval signals; the ~4k-token prefix is written once (alongside `trial_metadata.json`), mirroring the per-call `prefix_hash` dedup and avoiding 5× duplication of copyrighted text.
- These are **debug artifacts** — git-ignored, disposable, and may contain full paper text (copyright); co-located for convenience, not durable output.

### Output artifacts

- **`trace.json`** — one per **trial** (not per trial-outcome), with every span tagged `tier` / `outcome` so cost and latency can be sliced per outcome without separate files. At `summary` it carries spans + counts; at `full` it additionally carries the prompt/response bodies and repair ladders.
- A **`timing_summary`** view (embedded at the top of `trace.json` or a sibling) holding: total wall time, total LLM latency, **estimated non-LLM time**, LLM call/cache/repair counts, slowest nodes, and per-node latency (calls / total / mean / max / errors). This is the shape the batch summary and eval harness read.

### Cost & the `null`-not-`0` discipline

Cost is computed in `cost.py` from `MODEL_REGISTRY` prices × token counts ([REQ-01](#req-01-dependency-and-project-setup)). **Everywhere a value is genuinely unavailable, record `null` ("unknown"), never `0`:** a non-caching route reports `cache_hit: null` (not `false`); a model absent from the price table reports `cost: null` + `pricing_unknown` (not `$0.00`); only a genuinely free model reports `0`. This protects the model/config comparisons the instrumentation exists to enable from being silently corrupted by a `0` standing in for "not measured."

### Acceptance Criteria

- A run at `--trace-level full` produces a per-trial `trace.json` containing node spans (tier/outcome-tagged), per-call records with token counts and latency, the prefix registered once and referenced by hash, and — for a forced schema-repair — the full repair ladder.
- The **`Assessment` record, SQLite row, and LangGraph state contain no trace/timing/cost fields** (side-channel invariant); the same trial assessed twice yields byte-identical `data.json` D1 blocks despite differing traces.
- At `--trace-level summary`, no prompt/response bodies and no `artifacts/` dump are written, but timings + counts are present; at `off`, no trace artifact at all and recording is a no-op.
- At `full`, the intermediate `artifacts/` are written (segments not index; no non-serialisable handles).
- A non-caching route records `cache_hit: null`; a model with no registry price records `cost: null` + `pricing_unknown`; a free model records `0`.
- The batch summary and `timing_summary` report total wall time, LLM latency, call/cache/repair counts, and slowest nodes.

---

## REQ-24: Reviewer-Facing Markdown Report

**Module:** `arbiter/output/report_writer.py`
**Signature:** `def write_assessment_report(assessment: Assessment, output_dir: Path, timing_summary: dict | None = None) -> Path`

### Why this exists

JSON and SQLite serve downstream tooling, but clinicians, reviewers, supervisors, and collaborators need a **readable artifact**. This renders the existing `Assessment` (plus an optional cost/timing footer) to Markdown. It is **pure presentation** — no LLM calls, no network, sub-millisecond — and reads **only data ARBITER already produces** (no new extraction).

### Output

One file per trial-outcome at `{output_dir}/{trial_id}/{outcome_slug}__{effect}/report.md` (alongside `data.json`, [REQ-16](#req-16-output-json-and-sqlite-writers)). The `{effect}` segment mirrors the JSON/DB key so `assignment` and `adhering` reports don't collide. **Default-on**; `--no-report` / `ARBITER_REPORT_ENABLED=false` suppresses.

### Structure (triage-oriented — surface the weak points first)

1. **Header** — trial id, NCT, outcome, effect (human-readable, e.g. "Effect of assignment to intervention (intention-to-treat)"), **overall judgment (prominent)**, models used, pipeline version, timestamp, and "sources consulted" (the domain sections/segments used). Rendered from `TrialMetadata` + `DomainContext` only. **No numerical effect estimate** (HR/CI) in v0.1 — that would need new, validated extraction and carries asymmetric wrong-number risk; logged as a future enhancement.
2. **⚠ Needs attention** — `requires_human_review` if set, plus every `FLAGGED`/`UNCERTAIN` SQ collected into one list with its `flag_reason`. A reviewer sees the soft spots before reading 22 questions.
3. **Domain summary table** — D1–D5 judgments at a glance + overall.
4. **Per-domain detail** — each domain's judgment, its **algorithm rationale rendered as the SQ-answer pattern → judgment mapping** (e.g. `1.1=Y/PY, 1.2=Y/PY/NI, 1.3=N/PN → Low`), then a per-SQ table: id · question · answer · **confidence flag** · quote (+page) · justification. Flagged rows visually marked.
5. **Outcome comparison** (registered vs published, [REQ-13](#req-13-pre-d5-outcome-comparison)) in the D5 section.
6. **Sources + parsing quality**, and — when a `timing_summary` is passed — a small **cost/timing footer** (this outcome's tier cost + the shared trial-tier cost labelled "counted once per trial").
7. A one-line **v0.1 advisory** footer ("first-pass draft; flags are advisory/uncalibrated", echoing [§3](#3-users-and-workflow)).

### Two rendering rules

- **Confidence column = the deterministic flag, not a graded "evidence support."** Render `ConfidenceSignals.flag` (`✓ Confident` / `⚠ Uncertain` / `🚩 Flagged`) with `flag_reason` as the "why". ARBITER does **not** produce a Strong/Moderate/Unsupported LLM-graded support level — introducing one would be new LLM work and would violate the "LLM only finds a quote and picks a code" constraint.
- **Label LLM-authored vs deterministic reasoning distinctly.** The per-SQ **Justification** column is the **LLM's** stated reason for the SQ answer (best-effort, advisory). The per-domain **Algorithm rationale** and the overall **Rationale** are **deterministic** ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)) and are what actually decided the judgments. The report must make this boundary legible (a header note and/or styling) — it is the "LLM answers SQs, the algorithm judges" principle made visible to a clinician.

### Acceptance Criteria

- `arbiter assess --paper <fixture_a>` writes a `report.md` next to `data.json` with: a prominent overall judgment, a Needs-attention section listing any flagged/uncertain SQs, a D1–D5 summary table, and per-domain SQ tables including `justification` and the deterministic confidence flag.
- The per-domain algorithm rationale renders the SQ-pattern → judgment mapping; the per-SQ Justification is labelled as the LLM's reasoning, distinct from the deterministic rationales.
- The confidence column renders `CONFIDENT/UNCERTAIN/FLAGGED` (no graded "evidence support"); no numerical effect estimate appears.
- `--no-report` suppresses the file; the writer makes no LLM or network call.
- When a `timing_summary` is supplied, the footer shows this-outcome vs shared trial-tier cost; otherwise the footer is omitted.

---

## Implementation Order

Build and test in this order; each step depends on the previous.

| Step | Build                                                                                                                                                                                                                                                                                                                                                                                                                                     | Why                                                                                                                                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | REQ-01 setup + §5 data models **incl. the `RunTrace` / `CallRecord` / `NodeSpan` model shells** (types only; collection/cost logic lands in step 11)                                                                                                                                                                                                                                                                                      | Everything imports from `config.py` / `models.py`; the trace **types** must exist now because steps 7 and 9 wire the optional `trace` seam against them                                                      |
| 2    | **Vendor `docs/rob2/`** (IRPG; fetch binaries, commit only README)                                                                                                                                                                                                                                                                                                                                                                        | Source of truth for steps 3–5                                                                                                                                                                                |
| 3    | REQ-07 decision tables + rollup                                                                                                                                                                                                                                                                                                                                                                                                           | Pure Python, easiest to test first; implement from vendored source                                                                                                                                           |
| 4    | REQ-08 branching                                                                                                                                                                                                                                                                                                                                                                                                                          | Needed by orchestration                                                                                                                                                                                      |
| 5    | REQ-09 SQ prompts + Appendix A reconciliation                                                                                                                                                                                                                                                                                                                                                                                             | Needed by SQ worker                                                                                                                                                                                          |
| 6    | REQ-10 quote verifier, REQ-11 confidence                                                                                                                                                                                                                                                                                                                                                                                                  | Needed by SQ worker                                                                                                                                                                                          |
| 7    | REQ-06 LLM layer (incl. hybrid structured output; `complete_vision` left as a stub). **Wire only the `trace: RunTrace \| None` seam** against the step-1 shells — `None` is a no-op; the collector behaviour lands in step 11                                                                                                                                                                                                             | Needed by all LLM callers; vision impl deferred to v0.2                                                                                                                                                      |
| 8    | REQ-02 paper, REQ-03 supplements, REQ-04 CT.gov, REQ-05 metadata                                                                                                                                                                                                                                                                                                                                                                          | Ingestion                                                                                                                                                                                                    |
| 9    | **Leaf nodes first, then orchestrate:** REQ-13 pre_d5 + REQ-14 context assembly + REQ-15 SQ worker (each mock-testable standalone), **then** REQ-12 two-tier graphs + `resolve_loop` + parallel domain branches. (D3 **text-only**; CONSORT vision node deferred to v0.2.) **Wire only the `trace` seam** in the builder against the step-1 shells                                                                                        | Core assessment — build and pin the leaf nodes with `MockLLMClient` before standing up the novel orchestration (cyclic `resolve_loop`, `Send` fan-out, parallel domain fan-in) over them                     |
| 10   | REQ-16 writers (with idempotency key; incl. the dedicated `write_skip_record`) + REQ-24 Markdown report. REQ-24's optional `timing_summary` footer is `\| None` until step 11 — build/test with `None`, the footer lights up at step 11/12 (no reorder)                                                                                                                                                                                   | Persist results; readable artifact (pure render over `Assessment`)                                                                                                                                           |
| 11   | REQ-23 trace/timing — **fill in** the step-1 `RunTrace`/`CallRecord`/`NodeSpan` shells with collection, the prefix registry, node-span wrapping, and `cost.py`                                                                                                                                                                                                                                                                            | Side-channel; the seams in REQ-06 clients (step 7) + REQ-12 builder (step 9) now light up — no rework, just the collector behind the handle                                                                  |
| 12   | REQ-19 Python API: **`ingest_trial` (Phase 1 → `TrialContext`) split from `assess_trial` (consumes the context, gate-free)**                                                                                                                                                                                                                                                                                                              | Split so the caller can run the eligibility gate between ingestion and assessment with no double-ingestion; wire trial+outcome tiers; create the per-trial `RunTrace`                                        |
| 13   | REQ-17 batch runner + manifest, REQ-18 CLI — **incl. the shared `check_eligibility` gate** wired between `ingest_trial` and `assess_trial` in both entry points (writes a `SkipRecord` via `write_skip_record` on a non-`parallel_rct`)                                                                                                                                                                                                   | Primary interface; the gate lands here with its callers; batch summary reads `timing_summary`                                                                                                                |
| 14   | REQ-20 error handling — **consolidate & test, not first-build.** Its three layers land earlier with the modules they live in: the layer-1 network-retry wrapper (`max_retries=0` clients) with the LLM client in **step 7**; the per-SQ `NI`/`FLAGGED` fallback with the SQ worker in **step 9**; batch continue-on-error with the runner in **step 13**. Step 14 reconciles them into the two-layer model and adds the acceptance tests. | Harden — consolidate the cross-cutting error paths built in steps 7/9/13; add the REQ-20 acceptance tests (auth aborts immediately; 3× rate-limit transparent; 3× failure → `NI`/`FLAGGED`, graph completes) |
| 15   | REQ-21 eval harness                                                                                                                                                                                                                                                                                                                                                                                                                       | Measure agreement                                                                                                                                                                                            |
| 16   | REQ-22 tests                                                                                                                                                                                                                                                                                                                                                                                                                              | Alongside each step; full suite at end                                                                                                                                                                       |
| —    | **CONSORT vision vertical — DEFERRED TO v0.2** (detector, `consort_extract` node, `consort_vision.py` prompt, `ChatOpenRouter.complete_vision` impl, mock-vision fixtures + the three CONSORT integration tests)                                                                                                                                                                                                                          | Not in v0.1. The seam (inventory in [§3](#3-users-and-workflow)) ships in steps 1/7, so the v0.2 vertical plugs in without rework.                                                                           |

---

## Acceptance Checklist

Each item is a checkable gate; detail lives in the referenced REQ.

- [ ] `uv sync --extra anthropic` / `--extra openai` / `--extra openrouter` all succeed; `arbiter --help` works. (REQ-01)
- [ ] `docs/rob2/` README committed (version `IRPG beta v9`, date, URLs, licence); binaries git-ignored, fetched locally. (REQ-07)
- [ ] Decision tables reproduce vendored RoB 2 exactly; D1.3-direction regression test passes. (REQ-07)
- [ ] `arbiter assess --paper <fixture_a>` → valid JSON, 22 SQ answers (non-empty quotes except NI/NA), D1–D5. (REQ-16)
- [ ] Deterministic core is bit-reproducible (fixed SQ answers → fixed judgment); idempotency rests on deterministic `trial_id` + DB key, **not** LLM determinism. (invariants 1 & 4, §2)
- [ ] Multi-outcome run: D1 identical across outcomes; D2–D5 per-outcome; ingestion once. (REQ-12/19)
- [ ] `arbiter batch` runs unattended; re-run **resumes** (skips completed pairs); `--force` re-runs. (REQ-17)
- [ ] Corrupt manifest entry → error record, batch continues. (REQ-17)
- [ ] Supplements as a directory ingest without categorisation; NCT derived from paper when omitted. (REQ-03/05)
- [ ] Supplements parsed in bounded windows with per-window degraded fallback; one bad page ≠ lost document. (REQ-03)
- [ ] Eligibility gate is a shared precondition applied by **both** `batch` and `assess`: non-`parallel_rct` (incl. `unclear`) → `SkipRecord` (one trial row, sentinel `outcome`, nullable judgment columns, `requires_human_review=True`, no fabricated D1–D5), no new LLM call. (REQ-05/17/18)
- [ ] Structured output validated for native-schema and gpt-oss/OpenRouter (repair-ladder) paths. (REQ-06)
- [ ] `quote_verified=False`→`FLAGGED`; `NI`+supplements-available (`segments_available>0`)+weak best passage (`retrieval_top_score<threshold`)→`FLAGGED`; no `answer_consistency` field. (REQ-11)
- [ ] D2 keyed by `(sq_id, effect)`; adhering's only `NA` is `2.7`; `2.3–2.6` re-worded not NA'd; D4 `4.3` gated on `N/PN/NI to 4.1 and 4.2`. (REQ-08/09)
- [ ] D3 is **text-only** in v0.1 (no vision call; `consort` stays `None`); CONSORT vision deferred to v0.2, seam-only (inventory in [§3](#3-users-and-workflow)). (REQ-14/§3)
- [ ] Pre-D5 matches assessed outcome against full registered set (primaries ∪ secondaries); registered secondary not flagged. (REQ-13)
- [ ] SQLite table created on first run with unique `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` over deterministic `trial_id` (never `uuid4()`); non-keyed `inputs_hash` column present. (REQ-16/17)
- [ ] Eval forks: dev smoke-test (mHSPC-28, no published number) + paper eval (headline on Cochrane-mined, per-SQ + ablations on ARBITER-Depth). (REQ-21)
- [ ] Headline metrics on mined set: per-domain + rollup-normalised overall agreement; quote faithfulness as integrity property (not vs reviewer quote); per-SQ on ARBITER-Depth only (collapsed classes, `NA` excluded, `effect=assignment`, parsed-only + end-to-end, NI split). (REQ-21)
- [ ] Execution mode explicit: dev=free tier; headline=pinned paid provider/snapshot (incl. paid `gpt-oss-120b`); free-tier = labelled secondary footnote. (REQ-21)
- [ ] Primary set mined from published RoB 2 reviews (traceable; copyright-safe derived-only release); ARBITER-Depth carries per-SQ + the blinded divergent-cell grounding adjudication. (REQ-21)
- [ ] Agreement reported as triad (% / Gwet AC2 / Cohen κ) with bootstrap CIs, vs human-IRR ceiling (κ≈0.40). (REQ-21)
- [ ] Open roster vs frontier control as **whole-pipeline arms** (each arm's model is both sq + aux; the harness assigns each arm a distinct **auto-derived `pipeline_version`** by hashing non-keyed config dims, so no two arms collide on the resume key), same pipeline code, with cost/assessment + schema-repair rate; `--repeats K` opt-in. (REQ-21)
- [ ] Ablations run: retrieval recall@k (BM25/dense/hybrid) on ARBITER-Depth passages; D5 registry-stratified; enriched-vs-main-text (adjudicated); NI-vs-human. (CONSORT A/B is v0.2.) (REQ-21)
- [ ] Every report stamps models / dated snapshots / pinned `provider` / dataset+arm; limitations printed. (REQ-21)
- [ ] Retrieval is local hybrid (BM25 + BGE-M3, RRF) **by default** (not ablation-gated); BM25 the deterministic baseline arm; recall@k reported as per-domain characterisation; reranker parked. (REQ-03)
- [ ] PY/PN-bridge wording present; NA cannot be LLM-emitted (deterministic branching); NI rate measured vs human. (REQ-09/11)
- [ ] `pytest tests/unit/` + `tests/integration/` green; no real API calls; <60 s. (REQ-22)
- [ ] PyMuPDF AGPL licence documented in README (internal use; revisit before redistribution).
- [ ] Run trace side-channel (invariant 3): `full`→per-trial `trace.json` + `artifacts/`; `summary`→timings/counts; `off`→nothing; `Assessment`/SQLite/state carry no trace fields; cost `null`-not-`0`, `cache_hit: null` on non-caching routes. (REQ-23)
- [ ] `report.md` default-on (next to `data.json`; `--no-report` suppresses): overall judgment + Needs-attention + D1–D5/per-SQ tables + deterministic confidence flag (no graded support, no effect estimate); no LLM/network call. (REQ-24)
- [ ] Vendored binaries git-ignored with committed fetch README; rollup policy in an ADR (`OVERALL_HIGH_SC_THRESHOLD`=3 + `requires_human_review` on both policy paths). (REQ-07)

---

## Appendix A: The 22 Signaling Questions

This appendix is the **LLM-facing reference** for the SQ prompt templates ([REQ-09](#req-09-signaling-question-prompt-templates)). The **authoritative wording and the answer-to-judgment logic are the vendored official RoB 2 tool** in `docs/rob2/`; the `question_text` and `answer_definitions` here must be **reconciled verbatim against that source** at implementation time. SQ IDs and their meanings must be used consistently across the algorithm ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)), branching ([REQ-08](#req-08-conditional-branching)), and these templates.

> **D2 has two variants.** For a given assessment only one effect's SQ set applies; the other effect's unique SQs are `NA`. Do not mix the assignment and adhering SQ meanings (a prior version of this document did — that is the bug this appendix exists to prevent).

### Domain 1 — Randomisation process (trial-level; SQs 1.1–1.3)

| SQ  | Question (verbatim per `docs/rob2/`)                                                                   | `effect` |
| --- | ------------------------------------------------------------------------------------------------------ | -------- |
| 1.1 | Was the allocation sequence random?                                                                    | both     |
| 1.2 | Was the allocation sequence concealed until participants were enrolled and assigned to interventions?  | both     |
| 1.3 | Did baseline differences between intervention groups suggest a problem with the randomisation process? | both     |

> **Direction note (1.3):** `Y/PY` (differences suggest a problem) → toward **High**; `N/PN` (no problem) → required for **Low**.

`key_terms` — 1.1: random, randomis/z, sequence generation, computer generated, random number, minimisation, stratified. 1.2: concealment, allocation concealment, sealed/opaque envelope, central pharmacy, IWRS, telephone/central randomisation, sequentially numbered. 1.3: baseline characteristics, table 1, demographics, imbalance, covariate.

### Domain 2 — Deviations from intended interventions (outcome-level; SQs 2.1–2.7)

(Inputs such as blinding are trial-wide, but the judgment is outcome-scoped — see [§1](#1-domain-background-rob-2) and [§2](#2-system-overview).)

Shared triggers (both effects):

| SQ  | Question                                                                                                           | `effect` |
| --- | ------------------------------------------------------------------------------------------------------------------ | -------- |
| 2.1 | Were participants aware of their assigned intervention during the trial?                                           | both     |
| 2.2 | Were carers and people delivering the interventions aware of participants' assigned intervention during the trial? | both     |

**Effect = assignment (ITT):**

| SQ  | Question                                                                                                                                                            | `effect`   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 2.3 | [If Y/PY/NI to 2.1 or 2.2] Were there deviations from the intended intervention that arose because of the experimental context?                                     | assignment |
| 2.4 | [If Y/PY to 2.3] Were these deviations likely to have affected the outcome?                                                                                         | assignment |
| 2.5 | [If Y/PY/NI to 2.4] Were these deviations from intended intervention balanced between groups?                                                                       | assignment |
| 2.6 | Was an appropriate analysis used to estimate the effect of assignment to intervention?                                                                              | assignment |
| 2.7 | [If N/PN/NI to 2.6] Was there potential for a substantial impact (on the result) of the failure to analyse participants in the group to which they were randomised? | both¹      |

> ¹ `effect` is the **wording-lookup discriminator** ([REQ-09](#req-09-signaling-question-prompt-templates)), not the scope. 2.7 has no per-effect wording variant, so it is keyed `effect="both"` — but it is **assignment-only in scope** and branching ([REQ-08](#req-08-conditional-branching)) NAs it under adhering. Scope lives in branching, never in the template.

**Effect = adhering (per-protocol):** SQs **2.3–2.6 are different questions** under the same IDs — they are _not_ the assignment questions, and they are _not_ `NA`. Only **2.7 is assignment-only** (→ `NA` under adhering). Verbatim from the vendored tool (`Print_format (PP)` / `Function Tab` col H):

| SQ  | Question                                                                                                                                   | `effect` |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| 2.3 | [If Y/PY/NI to 2.1 or 2.2] Were important non-protocol interventions balanced across intervention groups?                                  | adhering |
| 2.4 | [If applicable] Were there failures in implementing the intervention that could have affected the outcome?                                 | adhering |
| 2.5 | [If applicable] Was there non-adherence to the assigned intervention regimen that could have affected participants' outcomes?              | adhering |
| 2.6 | [If N/PN/NI to 2.3, **or** Y/PY/NI to 2.4 or 2.5] Was an appropriate analysis used to estimate the effect of adhering to the intervention? | adhering |

> **Shared-ID warning.** `2.3`–`2.6` carry **distinct `question_text`/`answer_definitions` per effect**, so `SQ_PROMPTS` must key these by `(sq_id, effect)`, not `sq_id` alone (see [REQ-09](#req-09-signaling-question-prompt-templates)). For a given assessment exactly one effect's wording applies; **`2.7` is the only genuinely effect-exclusive ID** (assignment-only → `NA` under adhering). The adhering branch has **no 2.7**, and its **2.6 uses a compound gate** (`N/PN/NI to 2.3` **or** `Y/PY/NI to 2.4/2.5`), not a single chain. Adhering **2.4 and 2.5 carry the literal "[If applicable]" tag with no logical gate** (`Function Tab` H4/H5), so ARBITER asks them **whenever the adhering effect is in scope** (i.e. always — there is no machine-evaluable predicate to gate on); and `NA` never satisfies the 2.6 compound gate (compound-gate NA rule, [REQ-08](#req-08-conditional-branching)).

`key_terms` — blinding/masking/open-label/placebo (2.1, 2.2); deviation, protocol deviation, co-intervention, concomitant, crossover, discontinued (2.3); adherence, compliance, per-protocol, dose received, fidelity (adhering 2.x); intention-to-treat, ITT, modified ITT, full analysis set, per-protocol, instrumental variable (2.6).

### Domain 3 — Missing outcome data (outcome-level; SQs 3.1–3.4)

| SQ  | Question                                                                                      | `effect` |
| --- | --------------------------------------------------------------------------------------------- | -------- |
| 3.1 | Were data for this outcome available for all, or nearly all, randomised participants?         | both     |
| 3.2 | [If N/PN/NI to 3.1] Is there evidence that the result was not biased by missing outcome data? | both     |
| 3.3 | [If N/PN to 3.2] Could missingness in the outcome depend on its true value?                   | both     |
| 3.4 | [If Y/PY/NI to 3.3] Is it likely that missingness in the outcome depended on its true value?  | both     |

`key_terms` — missing data, lost to follow-up, dropout, withdrawal, analysed, completeness; sensitivity analysis, imputation, tipping point, MAR/MNAR/MCAR; informative censoring, reason for withdrawal, death, disease progression.

### Domain 4 — Measurement of the outcome (outcome-level; SQs 4.1–4.5)

| SQ  | Question                                                                                                                  | `effect` |
| --- | ------------------------------------------------------------------------------------------------------------------------- | -------- |
| 4.1 | Was the method of measuring the outcome inappropriate?                                                                    | both     |
| 4.2 | Could measurement or ascertainment of the outcome have differed between intervention groups?                              | both     |
| 4.3 | [If N/PN/NI to 4.1 **and** 4.2] Were outcome assessors aware of the intervention received by study participants?          | both     |
| 4.4 | [If Y/PY/NI to 4.3] Could assessment of the outcome have been influenced by knowledge of the intervention received?       | both     |
| 4.5 | [If Y/PY/NI to 4.4] Is it likely that assessment of the outcome was influenced by knowledge of the intervention received? | both     |

> **Meaning note:** 4.1 is about the **measurement method**, 4.3 about **assessor awareness**. Branching/judgment must use these meanings (an earlier draft conflated 4.1 with blinding — do not).

`key_terms` — outcome measure, validated instrument, scale, endpoint definition (4.1); assessment schedule, frequency, differed between groups (4.2); outcome assessor, blinded/masked assessor, adjudication/endpoint committee, central review (4.3); subjective vs objective, patient-reported, clinician-assessed, hard endpoint, mortality (4.4, 4.5).

### Domain 5 — Selection of the reported result (outcome-level; SQs 5.1–5.3)

| SQ  | Question                                                                                                                                                           | `effect` |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| 5.1 | Were the data that produced this result analysed in accordance with a pre-specified analysis plan that was finalised before unblinded outcome data were available? | both     |
| 5.2 | [Is the result] selected from multiple eligible outcome **measurements** on the basis of the results?                                                              | both     |
| 5.3 | [Is the result] selected from multiple eligible **analyses** of the data on the basis of the results?                                                              | both     |

> **Direction note:** for 5.2/5.3, evidence of selection-on-results pushes toward **High**. Reconcile exact wording/polarity with `docs/rob2/`. ARBITER also injects the deterministic registered-vs-published outcome comparison ([REQ-13](#req-13-pre-d5-outcome-comparison)) into the D5 context.

`key_terms` — pre-specified, pre-registered, protocol, SAP, statistical analysis plan, registry, ClinicalTrials.gov (5.1); outcome switching, selective reporting, multiple measurements/scales/timepoints (5.2); post hoc, unplanned analysis, multiplicity, subgroup, multiple analyses (5.3).

---

_End of PRD._

_References:_

- _Sterne JAC et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ 2019;366:l4898._
- _Higgins JPT et al. Revised Cochrane risk-of-bias tool for randomized trials (RoB 2) — guidance document, 22 August 2019. (Vendored in `docs/rob2/`.)_
- _Official RoB 2 Excel tool, riskofbias.info. (Vendored in `docs/rob2/`; version + retrieval date in `docs/rob2/README.md`.)_
- _Huang J et al. Large Language Model–Assisted Risk-of-Bias Assessment in RCTs Using RoB 2. JMIR 2025;27:e70450. (SQ-decomposition + algorithm rollup beat direct judgment; SQ accuracy 83.2%; D2 worst; NA over-selection; reference set of 46 RCTs.)_
- _Hultin S et al. ROBoto2: An Interactive System and Dataset for LLM-assisted Clinical Trial Risk of Bias Assessment. EMNLP 2025 (System Demonstrations); arXiv:2511.03048. (**Related work — NOT used as our eval data:** the released set exposes no trial identifiers, so trials and the registry/supplement inputs ARBITER needs cannot be located; single-annotator main set, dual-annotated only on a 20-trial reliability subset at κ≈0.40. Cited for the "No Information" over-selection finding and the human-IRR ceiling.)_
- _Dhrangadhariya A et al. RoBuster-Corpus Annotated With Risk of Bias Text Spans in RCTs in Physiotherapy and Rehabilitation. JMIR Form Res 2026;10:e55127. doi:10.2196/55127. (**Related work — NOT used as our eval data** for the same traceability reason; grounding is instead operationalised as verbatim quote faithfulness against the trial full text.)_
- _Eisele-Metzger A et al. Exploring the potential of Claude 2 for risk of bias assessment. Res Synth Methods 2025;16(3):491-508. doi:10.1017/rsm.2025.12. (Direct-judgment baseline: overall agreement 41%, κ=0.22, D5 κ=0.10 — the control condition the SQ-decomposition architecture must beat.)_
- _Nagao T, Kawakita T. Assessing the Reliability of LLMs for Evaluation of Risk of Bias in RCTs. Am J Perinatol 2026. doi:10.1055/a-2793-9092. (GPT-5/4o/Claude on 180 RCTs; D5 worst at 47-51% "likely due to inability to cross-reference external trial registries" — motivates REQ-13.)_
- _Related (NOT a RoB 2 benchmark — distinct 6-item instrument; do not conflate): ROBUST-RCT feasibility study, Sci Rep 2026 (preprint medRxiv 2025.08.12.25333520)._
- _Retrieval evidence for the hybrid choice: BGE-M3 (Chen et al., multi-functionality embedding); hybrid BM25+dense+RRF improves recall +15–30% across 2026 RAG benchmarks (e.g. arXiv:2604.01733)._
