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
28. [Implementation Order](#implementation-order)
29. [Acceptance Checklist](#acceptance-checklist)
30. [Appendix A: The 22 Signaling Questions](#appendix-a-the-22-signaling-questions)

---

## 1. Domain Background: RoB 2

### What RoB 2 Is

Cochrane Risk of Bias 2 (RoB 2) is the standard methodology for assessing whether a randomised controlled trial (RCT) is at risk of producing a biased result. It is applied **per outcome**: the same trial assessed for two different outcomes (e.g., Overall Survival and Progression-Free Survival) can receive different RoB 2 scores.

### The Five Domains

| Domain | What it assesses              | Key question                                                              | Scope             |
| ------ | ----------------------------- | ------------------------------------------------------------------------- | ----------------- |
| D1     | Randomisation process         | Was the sequence truly random and properly concealed?                     | **Trial-level**   |
| D2     | Deviations from interventions | Were participants/clinicians unblinded, or did protocol deviations occur? | **Trial-level**   |
| D3     | Missing outcome data          | Were there dropouts or exclusions that could bias _this outcome_?         | **Outcome-level** |
| D4     | Outcome measurement           | Was _this outcome_ measured without knowledge of allocation?              | **Outcome-level** |
| D5     | Selection of reported result  | Does _this reported result_ match what was pre-registered?                | **Outcome-level** |

The trial-level / outcome-level split is a core architectural fact for ARBITER — see [§2](#2-system-overview) and [REQ-12](#req-12-assessment-orchestration-two-tier-graph).

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

### The Two-Tier Structure

Because D1 and D2 are properties of the _trial_ and D3/D4/D5 are properties of the _outcome_, ARBITER assesses each trial in two tiers:

1. **Trial tier (once per trial):** ingest the paper + supplements, fetch CT.gov, extract metadata, then assess **D1 and D2**. These results are computed a single time.
2. **Outcome tier (once per requested outcome):** for each outcome, assess **D3, D4, D5**, then roll up the overall judgment using the _shared_ D1/D2 plus this outcome's D3/D4/D5.

This guarantees D1/D2 are identical across a trial's outcomes (as real RoB 2 requires) and avoids re-spending LLM calls on outcome-invariant domains.

> **Documented assumption:** Strictly, RoB 2 is applied per outcome and D1/D2 _could_ in principle be reconsidered per outcome. In practice D1 (randomisation) and D2 (deviations) are outcome-invariant, and treating them as trial-level is a deliberate, documented simplification.

### Determinism

ARBITER has a **bit-reproducible deterministic core** and a **best-effort LLM layer** — be precise about which is which:

- **Deterministic core (bit-reproducible):** BM25 retrieval, context assembly, quote verification, conditional branching, the RoB 2 decision tables, and the overall rollup. Given a **fixed set of SQ answers**, ARBITER produces a **fixed judgment**. This is the reproducibility property that matters for auditability.
- **LLM SQ-answering (best-effort):** every LLM call runs at **temperature 0** with **no resampling** to _minimise_ variance, but temperature 0 is _most-likely-token_ sampling, **not** bit-identical output. It is **more** variable on OpenRouter multi-provider routing (different hardware/quantisation per route); in prod, pin a single `provider` to tighten reproducibility. For the eval, pin one provider so reported numbers are reproducible.

**Batch idempotency does not depend on LLM determinism.** It derives from the DB unique key `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` ([REQ-16](#req-16-output-json-and-sqlite-writers)/[REQ-17](#req-17-batch-runner-and-manifest)) — over a **deterministic** `trial_id` ([REQ-05](#req-05-trial-metadata-extractor)) — which skips already-completed pairs regardless of whether a re-run would reproduce the exact bytes. _(A prompt-hash response cache is a planned enhancement if harder replay is ever wanted.)_

### What the LLM Does and Does Not Do

The LLM has exactly one job: for each signaling question, **find and quote the specific sentence(s)** in the provided text that most directly address the question, and **select an answer code**.

The LLM does **not**:

- Make domain judgments or the overall judgment (deterministic algorithm)
- Decide which sections of the paper are relevant (context assembly is deterministic)
- Decide which supplement segments to retrieve (BM25 retrieval is deterministic)
- Run the RoB 2 decision algorithm

### The Context Engineering Principle

LLM accuracy on RoB 2 SQs degrades when the context is too broad. ARBITER counters this with:

1. **Per-SQ context assembly** — each SQ call receives only the paper sections and supplement passages relevant to its domain, never the full paper.
2. **Cached static prefix** — the static part of each prompt (trial metadata, Methods/Results, CT.gov data) is marked cacheable and shared across SQ calls. _(Caching is provider-dependent. Anthropic/OpenAI cache natively; **OpenRouter caches via `cache_control` breakpoints and routing — including for gpt-oss when routed to a caching-capable provider**, so the client must **forward** `cache_control`, not strip it, and prod should pin such a provider. Where a route genuinely can't cache, the prefix structure is retained at no benefit but no harm.)_
3. **One question per call** — each call answers exactly one SQ.

### Three Phases (per trial)

- **Phase 1 — Ingestion (synchronous):** parse the main paper and supplements, fetch CT.gov, extract metadata.
- **Phase 2 — Assessment:** trial-tier (D1, D2) then outcome-tier (D3, D4, D5 + rollup) per outcome.
- **Phase 3 — Output:** one JSON file and one SQLite row per trial-outcome pair.

---

## 3. Users and Workflow

### Primary user

A **systematic-review / evidence-synthesis team** running ARBITER as an **automated batch tool** over a set of trials (e.g., a tumour-type literature base). Output is consumed **directly** — there is no required human-in-the-loop step in v0.1. Confidence flags and `requires_human_review` are **advisory metadata** attached to each answer/assessment, not a workflow gate.

### Workflow

1. The team assembles a **manifest** (one row per trial: main paper, optional supplements/NCT/outcomes).
2. They run `arbiter batch <manifest>`.
3. ARBITER produces, per trial-outcome, a JSON record (with verbatim supporting quotes + page numbers + confidence flags) and a SQLite row.
4. Downstream tooling consumes the SQLite table (e.g., to build evidence maps or RoB summary figures). Flagged items _may_ be spot-checked but are not required to be.

### Honest limitation (v0.1)

Because output is consumed without a mandatory human review, accuracy matters — yet broad accuracy validation is **deferred** (see [REQ-21](#req-21-evaluation-harness)). **v0.1 is a "does the pipeline run end-to-end and roughly agree with reference judgments" milestone, not a "trust the numbers in production" milestone.** The confidence flags are advisory and uncalibrated until the evaluation set is expanded. Teams should treat v0.1 output as a first-pass draft.

**CONSORT vision in v0.1 (scope statement).** v0.1 ships CONSORT participant-flow vision extraction as a **live, on-by-default** capability that feeds D3 ([REQ-14](#req-14-context-assembly)), with a kill switch to disable it (`consort_vision_enabled` / `ARBITER_CONSORT_ENABLED` / `--no-consort`). It is a **measured-but-uncalibrated** D3 aid: its contribution is **reported as a vision-on vs text-only D3-agreement delta** against the reference D3 column ([REQ-21](#req-21-evaluation-harness)), **not assumed positive**. Because the asymmetric risk of a _wrong_ figure exceeds that of a _missed_ one, the detector gate is conservative and a low-confidence detection falls back to text-only + an `UNCERTAIN` flag. This is the one accuracy-affecting component v0.1 adds beyond the end-to-end baseline; it is included on the explicit understanding that the eval, not assertion, decides whether it earns its place.

---

## 4. Repository Structure

Implement the following layout. Module paths referenced throughout correspond to it.

```
arbiter/
├── arbiter/
│   ├── __init__.py               # Exports: assess_trial(), AssessmentConfig
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
│   │   └── bm25_index.py         # SupplementIndex class
│   ├── graph/
│   │   ├── state.py              # TrialState / OutcomeState (TypedDict) + reducers
│   │   ├── builder.py            # build_trial_graph(), build_outcome_graph()
│   │   └── nodes/                # context_assembly, consort_extract, sq_node, fanin, judgment, pre_d5, overall
│   ├── arbiter_algorithm/
│   │   ├── rob2_reference.py     # Thin wrapper over the vendored RoB 2 algorithm
│   │   ├── decision_tables.py    # judge_domain_1..5() — implement FROM docs/rob2/
│   │   ├── rollup.py             # compute_overall_judgment()
│   │   └── branching.py          # get_applicable_sqs(), get_na_sqs()
│   ├── llm/
│   │   ├── base.py               # LLMClient ABC + structured-output enforcement
│   │   ├── openai_compatible.py  # OpenAI-compatible client (OpenAI, OpenRouter)
│   │   ├── anthropic_client.py   # AnthropicClient
│   │   └── mock_client.py        # MockLLMClient for tests
│   ├── prompts/
│   │   ├── system.py             # build_system_prompt()
│   │   ├── metadata_extraction.py
│   │   ├── supplement_annotation.py
│   │   ├── consort_vision.py     # CONSORT-figure vision prompt → ConsortFlow
│   │   └── sq_prompts.py         # SQ_PROMPTS keyed by (sq_id, effect), SQPromptTemplate
│   ├── confidence/
│   │   ├── quote_verifier.py     # verify_quote() → bool
│   │   └── signals.py            # compute_confidence() → ConfidenceSignals
│   └── output/
│       ├── json_writer.py        # write_assessment_json()
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
│   ├── reference/                # overall_survival.csv, progression_free_survival.csv, adverse_events.csv
│   └── run_eval.py               # Domain + overall agreement harness (REQ-21)
├── tests/
│   ├── unit/  integration/  fixtures/
├── pyproject.toml
├── .env.example
└── README.md
```

> **In v0.1:** CONSORT-figure vision extraction is **part of v0.1**, **on by default** with a kill switch (`consort_vision_enabled` / `ARBITER_CONSORT_ENABLED` / `--no-consort`) — a deterministic detector gates a conservative vision call that fills `ConsortFlow`, run once per trial and threaded into D3 (see [REQ-06](#req-06-llm-abstraction-layer), [REQ-14](#req-14-context-assembly), and the §3 scope statement). Its contribution is **measured** as a vision-on vs text-only D3-agreement delta ([REQ-21](#req-21-evaluation-harness)), not assumed. Add `prompts/consort_vision.py` and a trial-tier `graph/nodes/consort_extract.py`.
>
> **Not in v0.1 (planned enhancements):** graph-level checkpoint/resume; hybrid (BM25 + embeddings) retrieval, eval-gated; a prompt-hash LLM response cache for harder replay. Interfaces are kept extension-ready.

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
    all_outcomes: list[str]        # primary first, then secondary; max 10
    effect_of_interest: EffectOfInterest
    blinding: BlindingStatus
    nct_number: str | None = None

class ConfidenceSignals(BaseModel):
    supplement_segments_retrieved: int
    supplement_segments_available: int
    bm25_top_score: float | None
    quote_verified: bool
    flag: ConfidenceFlag
    flag_reason: str | None = None

class SQAnswer(BaseModel):
    sq_id: str                     # "1.1", "2.3", "5.2" ...
    answer: AnswerCode
    quote: str                     # verbatim; empty for NA/NI
    page: int | None               # 0-based; None for NA/NI
    justification: str             # exactly one sentence
    confidence: ConfidenceSignals

class SQRawAnswer(BaseModel):
    """Schema the LLM SQ call must satisfy (post-validation)."""
    answer: AnswerCode
    quote: str = Field(max_length=400)
    page: int | None
    justification: str = Field(max_length=200)

class DomainJudgment(BaseModel):
    domain: str                    # "D1" ... "D5"
    scope: Literal["trial", "outcome"]
    judgment: Judgment
    algorithm_rationale: str
    sq_answers: list[SQAnswer]

class ConsortFlow(BaseModel):          # populated by the trial-tier CONSORT vision node (REQ-14)
    randomised: int | None = None
    allocated_intervention: int | None = None
    allocated_control: int | None = None
    lost_intervention: int | None = None
    lost_control: int | None = None
    analysed_intervention: int | None = None
    analysed_control: int | None = None
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

    trial_id: str
    nct_number: str | None
    outcome: str                   # THE outcome this record assesses

    requires_human_review: bool    # advisory
    config_summary: dict
    trial_metadata: TrialMetadata
    ct_gov_data: dict | None
    outcome_comparison: dict | None

    # D1 + D2 are trial-level (reused identically across this trial's outcomes);
    # D3 + D4 + D5 are specific to `outcome`.
    domain_judgments: list[DomainJudgment]   # exactly D1..D5
    overall_judgment: Judgment
    overall_rationale: str

    sources_manifest: SourcesManifest
    errors: list[str]
```

`domain_judgments` always contains five entries; the D1 and D2 entries carry `scope="trial"` and are byte-identical across a trial's outcome records by construction.

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

Two TypedDicts — one per tier. **No LLM client, BM25 index, or other non-serialisable object is stored as durable state**; they are passed as runtime handles only (no checkpointing in v0.1).

```python
class TrialState(TypedDict):           # trial tier: D1, D2
    config_summary: dict
    trial_metadata: TrialMetadata
    section_map: SectionMap
    raw_char_stream: str
    supplement_index: SupplementIndex      # runtime handle
    ct_gov_data: dict | None
    llm_client_sq: LLMClient               # runtime handle
    llm_client_aux: LLMClient              # runtime handle
    llm_client_vision: LLMClient           # runtime handle (CONSORT extraction)
    consort_flow: ConsortFlow | None       # extracted once per trial; None if vision didn't fire
    consort_detected: bool                 # detector fired vision (audit)
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]   # D1, D2
    errors: Annotated[list[str], operator.add]

class OutcomeState(TypedDict):         # outcome tier: D3, D4, D5 + rollup
    # all TrialState ingestion fields (incl. consort_flow / consort_detected), plus:
    outcome: str
    trial_domain_judgments: list[DomainJudgment]   # the reused D1, D2
    outcome_change_detected: bool | None
    registered_outcome: str | None
    published_outcome: str | None
    outcome_similarity_score: float | None
    registered_as_primary: bool | None
    domain_contexts: Annotated[dict[str, DomainContext], merge_dict]
    sq_answers: Annotated[dict[str, SQAnswer], merge_dict]
    domain_judgments: Annotated[list[DomainJudgment], operator.add]   # D3, D4, D5
    overall_judgment: Judgment | None
    overall_rationale: str | None
    requires_human_review: bool | None
    errors: Annotated[list[str], operator.add]

class DomainContext(BaseModel):
    domain: str
    main_paper_text: str
    ct_gov_block: str | None = None
    supplement_block: str | None = None
    outcome_comparison_block: str | None = None
    bm25_top_score: float | None = None
    segments_retrieved: int = 0
    segments_available: int = 0
```

---

## REQ-01: Dependency and Project Setup

### What to Build

A `pyproject.toml` declaring dependencies, optional provider extras, and the CLI entrypoint; and a `.env.example` with all environment variables.

### Dependencies (intent, not a lockfile)

- **Orchestration:** `langgraph`, `langchain-core` _(no `langgraph-checkpoint-sqlite` — v0.1 does not checkpoint)_
- **PDF parsing:** `pymupdf`, `pymupdf4llm`
- **Retrieval:** `bm25s`, `nltk`
- **Fuzzy matching:** `rapidfuzz`
- **Validation:** `pydantic >= 2`
- **HTTP:** `httpx`
- **CLI:** `click`
- **Env:** `python-dotenv`
- **LLM providers (optional extras):** `anthropic` (extra `anthropic`), `openai` (extra `openai`, also used for OpenRouter). `all` installs both.
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
    sq_model: str = "gpt-oss-120b"               # accuracy-critical SQ judgments (DEV default; see note)
    aux_model: str = "gpt-oss-120b"              # metadata + supplement annotation (DEV default)
    vision_model: str = "google/gemma-4-31b-it:free"  # CONSORT-figure extraction (DEV default)
    consort_vision_enabled: bool = True          # kill switch for CONSORT vision; text-only D3 when False
    output_dir: Path = Path("./output")
    db_path: Path = Path("./arbiter.db")
    force: bool = False                          # re-run even if already in DB
```

> **Model defaults are DEV defaults.** `gpt-oss-120b` and the free Gemma vision model are chosen because they are **free on OpenRouter**, for local development and CI. **Production and the real (TBD) eval use frontier models** (e.g. `claude-sonnet-4-6` / `gpt-4o` for `sq_model`). The weakest models being the _defaults_ is a dev convenience, **not** the intended production configuration — and v0.1 accuracy numbers should be reported on the prod models.

`MODEL_REGISTRY` maps model name → `{provider, base_url?, supports_cache, supports_native_schema, supports_vision}`. v0.1 entries:

| model                        | provider   | supports_cache     | supports_native_schema | supports_vision |
| ---------------------------- | ---------- | ------------------ | ---------------------- | --------------- |
| `gpt-oss-120b`               | openrouter | routing-dependent¹ | False                  | False           |
| `google/gemma-4-31b-it:free` | openrouter | routing-dependent¹ | False                  | **True**        |
| `claude-haiku-4-5`           | anthropic  | True               | True                   | True            |
| `claude-sonnet-4-6`          | anthropic  | True               | True                   | True            |
| `gpt-4o` / `gpt-4o-mini`     | openai     | True               | True                   | True            |

¹ OpenRouter caches via `cache_control` breakpoints **when routed to a caching-capable provider**; pin one via OpenRouter's `provider` routing arg in prod. The client **forwards** `cache_control` rather than stripping it (see [REQ-06](#req-06-llm-abstraction-layer)). The Gemma vision slug is confirmed; mind its free-tier rate limits under batch use.

`provider == "openrouter"` uses the OpenAI-compatible client with `base_url = https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`.

### .env.example (variables)

`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`; `ARBITER_SQ_MODEL`, `ARBITER_AUX_MODEL`, `ARBITER_VISION_MODEL`; `ARBITER_OUTPUT_DIR`, `ARBITER_DB_PATH`; `ARBITER_BM25_TOP_K` (default 5), `ARBITER_SMALL_SEGMENT_TOKEN_THRESHOLD` (1500); `ARBITER_QUOTE_VERIFY_THRESHOLD` (85); `ARBITER_BM25_UNCERTAIN_THRESHOLD` (3.0); `ARBITER_MAX_ANNOTATIONS_PER_DOC` (default 40, per-document annotation-call cap); `ARBITER_CONSORT_DETECT_THRESHOLD` (CONSORT-figure detection score gate; conservative default); `ARBITER_CONSORT_ENABLED` (default `true`; the CONSORT-vision kill switch — set `false` for text-only D3); `ARBITER_MAX_CONCURRENCY` (bounded batch concurrency for OpenRouter free-tier rate limits).

### Acceptance Criteria

- `pip install -e ".[anthropic]"` and `pip install -e ".[openai]"` complete without error.
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
- **Domain-tag** each section using `SECTION_KEYWORDS` (one keyword list per domain plus METHODS/RESULTS). A section is tagged for domain D if any of D's keywords appears in its normalised label or first 300 characters of body. Multiple tags allowed.
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

For each supplementary PDF, produce a `SupplementIndex` of domain-tagged, contextually annotated segments backed by an in-memory BM25S index. Supplements can be hundreds of pages; injecting them wholesale into every SQ call is too expensive and harms accuracy (lost-in-the-middle). Instead, segment → tag → annotate → index, then retrieve only relevant passages at query time.

### Input Handling

`paths` may contain **files or directories**. For any directory, glob all `*.pdf` within it (non-recursive is sufficient; document the choice). **The caller never categorises PDFs** — ARBITER infers each document's type itself (below).

### Pipeline (per document)

- **Parse** via the same layout extraction as `ingest_paper`.
- **Detect document type (rule-based, no LLM):** score the text of `section-header` boxes on the first 10 pages against three lexicons (`sap`, `protocol`, `appendix`); highest score wins; tie → `sap`; no headers → `unknown` (default SAP lexicon).
- **Segment** at each `section-header` boundary (heading, page range, body until next header). Domain-tag each segment with `SECTION_KEYWORDS` (heading + first 300 chars). If a document yields fewer than 3 segments, treat the whole document as one segment tagged for all domains.
- **Annotate (one aux LLM call per _domain-tagged_ segment — bounded):** annotation is an **enrichment, not a gate**. To avoid a cost/latency explosion on fat protocols (a 200-page SAP can segment into 100+ chunks), annotate **only segments carrying ≥1 `domain_tag`**, and cap the number of annotation calls per document at `ARBITER_MAX_ANNOTATIONS_PER_DOC` (default 40), prioritising segments by domain-tag count. Untagged or over-cap segments get an empty annotation and are **still indexed on raw text**. For each annotated segment: with a static document-preamble prefix (title page + first ~500 tokens) and a per-segment suffix, ask for 2–3 sentences naming the methods/populations/procedures relevant to randomisation, blinding, missing data, outcome assessment, or selective reporting. If a segment has no RoB-relevant content, the model returns exactly `"No risk-of-bias relevant content."`. Store as `annotation`.
- **Index:** build a `SupplementIndex` over each segment's `annotated_text` (annotation + raw text). Because **every** segment is indexed on raw text regardless of annotation, BM25 recall is unaffected by the annotation cap.

> **Retrieval is BM25-only in v0.1** (the curated `key_terms` are rare domain jargon — exactly where lexical IDF beats embeddings — and the annotation step already bridges vocabulary gaps in canonical RoB wording). **Hybrid retrieval** (local pinned embeddings + RRF fusion) is a planned enhancement, **gated on eval evidence** that a domain (likely D3/D5) answers `NI`/wrong because a real passage wasn't retrieved.

### SupplementIndex

`SupplementIndex.retrieve(query_terms: list[str], domain: str, top_k: int = 5) -> tuple[list[SupplementSegment], float]` returns the top-k segments and the best BM25 score. It filters to segments tagged with `domain` first; if fewer than 2 such segments exist, it falls back to the full set. The index is in-memory only; **never serialised to disk**.

### Acceptance Criteria

- Returns a `SupplementIndex` for any list of paths, including `[]` → empty index, and a directory of PDFs.
- Each segment has non-empty `annotation` and `raw_text` (`domain_tags` may be empty).
- `retrieve(["concealment","allocation"], "D1", top_k=5)` returns ≤5 segments.
- A document with no section headers yields one full-document segment.

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

Extract structured trial metadata from the main paper using one **aux** LLM call over the abstract + methods text (capped ~3,000 tokens; fall back to the first 3,000 tokens of `full_text` if those sections are absent). The model returns: `title`, `intervention`, `comparator`, `primary_outcome`, `all_outcomes` (primary first, ≤10), `blinding`, `nct_number`. It does **not** return an effect-of-interest hint: `effect_of_interest` comes from config (default `assignment`, `--effect` override), so an extracted hint could never win and is omitted.

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

An abstract `LLMClient` plus concrete clients for Anthropic and OpenAI-compatible providers (OpenAI **and** OpenRouter), and a mock for tests. The layer **guarantees a validated Pydantic instance** regardless of whether the provider natively enforces a schema.

### Base Class (`llm/base.py`)

```python
class LLMClient(ABC):
    def __init__(self, model: str): self.model = model

    @abstractmethod
    async def complete_structured(
        self, messages: list[dict], schema: type[BaseModel],
        temperature: float = 0.0, max_tokens: int = 512,
    ) -> BaseModel: ...

    @abstractmethod
    def supports_prompt_caching(self) -> bool: ...
    @abstractmethod
    def supports_native_schema(self) -> bool: ...
    @abstractmethod
    def supports_vision(self) -> bool: ...

    # v0.1: implemented on vision-capable clients (CONSORT extraction, REQ-14);
    # clients whose model has supports_vision == False still raise NotImplementedError.
    async def complete_vision(self, image_bytes: bytes, prompt: str,
                              schema: type[BaseModel]) -> BaseModel:
        raise NotImplementedError("This client's model does not support vision.")
```

`complete_vision` is **implemented in v0.1** on the OpenAI-compatible client for vision-capable OpenRouter models (e.g. the free Gemma vision model): the image is sent as a base64 **data-URL** content block, and the same hybrid-structured-output contract applies (return a validated `schema` instance — here `ConsortFlow`). It is used by the trial-tier CONSORT-extraction node ([REQ-14](#req-14-context-assembly)).

### Hybrid Structured Output (the critical contract)

`complete_structured` must return a validated `schema` instance, never a dict or string:

1. **If `supports_native_schema()`:** use native enforcement (Anthropic tool-use / OpenAI `response_format={"type":"json_schema",...}`).
2. **Else (e.g. gpt-oss via OpenRouter):** inject the JSON schema into the prompt, extract the first JSON object from the response, validate with Pydantic. On validation failure, **retry up to N times** (bounded; default 2) feeding the validation error back into the prompt. These **schema-validation retries are separate** from the network-error retries in [REQ-20](#req-20-error-handling-and-retry).
3. If still invalid after retries, raise `ValueError` with a descriptive message.

### Caching

Messages may contain content blocks with `"cache_control": {"type":"ephemeral"}`. The Anthropic client passes these through natively. **The OpenAI-compatible client must FORWARD `cache_control` blocks when `base_url` is OpenRouter** (OpenRouter honours breakpoint caching, including for gpt-oss when routed to a caching-capable provider) — do **not** strip them. For **vanilla OpenAI** `base_url`, strip `cache_control` (OpenAI caches automatically and rejects the unknown key). Where a route genuinely cannot cache, the prefix structure is preserved at no benefit and no harm.

### Factory

`create_llm_client(model: str) -> LLMClient` looks up `MODEL_REGISTRY`, constructs the right client (anthropic / openai / openrouter via `base_url`), and raises `ValueError` for unknown models.

### Acceptance Criteria

- `complete_structured` returns a validated model instance for **both** a native-schema provider and gpt-oss/OpenRouter (non-native path exercised by a test).
- Validation failures after bounded retries raise `ValueError`.
- `MockLLMClient` returns deterministic fixture responses (keyed by `(sq_id, effect)` for D2, else `sq_id`; vision responses keyed separately) with no network calls.
- `complete_vision` returns a validated `ConsortFlow` for a vision-capable model; the mock vision path is exercised by a test (see [REQ-22](#req-22-testing)). On a client whose model has `supports_vision == False`, it raises `NotImplementedError`.

---

## REQ-07: RoB 2 Algorithm — Deterministic Judgments

**Module:** `arbiter/arbiter_algorithm/decision_tables.py`, `rollup.py`, `rob2_reference.py`

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
  - The **overall rollup's "multiple domains" boundary is an ARBITER policy decision**, _not_ a verbatim rule. The official guidance defines overall **High** as "High in ≥1 domain **or** some concerns for _multiple domains in a way that substantially lowers confidence_" — and explicitly leaves "multiple domains / substantially lowers confidence" to **reviewer judgment**. Because ARBITER runs unattended, it **operationalises** that judgment deterministically using a named constant `OVERALL_HIGH_SC_THRESHOLD` (default **3**): **Low** if all five Low; **High** if any domain High **or** `≥ OVERALL_HIGH_SC_THRESHOLD` domains Some concerns; otherwise **Some concerns**. `requires_human_review = True` is set on **exactly the two policy-driven paths** — the cases where ARBITER's rule, not the verbatim tables, decided the outcome: (a) **exactly-2** Some-concerns → Some concerns (the SC/High boundary), and (b) `≥ OVERALL_HIGH_SC_THRESHOLD` Some-concerns with **no** domain High → High (High-by-accumulation — the "multiple domains substantially lower confidence" case the guidance leaves to reviewer judgment). The table-clean outcomes (all-five-Low → Low; any-domain-High → High) are **not** flagged, so `requires_human_review` is a precise "policy decided this" signal, not a one-sided boundary detector. Document this operationalisation — the `OVERALL_HIGH_SC_THRESHOLD` default and **both** review-flag paths — in an **ADR** (`docs/adr/`) so the policy is reviewable and the threshold is owned by ARBITER, not misattributed to Cochrane. The eval's rollup-normalisation ([REQ-21](#req-21-evaluation-harness)) reads the **same constant**.
  - SQ-direction still matters and must not invert (e.g. D1.3 _"baseline differences suggest a problem"_: `Y/PY` → toward **High**, `N/PN` required for **Low**). Every SQ ID used here means what it means in [Appendix A](#appendix-a-the-22-signaling-questions) and the prompt templates.
- **No function makes an LLM or network call.** All are pure (no side effects, no global state).

### Acceptance Criteria

- Each domain function is tested **exhaustively** against the vendored RoB 2 Excel logic; every reachable path is covered (a **synthetic conformance test** enumerating the SQ-answer combinations the Excel maps to each judgment — including the `High` paths the human eval set cannot reach).
- A regression test asserts D1.3 direction is **not** inverted (a known prior bug).
- `compute_overall_judgment` matches the **documented ARBITER rollup policy** (above) on all 5-domain combinations, including `requires_human_review = True` on **both** policy-driven paths (exactly-2 Some-concerns, **and** `≥ OVERALL_HIGH_SC_THRESHOLD` Some-concerns with no domain High) and `False` on the table-clean outcomes; the ADR is referenced in its docstring and the threshold is read from `OVERALL_HIGH_SC_THRESHOLD`, not hardcoded.

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
- **D2 (adhering effect)** — `2.1, 2.2` always; `2.3` gated on `Y/PY/NI to 2.1 or 2.2`; `2.4`, `2.5` asked when applicable; `2.6` gated on the **compound** condition `N/PN/NI to 2.3` **or** `Y/PY/NI to 2.4 or 2.5`. `2.3`–`2.6` carry their **adhering** wording. **`2.7` does not exist for adhering → it is the _only_ D2 SQ in `get_na_sqs` under this effect.** (`2.3`–`2.6` are answered under _both_ effects with different wording — they are **not** NA; see [Appendix A](#appendix-a-the-22-signaling-questions).)
- **D3** — `3.1` always; `3.2` gated on `N/PN/NI to 3.1`; `3.3` gated on `N/PN to 3.2`; `3.4` gated on `Y/PY/NI to 3.3`.
- **D4** — `4.1` **and** `4.2` always; `4.3` gated on `N/PN/NI to 4.1 **and** 4.2`; `4.4` gated on `Y/PY/NI to 4.3`; `4.5` gated on `Y/PY/NI to 4.4`. **Note the official meaning of each D4 SQ ID** (4.1 = appropriateness of measurement method; 4.3 = assessor awareness; etc.) — branching must use those meanings, not a relabelled set. (4.2 is **always asked**, not gated.)
- **D5** — `5.1, 5.2, 5.3` per the official gating.

> Because the exact conditional wording and gating are defined by the vendored tool, implement branching **from `docs/rob2/`** and keep the SQ-ID meanings consistent with Appendix A and the prompt templates.

### Two-Round Fan-Out

D2 and D4 have intra-domain conditionals, so their SQ workers run in **two rounds**: round 1 answers the trigger SQ(s); ARBITER then computes the applicable round-2 SQs from those answers and fans them out. SQs determined to be NA are recorded with `answer = NA`, empty quote, `page = None`.

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
    key_terms: list[str]     # BM25 query terms for supplement retrieval
    applies_to: list[str]    # ["assignment"] | ["adhering"] | ["both"]
```

**Keying.** `SQ_PROMPTS` is keyed by `(sq_id, effect)`, **not** `sq_id` alone. This is mandatory: for D2, IDs `2.3`–`2.6` are **different questions** under the assignment vs adhering effect (see [Appendix A](#appendix-a-the-22-signaling-questions)) and cannot share one entry. IDs that are effect-invariant (`1.x`, `2.1`, `2.2`, `2.7`, `3.x`, `4.x`, `5.x`) use `effect="both"`. The SQ worker looks up `(sq_id, config.effect_of_interest)` with a `"both"` fallback.

The authoritative `question_text` and `answer_definitions` wording is the vendored RoB 2 guidance (`docs/rob2/`); the full reference set is reproduced in [Appendix A](#appendix-a-the-22-signaling-questions) and **must be reconciled verbatim against the vendored source** at implementation time. Each SQ ID and its meaning must match the algorithm ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)) and branching ([REQ-08](#req-08-conditional-branching)).

### Acceptance Criteria

- `SQ_PROMPTS` covers the **22 SQ positions** (`1.1–1.3`, `2.1–2.7`, `3.1–3.4`, `4.1–4.5`, `5.1–5.3`) across **26 templates** — D2 `2.3`–`2.6` each appear twice (assignment + adhering), all other IDs once (`effect="both"`).
- Looking up `(sq_id="2.4", effect="adhering")` returns the **per-protocol** wording (failures in implementation), distinct from `(sq_id="2.4", effect="assignment")` (deviations affecting the outcome).
- Every entry has non-empty `question_text`, `answer_definitions`, and `key_terms`, with explicit criteria for each answer code applicable to that question.
- Every `(sq_id, effect)` referenced by the algorithm and branching modules exists in `SQ_PROMPTS` with a consistent meaning.

---

## REQ-10: Quote Verifier

**Module:** `arbiter/confidence/quote_verifier.py`
**Signature:** `def verify_quote(quote: str, raw_char_stream: str, threshold: int = 85) -> bool`

### What to Build

Fuzzy-match the LLM-returned quote against the raw PDF character stream (via `rapidfuzz`). Normalise both (collapse whitespace, lowercase), slide a window over the stream, and return whether the best `partial_ratio` ≥ `threshold`. Quotes shorter than 15 characters (typical of NA/NI answers) are trivially verified as `True`.

### Design Notes

- Threshold 85 absorbs OCR artefacts, hyphenation, and ligature differences.
- Must complete in well under a second for a ~50,000-character stream.

### Acceptance Criteria

- `True` for an exact quote and for quotes differing only in whitespace/case.
- `False` for a quote absent from the stream.
- `True` for empty or <15-char quotes.
- Completes in <500 ms for a 50,000-char stream.

---

## REQ-11: Confidence Signal System

**Module:** `arbiter/confidence/signals.py`
**Signature:** `def compute_confidence(answer, quote_verified, segments_retrieved, segments_available, bm25_top_score) -> ConfidenceSignals`

### What to Build

A **deterministic** flag derived from retrieval and verification signals. There is **no self-consistency / second-sample mechanism** — the pipeline is temperature-0 and single-sample, so resampling would be identical and carries no information.

### Flag Rules (priority order)

1. `answer == NA` → always `CONFIDENT` (set by the algorithm, not the LLM).
2. `FLAGGED` if: the quote could not be verified on a non-NI/NA answer; **or** the answer is `NI` while 0 segments were retrieved but domain-relevant segments were available.
3. `UNCERTAIN` if: `bm25_top_score` is below `ARBITER_BM25_UNCERTAIN_THRESHOLD` (default 3.0); **or** the answer is `NI` with no supplementary materials for the domain.
4. Otherwise `CONFIDENT`.

`flag_reason` is a human-readable string for any non-CONFIDENT flag. All flags are **advisory** (see [§3](#3-users-and-workflow)).

### Acceptance Criteria

- `NA` → `CONFIDENT` always.
- Unverified quote on a non-NI/NA answer → `FLAGGED` with reason.
- `NI` + 0 retrieved + supplements available → `FLAGGED`.
- Low BM25 score → `UNCERTAIN`.
- Verified quote with adequate retrieval → `CONFIDENT`.
- `ConfidenceSignals` has **no** `answer_consistency` field.

---

## REQ-12: Assessment Orchestration (Two-Tier Graph)

**Modules:** `arbiter/graph/state.py`, `arbiter/graph/builder.py`, `arbiter/graph/nodes/`

### What to Build

Two LangGraph graphs reflecting the trial/outcome tiers. **No checkpointer** — graphs run in-memory; resilience is at the batch layer ([REQ-17](#req-17-batch-runner-and-manifest)).

**Trial graph** (`build_trial_graph`) — runs once per trial:

```
START → context_D1 → [Send] sq_worker_D1 ×N → fanin_D1 → judgment_D1
      → context_D2 → [Send] sq_worker_D2_round1 → fanin_D2_r1
                   → [Send] sq_worker_D2_round2 (conditional) → fanin_D2 → judgment_D2 → END
```

Output: `domain_judgments = [D1, D2]` with `scope="trial"`.

**Outcome graph** (`build_outcome_graph`) — runs once per outcome, seeded with the reused D1/D2:

```
START → pre_d5 → context_D3 → [Send] sq_worker_D3 ×4 → fanin_D3 → judgment_D3
      → context_D4 → [Send] sq_worker_D4_round1 → fanin_D4_r1
                   → [Send] sq_worker_D4_round2 (conditional) → fanin_D4 → judgment_D4
      → context_D5 → [Send] sq_worker_D5 ×3 → fanin_D5 → judgment_D5
      → overall_judgment → END
```

Output: `domain_judgments = [D3, D4, D5]` (scope `"outcome"`) + overall judgment/rationale + `requires_human_review`, rolled up over the reused D1/D2 plus D3/D4/D5.

### State Reducers

`sq_answers` and `domain_contexts` use a dict-merge reducer (never overwrite). `domain_judgments` and `errors` use `operator.add`. All other fields are last-write-wins. Runtime handles (LLM clients, BM25 index) are passed through state but are **not** durable/checkpointed.

### Fan-In and Judgment Nodes

- **Fan-in** validates that all expected SQ answers for the domain are present; appends a descriptive string to `errors` for any missing SQ (does **not** abort).
- **Judgment** calls the matching `judge_domain_*` and emits a `DomainJudgment` with the correct `scope`.

### Acceptance Criteria

- Both graphs compile and run; a full trial produces D1–D5 and an overall judgment per outcome.
- D1/D2 in every outcome record for a trial are byte-identical (reused, not re-judged).
- D2 branching: with `effect=assignment` and `2.1=Y`, the gated next-round SQ is answered, `2.6` is asked, and `2.7` is gated on `2.6`; with `effect=adhering`, `2.3`–`2.6` use their per-protocol wording and **`2.7` is the only `NA`** (see [REQ-08](#req-08-conditional-branching) — `2.3`–`2.6` are _not_ NA'd across effects, they are re-worded).
- **CONSORT extraction runs once per trial:** the trial tier locates and (conservatively) vision-extracts the participant-flow `ConsortFlow`, stored on `TrialState` and threaded into each `OutcomeState`; it is **not** re-run per outcome.

---

## REQ-13: Pre-D5 Outcome Comparison

**Module:** `arbiter/graph/nodes/pre_d5.py`

### What to Build

A deterministic node (no LLM) that matches the **assessed** outcome (the current outcome string) against the **full set of registered outcomes** from CT.gov — `primaryOutcomes[*] ∪ secondaryOutcomes[*]` — via `rapidfuzz.ratio`, keeping the **best** match. This avoids the false positive of comparing a secondary outcome (PFS, AE) against `primaryOutcomes[0]` (OS) and spuriously flagging a switch on every non-primary outcome. Returns:

- `registered_outcome` — the best-matching registered measure,
- `published_outcome` — the assessed outcome string,
- `outcome_similarity_score` (0–1, rounded) — that best score,
- `outcome_change_detected = best_score < 0.85` — i.e. the assessed outcome is **not found in the registry at all** (possible selective reporting / unregistered outcome),
- `registered_as_primary: bool | None` — whether the best match came from the **primary** list (lets D5 distinguish "a registered secondary was promoted to the headline result" from "absent from registry").

If CT.gov data is absent or the outcomes module is missing, return all fields as `None`.

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

For each domain, assemble the `DomainContext` shared by that domain's SQ workers. Runs once per domain before fan-out.

- **Main-paper selection:** collect `DocumentSection`s whose label partially matches the domain's heading list (`DOMAIN_SECTIONS`), case-insensitive; **always include METHODS and RESULTS**; cap at ~4,000 tokens. If the selection is short, prepend the abstract.
- **Supplement selection:** query `SupplementIndex.retrieve(union_of_domain_key_terms, domain, top_k)`. Include small segments verbatim; for large segments (`char_count ≥ ~6,000`), sub-rank the segment's sentences by BM25 against the domain key terms and include the top few. Cap supplement context at ~2,000 tokens. Record `bm25_top_score`, `segments_retrieved`, `segments_available`.
- **D3 — participant flow from BOTH text and vision (complementary, never either/or):** inject (a) the Results-text flow numbers / flow-diagram captions, (b) the **CT.gov `enrollmentInfo.count`** as a randomised-N denominator hint, and (c) the trial-tier **`ConsortFlow`** when CONSORT vision fired (see the CONSORT node below). All three coexist in the D3 context; the LLM sees text _and_ the extracted figure counts. A **disagreement** between text-derived and vision-derived counts is surfaced as a confidence signal (toward `UNCERTAIN`/`FLAGGED`), not silently resolved in favour of one source.
- **D5 only:** inject the outcome-comparison block when `outcome_change_detected is not None`.
- **Total** `main_paper_text + supplement_block` must not exceed ~6,000 tokens.

### CONSORT extraction (trial tier, v0.1)

A trial-tier node locates the CONSORT participant-flow figure and conditionally vision-extracts it into `ConsortFlow` (once per trial; threaded into every `OutcomeState`):

- **Deterministic detector (no LLM):** score candidate pages/images using `PageBox` (`boxclass == "picture"` + `bbox` + `page`) plus nearby text (caption / within ~300 chars) for CONSORT-flow vocabulary (`CONSORT`, `flow`, `enrol(l)`, `randomi[sz]ed`, `allocated`, `discontinued`, `lost to follow-up`, `analy[sz]ed`, `assessed for eligibility`), biased toward Methods/early-Results, low text density.
- **Conservative gate:** only when the best score ≥ `ARBITER_CONSORT_DETECT_THRESHOLD` (conservative default) render that page to an image and call `vision_model.complete_vision(..., schema=ConsortFlow)`. **Asymmetric risk justifies conservatism:** a _missed_ figure falls back to text-only + an `UNCERTAIN` flag (safe), but a _wrong_ image injects bogus counts and can make D3 confidently wrong.
- **No confident candidate → text-only fallback** (no vision call); record `consort_detected=False`, detection score, and `vision_used=False` for auditability.
- **Caveat to document:** the CONSORT flow is the trial-level/primary-analysis flow; per-outcome missingness can differ — treat it as a denominator _hint_, with outcome-specific analysed-N still taken from the Results text.

### Acceptance Criteria

- Returns a `DomainContext` for every domain with non-empty `main_paper_text` for any parseable paper.
- D3 context contains the text flow block and the CT.gov enrolment denominator when available, **and** the `ConsortFlow` block when CONSORT vision fired; a text/vision count mismatch raises the confidence flag.
- The CONSORT detector fires vision **only** above the threshold; below it, D3 is text-only and self-flags `UNCERTAIN` when flow numbers are absent.
- D5 context includes the outcome-comparison block when CT.gov data is present.
- Combined main+supplement context does not exceed ~6,000 tokens.

---

## REQ-15: SQ Worker

**Module:** `arbiter/graph/nodes/sq_node.py`

### What to Build

The core LLM node: processes **one** SQ per invocation, fanned out in parallel within a domain (except the round-1 triggers of D2/D4).

- Build messages with a **cacheable static prefix** (trial context, main-paper source text, CT.gov block) and a **dynamic suffix** (domain supplement block, the SQ's `question_text` + `answer_definitions`, and the task instructions). The task: find the most relevant verbatim sentence(s) in the SOURCE TEXT, copy them exactly, note the 0-based page, choose the answer code, and write exactly one justification sentence. If no relevant text exists in any provided source, answer `NI` with an empty quote.
- Call `sq_model`'s `complete_structured(..., schema=SQRawAnswer)` (temperature 0).
- **Verify** the quote against `raw_char_stream` ([REQ-10](#req-10-quote-verifier)).
- **Compute** confidence ([REQ-11](#req-11-confidence-signal-system)) from verification + retrieval signals. **No second sample.**
- Emit `{"sq_answers": {sq_id: SQAnswer(...)}}`.

LLM API errors are handled by the retry wrapper ([REQ-20](#req-20-error-handling-and-retry)), not here.

### Acceptance Criteria

- Returns `{"sq_answers": {sq_id: SQAnswer(...)}}` every invocation, with `confidence` always populated.
- Exactly one LLM call per SQ (no resampling).
- Uses `sq_model`, not `aux_model`.

---

## REQ-16: Output — JSON and SQLite Writers

**Modules:** `arbiter/output/json_writer.py`, `arbiter/output/sqlite_writer.py`

### JSON Writer

`write_assessment_json(assessment: Assessment, output_dir: Path) -> Path` writes one file per trial-outcome to `{output_dir}/{trial_id}__{outcome_slug}__{effect}.json` (creating the directory). The `{effect}` segment keeps the two effect-of-interest assessments of one trial-outcome from colliding (mirrors the unique key below). The JSON contains: identifiers, models used, `trial_metadata`, `outcome_comparison`, a `domains` object (`D1`–`D5`, each with judgment, rationale, and per-SQ `{answer, quote, page, justification, confidence}`), the `overall` block, `sources_manifest`, and `errors`. D1/D2 blocks are present and identical across the trial's outcome files.

### SQLite Writer

`write_assessment_sqlite(assessment: Assessment, db_path: Path) -> None` upserts one row per trial-outcome into `arbiter_assessments` (created on first write). Columns include: `assessment_id` (PK), `created_at`, `trial_id`, `nct_number`, `title`, `outcome`, `effect_of_interest`, `overall_judgment`, `d1..d5_judgment`, `flagged_sq_count`, `uncertain_sq_count`, `requires_human_review`, `model_sq`, `model_aux`, `pipeline_version`, `json_path`, `errors` (JSON array). A **unique key** on `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` backs batch idempotency ([REQ-17](#req-17-batch-runner-and-manifest)). `effect_of_interest` is in the key because it changes D2 (and therefore potentially the overall judgment): without it, assessing one trial-outcome under `assignment` then `adhering` would clobber or skip the second. Use `INSERT … ON CONFLICT … DO UPDATE` (or `INSERT OR REPLACE`) so re-writes don't raise.

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

- For each entry: ingest once (trial tier), assess D1/D2 once, then loop the entry's outcomes (outcome tier), writing one record per trial-outcome.
- **Idempotency:** before assessing a trial-outcome, check the DB unique key `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)`; **skip** if already present unless `--force`. Because `trial_id` is deterministic ([REQ-05](#req-05-trial-metadata-extractor)) — including the NCT-less content-hash fallback — re-running a batch resumes by doing only missing/failed pairs even for trials with no registry metadata.
- **Continue-on-error:** any exception assessing one trial or outcome is caught, recorded (error row / `errors` field), and the batch proceeds to the next. One bad PDF never halts the batch.
- **Bounded concurrency:** respect `ARBITER_MAX_CONCURRENCY` (default low) to stay within provider rate limits — important on the OpenRouter free tier. Combine with the backoff in [REQ-20](#req-20-error-handling-and-retry). Note this bounds **trial-level** parallelism; within a trial a domain still fans out up to ~7 SQ calls at once (D2), so intra-domain rate-limit politeness relies on the REQ-20 backoff rather than a separate cap — correct, occasionally slow on the free tier.

### Acceptance Criteria

- `arbiter batch manifest.csv` processes every entry; a manifest with only `main_paper` columns runs (NCT derived, primary-outcome default).
- A directory in `supplements` ingests all PDFs within it without manual categorisation.
- Re-running the same batch skips already-completed trial-outcome pairs; `--force` re-runs them.
- A deliberately corrupt entry produces an error record and does not stop the batch.

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

`--paper PATH` (required); `--supplement PATH` (repeatable; file or dir); `--nct TEXT`; `--outcome TEXT` (repeatable; default = extracted primary); `--effect [assignment|adhering]` (default assignment); `--sq-model TEXT` / `--aux-model TEXT` / `--vision-model TEXT` (default from env); `--no-consort` (disable CONSORT vision → text-only D3); `--output-dir PATH`; `--db PATH`; `--force`.

### `arbiter batch` options

`--manifest PATH` (required); `--sq-model` / `--aux-model` / `--vision-model`; `--no-consort` (disable CONSORT vision → text-only D3); `--output-dir`; `--db`; `--max-concurrency INT`; `--force`.

### Stdout

`assess` prints the trial id and, per outcome on completion, the outcome and overall judgment plus the JSON path. `batch` prints a per-trial progress line and a final summary (counts: completed / skipped / errored).

### Acceptance Criteria

- `arbiter assess --paper paper.pdf` runs end-to-end (primary outcome).
- `arbiter assess --paper paper.pdf --supplement sap.pdf --nct NCT01234567 --outcome "Overall Survival" --outcome "Progression-Free Survival"` produces two records.
- `arbiter batch --manifest m.csv` runs and is idempotent on re-run.
- Commands exit 0 on success, non-zero on unrecoverable error (e.g., auth failure).

---

## REQ-19: Python API

**Module:** `arbiter/__init__.py`

### What to Build

```python
async def assess_trial(config: AssessmentConfig) -> list[Assessment]:
    """Assess one trial across config.outcomes (or its primary outcome).
    Returns one Assessment per outcome. Ingests and judges D1/D2 ONCE,
    reusing them across outcomes."""
```

Flow: create `sq_model`, `aux_model`, and `vision_model` clients → Phase 1 ingestion (paper, supplements, CT.gov, metadata) → Phase 2 trial graph (D1, D2 **and CONSORT extraction**) once → for each outcome, outcome graph (D3, D4, D5 + rollup) seeded with the reused D1/D2 **and the shared `ConsortFlow`** → Phase 3 build `Assessment`, write JSON + SQLite. Idempotency and concurrency are enforced by the batch runner; `assess_trial` itself always computes.

### Acceptance Criteria

- Returns one `Assessment` per requested outcome; D1/D2 identical across them.
- Ingestion (PDF parsing, supplement indexing) happens once per trial regardless of outcome count.

---

## REQ-20: Error Handling and Retry

**Module:** `arbiter/llm/base.py` (applied in all clients) and the batch runner.

### Two Independent Retry Layers

1. **Network/transient retries (LLM transport):** wrap `complete_structured` calls with bounded exponential backoff + jitter (default max 3 attempts) for rate-limit / timeout / connection errors. **Authentication / invalid-request errors raise immediately** (no retry) and abort the run with a clear message.
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

Measure agreement between ARBITER and human reference RoB 2 judgments. **v0.1 goal is to confirm the pipeline runs end-to-end and roughly agrees** — not to certify production accuracy.

### Reference Set (current)

`eval/reference/{overall_survival,progression_free_survival,adverse_events}.csv`, one row per trial with columns `Trial, D1, D2, D3, D4, D5, Overall Risk` (`L`/`S`/`H` → Low/Some concerns/High). mHSPC trials across 3 outcomes — **OS 10, PFS 10, AE 8 = 28 trial-outcome reference assessments** (not every trial reports every outcome, so the total is 28, not 30). Add a `rationale` column (or sibling notes file) capturing the **published per-domain reasons** (e.g. PEACE-1 D2 = protocol modified to add docetaxel; STAMPEDE/LATITUDE/ARCHES D3 = ≥10% missing; open-label PFS/AE D4 = unmasked assessment) for **error analysis** of disagreements.

> **The reference used a NON-STANDARD overall rollup.** The source publication defined overall **Low** as "low in all domains **or some concerns in one domain**" — i.e. it tolerates a single Some-concerns domain as overall Low, which **disagrees with both official RoB 2 and ARBITER's rollup** (where one Some-concerns domain → overall Some concerns). Every `Some Concerns` overall in the set is therefore an exactly-≥2-domain case _by the authors' rule_. The metrics below are designed so this rule mismatch does not masquerade as model error.

### Metrics

- **Per-domain agreement (PRIMARY):** exact-match rate over all judged domain cells (5 × 28 = 140), reported overall and as a confusion matrix vs human. This is **rollup-independent** and is the real test of ARBITER's SQ-answering + decision tables.
- **Overall-judgment agreement (rollup-normalised):** the **primary** overall metric recomputes the reference overall **from the reference domain cells using ARBITER's own rollup policy**, then compares to ARBITER — so any disagreement reflects domain disagreement, not the rule mismatch. **Also** report "as-published overall agreement" against the raw `Overall Risk` column, explicitly labelled as using the publication's lenient rule.
- **Confidence-flag calibration (diagnostic):** whether human–ARBITER disagreements concentrate in `FLAGGED`/`UNCERTAIN` domains (expected especially in D3, where image-only participant flow drives the ≥10%-missing signal).
- **CONSORT-vision contribution (D3 A/B, diagnostic):** because vision is on-by-default and gated by a kill switch ([REQ-14](#req-14-context-assembly)), the harness can run two passes — **vision-on** and **text-only** (`--no-consort`) — and report the **D3 domain-agreement delta** against the reference D3 column. This is the only measurable check that the vision node helps rather than hurts (the reference has D3 labels but no SQ-level ground truth). A non-positive delta is the signal to raise `ARBITER_CONSORT_DETECT_THRESHOLD` or reconsider the node — measured, not assumed.

The **numeric pass threshold is set after a baseline run** rather than asserted up front.

### Documented Limitations (record in the harness output and README)

- Reference labels are **domain-level only** — no SQ-level ground truth, so **per-SQ accuracy cannot be measured** with this set. (The published per-domain _rationale_ gives partial, domain-level expectations — e.g. the 4 double-blind trials ARASENS/ARCHES/LATITUDE/TITAN should come out D4 = Low on PFS/AE — usable as sanity oracles, not SQ ground truth.)
- The reference's **overall rollup is non-standard** (above); overall agreement is therefore reported rollup-normalised **and** as-published.
- Labels are **only L and S (no High)** and a **single tumour type** — High-risk detection and cross-disease generalisation are **unvalidated** by the human set. High-path _algorithm_ correctness is instead covered by the synthetic conformance test ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)/[REQ-22](#req-22-testing)). A more granular, broader eval set is planned.
- **D3 is the most likely domain to disagree** in v0.1: participant-flow counts are frequently image-only, so even with the CONSORT-vision node, low-confidence detections fall back to text and should self-flag `UNCERTAIN` rather than guess.

### Acceptance Criteria

- `python eval/run_eval.py` runs ARBITER over the reference trials (given their PDFs/NCTs) and prints **per-domain agreement** (primary), **rollup-normalised overall agreement** _and_ **as-published overall agreement**, and the confusion matrices.
- **Every report (stdout and any written file) stamps the exact `model_sq` / `model_aux` / `model_vision` and the pinned provider at the top**, so no agreement number can be quoted without the models that produced it. (The free dev models are fine for dev-time eval; the real, TBD eval swaps in prod models — the stamp is what keeps the two from being confused.)
- The harness pins a single OpenRouter provider (or uses prod models) so reported numbers are reproducible.
- The harness explicitly reports the limitations above, including the reference's non-standard rollup.

---

## REQ-22: Testing

### Unit (`tests/unit/`) — every pure function

Priority: decision tables (exhaustive vs vendored RoB 2, incl. the D1.3-direction regression); rollup (all 5-domain combinations incl. **both** `requires_human_review` paths — exactly-2 SC and `≥ OVERALL_HIGH_SC_THRESHOLD` SC with no domain High — reading the threshold from the constant); branching (all conditional triggers, both D2 effects); quote verifier (exact / fuzzy / OCR / mismatch); confidence signals (all flag conditions; assert no `answer_consistency`); segmenter (doc-type scoring, domain tagging); SQ prompts (22 entries, required fields); **CONSORT detector** (pure, deterministic — feed synthetic `list[PageBox]`: a `picture` box on an early page with CONSORT-flow caption vocabulary scores **above** threshold; a body-text-only page scores **below** — no image, no LLM).

### Integration (`tests/integration/`) — `MockLLMClient`, no network

1. Full trial on **Fixture A** (simple RCT, no supplements, NCT present) — overall judgment matches reference.
2. Full trial on **Fixture B** (SAP, multi-outcome, D5 concerns) — `outcome_change_detected = True`; D5 High where expected.
3. **Two-tier reuse:** a trial with 2 outcomes — D1/D2 judged once, identical across both outcome records; ingestion runs once.
4. **LLM failure on one SQ** — mock raises rate-limit 3× for one SQ; graph completes; that SQ is `NI`/`FLAGGED`; `errors` populated.
5. **Quote-verification failure** — mock returns an absent quote; `quote_verified = False`, `flag = FLAGGED`.
6. **D2 branching assignment** — `2.1 = Y`; gated SQ answered; adhering-only SQs `NA`.
7. **D2 branching adhering** — `effect = adhering`; assignment-only SQs `NA`.
8. **Batch idempotency** — run a 1-trial manifest twice; second run skips the completed pair; `--force` re-runs it.
9. **Non-native structured output** — a mock simulating a model that emits slightly malformed JSON; the validate-and-retry path recovers a valid `SQRawAnswer`.
10. **CONSORT vision — detector fires** — Fixture B's layout trips the detector above threshold; **mock `complete_vision`** returns a fixture `ConsortFlow` (keyed separately); assert the `ConsortFlow` block lands in D3 context and is threaded into **every** `OutcomeState` (extracted once per trial, `consort_detected = True`), not re-run per outcome.
11. **CONSORT vision — safety paths** — (a) **below-threshold → text-only:** no vision call, `consort_detected = False`, `vision_used = False`, and D3 self-flags `UNCERTAIN` when flow numbers are absent; (b) **text/vision disagreement:** the mock `ConsortFlow` counts contradict the Results-text flow numbers → a confidence flag is raised (the case [REQ-14](#req-14-context-assembly) says must not be silently resolved).
12. **CONSORT kill switch** — `consort_vision_enabled = False` (`--no-consort`): the detector and vision call are skipped entirely; D3 is text-only; no vision mock is consulted.

### Fixtures

`tests/fixtures/fixture_a|b/`: open-access `paper.pdf` (+ `sap.pdf` for B), `expected_assessment.json` (human-reviewed), `mock_llm_responses.yaml` (per-SQ `SQRawAnswer`). **Fixture B is extended** to carry a CONSORT participant-flow figure (so the detector fires) and a `mock_vision_responses.yaml` (a fixture `ConsortFlow`, keyed separately); the CONSORT detector unit test uses **synthetic `PageBox`es in code**, no PDF. No dedicated Fixture C.

### Acceptance Criteria

- `pytest tests/unit/` and `pytest tests/integration/` are fully green.
- No test makes a real LLM or network call.
- Suite completes in under 60 seconds.

---

## Implementation Order

Build and test in this order; each step depends on the previous.

| Step | Build                                                              | Why                                                                |
| ---- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 1    | REQ-01 setup + REQ-05 data models                                  | Everything imports from `config.py` / `models.py`                  |
| 2    | **Vendor `docs/rob2/`** (IRPG; fetch binaries, commit only README) | Source of truth for steps 3–5                                      |
| 3    | REQ-07 decision tables + rollup                                    | Pure Python, easiest to test first; implement from vendored source |
| 4    | REQ-08 branching                                                   | Needed by orchestration                                            |
| 5    | REQ-09 SQ prompts + Appendix A reconciliation                      | Needed by SQ worker                                                |
| 6    | REQ-10 quote verifier, REQ-11 confidence                           | Needed by SQ worker                                                |
| 7    | REQ-06 LLM layer (incl. hybrid structured output)                  | Needed by all LLM callers                                          |
| 8    | REQ-02 paper, REQ-03 supplements, REQ-04 CT.gov, REQ-05 metadata   | Ingestion                                                          |
| 9    | REQ-12 two-tier graphs + REQ-13/14/15 nodes                        | Core assessment                                                    |
| 10   | REQ-16 writers (with idempotency key)                              | Persist results                                                    |
| 11   | REQ-19 Python API (`assess_trial`)                                 | Wire trial+outcome tiers                                           |
| 12   | REQ-17 batch runner + manifest, REQ-18 CLI                         | Primary interface                                                  |
| 13   | REQ-20 error handling                                              | Harden                                                             |
| 14   | REQ-21 eval harness                                                | Measure agreement                                                  |
| 15   | REQ-22 tests                                                       | Alongside each step; full suite at end                             |

---

## Acceptance Checklist

- [ ] `pip install -e ".[anthropic]"` and `".[openai]"` succeed.
- [ ] `docs/rob2/` README (committed) records version (`IRPG beta v9`), retrieval date, source URLs, and the licence note; the binaries are git-ignored and fetched locally (not committed to the public repo).
- [ ] Decision tables reproduce the vendored RoB 2 logic exactly; D1.3 direction is **not** inverted (regression test passes).
- [ ] `arbiter assess --paper <fixture_a>` produces valid JSON with all 22 SQ answers (non-empty quotes except NI/NA) and D1–D5 judgments.
- [ ] Re-running ARBITER on identical inputs produces identical judgments (determinism).
- [ ] Multi-outcome run: D1/D2 identical across a trial's outcome records; ingestion runs once.
- [ ] `arbiter batch --manifest m.csv` runs unattended; re-run skips completed trial-outcome pairs; `--force` re-runs.
- [ ] A corrupt manifest entry yields an error record without halting the batch.
- [ ] Supplements given as a directory are ingested without manual categorisation; NCT is derived from the paper when omitted.
- [ ] Structured output is validated for both native-schema and gpt-oss/OpenRouter providers.
- [ ] `quote_verified = False` → `FLAGGED`; `NI` + 0 retrieved + supplements available → `FLAGGED`; no `answer_consistency` field anywhere.
- [ ] D2 branching: `SQ_PROMPTS` keyed by `(sq_id, effect)`; adhering's only `NA` is `2.7`; `2.3`–`2.6` carry effect-specific wording (not NA'd). D4 `4.3` gated on `N/PN/NI to 4.1 and 4.2`.
- [ ] CONSORT vision is on by default with a working kill switch (`--no-consort`); fires **only** above the detector threshold; below it D3 is text-only and self-flags `UNCERTAIN`; `ConsortFlow` is extracted once per trial and reused across outcomes; D3 uses text + denominator + vision together; `complete_vision` returns a validated `ConsortFlow` and is exercised by a mock-vision test.
- [ ] Pre-D5 matches the assessed outcome against the **full** registered set (primaries ∪ secondaries); a registered secondary is **not** flagged as a switch.
- [ ] Overall determinism claim is scoped to the deterministic core; idempotency is backed by the DB key, not LLM determinism.
- [ ] SQLite `arbiter_assessments` table is created on first run with a unique `(trial_id, outcome, effect_of_interest, model_sq, pipeline_version)` key over a **deterministic** `trial_id` (NCT → slugified `trial_label` → `sha256(paper)[:12]`; never `uuid4()`); re-run idempotency holds for NCT-less trials.
- [ ] `eval/run_eval.py` reports per-domain agreement (primary) + rollup-normalised **and** as-published overall agreement + confusion matrices + the vision-on vs text-only **D3 A/B delta**, **stamps the exact `model_sq`/`model_aux`/`model_vision` + provider at the top of every report**, and states the eval-set limitations (incl. the reference's non-standard rollup).
- [ ] `pytest tests/unit/` and `tests/integration/` are green; no real API calls; <60 s.
- [ ] PyMuPDF's AGPL licence is documented in the README (accepted for internal use; revisit before any redistribution).
- [ ] The vendored RoB 2 binaries are git-ignored (non-commercial/no-derivatives) with a committed `docs/rob2/README.md` fetch instruction; the rollup policy is captured in an ADR — the `OVERALL_HIGH_SC_THRESHOLD` (default 3) constant **and** `requires_human_review` firing on **both** policy-driven paths (exactly-2 SC and ≥-threshold SC with no domain High).

---

## Appendix A: The 22 Signaling Questions

This appendix is the **LLM-facing reference** for the SQ prompt templates ([REQ-09](#req-09-signaling-question-prompt-templates)). The **authoritative wording and the answer-to-judgment logic are the vendored official RoB 2 tool** in `docs/rob2/`; the `question_text` and `answer_definitions` here must be **reconciled verbatim against that source** at implementation time. SQ IDs and their meanings must be used consistently across the algorithm ([REQ-07](#req-07-rob-2-algorithm-deterministic-judgments)), branching ([REQ-08](#req-08-conditional-branching)), and these templates.

> **D2 has two variants.** For a given assessment only one effect's SQ set applies; the other effect's unique SQs are `NA`. Do not mix the assignment and adhering SQ meanings (a prior version of this document did — that is the bug this appendix exists to prevent).

### Domain 1 — Randomisation process (trial-level; SQs 1.1–1.3)

| SQ  | Question (verbatim per `docs/rob2/`)                                                                   | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------ | ------------ |
| 1.1 | Was the allocation sequence random?                                                                    | both         |
| 1.2 | Was the allocation sequence concealed until participants were enrolled and assigned to interventions?  | both         |
| 1.3 | Did baseline differences between intervention groups suggest a problem with the randomisation process? | both         |

> **Direction note (1.3):** `Y/PY` (differences suggest a problem) → toward **High**; `N/PN` (no problem) → required for **Low**.

`key_terms` — 1.1: random, randomis/z, sequence generation, computer generated, random number, minimisation, stratified. 1.2: concealment, allocation concealment, sealed/opaque envelope, central pharmacy, IWRS, telephone/central randomisation, sequentially numbered. 1.3: baseline characteristics, table 1, demographics, imbalance, covariate.

### Domain 2 — Deviations from intended interventions (trial-level)

Shared triggers (both effects):

| SQ  | Question                                                                                                           | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------ |
| 2.1 | Were participants aware of their assigned intervention during the trial?                                           | both         |
| 2.2 | Were carers and people delivering the interventions aware of participants' assigned intervention during the trial? | both         |

**Effect = assignment (ITT):**

| SQ  | Question                                                                                                                                                            | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 2.3 | [If Y/PY/NI to 2.1 or 2.2] Were there deviations from the intended intervention that arose because of the trial context?                                            | assignment   |
| 2.4 | [If Y/PY to 2.3] Were these deviations likely to have affected the outcome?                                                                                         | assignment   |
| 2.5 | [If Y/PY/NI to 2.4] Were these deviations from intended intervention balanced between groups?                                                                       | assignment   |
| 2.6 | Was an appropriate analysis used to estimate the effect of assignment to intervention?                                                                              | assignment   |
| 2.7 | [If N/PN/NI to 2.6] Was there potential for a substantial impact (on the result) of the failure to analyse participants in the group to which they were randomised? | assignment   |

**Effect = adhering (per-protocol):** SQs **2.3–2.6 are different questions** under the same IDs — they are _not_ the assignment questions, and they are _not_ `NA`. Only **2.7 is assignment-only** (→ `NA` under adhering). Verbatim from the vendored tool (`Print_format (PP)` / `Function Tab` col H):

| SQ  | Question                                                                                                                                   | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| 2.3 | [If Y/PY/NI to 2.1 or 2.2] Were important non-protocol interventions balanced across intervention groups?                                  | adhering     |
| 2.4 | [If applicable] Were there failures in implementing the intervention that could have affected the outcome?                                 | adhering     |
| 2.5 | [If applicable] Was there non-adherence to the assigned intervention regimen that could have affected participants' outcomes?              | adhering     |
| 2.6 | [If N/PN/NI to 2.3, **or** Y/PY/NI to 2.4 or 2.5] Was an appropriate analysis used to estimate the effect of adhering to the intervention? | adhering     |

> **Shared-ID warning.** `2.3`–`2.6` carry **distinct `question_text`/`answer_definitions` per effect**, so `SQ_PROMPTS` must key these by `(sq_id, effect)`, not `sq_id` alone (see [REQ-09](#req-09-signaling-question-prompt-templates)). For a given assessment exactly one effect's wording applies; **`2.7` is the only genuinely effect-exclusive ID** (assignment-only → `NA` under adhering). The adhering branch has **no 2.7**, and its **2.6 uses a compound gate** (`N/PN/NI to 2.3` **or** `Y/PY/NI to 2.4/2.5`), not a single chain.

`key_terms` — blinding/masking/open-label/placebo (2.1, 2.2); deviation, protocol deviation, co-intervention, concomitant, crossover, discontinued (2.3); adherence, compliance, per-protocol, dose received, fidelity (adhering 2.x); intention-to-treat, ITT, modified ITT, full analysis set, per-protocol, instrumental variable (2.6).

### Domain 3 — Missing outcome data (outcome-level; SQs 3.1–3.4)

| SQ  | Question                                                                                      | `applies_to` |
| --- | --------------------------------------------------------------------------------------------- | ------------ |
| 3.1 | Were data for this outcome available for all, or nearly all, randomised participants?         | both         |
| 3.2 | [If N/PN/NI to 3.1] Is there evidence that the result was not biased by missing outcome data? | both         |
| 3.3 | [If N/PN to 3.2] Could missingness in the outcome depend on its true value?                   | both         |
| 3.4 | [If Y/PY/NI to 3.3] Is it likely that missingness in the outcome depended on its true value?  | both         |

`key_terms` — missing data, lost to follow-up, dropout, withdrawal, analysed, completeness; sensitivity analysis, imputation, tipping point, MAR/MNAR/MCAR; informative censoring, reason for withdrawal, death, disease progression.

### Domain 4 — Measurement of the outcome (outcome-level; SQs 4.1–4.5)

| SQ  | Question                                                                                                                  | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 4.1 | Was the method of measuring the outcome inappropriate?                                                                    | both         |
| 4.2 | Could measurement or ascertainment of the outcome have differed between intervention groups?                              | both         |
| 4.3 | [If N/PN/NI to 4.1 **and** 4.2] Were outcome assessors aware of the intervention received by study participants?          | both         |
| 4.4 | [If Y/PY/NI to 4.3] Could assessment of the outcome have been influenced by knowledge of the intervention received?       | both         |
| 4.5 | [If Y/PY/NI to 4.4] Is it likely that assessment of the outcome was influenced by knowledge of the intervention received? | both         |

> **Meaning note:** 4.1 is about the **measurement method**, 4.3 about **assessor awareness**. Branching/judgment must use these meanings (an earlier draft conflated 4.1 with blinding — do not).

`key_terms` — outcome measure, validated instrument, scale, endpoint definition (4.1); assessment schedule, frequency, differed between groups (4.2); outcome assessor, blinded/masked assessor, adjudication/endpoint committee, central review (4.3); subjective vs objective, patient-reported, clinician-assessed, hard endpoint, mortality (4.4, 4.5).

### Domain 5 — Selection of the reported result (outcome-level; SQs 5.1–5.3)

| SQ  | Question                                                                                                                                                           | `applies_to` |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| 5.1 | Were the data that produced this result analysed in accordance with a pre-specified analysis plan that was finalised before unblinded outcome data were available? | both         |
| 5.2 | [Is the result] selected from multiple eligible outcome **measurements** on the basis of the results?                                                              | both         |
| 5.3 | [Is the result] selected from multiple eligible **analyses** of the data on the basis of the results?                                                              | both         |

> **Direction note:** for 5.2/5.3, evidence of selection-on-results pushes toward **High**. Reconcile exact wording/polarity with `docs/rob2/`. ARBITER also injects the deterministic registered-vs-published outcome comparison ([REQ-13](#req-13-pre-d5-outcome-comparison)) into the D5 context.

`key_terms` — pre-specified, pre-registered, protocol, SAP, statistical analysis plan, registry, ClinicalTrials.gov (5.1); outcome switching, selective reporting, multiple measurements/scales/timepoints (5.2); post hoc, unplanned analysis, multiplicity, subgroup, multiple analyses (5.3).

---

_End of PRD._

_References:_

- _Sterne JAC et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ 2019;366:l4898._
- _Higgins JPT et al. Revised Cochrane risk-of-bias tool for randomized trials (RoB 2) — guidance document, 22 August 2019. (Vendored in `docs/rob2/`.)_
- _Official RoB 2 Excel tool, riskofbias.info. (Vendored in `docs/rob2/`; version + retrieval date in `docs/rob2/README.md`.)_
- _Huang J et al. Large Language Model–Assisted Risk-of-Bias Assessment in RCTs Using RoB 2. JMIR 2025;27:e70450._
