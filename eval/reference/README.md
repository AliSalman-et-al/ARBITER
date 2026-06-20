# mHSPC-28 — dev smoke-test

The end-to-end / sanity check for the ARBITER pipeline (REQ-21). **Dev-only:** it confirms the
pipeline runs and produces sane output. Accuracy is **never tuned against it** and it **never
produces a published number** — powered claims come from the paper eval.

Single tumour type (metastatic hormone-sensitive prostate cancer), Low/Some-concerns only (no
High), and a **non-standard overall rollup** (see REQ-21). 10 trials × per-trial outcome lists =
**28 trial-outcome pairs**.

## Layout (all committed except `runs/`)

| Path                            | Committed?           | What                                                                                      |
| ------------------------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| `manifest.csv`                  | yes                  | 10-trial batch manifest: paths, pinned NCTs, per-trial `;`-delimited outcome lists        |
| `overall_survival.csv`          | yes                  | gold labels, OS (10 trials) — `Trial,D1,D2,D3,D4,D5,Overall Risk`                         |
| `progression_free_survival.csv` | yes                  | gold labels, PFS (10 trials)                                                              |
| `adverse_events.csv`            | yes                  | gold labels, AE (8 trials — no CHAARTED, no GETUG-AFU 15)                                 |
| `pdfs/<TRIAL>.pdf`              | yes                  | main papers                                                                               |
| `pdfs/supplement/<TRIAL>/*.pdf` | yes                  | per-trial supplements (protocol/appendix; SAP where published)                            |
| `runs/<run_id>/`                | **no (git-ignored)** | disposable outputs: `output/<trial_id>/` (`data.json`, `report.md`, debug) + `arbiter.db` |

Inputs are committed because this repo is **private**; they would be excluded from any public
mirror. (The `docs/rob2/` binaries and the mined-set review prose stay git-ignored for their own
licences — different rule, see REQ-21 / `.gitignore`.)

## Run

From the repo root (manifest paths are repo-root-relative):

```
uv run python eval/run_eval.py --smoke      # drives the batch + scores vs the gold CSVs
```

`run_eval` invokes, under the hood:

```
arbiter batch eval/reference/manifest.csv \
  --output-dir eval/reference/runs/<run_id>/output \
  --db         eval/reference/runs/<run_id>/arbiter.db
```

then reads that run's SQLite/JSON back and reports per-domain + rollup-normalised overall
agreement + confusion matrices, joining on `(trial_label, outcome)`.

## Trials & NCTs

| trial_label (gold spelling) | NCT         | on-disk note                                                               |
| --------------------------- | ----------- | -------------------------------------------------------------------------- |
| ARASENS                     | NCT02799602 |                                                                            |
| ARCHES                      | NCT02677896 |                                                                            |
| CHAARTED                    | NCT00309985 | OS + PFS only (no AE gold)                                                 |
| ENZAMET                     | NCT02446405 |                                                                            |
| GETUG-AFU 15                | NCT00104715 | folder was mislabelled `GETUG-AFU1`; corrected. OS + PFS only              |
| LATITUDE                    | NCT01715285 |                                                                            |
| PEACE-1                     | NCT01957436 | platform trial; canonical registry ID pinned                               |
| STAMPEDE                    | NCT00268476 | platform trial; canonical registry ID pinned                               |
| SWOG-1216                   | NCT01809691 | files live under `SWOG 1216/` (space); label uses the hyphen to match gold |
| TITAN                       | NCT02489318 |                                                                            |

The two naming mismatches (`SWOG-1216` ↔ `SWOG 1216`, `GETUG-AFU 15` ↔ old `GETUG-AFU1`) are
handled in `manifest.csv`: `trial_label` always uses the gold-CSV spelling so the score join is an
identity match, while the path columns point at the real files.
