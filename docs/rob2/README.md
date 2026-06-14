# Vendored RoB 2 reference (pinned)

This directory is the **single source of truth** for ARBITER's deterministic decision
logic (REQ-07), conditional branching (REQ-08), and the signaling-question prompt
wording (REQ-09 / Appendix A). All of those are implemented **directly from these
files** so that every ARBITER judgment is traceable to a fixed, citable algorithm
version.

## Variant scope

ARBITER v0.1 implements the **IRPG variant only** — _Individually Randomized,
Parallel-Group_ trials. The RoB 2 family also publishes **cluster-randomized** and
**crossover** variants, which use different signaling questions and decision tables.
Those are **out of scope** for v0.1; do not implement branching/tables from them.

## Pinned version

| Field              | Value                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| Tool               | Cochrane Risk of Bias 2 (RoB 2) — IRPG                                                                 |
| Algorithm workbook | `ROB2_IRPG_beta_v9` (beta v9)                                                                          |
| Guidance document  | Higgins JPT et al., _Revised Cochrane risk-of-bias tool for randomized trials (RoB 2)_, 22 August 2019 |
| Retrieved          | 2026-06-13                                                                                             |
| Source             | https://www.riskofbias.info/welcome/rob-2-0-tool/current-version-of-rob-2                              |

## Files (not committed — see below)

| Local filename             | Original                 | Purpose                                                                                                                                           |
| -------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rob2_irpg_algorithm.xlsm` | `ROB2_IRPG_beta_v9.xlsm` | Decision-table cell logic (REQ-07) and SQ routing/gating (REQ-08). Sheets of interest: `Print_format (ITT)`, `Print_format (PP)`, `Function Tab`. |
| `rob2_guidance.pdf`        | `RoB_2.0_guidance.pdf`   | Verbatim SQ wording + answer definitions (REQ-09 / Appendix A)                                                                                    |
| `rob2_cribsheet.pdf`       | `RoB_2.0_cribsheet.pdf`  | Condensed reviewer guidance (cross-reference)                                                                                                     |
| `rob2_template.pdf`        | `RoB_2.0_template.pdf`   | Blank assessment form (cross-reference)                                                                                                           |

## Licensing — why these binaries are git-ignored

The RoB 2 tool, guidance, crib sheet, and template are © the RoB 2 Development Group
and distributed from riskofbias.info under **non-commercial / no-derivatives** terms.
This repository is **public**, so the binaries are **not committed** (they are listed
in `.gitignore`). To populate this directory, download the four files from the source
URL above and rename them to the local filenames in the table.

> If ARBITER's licensing posture changes (e.g. the project obtains redistribution
> permission, or moves to a private repo), revisit this decision.
