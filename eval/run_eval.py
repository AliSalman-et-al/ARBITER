"""Evaluation harness for ARBITER REQ-21."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from arbiter.config import AssessmentConfig, MODEL_REGISTRY
from arbiter.manifest import run_batch
from arbiter.models import Judgment

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "eval" / "reference"
BENCHMARK_DIR = REPO_ROOT / "eval" / "benchmarks"
RUNS_DIR = REFERENCE_DIR / "runs"
DOMAIN_COLUMNS = ("D1", "D2", "D3", "D4", "D5")
GOLD_FILES = {
    "Overall Survival": REFERENCE_DIR / "overall_survival.csv",
    "Progression-Free Survival": REFERENCE_DIR / "progression_free_survival.csv",
    "Adverse Events": REFERENCE_DIR / "adverse_events.csv",
}
JUDGMENT_SHORT = {"L": "Low", "S": "Some concerns", "H": "High"}
JUDGMENT_ORDER = ["Low", "Some concerns", "High"]
LIMITATIONS = [
    "mHSPC-28 is dev-only; no published number rests on it.",
    "Mined-set labels are domain-level only; per-SQ accuracy belongs to ARBITER-Depth.",
    "Grounding correctness requires blinded divergent-cell adjudication on ARBITER-Depth.",
    "Overall agreement is reported rollup-normalised and as-published where available.",
    "Human RoB 2 IRR is only fair-to-moderate; interpret chance-corrected agreement against that ceiling.",
    "High-path algorithm correctness is covered by synthetic conformance tests, not by label prevalence here.",
]


@dataclass(frozen=True)
class EvalArm:
    name: str
    sq_model: str
    aux_model: str
    provider: str | None = None
    snapshot: str | None = None
    execution_mode: str = "dev-free-tier"
    pinned_paid: bool = False


@dataclass(frozen=True)
class GoldCell:
    trial_label: str
    outcome: str
    domain_gold: dict[str, str]
    overall_as_published: str
    overall_rollup_normalized: str


@dataclass(frozen=True)
class PredictionCell:
    trial_label: str
    outcome: str
    domain_pred: dict[str, str | None]
    overall_pred: str | None
    json_path: Path | None


def derive_pipeline_version(base_version: str, arm: EvalArm, *, retriever: str = "hybrid", consort_enabled: bool = False) -> str:
    """Derive a stable arm-specific pipeline version from non-keyed config dimensions."""

    payload = {
        "base_version": base_version,
        "aux_model": arm.aux_model,
        "provider": arm.provider,
        "snapshot": arm.snapshot,
        "retriever": retriever,
        "consort_enabled": consort_enabled,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{base_version}+eval.{digest}"


def smoke_arm() -> EvalArm:
    return EvalArm(
        name="smoke-free-gpt-oss-120b",
        sq_model="gpt-oss-120b",
        aux_model="gpt-oss-120b",
        provider="openrouter-free-tier",
        snapshot="unversioned-free-route",
        execution_mode="dev-free-tier",
        pinned_paid=False,
    )


def paper_roster() -> list[EvalArm]:
    arms: list[EvalArm] = []
    for name, info in MODEL_REGISTRY.items():
        if name == "google/gemma-4-31b-it:free":
            continue
        provider = str(info.get("provider")) if info.get("provider") is not None else None
        arms.append(
            EvalArm(
                name=name,
                sq_model=name,
                aux_model=name,
                provider=provider,
                snapshot=str(info.get("model_id") or name),
                execution_mode="headline-pinned-paid",
                pinned_paid=True,
            )
        )
    return arms


async def run_smoke_eval(
    *,
    reference_dir: Path = REFERENCE_DIR,
    run_id: str | None = None,
    force: bool = False,
    run_batch_fn=run_batch,
) -> dict[str, Any]:
    arm = smoke_arm()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = reference_dir / "runs" / run_id
    output_dir = run_dir / "output"
    db_path = run_dir / "arbiter.db"
    pipeline_version = derive_pipeline_version("0.1.0", arm)
    config = AssessmentConfig.from_env(
        paper_path=reference_dir / "manifest.csv",
        sq_model=arm.sq_model,
        aux_model=arm.aux_model,
        pipeline_version=pipeline_version,
        output_dir=output_dir,
        db_path=db_path,
        force=force,
        trace_level="summary",
        report_enabled=True,
    )
    progress: list[str] = []
    summary = await run_batch_fn(reference_dir / "manifest.csv", config, progress_callback=progress.append)
    gold = load_smoke_gold(reference_dir)
    predictions = load_predictions(db_path, reference_dir / "manifest.csv")
    report = score_domain_overall(gold, predictions)
    report.update(
        {
            "kind": "dev-smoke-test",
            "dev_only": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "arm": arm.__dict__,
            "pipeline_version": pipeline_version,
            "batch_summary": summary.model_dump() if hasattr(summary, "model_dump") else dict(summary),
            "progress": progress,
            "limitations": LIMITATIONS,
        }
    )
    write_report(run_dir / "eval_report.json", report)
    return report


def run_paper_eval(
    *,
    benchmark_dir: Path = BENCHMARK_DIR,
    repeats: int = 1,
    roster: Sequence[EvalArm] | None = None,
) -> dict[str, Any]:
    mined = benchmark_dir / "cochrane_mined"
    depth = benchmark_dir / "depth"
    missing = [str(path) for path in (mined, depth) if not path.exists()]
    arms = list(roster or paper_roster())
    report = {
        "kind": "paper-eval",
        "execution_mode": "headline-pinned-paid",
        "repeats": repeats,
        "arms": [
            {
                **arm.__dict__,
                "pipeline_version": derive_pipeline_version("0.1.0", arm),
            }
            for arm in arms
        ],
        "metrics_planned": [
            "mined per-domain agreement",
            "mined rollup-normalised and as-published overall agreement",
            "quote faithfulness locate-rate and match-score distribution",
            "ARBITER-Depth per-SQ parsed-only and end-to-end accuracy",
            "retrieval recall@k on ARBITER-Depth evidence passages",
            "D5 registry/protocol stratification",
            "enriched vs main-text-only with blinded divergent-cell adjudication",
            "NI-rate split into substantive and format-failure NI",
            "schema-repair rate and cost per assessment",
        ],
        "limitations": LIMITATIONS,
    }
    if missing:
        report["status"] = "blocked-missing-benchmarks"
        report["missing"] = missing
        return report
    report["status"] = "ready"
    return report


def load_smoke_gold(reference_dir: Path = REFERENCE_DIR) -> list[GoldCell]:
    cells: list[GoldCell] = []
    for outcome, path in GOLD_FILES.items():
        path = reference_dir / path.name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                domain_gold = {domain: _expand_judgment(row[domain]) for domain in DOMAIN_COLUMNS}
                cells.append(
                    GoldCell(
                        trial_label=row["Trial"],
                        outcome=outcome,
                        domain_gold=domain_gold,
                        overall_as_published=_expand_judgment(row["Overall Risk"]),
                        overall_rollup_normalized=rollup_normalized(domain_gold.values()),
                    )
                )
    return cells


def load_predictions(db_path: Path, manifest_path: Path) -> list[PredictionCell]:
    trial_by_id = _manifest_trial_labels(manifest_path)
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT trial_id, outcome, overall_judgment, d1_judgment, d2_judgment, d3_judgment,
                   d4_judgment, d5_judgment, json_path
            FROM arbiter_assessments
            WHERE outcome != '__TRIAL__'
            """
        ).fetchall()
    predictions = []
    for row in rows:
        predictions.append(
            PredictionCell(
                trial_label=trial_by_id.get(str(row["trial_id"]), str(row["trial_id"])),
                outcome=str(row["outcome"]),
                domain_pred={domain: row[f"{domain.lower()}_judgment"] for domain in DOMAIN_COLUMNS},
                overall_pred=row["overall_judgment"],
                json_path=Path(row["json_path"]) if row["json_path"] else None,
            )
        )
    return predictions


def score_domain_overall(gold: Sequence[GoldCell], predictions: Sequence[PredictionCell]) -> dict[str, Any]:
    pred_by_key = {(item.trial_label, item.outcome): item for item in predictions}
    missing = [cell for cell in gold if (cell.trial_label, cell.outcome) not in pred_by_key]
    domain_metrics = {}
    for domain in DOMAIN_COLUMNS:
        pairs = [
            (cell.domain_gold[domain], pred_by_key[(cell.trial_label, cell.outcome)].domain_pred.get(domain))
            for cell in gold
            if (cell.trial_label, cell.outcome) in pred_by_key
        ]
        domain_metrics[domain] = agreement_report(pairs, labels=JUDGMENT_ORDER)

    overall_pairs = [
        (cell.overall_rollup_normalized, pred_by_key[(cell.trial_label, cell.outcome)].overall_pred)
        for cell in gold
        if (cell.trial_label, cell.outcome) in pred_by_key
    ]
    return {
        "n_gold": len(gold),
        "n_predictions": len(predictions),
        "n_joined": len(gold) - len(missing),
        "missing_predictions": [{"trial": item.trial_label, "outcome": item.outcome} for item in missing],
        "domains": domain_metrics,
        "overall_rollup_normalized": agreement_report(overall_pairs, labels=JUDGMENT_ORDER),
        "stamp": {
            "dataset": "mHSPC-28",
            "dataset_role": "dev smoke-test only",
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def agreement_report(pairs: Sequence[tuple[str, str | None]], *, labels: Sequence[str]) -> dict[str, Any]:
    clean = [(a, b) for a, b in pairs if b is not None]
    return {
        "n": len(clean),
        "percent_agreement": percent_agreement(clean),
        "cohen_kappa": cohen_kappa(clean, labels),
        "gwet_ac2": gwet_ac2(clean, labels),
        "confusion_matrix": confusion_matrix(clean, labels),
        "bootstrap_ci": bootstrap_ci(clean, labels),
    }


def percent_agreement(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(1 for gold, pred in pairs if gold == pred) / len(pairs)


def cohen_kappa(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float | None:
    if not pairs:
        return None
    observed = percent_agreement(pairs)
    assert observed is not None
    total = len(pairs)
    gold_counts = Counter(gold for gold, _ in pairs)
    pred_counts = Counter(pred for _, pred in pairs)
    expected = sum((gold_counts[label] / total) * (pred_counts[label] / total) for label in labels)
    if math.isclose(1.0, expected):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def gwet_ac2(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float | None:
    """Unweighted multiclass Gwet agreement coefficient for nominal RoB labels."""

    if not pairs:
        return None
    observed = percent_agreement(pairs)
    assert observed is not None
    total_ratings = len(pairs) * 2
    proportions = Counter(value for pair in pairs for value in pair)
    q = len(labels)
    if q <= 1:
        return 1.0
    chance = sum((proportions[label] / total_ratings) * (1 - proportions[label] / total_ratings) for label in labels)
    chance = chance / (q - 1)
    if math.isclose(1.0, chance):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - chance) / (1.0 - chance)


def confusion_matrix(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in pairs:
        matrix.setdefault(gold, {label: 0 for label in labels})
        matrix[gold][pred] = matrix[gold].get(pred, 0) + 1
    return matrix


def bootstrap_ci(
    pairs: Sequence[tuple[str, str]],
    labels: Sequence[str],
    *,
    iterations: int = 200,
) -> dict[str, list[float | None]]:
    if len(pairs) < 2:
        value = percent_agreement(pairs)
        return {"percent_agreement": [value, value], "gwet_ac2": [gwet_ac2(pairs, labels), gwet_ac2(pairs, labels)]}
    samples = {"percent_agreement": [], "gwet_ac2": []}
    n = len(pairs)
    for i in range(iterations):
        sample = [pairs[(hash((i, j, n)) % n)] for j in range(n)]
        samples["percent_agreement"].append(percent_agreement(sample))
        samples["gwet_ac2"].append(gwet_ac2(sample, labels))
    return {
        key: [_quantile([v for v in values if v is not None], 0.025), _quantile([v for v in values if v is not None], 0.975)]
        for key, values in samples.items()
    }


def rollup_normalized(domain_judgments: Iterable[str]) -> str:
    values = list(domain_judgments)
    if Judgment.HIGH.value in values:
        return Judgment.HIGH.value
    some_count = sum(value == Judgment.SOME_CONCERNS.value for value in values)
    if some_count >= 3:
        return Judgment.HIGH.value
    if some_count:
        return Judgment.SOME_CONCERNS.value
    return Judgment.LOW.value


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_console_report(report: dict[str, Any]) -> str:
    lines = [
        "ARBITER evaluation harness",
        f"Mode: {report['kind']}",
        f"Dataset: {report.get('stamp', {}).get('dataset', 'paper benchmarks')}",
    ]
    if report.get("dev_only"):
        lines.append("DEV-ONLY: do not quote as a published accuracy number.")
    if "overall_rollup_normalized" in report:
        overall = report["overall_rollup_normalized"]
        lines.append(f"Joined: {report['n_joined']}/{report['n_gold']}")
        lines.append(f"Overall rollup-normalised agreement: {_fmt(overall['percent_agreement'])}")
        for domain in DOMAIN_COLUMNS:
            metric = report["domains"][domain]
            lines.append(f"{domain}: agreement {_fmt(metric['percent_agreement'])}, n={metric['n']}")
    if report.get("status") == "blocked-missing-benchmarks":
        lines.append("Paper eval datasets are not present yet:")
        lines.extend(f"- {path}" for path in report["missing"])
    lines.append("Limitations:")
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines)


def _expand_judgment(value: str) -> str:
    value = value.strip()
    return JUDGMENT_SHORT.get(value, value)


def _manifest_trial_labels(manifest_path: Path) -> dict[str, str]:
    mapping = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            nct = (row.get("nct_number") or "").strip().upper()
            label = (row.get("trial_label") or nct).strip()
            if nct:
                mapping[nct] = label
    return mapping


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ARBITER evaluation harnesses.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the mHSPC-28 dev smoke-test.")
    mode.add_argument("--paper", action="store_true", help="Prepare/run the paper evaluation harness.")
    parser.add_argument("--run-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    if args.smoke:
        report = asyncio.run(run_smoke_eval(run_id=args.run_id, force=args.force))
    else:
        report = run_paper_eval(repeats=args.repeats)
    print(format_console_report(report))
    return 0 if report.get("status") != "blocked-missing-benchmarks" else 2


if __name__ == "__main__":
    raise SystemExit(main())
