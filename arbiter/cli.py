"""Command-line interface for ARBITER."""

import asyncio
from pathlib import Path
from typing import cast

import click

from . import assess_trial, ingest_trial
from .config import AssessmentConfig, EffectOfInterest, TraceLevel
from .manifest import check_eligibility, run_batch
from .output.json_writer import assessment_json_path
from .output.sqlite_writer import write_skip_record


@click.group()
@click.version_option()
def cli() -> None:
    """Automated Cochrane RoB 2 assessment pipeline."""


@cli.command()
@click.option("--paper", "paper_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--supplement", "supplement_paths", type=click.Path(path_type=Path, exists=True), multiple=True)
@click.option("--nct", "--nct-number", "nct_number")
@click.option("--outcome", "outcomes", multiple=True)
@click.option("--effect", "--effect-of-interest", "effect_of_interest", type=click.Choice(["assignment", "adhering"]), default="assignment", show_default=True)
@click.option("--sq-model")
@click.option("--aux-model")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--db", "--db-path", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.option("--trace-level", type=click.Choice(["off", "summary", "full"]), default="full", show_default=True)
@click.option("--report/--no-report", "report_enabled", default=True, show_default=True)
def assess(
    paper_path: Path,
    supplement_paths: tuple[Path, ...],
    nct_number: str | None,
    outcomes: tuple[str, ...],
    effect_of_interest: str,
    sq_model: str | None,
    aux_model: str | None,
    output_dir: Path | None,
    db_path: Path | None,
    force: bool,
    trace_level: str,
    report_enabled: bool,
) -> None:
    """Assess one paper.

    The setup slice wires the CLI and config shape. The assessment engine is
    implemented in later requirements.
    """
    config = AssessmentConfig.from_env(
        paper_path=paper_path,
        supplement_paths=list(supplement_paths),
        nct_number=nct_number,
        outcomes=list(outcomes) or None,
        effect_of_interest=cast(EffectOfInterest, effect_of_interest),
        sq_model=sq_model,
        aux_model=aux_model,
        output_dir=output_dir,
        db_path=db_path,
        force=force,
        trace_level=cast(TraceLevel, trace_level),
        report_enabled=report_enabled,
    )
    result = asyncio.run(_assess_one(config))
    if result["skipped"]:
        click.echo(f"Skipped {result['trial_id']}: wrote {result['json_path']}.")
        return
    click.echo(f"Trial {result['trial_id']}")
    for item in result["assessments"]:
        click.echo(f"{item['outcome']}: {item['overall_judgment']} -> {item['json_path']}")


@cli.command()
@click.argument("manifest_arg", type=click.Path(path_type=Path, exists=True), required=False)
@click.option("--manifest", "manifest_option", type=click.Path(path_type=Path, exists=True))
@click.option("--sq-model")
@click.option("--aux-model")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--db", "--db-path", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--max-concurrency", type=int)
@click.option("--force", is_flag=True)
@click.option("--trace-level", type=click.Choice(["off", "summary", "full"]), default="summary", show_default=True)
@click.option("--report/--no-report", "report_enabled", default=True, show_default=True)
def batch(
    manifest_arg: Path | None,
    manifest_option: Path | None,
    sq_model: str | None,
    aux_model: str | None,
    output_dir: Path | None,
    db_path: Path | None,
    max_concurrency: int | None,
    force: bool,
    trace_level: str,
    report_enabled: bool,
) -> None:
    """Assess a manifest of trials."""
    manifest = manifest_option or manifest_arg
    if manifest is None:
        raise click.UsageError("Provide a manifest path with --manifest PATH or as the positional argument.")
    config = AssessmentConfig.from_env(
        paper_path=manifest,
        sq_model=sq_model,
        aux_model=aux_model,
        output_dir=output_dir,
        db_path=db_path,
        max_concurrency=max_concurrency,
        force=force,
        trace_level=cast(TraceLevel, trace_level),
        report_enabled=report_enabled,
    )
    summary = asyncio.run(run_batch(manifest, config, progress_callback=click.echo))
    click.echo(
        "Processed {processed} entries; assessed {assessed} pair(s); skipped {skipped_entries} entry/entries "
        "and {skipped_pairs} pair(s); errors {errors}; wall {wall:.3f}s; LLM latency {latency:.3f}s; "
        "LLM calls {calls}; tokens {tokens}; cost {cost}.".format(
            processed=summary.processed_entries,
            assessed=summary.assessed_pairs,
            skipped_entries=summary.skipped_entries,
            skipped_pairs=summary.skipped_pairs,
            errors=summary.error_count,
            wall=summary.total_wall_time_s,
            latency=summary.total_llm_latency_s,
            calls=summary.total_llm_calls,
            tokens=summary.total_tokens if summary.total_tokens is not None else "null",
            cost=summary.total_cost if summary.total_cost is not None else "null",
        )
    )
    if summary.slowest_trials:
        slowest = ", ".join(f"{item['trial_id']} {item['wall_time_s']:.3f}s" for item in summary.slowest_trials)
        click.echo(f"Slowest trials: {slowest}")


async def _assess_one(config: AssessmentConfig):
    ctx = await ingest_trial(config)
    skip = check_eligibility(ctx.trial_metadata, config)
    if skip is not None:
        skip = skip.model_copy(update={"inputs_hash": ctx.config_summary.get("inputs_hash")})
        json_path = write_skip_record(skip, config.output_dir, config.db_path)
        return {
            "skipped": True,
            "trial_id": skip.trial_id,
            "json_path": json_path,
            "assessments": [],
        }
    assessments = await assess_trial(ctx, config)
    return {
        "skipped": False,
        "trial_id": ctx.trial_metadata.trial_id,
        "json_path": None,
        "assessments": [
            {
                "outcome": assessment.outcome,
                "overall_judgment": assessment.overall_judgment.value,
                "json_path": assessment_json_path(assessment, config.output_dir),
            }
            for assessment in assessments
        ],
    }
