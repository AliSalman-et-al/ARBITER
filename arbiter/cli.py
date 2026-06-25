"""Command-line interface for ARBITER."""

import asyncio
from pathlib import Path
from typing import cast

import click

from . import assess_trial, ingest_trial
from .config import AssessmentConfig, EffectOfInterest, TraceLevel
from .manifest import check_eligibility, run_batch
from .output.sqlite_writer import write_skip_record


@click.group()
@click.version_option()
def cli() -> None:
    """Automated Cochrane RoB 2 assessment pipeline."""


@cli.command()
@click.option("--paper", "paper_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--supplement", "supplement_paths", type=click.Path(path_type=Path, exists=True), multiple=True)
@click.option("--nct-number")
@click.option("--outcome", "outcomes", multiple=True)
@click.option("--effect-of-interest", type=click.Choice(["assignment", "adhering"]), default="assignment", show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--db-path", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.option("--trace-level", type=click.Choice(["off", "summary", "full"]), default="full", show_default=True)
@click.option("--report/--no-report", "report_enabled", default=True, show_default=True)
def assess(
    paper_path: Path,
    supplement_paths: tuple[Path, ...],
    nct_number: str | None,
    outcomes: tuple[str, ...],
    effect_of_interest: str,
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
        output_dir=output_dir,
        db_path=db_path,
        force=force,
        trace_level=cast(TraceLevel, trace_level),
        report_enabled=report_enabled,
    )
    assessments = asyncio.run(_assess_one(config))
    click.echo(f"Wrote {len(assessments)} assessment(s).")


@cli.command()
@click.argument("manifest", type=click.Path(path_type=Path, exists=True))
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--db-path", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.option("--trace-level", type=click.Choice(["off", "summary", "full"]), default="summary", show_default=True)
@click.option("--report/--no-report", "report_enabled", default=True, show_default=True)
def batch(
    manifest: Path,
    output_dir: Path | None,
    db_path: Path | None,
    force: bool,
    trace_level: str,
    report_enabled: bool,
) -> None:
    """Assess a manifest of trials."""
    config = AssessmentConfig.from_env(
        paper_path=manifest,
        output_dir=output_dir,
        db_path=db_path,
        force=force,
        trace_level=cast(TraceLevel, trace_level),
        report_enabled=report_enabled,
    )
    summary = asyncio.run(run_batch(manifest, config))
    click.echo(
        "Processed {processed} entries; assessed {assessed} pair(s); skipped {skipped_entries} entry/entries "
        "and {skipped_pairs} pair(s); errors {errors}.".format(
            processed=summary.processed_entries,
            assessed=summary.assessed_pairs,
            skipped_entries=summary.skipped_entries,
            skipped_pairs=summary.skipped_pairs,
            errors=summary.error_count,
        )
    )


async def _assess_one(config: AssessmentConfig):
    ctx = await ingest_trial(config)
    skip = check_eligibility(ctx.trial_metadata, config)
    if skip is not None:
        skip = skip.model_copy(update={"inputs_hash": ctx.config_summary.get("inputs_hash")})
        write_skip_record(skip, config.output_dir, config.db_path)
        return []
    return await assess_trial(ctx, config)
