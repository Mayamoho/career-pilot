"""Command line entry point.

The default mode never touches a model: prompts have to be recorded first, and the
recordings are what CI runs against. `--live` opts in, and `record-fixtures` exists so
the recordings are produced deliberately rather than by accident.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from careerpilot import pipeline
from careerpilot.ai.gateway import get_gateway, set_gateway
from careerpilot.config import FIXTURES_DIR, live_mode_enabled, profile_path
from careerpilot.ingest.boards import BOARDS
from careerpilot.ingest.store import JobStore
from careerpilot.profile.model import Profile, load_profile
from careerpilot.report import render

app = typer.Typer(help="CareerPilot — grounded job fit and CV tailoring.", no_args_is_help=True)

ProfileOption = Annotated[
    Path | None, typer.Option("--profile", help="Candidate fact base YAML.")
]
DbOption = Annotated[Path | None, typer.Option("--db", help="SQLite job store path.")]


def _profile(path: Path | None) -> Profile:
    return load_profile(path or profile_path())


def _store(path: Path | None) -> JobStore:
    return JobStore(path)


@app.command()
def collect(
    *,
    db: DbOption = None,
    only_verified: Annotated[bool, typer.Option("--only-verified")] = False,
    keep_all: Annotated[bool, typer.Option("--keep-all", help="Skip the role filter.")] = False,
) -> None:
    """Fetch every curated board and store the postings that match the role filter."""
    store = _store(db)
    try:
        report = pipeline.collect(
            BOARDS, only_verified=only_verified, store=store, keep_all=keep_all
        )
        typer.echo(json.dumps(report.model_dump(), indent=2, default=str))
        typer.echo(f"{store.count()} job(s) in store")
    finally:
        store.close()


@app.command()
def jobs(
    *,
    query: Annotated[str, typer.Argument(help="FTS keyword search.")] = "",
    db: DbOption = None,
    limit: Annotated[int, typer.Option("--limit")] = 25,
) -> None:
    """List or search stored postings."""
    store = _store(db)
    try:
        rows = store.search(query, limit=limit) if query else store.all(limit=limit)
        for job in rows:
            typer.echo(f"{job.uid}\t{job.company} — {job.title}\t{job.location or ''}")
        typer.echo(f"{len(rows)} shown")
    finally:
        store.close()


@app.command()
def score_job(
    uid: Annotated[str, typer.Argument(help="Job uid, as printed by `jobs`.")],
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
) -> None:
    """Extract requirements and compute the fit score for one posting."""
    candidate = _profile(profile)
    store = _store(db)
    try:
        job = store.get(uid)
        if job is None:
            raise typer.BadParameter(f"no job with uid {uid!r}")
        scored = pipeline.score(job, candidate, store=store)
        typer.echo(f"{scored.fit.total:.1f}/100 — {scored.fit.band}")
        for outcome in scored.fit.outcomes:
            mark = "yes" if outcome.matched else " no"
            typer.echo(f" [{mark}] ({outcome.weight:.1f}) {outcome.requirement.skill}: "
                       f"{outcome.requirement.text}")
        if scored.fit.unmet_must_haves:
            typer.echo("unmet must-haves:")
            for text in scored.fit.unmet_must_haves:
                typer.echo(f"  - {text}")
    finally:
        store.close()


@app.command()
def tailor(
    uid: Annotated[str, typer.Argument(help="Job uid, as printed by `jobs`.")],
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
    out: Annotated[Path | None, typer.Option("--out", help="Artifact directory.")] = None,
) -> None:
    """Score, draft, gate and write the CV, cover letter and audit trail."""
    candidate = _profile(profile)
    store = _store(db)
    try:
        job = store.get(uid)
        if job is None:
            raise typer.BadParameter(f"no job with uid {uid!r}")
        result = pipeline.tailor(job, candidate, store=store)
        artifacts = render.write(result, candidate, out)
        bullets = result.bullets.report
        typer.echo(f"fit {result.fit.total:.1f}/100 ({result.fit.band})")
        typer.echo(f"bullets shipped {len(bullets.shipped)}/{bullets.total}")
        for verdict in bullets.rejected:
            typer.echo(f"  dropped: {verdict.failures[0]}")
        typer.echo(f"letter paragraphs shipped {len(result.letter_report.shipped)}"
                   f"/{result.letter_report.total}")
        typer.echo(f"wrote {artifacts.cv_path}")
        typer.echo(f"wrote {artifacts.letter_path}")
        typer.echo(f"wrote {artifacts.audit_path}")
    finally:
        store.close()


@app.command()
def traces() -> None:
    """Summarise recorded AI calls: failures per task, fallback use, cost."""
    typer.echo(json.dumps(get_gateway().sink.summary(), indent=2))


@app.command(name="record-fixtures")
def record_fixtures(
    uids: Annotated[list[str], typer.Argument(help="Job uids to record prompts for.")],
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
) -> None:
    """Run the pipeline against real models to record cassettes for CI.

    Recording is the only place where a live call is expected without the operator
    having to remember the environment variable, so the cost is a deliberate act.
    """
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise typer.BadParameter("no provider credential in the environment")
    os.environ["CAREERPILOT_LIVE"] = "1"
    live_mode_enabled.cache_clear()
    cassette_dir = FIXTURES_DIR / "cassettes"
    cassette_dir.mkdir(parents=True, exist_ok=True)
    candidate = _profile(profile)
    store = _store(db)
    try:
        for uid in uids:
            job = store.get(uid)
            if job is None:
                typer.echo(f"skipping {uid}: not in store")
                continue
            pipeline.tailor(job, candidate, store=store)
            typer.echo(f"recorded {uid}")
    finally:
        store.close()
        set_gateway(None)
        live_mode_enabled.cache_clear()
