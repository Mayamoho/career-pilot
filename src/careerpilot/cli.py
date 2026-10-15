"""Command line entry point.

The default mode never touches a model: prompts have to be recorded first, and the
recordings are what CI runs against. `--live` opts in, and `record-fixtures` exists so
the recordings are produced deliberately rather than by accident.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from careerpilot import pipeline
from careerpilot.ai.errors import AIError
from careerpilot.ai.gateway import get_gateway, set_gateway
from careerpilot.config import (
    FIXTURES_DIR,
    MissingCredentialError,
    live_mode_enabled,
    load_env_file,
    profile_path,
)
from careerpilot.domain.job import JobPosting
from careerpilot.domain.place import Place, home_country, place_of
from careerpilot.ingest.boards import BOARDS
from careerpilot.ingest.store import JobStore
from careerpilot.profile.model import Profile, load_profile
from careerpilot.report import render

app = typer.Typer(help="CareerPilot — grounded job fit and CV tailoring.", no_args_is_help=True)


def _recency(job: JobPosting) -> float:
    """Sort key that keeps an undated posting newest rather than oldest."""
    return job.published_at.timestamp() if job.published_at else float("inf")

ProfileOption = Annotated[
    Path | None, typer.Option("--profile", help="Candidate fact base YAML.")
]
DbOption = Annotated[Path | None, typer.Option("--db", help="SQLite job store path.")]


def _profile(path: Path | None) -> Profile:
    return load_profile(path or profile_path())


def _store(path: Path | None) -> JobStore:
    return JobStore(path)


@contextmanager
def _spoken_not_dumped() -> Iterator[None]:
    """Turn an AI-layer failure into a sentence, not a stack trace.

    Every error caught here is a state of the user's environment — no key, live mode off,
    no recording for this prompt — and none of them are fixed by reading Python internals.
    A traceback for those is not diagnostics, it is the tool failing to explain itself.
    """
    try:
        yield
    except (AIError, MissingCredentialError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


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


_SCAN = 5000


@app.command()
def jobs(
    *,
    query: Annotated[str, typer.Argument(help="FTS keyword search.")] = "",
    db: DbOption = None,
    profile: ProfileOption = None,
    limit: Annotated[int, typer.Option("--limit")] = 25,
    where: Annotated[
        str,
        typer.Option("--where", help="'bd' in home, 'open' workable from home, 'all'."),
    ] = "open",
    home: Annotated[
        str | None, typer.Option("--home", help="Override the country in the fact base.")
    ] = None,
) -> None:
    """List stored postings with the location read honestly.

    Defaults to `--where open`: what a person at the address in the fact base can
    actually take. A posting is open if it is in that country, or if it claims remote
    and names no other country. The stored `is_remote` flag is not trusted on its own:
    Ashby sets it for roles whose only stated location is an office.
    """
    if where not in {"bd", "open", "all"}:
        raise typer.BadParameter("choose one of: bd, open, all")
    reference = home or home_country(_profile(profile))
    store = _store(db)
    try:
        rows = store.search(query, limit=_SCAN) if query else store.all(limit=_SCAN)
    finally:
        store.close()

    shown: list[tuple[int, JobPosting, Place]] = []
    for job in rows:
        place = place_of(job, home=reference)
        keep = (
            where == "all"
            or (where == "bd" and place.at_home)
            or (where == "open" and place.workable)
        )
        if keep:
            shown.append((place.rank, job, place))
    if where != "all":
        shown.sort(key=lambda item: (item[0], -_recency(item[1])))
    typer.echo("# workable from " + reference if where != "all" else "# all stored postings")
    for _, job, place in shown[:limit]:
        typer.echo(
            f"{job.uid}\t{job.company} — {job.title}"
            f"\t[{place.mode}] {job.location or 'no location'}\t{place.note}"
        )
    typer.echo(f"{min(len(shown), limit)} of {len(shown)} matching, {len(rows)} scanned")


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
        with _spoken_not_dumped():
            scored = pipeline.score(job, candidate, store=store)
        loc = scored.fit.location
        verdict = "workable" if loc.workable else "NOT WORKABLE"
        typer.echo(f"{scored.fit.total:.1f}/100 — {scored.fit.band}")
        typer.echo(f"location [{verdict}] from {loc.home}: {loc.note}")
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
    ignore_location: Annotated[
        bool,
        typer.Option("--ignore-location", help="Tailor a posting you cannot work from home."),
    ] = False,
) -> None:
    """Score, draft, gate and write the CV, cover letter and audit trail.

    Refuses a posting that cannot be worked from the address in the fact base, before
    spending a single model call: a tailored CV for a job restricted to another country
    is not a near-miss, it is a document nobody can hire you under.
    """
    candidate = _profile(profile)
    store = _store(db)
    try:
        job = store.get(uid)
        if job is None:
            raise typer.BadParameter(f"no job with uid {uid!r}")
        place = place_of(job, home=home_country(candidate))
        if not place.workable and not ignore_location:
            typer.echo(f"refusing: {place.note} — home is {place.home}.")
            typer.echo("pass --ignore-location to tailor it anyway.")
            raise typer.Exit(code=2)
        with _spoken_not_dumped():
            result = pipeline.tailor(job, candidate, store=store)
        artifacts = render.write(result, candidate, out)
        bullets = result.bullets.report
        typer.echo(f"fit {result.fit.total:.1f}/100 ({result.fit.band})")
        typer.echo(f"location: {result.fit.location.note}")
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
    load_env_file()
    if not any(os.environ.get(name, "").strip() for name in ("GROQ_API_KEY", "GEMINI_API_KEY")):
        raise typer.BadParameter(
            "no provider credential in the environment or in .env — see the README"
        )
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
            with _spoken_not_dumped():
                pipeline.tailor(job, candidate, store=store)
            typer.echo(f"recorded {uid}")
    finally:
        store.close()
        set_gateway(None)
        live_mode_enabled.cache_clear()
