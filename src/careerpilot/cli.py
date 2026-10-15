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
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from careerpilot import pipeline
from careerpilot.ai.errors import AIError
from careerpilot.ai.gateway import get_gateway, set_gateway
from careerpilot.ai.providers import CREDENTIAL_ENV, credential_present
from careerpilot.ats.reader import read_cv, score_cv
from careerpilot.config import (
    DATA_DIR,
    FIXTURES_DIR,
    MODEL_CHAIN,
    REPO_ROOT,
    MissingCredentialError,
    live_mode_enabled,
    load_env_file,
    profile_path,
)
from careerpilot.domain.job import JobPosting
from careerpilot.domain.place import Place, home_country, place_of
from careerpilot.ingest.boards import BOARDS
from careerpilot.ingest.store import JobStore
from careerpilot.profile import questions
from careerpilot.profile.model import Profile, load_profile
from careerpilot.report import render, templates

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _guide() -> None:
    """Grounded job fit and CV tailoring for someone starting out in Bangladesh.

    \b
    WHICH COMMANDS NEED A JOB UID, AND WHERE IT COMES FROM
      A uid is `source:id` exactly as printed: bdjobs:1545472, bdtechjobs:52,
      greenhouse:openai:1234, jobicy:155389. Get one from `careerpilot jobs` (or
      `careerpilot rank`, `careerpilot closing`). Commands that take one:
        jobs [QUERY]        optional keyword filter, no uid
        score-job UID       one posting  -> the fit score and its reasons
        tailor UID          one posting  -> CV, cover letter, audit trail
        ats [UID]           optional uid; with none it scores the CV alone
        record-fixtures UID...  several, to record cassettes for CI
      No uid at all: collect, closing, rank, start-here, traces, keys.

    \b
    WHAT EACH RUN COSTS
      Free, no key, no model:  collect, jobs, ats, closing, traces, keys
      Also free:  start-here, a questionnaire that writes your fact base
      rank:  1 extract + one check per requirement, capped by --max-new (5)
      Several model calls:  score-job, tailor, record-fixtures
      A posting scored once is reused, so `rank` twice costs nothing.
      `gave up after N attempt(s)` is the model refusing, not crashing.
      Run `careerpilot keys` to see which hop is out of tokens today.

    \b
    FIRST RUN, IN ORDER
      careerpilot start-here     then     careerpilot collect
      careerpilot ats            then     careerpilot jobs --where bd python
      careerpilot rank --limit 10        then     careerpilot tailor <uid>
    """


def _recency(job: JobPosting) -> float:
    """Sort key that keeps an undated posting newest rather than oldest."""
    return job.published_at.timestamp() if job.published_at else float("inf")


def _naive_utc(moment: datetime) -> datetime:
    """`coerce_datetime` stores deadlines as naive UTC; the clock here is aware."""
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


def _days_left(job: JobPosting, now: datetime | None = None) -> float | None:
    """Whole days until the employer stops reading. `None` means the board says nothing."""
    if job.deadline is None:
        return None
    deadline = _naive_utc(job.deadline)
    return (deadline - _naive_utc(now or datetime.now(UTC))).total_seconds() / 86400.0


def _place(job: JobPosting, home: str, where: str) -> Place | None:
    """The place this posting maps to under the `--where` filter, or None to drop it."""
    place = place_of(job, home=home)
    if where == "all" or (where == "bd" and place.at_home) or (where == "open" and place.workable):
        return place
    return None


_WHERE = ("bd", "open", "all")


def _check_where(where: str) -> str:
    if where not in _WHERE:
        raise typer.BadParameter(f"choose one of: {', '.join(_WHERE)}")
    return where

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


def _working(message: str) -> None:
    """One line of progress on stderr while a slow stage runs.

    stderr, not stdout, because stdout is the result: `careerpilot rank > shortlist.txt`
    has to stay clean while the person watching the terminal can still see that nothing
    is stuck. A model call is tens of seconds, and silence over that window reads as a
    hang — which is how a working tool gets Ctrl+C'd.
    """
    typer.echo(f"… {message}", err=True)


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
            BOARDS, only_verified=only_verified, store=store, keep_all=keep_all,
            progress=_working,
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
    source: Annotated[
        str | None,
        typer.Option("--source", help="One board family only: bdjobs, jobicy, ashby…"),
    ] = None,
) -> None:
    """List stored postings with the location read honestly.

    Defaults to `--where open`: what a person at the address in the fact base can
    actually take. A posting is open if it is in that country, or if it claims remote
    and names no other country. The stored `is_remote` flag is not trusted on its own:
    Ashby sets it for roles whose only stated location is an office.
    """
    _check_where(where)
    reference = home or home_country(_profile(profile))
    store = _store(db)
    try:
        rows = store.search(query, limit=_SCAN) if query else store.all(limit=_SCAN)
    finally:
        store.close()

    shown: list[tuple[int, JobPosting, Place]] = []
    for job in rows:
        if source and job.source != source:
            continue
        place = _place(job, reference, where)
        if place is not None:
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
    uid: Annotated[
        str, typer.Argument(help="Job uid as printed by `jobs`, e.g. bdjobs:1545472.")
    ],
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
            raise typer.BadParameter(
                f"no job with uid {uid!r} — list what is stored with `careerpilot jobs`",
            )
        with _spoken_not_dumped():
            scored = pipeline.score(job, candidate, store=store, progress=_working)
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
    uid: Annotated[
        str,
        typer.Argument(help="Job uid as printed by `jobs`/`rank`/`closing`, e.g. bdjobs:1545472."),
    ],
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
    out: Annotated[Path | None, typer.Option("--out", help="Artifact directory.")] = None,
    style: Annotated[
        str,
        typer.Option("--style", help="Print forms: ats, oxford, both, or md for markdown only."),
    ] = "both",
    ignore_location: Annotated[
        bool,
        typer.Option("--ignore-location", help="Tailor a posting you cannot work from home."),
    ] = False,
    ignore_deadline: Annotated[
        bool, typer.Option("--ignore-deadline", help="Tailor a posting whose deadline passed.")
    ] = False,
) -> None:
    """Score, draft, gate and write the CV, cover letter and audit trail.

    Refuses a posting that cannot be worked from the address in the fact base, before
    spending a single model call: a tailored CV for a job restricted to another country
    is not a near-miss, it is a document nobody can hire you under. The same is true of
    a deadline that has passed — and on Bangladeshi boards, most of them have.
    """
    styles = {"md": (), "ats": ("ats",), "oxford": ("oxford",), "both": templates.STYLES}.get(style)
    if styles is None:
        raise typer.BadParameter("choose one of: md, ats, oxford, both")
    candidate = _profile(profile)
    store = _store(db)
    try:
        job = store.get(uid)
        if job is None:
            raise typer.BadParameter(
                f"no job with uid {uid!r} — list what is stored with `careerpilot jobs`",
            )
        place = place_of(job, home=home_country(candidate))
        if not place.workable and not ignore_location:
            typer.echo(f"refusing: {place.note} — home is {place.home}.")
            typer.echo("pass --ignore-location to tailor it anyway.")
            raise typer.Exit(code=2)
        left = _days_left(job)
        if left is not None and left < 0 and not ignore_deadline:
            typer.echo(f"refusing: the application deadline was {abs(left):.0f} day(s) ago.")
            typer.echo("pass --ignore-deadline to tailor it anyway.")
            raise typer.Exit(code=2)
        with _spoken_not_dumped():
            result = pipeline.tailor(job, candidate, store=store, progress=_working)
        artifacts = render.write(result, candidate, out, styles)
        bullets = result.bullets.report
        typer.echo(f"fit {result.fit.total:.1f}/100 ({result.fit.band})")
        typer.echo(f"location: {result.fit.location.note}")
        if left is not None:
            typer.echo(f"deadline: {job.deadline:%Y-%m-%d}, {left:.0f} day(s) left")
        typer.echo(f"bullets shipped {len(bullets.shipped)}/{bullets.total}")
        for verdict in bullets.rejected:
            typer.echo(f"  dropped: {verdict.failures[0]}")
        typer.echo(f"letter paragraphs shipped {len(result.letter_report.shipped)}"
                   f"/{result.letter_report.total}")
        typer.echo(f"wrote {artifacts.cv_path}")
        typer.echo(f"wrote {artifacts.letter_path}")
        typer.echo(f"wrote {artifacts.audit_path}")
        for path in artifacts.extra_paths:
            typer.echo(f"wrote {path}")
    finally:
        store.close()


@app.command()
def traces() -> None:
    """Summarise recorded AI calls: failures per task, fallback use, cost."""
    sink = get_gateway().sink
    typer.echo(json.dumps(sink.summary(sink.read_disk()), indent=2))


_KEY_SOURCE = {
    "groq": ("console.groq.com", "fastest hop, but its free tier caps tokens per day"),
    "gemini": ("aistudio.google.com", "a separate daily window: the key that keeps you working"),
    "cerebras": ("cerebras.ai", "a third free token budget, same wire format as Groq"),
    "openrouter": ("openrouter.ai", "one key in front of many free models, metered per request"),
    "cohere": ("docs.cohere.com", "asked for JSON rather than forced into it, so the last resort"),
}


@app.command()
def keys() -> None:
    """Say which provider keys are set, and which one to get next.

    Reading the fallback chain out of the configuration rather than restating it means this
    cannot drift from what `tailor` will actually try.
    """
    load_env_file()
    live = live_mode_enabled()
    typer.echo(f"live mode: {'ON' if live else 'OFF — set CAREERPILOT_LIVE=1 in .env'}\n")
    windows = 0
    hosts: set[str] = set()
    for hop, spec in enumerate(MODEL_CHAIN, start=1):
        secret = CREDENTIAL_ENV[spec.provider]
        source, why = _KEY_SOURCE[spec.provider]
        ready = credential_present(spec.provider)
        if ready:
            windows += 1
            hosts.add(spec.provider)
        status = "set" if ready else "NOT SET"
        detail = spec.model if ready else f"free key at {source}: {why}"
        typer.echo(f"{hop}. {spec.provider:<11} {status:<8} {secret}  — {detail}")
    if windows == 0:
        typer.echo(
            "\nNothing is configured, so every model-dependent command refuses. "
            "Groq alone is enough to start."
        )
    elif windows == 1:
        typer.echo("\nOne model window is usable: when it runs out the command stops. Add hop 2.")
    else:
        span = "one key" if len(hosts) == 1 else f"{len(hosts)} providers"
        typer.echo(
            f"\n{windows} model windows on {span} are usable, so a spent model moves on "
            "instead of stopping the command."
        )


@app.command(name="record-fixtures")
def record_fixtures(
    uids: Annotated[
        list[str],
        typer.Argument(help="Job uids to record prompts for, as printed by `jobs`."),
    ],
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
) -> None:
    """Run the pipeline against real models to record cassettes for CI.

    Recording is the only place where a live call is expected without the operator
    having to remember the environment variable, so the cost is a deliberate act.
    """
    load_env_file()
    if not any(credential_present(provider) for provider in CREDENTIAL_ENV):
        raise typer.BadParameter(
            "no provider credential in the environment or in .env — see `careerpilot keys`"
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
                pipeline.tailor(job, candidate, store=store, progress=_working)
            typer.echo(f"recorded {uid}")
    finally:
        store.close()
        set_gateway(None)
        live_mode_enabled.cache_clear()


@app.command()
def ats(
    uid: Annotated[
        str, typer.Argument(help="Posting to score the CV against, as printed by `jobs`.")
    ] = "",
    *,
    cv: Annotated[
        Path | None,
        typer.Option("--cv", help="Your CV as .txt/.md/.html. Defaults to your fact base."),
    ] = None,
    jd: Annotated[
        Path | None,
        typer.Option("--jd", help="A job description in a file, instead of a uid."),
    ] = None,
    db: DbOption = None,
    profile: ProfileOption = None,
    write_cv: Annotated[
        bool, typer.Option("--write-cv", help="Write the plain-text CV this score is based on.")
    ] = False,
) -> None:
    """Will a machine be able to read this CV? Computed in code: no model, no network.

    The keyword half of the answer is split by what your fact base actually contains.
    A term you can prove goes under "you may add"; a term you cannot goes under "not in
    your fact base — do not add". This tool will not hand you a lie and call it optimisation.
    """
    candidate = _profile(profile)
    job_text = ""
    label = "the CV alone (no posting given)"
    if uid:
        store = _store(db)
        try:
            job = store.get(uid)
            if job is None:
                raise typer.BadParameter(
                    f"no job with uid {uid!r} — list what is stored with `careerpilot jobs`",
                )
            job_text = f"{job.title}\n{job.description}"
            label = f"{job.company} — {job.title}"
        finally:
            store.close()
    elif jd:
        job_text = jd.read_text(encoding="utf-8", errors="replace")
        label = jd.name

    text = read_cv(cv) if cv else templates.cv_txt(candidate, None)
    report = score_cv(text, candidate, job_text)

    typer.echo(f"ATS readiness {report.total}/100 — {report.band}  ({report.words} words)")
    typer.echo(f"scored against: {label}")
    for check in report.checks:
        if check.possible <= 0:
            typer.echo(f" {check.id:17}    —   {check.why}")
            continue
        typer.echo(
            f" {check.id:17} {check.points:5.1f}/{check.possible:4.1f}  {check.why}"
        )
        if check.fix:
            typer.echo(f"    → {check.fix}")
    if report.matched_terms:
        typer.echo("posting terms your CV already states: " + ", ".join(report.matched_terms))
    if report.addable_terms:
        typer.echo("YOU MAY ADD (they are in your fact base): " + ", ".join(report.addable_terms))
    if report.not_in_fact_base:
        typer.echo(
            "NOT IN YOUR FACT BASE — DO NOT ADD: " + ", ".join(report.not_in_fact_base)
        )
        typer.echo("  add the experience first, then this line: `careerpilot start-here` helps.")
    if write_cv:
        written = render.write_cv(candidate, _cv_dir(), tailored=None)
        for path in written:
            typer.echo(f"wrote {path}")


def _cv_dir() -> Path:
    return DATA_DIR / "out" / "cv"


@app.command()
def rank(
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
    limit: Annotated[int, typer.Option("--limit")] = 10,
    where: Annotated[str, typer.Option("--where", help="'bd', 'open' or 'all'.")] = "open",
    source: Annotated[str | None, typer.Option("--source", help="Restrict to one feed.")] = None,
    query: Annotated[str, typer.Argument(help="FTS keyword filter.")] = "",
    max_new: Annotated[
        int,
        typer.Option(
            "--max-new",
            help="New postings to score this run. Each costs one extraction plus one "
            "citation check per requirement it names — a real bdjobs posting had 16 — "
            "so this is the day's spend cap, and it counts postings, not calls.",
        ),
    ] = 5,
    home: Annotated[str | None, typer.Option("--home")] = None,
) -> None:
    """Which postings are worth your time, cheapest first.

    A posting already in the `scores` table is read back rather than re-scored, so the
    second run of this command costs nothing. `--max-new` exists because Groq's free tier
    limits tokens per day, not just per minute: an uncapped batch is how a live run dies
    halfway through with a quota it cannot refill until midnight UTC. Each fresh score
    prints a line as it starts, so a run that is working is never mistaken for a hang.
    """
    _check_where(where)
    candidate = _profile(profile)
    reference = home or home_country(candidate)
    store = _store(db)
    fresh = reused = skipped = 0
    try:
        rows = store.search(query, limit=_SCAN) if query else store.all(limit=_SCAN)
        ranked: list[tuple[Place, JobPosting, float, str]] = []
        for job in rows:
            if source and job.source != source:
                continue
            place = _place(job, reference, where)
            if place is None:
                continue
            saved = store.saved_score(job.uid)
            if saved is not None:
                reused += 1
                ranked.append((place, job, saved.total, saved.band))
                continue
            if fresh >= max_new:
                skipped += 1
                continue
            fresh += 1
            _working(f"scoring {fresh}/{max_new}: {job.company} — {job.title}")
            with _spoken_not_dumped():
                fit = pipeline.score(job, candidate, store=store).fit
            ranked.append((place, job, fit.total, fit.band))
    finally:
        store.close()

    ranked.sort(key=lambda item: (item[0].rank, -item[2], _deadline_key(item[1])))
    typer.echo(
        f"# workable from {reference} — {fresh} fresh score(s), {reused} reused, "
        f"{skipped} over the --max-new budget"
    )
    for place, job, total, band in ranked[:limit]:
        left = _days_left(job)
        closing = f"{left:.0f}d left" if left is not None else "no deadline"
        typer.echo(
            f"{job.uid}\t{total:5.1f} {band:<8}\t[{place.mode}] {closing}\t"
            f"{job.company} — {job.title}"
        )


def _deadline_key(job: JobPosting) -> float:
    """Nearest deadline first; a posting that never states one goes last."""
    return job.deadline.timestamp() if job.deadline else float("inf")


@app.command()
def closing(
    *,
    db: DbOption = None,
    profile: ProfileOption = None,
    days: Annotated[int, typer.Option("--days", help="Window ahead.")] = 21,
    where: Annotated[str, typer.Option("--where", help="'bd', 'open' or 'all'.")] = "open",
    home: Annotated[str | None, typer.Option("--home")] = None,
) -> None:
    """Deadlines, nearest first, expired ones dropped.

    Only the Bangladeshi boards publish a deadline, which makes this command worth more
    here than anywhere else: on bdjobs a posting is dead the morning after its date, and
    nothing in the feed tells you so otherwise.
    """
    _check_where(where)
    reference = home or home_country(_profile(profile))
    now = datetime.now(UTC)
    store = _store(db)
    try:
        rows = store.all(limit=_SCAN)
    finally:
        store.close()

    open_now = [
        (left, job, place)
        for job in rows
        if (place := _place(job, reference, where)) is not None
        and (left := _days_left(job, now)) is not None
        and 0 <= left <= days
    ]
    open_now.sort(key=lambda item: item[0])
    expired = sum(1 for job in rows if (_days_left(job, now) or 1) < 0)
    typer.echo(
        f"# {len(open_now)} closing within {days} day(s); {expired} already expired and dropped"
    )
    for left, job, place in open_now:
        typer.echo(
            f"{job.deadline:%Y-%m-%d}\t{left:4.0f}d\t{job.uid}\t[{place.mode}]\t"
            f"{job.company} — {job.title}"
        )


@app.command(name="start-here")
def start_here(
    *,
    out: Annotated[
        Path, typer.Option("--out", help="Where to write the fact base.")
    ] = REPO_ROOT / "data" / "profile.yaml",
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
    show: Annotated[bool, typer.Option("--print", help="Print it instead of writing it.")] = False,
) -> None:
    """Turn a first job search into a fact base, by asking.

    Nothing here reads as generated text: every answer is yours, and the ids are made
    for you because the file format requires them. Validate-first, so a half-finished
    answer never becomes a broken profile on disk.
    """
    typer.echo("CareerPilot keeps one file of facts and prints CVs from it. Let's make yours.")
    typer.echo(questions.ENTRY_FORMAT)
    profile, dropped = questions.collect(_ask)
    for line in dropped:
        typer.echo(f"  skipped (too short to be a fact): {line!r}")
    text = questions.to_yaml(profile)
    if show:
        typer.echo(text)
        return
    if out.exists() and not force:
        typer.echo(f"{out} already exists. Pass --force to overwrite, or --print to see this.")
        raise typer.Exit(code=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {len(profile.atoms)} fact(s) to {out}")
    typer.echo("it is gitignored — this file holds your phone number, keep it that way")
    typer.echo("next: careerpilot ats          then   careerpilot collect")


def _ask(question: questions.Question) -> list[str]:
    """One prompt, with the hint printed above it rather than hidden in a `--help`.

    Ctrl+D ends the answer instead of aborting the run. A piped answer file has no
    trailing blank line, and losing every answer to one missing Enter is not a
    reasonable way to fail.
    """
    typer.echo(f"\n{question.prompt}")
    if question.hint:
        typer.echo(f"  {question.hint}")
    if question.answer != "lines":
        return [_answer("  answer")]
    typer.echo("  One per line; press Enter on an empty line when you are done.")
    lines: list[str] = []
    while True:
        line = _answer("  >")
        if not line.strip():
            return lines
        lines.append(line)


def _answer(text: str) -> str:
    """`input()` rather than `typer.prompt`, because Ctrl+D must end the answer.

    Under click a closed stdin raises `Abort`, which would throw away forty answers the
    user already gave. Here it is just the empty line that finishes the block.
    """
    typer.echo(text, nl=False)
    try:
        return input()
    except EOFError:
        return ""
