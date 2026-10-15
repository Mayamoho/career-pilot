"""The end-to-end run: collect, extract, score, tailor, gate.

Each stage is separately callable, which is what lets the eval suite pin a single
stage without dragging the network or the model along with it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from careerpilot.domain.job import JobPosting
from careerpilot.ground.gate import GateReport
from careerpilot.ingest.base import new_client
from careerpilot.ingest.boards import ADAPTERS, BOARDS, Board
from careerpilot.ingest.filter import wanted_title
from careerpilot.ingest.store import JobStore
from careerpilot.match.extract import RequirementSet, extract_requirements
from careerpilot.match.score import FitScore, score_job
from careerpilot.profile.model import Profile
from careerpilot.tailor.bullets import TailoredBullets, tailor_bullets
from careerpilot.tailor.letter import draft_letter, gate_letter


class CollectReport(BaseModel):
    fetched: int = 0
    inserted: int = 0
    unchanged_or_updated: int = 0
    kept: int = 0
    boards_ok: int = 0
    boards_dead: list[str] = Field(default_factory=list)


def collect(
    boards: Sequence[Board] = BOARDS,
    *,
    only_verified: bool = False,
    store: JobStore | None = None,
    keep_all: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CollectReport:
    owned = store is None
    job_store = store or JobStore()
    report = CollectReport()
    try:
        with new_client() as client:
            for board in boards:
                if only_verified and not board.verified:
                    continue
                adapter = ADAPTERS.get(board.source)
                if adapter is None:
                    continue
                if progress is not None:
                    progress(f"fetching {board.source}:{board.token}")
                rows = adapter().fetch_board(client, board.token, board.company)
                fetched = len(rows)
                kept = [row for row in rows if keep_all or wanted_title(row.title)]
                counts = job_store.upsert_many(kept)
                job_store.record_board_health(board.source, board.token, bool(rows), fetched)
                report.fetched += fetched
                report.inserted += counts["inserted"]
                report.unchanged_or_updated += counts["unchanged_or_updated"]
                report.kept += len(kept)
                if rows:
                    report.boards_ok += 1
                else:
                    report.boards_dead.append(f"{board.source}:{board.token}")
    finally:
        if owned:
            job_store.close()
    return report


class Scored(BaseModel):
    job: JobPosting
    requirements: RequirementSet
    fit: FitScore


def score(
    job: JobPosting,
    profile: Profile,
    *,
    store: JobStore | None = None,
    progress: Callable[[str], None] | None = None,
) -> Scored:
    if progress is not None:
        progress(f"extracting requirements from {job.uid}")
    requirements = extract_requirements(job)
    fit = score_job(job, requirements, profile)
    if store is not None:
        store.save_score(fit)
    return Scored(job=job, requirements=requirements, fit=fit)


class Tailored(BaseModel):
    scored: Scored
    bullets: TailoredBullets
    letter: list[str] = Field(default_factory=list)
    letter_report: GateReport = Field(default_factory=GateReport)

    @property
    def fit(self) -> FitScore:
        return self.scored.fit

    @property
    def shipped_bullets(self) -> list[str]:
        return [v.text for v in self.bullets.report.shipped]


def tailor(
    job: JobPosting,
    profile: Profile,
    *,
    store: JobStore | None = None,
    progress: Callable[[str], None] | None = None,
) -> Tailored:
    """Run every stage in order, naming each one to `progress` as it starts.

    The stages each spend a model call and the whole run is tens of seconds on a real
    link; without a word as they begin, the tool looks frozen exactly when it is working
    hardest. `progress` is how the CLI keeps the terminal honest about that.
    """
    scored = score(job, profile, store=store, progress=progress)
    if progress is not None:
        progress("drafting bullets from your facts")
    bullets = tailor_bullets(job, scored.fit, profile)
    if progress is not None:
        progress("drafting the cover letter")
    letter_draft = draft_letter(job, bullets.targets, scored.fit, profile)
    if progress is not None:
        progress("gating every line against the fact base")
    letter_report = gate_letter(letter_draft.paragraphs, profile)
    return Tailored(
        scored=scored,
        bullets=bullets,
        letter=[v.text for v in letter_report.shipped],
        letter_report=letter_report,
    )
