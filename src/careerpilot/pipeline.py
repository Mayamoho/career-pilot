"""The end-to-end run: collect, extract, score, tailor, gate.

Each stage is separately callable, which is what lets the eval suite pin a single
stage without dragging the network or the model along with it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from careerpilot.domain.job import JobPosting
from careerpilot.ground.gate import GateReport
from careerpilot.ingest.base import new_client
from careerpilot.ingest.boards import ADAPTERS, BOARDS, TITLE_EXCLUDE, TITLE_INCLUDE, Board
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


def wanted_title(title: str) -> bool:
    """Role filter in code, on the title, before anything costs a model call."""
    lowered = title.lower()
    if any(term in lowered for term in TITLE_EXCLUDE):
        return False
    return any(term in lowered for term in TITLE_INCLUDE)


def collect(
    boards: Sequence[Board] = BOARDS,
    *,
    only_verified: bool = False,
    store: JobStore | None = None,
    keep_all: bool = False,
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


def score(job: JobPosting, profile: Profile, *, store: JobStore | None = None) -> Scored:
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


def tailor(job: JobPosting, profile: Profile, *, store: JobStore | None = None) -> Tailored:
    scored = score(job, profile, store=store)
    bullets = tailor_bullets(job, scored.fit, profile)
    letter_draft = draft_letter(job, bullets.targets, scored.fit, profile)
    letter_report = gate_letter(letter_draft.paragraphs, profile)
    return Tailored(
        scored=scored,
        bullets=bullets,
        letter=[v.text for v in letter_report.shipped],
        letter_report=letter_report,
    )
