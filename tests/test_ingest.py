"""Ingestion and storage. No model is involved anywhere in this module's tests.

A posting is a fact, so it is read from an API and stored verbatim. What is tested
here is the parts that can silently corrupt data: dedupe, the retry policy, and the
keyword search that must not let a stray quote rewrite the query.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from careerpilot.domain.job import JobPosting
from careerpilot.ingest.adapters import (
    AshbyAdapter,
    BdTechJobsAdapter,
    GreenhouseAdapter,
    LeverAdapter,
)
from careerpilot.ingest.base import get_json
from careerpilot.ingest.boards import ADAPTERS, BOARDS, Board
from careerpilot.ingest.store import JobStore, fts_query
from careerpilot.pipeline import collect, wanted_title

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/acme"
LEVER_URL = "https://api.lever.co/v0/postings/acme"
BDTECHJOBS_URL = "https://www.bdtechjobs.com/api/jobs"


@pytest.fixture
def store(tmp_path: Path) -> Any:
    job_store = JobStore(tmp_path / "jobs.sqlite3")
    yield job_store
    job_store.close()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Retries must not make the suite slow, and the backoff itself is under test."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))
    return slept


def test_greenhouse_row_becomes_a_posting_with_plain_text_body(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(GREENHOUSE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 77,
                        "title": "  Backend Engineer ",
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/77",
                        "location": {"name": "Remote (US)"},
                        "content": "<p>We use <b>PostgreSQL</b></p><li>sharding</li>",
                        "first_published": "2026-10-01T09:00:00Z",
                        "departments": [{"name": "Engineering"}],
                    },
                    {"id": 78, "title": "", "content": "no title at all"},
                ]
            },
        )
    )

    with httpx.Client() as client:
        jobs = GreenhouseAdapter().fetch_board(client, "acme")

    assert len(jobs) == 1
    job = jobs[0]
    assert (job.uid, job.title) == ("greenhouse:77", "Backend Engineer")
    assert job.company == "Acme"
    assert job.is_remote is True
    assert job.department == "Engineering"
    assert "<" not in job.description and "PostgreSQL" in job.description
    assert job.published_at is not None and job.published_at.year == 2026


def test_bdtechjobs_row_uses_its_own_field_names(respx_mock: respx.MockRouter) -> None:
    """Its keys are `designation` and `applyLink`, not the obvious ones.

    Reading a wrong key here does not raise: every row loses its title, gets filtered
    out, and the board is reported dead. That is how this adapter shipped broken for a
    day while the live collector answered 200.
    """
    respx_mock.get(BDTECHJOBS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 42,
                        "designation": "Backend Engineer",
                        "companyName": "Relisource",
                        "applyLink": "https://relisource.example/job/42",
                        "location": "Gulshan-1, Dhaka",
                        "jobType": "Remote",
                        "experience": "2-3 years",
                        "experienceYears": "2-3 years",
                        "seniorityLevel": "Mid level",
                        "applicationDeadline": "2026-10-31",
                        "postedDate": "Not available",
                        "description": "<p>Owns the <b>PostgreSQL</b> estate</p>",
                        "jobCategory": ["System Administrator"],
                        "requiredSkills": ["Vmware", "Windows Server"],
                        "salary": "Competitive",
                    },
                    {"id": 43, "companyName": "a row with no designation"},
                ],
                "total": 55,
                "page": 1,
                "limit": 50,
            },
        )
    )

    with httpx.Client() as client:
        jobs = BdTechJobsAdapter().fetch_board(client, "1")

    assert len(jobs) == 1
    job = jobs[0]
    assert (job.uid, job.title, job.company) == (
        "bdtechjobs:42",
        "Backend Engineer",
        "Relisource",
    )
    assert (job.employment_type, job.seniority, job.department) == (
        "Remote",
        "Mid level",
        "System Administrator",
    )
    assert job.is_remote is True
    assert job.apply_url == "https://relisource.example/job/42"
    assert job.published_at is None
    # Skills and years have to survive into the prose, or the scorer has nothing to
    # read them from.
    assert "PostgreSQL" in job.description and "Vmware" in job.description
    assert "2-3 years" in job.description and "<" not in job.description


def test_a_multiline_designation_is_collapsed_to_one_line(
    respx_mock: respx.MockRouter,
) -> None:
    """The board sometimes puts the whole listing card in `designation`.

    Left raw, the embedded newlines break the tab-separated CLI row, and the content
    hash ends up keyed on the board's line wrapping rather than on its content.
    """
    respx_mock.get(BDTECHJOBS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 9,
                        "designation": "Backend\n\nEngineer\n\nOnsite\n\n/\n\nFull Time",
                        "companyName": "Bdtask",
                    }
                ],
                "total": 1,
                "page": 1,
                "limit": 50,
            },
        )
    )

    with httpx.Client() as client:
        jobs = BdTechJobsAdapter().fetch_board(client, "1")

    assert jobs[0].title == "Backend Engineer Onsite / Full Time"


def test_every_curated_board_has_an_adapter() -> None:
    """A board with a mistyped `source` is skipped in silence, not in error.

    `collect` looks the adapter up by name and `continue`s when the name is unknown, so
    the board simply never appears again and nothing says why. This is the only check
    that a curated token is reachable by code at all; whether the endpoint still answers
    is what `board_health` is for.
    """
    unbacked = sorted({board.source for board in BOARDS} - set(ADAPTERS))

    assert unbacked == []


def test_greenhouse_metadata_flags_remote_when_the_location_does_not(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(GREENHOUSE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Backend Engineer",
                        "location": {"name": "Berlin"},
                        "metadata": [{"name": "Remote", "value": "yes"}],
                    }
                ]
            },
        )
    )

    with httpx.Client() as client:
        jobs = GreenhouseAdapter().fetch_board(client, "acme")

    assert jobs[0].is_remote is True


def test_ashby_skips_unlisted_jobs_and_keeps_compensation(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(ASHBY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": "a1",
                        "title": "AI Engineer",
                        "isListed": False,
                        "jobUrl": "https://jobs.example/a1",
                    },
                    {
                        "jobId": "a2",
                        "title": "AI Engineer",
                        "jobUrl": "https://jobs.example/a2",
                        "organizationName": "Acme",
                        "address": {"postalAddress": {"country": "Bangladesh"}},
                        "compensation": {"compensationTierSummary": "$120k - $180k"},
                        "descriptionHtml": "<p>Build agents</p>",
                    },
                ]
            },
        )
    )

    with httpx.Client() as client:
        jobs = AshbyAdapter().fetch_board(client, "acme")

    assert [j.source_id for j in jobs] == ["a2"]
    assert jobs[0].location == "Bangladesh"
    assert jobs[0].compensation == "$120k - $180k"


def test_lever_reads_nested_categories(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(LEVER_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "lv1",
                    "text": "Software Engineer",
                    "hostedUrl": "https://jobs.example/lv1",
                    "categories": {"location": "Remote", "department": "Platform"},
                    "createdAt": 1759000000000,
                },
                {"text": "No id, no dedupe key"},
            ],
        )
    )

    with httpx.Client() as client:
        jobs = LeverAdapter().fetch_board(client, "acme")

    assert len(jobs) == 1
    assert jobs[0].is_remote is True
    assert jobs[0].department == "Platform"
    assert jobs[0].published_at is not None


def test_a_404_is_a_dead_board_not_a_retry(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    route = respx_mock.get(GREENHOUSE_URL).mock(return_value=httpx.Response(404))

    with httpx.Client() as client:
        assert get_json(client, GREENHOUSE_URL) is None

    assert route.call_count == 1
    assert _no_sleep == []


def test_a_server_error_is_retried_then_given_up_on(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    route = respx_mock.get(GREENHOUSE_URL).mock(return_value=httpx.Response(503))

    with httpx.Client() as client:
        assert get_json(client, GREENHOUSE_URL) is None

    assert route.call_count == 3
    assert _no_sleep == [1, 2, 4]


def test_rate_limit_waits_for_the_advertised_delay(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    respx_mock.get(GREENHOUSE_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(200, json={"jobs": []}),
        ]
    )

    with httpx.Client() as client:
        assert get_json(client, GREENHOUSE_URL) == {"jobs": []}

    assert _no_sleep == [3.0]


def test_collect_keeps_only_the_roles_we_want_and_reports_the_board(
    respx_mock: respx.MockRouter, store: JobStore
) -> None:
    respx_mock.get(GREENHOUSE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {"id": 1, "title": "Backend Engineer", "content": "<p>Python</p>"},
                    {"id": 2, "title": "Engineering Recruiter", "content": "<p>hello</p>"},
                ]
            },
        )
    )

    report = collect([Board("greenhouse", "acme", "Acme", True)], store=store)

    assert (report.fetched, report.kept, report.inserted) == (2, 1, 1)
    assert report.boards_ok == 1
    assert [j.title for j in store.all()] == ["Backend Engineer"]


def test_collect_survives_a_dead_board_and_remembers_it(
    respx_mock: respx.MockRouter, store: JobStore
) -> None:
    respx_mock.get(GREENHOUSE_URL).mock(return_value=httpx.Response(404))

    report = collect([Board("greenhouse", "acme", "Acme")], store=store)

    assert report.fetched == 0
    assert report.boards_dead == ["greenhouse:acme"]
    assert store.dead_boards() == [("greenhouse", "acme")]


@pytest.mark.parametrize(
    ("title", "wanted"),
    [
        ("Senior Backend Engineer", True),
        ("Machine Learning Engineer", True),
        ("Engineering Recruiter", False),
        ("Backend Engineering Manager", False),
        ("Sales Engineer", False),
        ("Staff Software Engineer", False),
        # The Bangladeshi market titles the same work differently, and the first
        # version of this list was written from US boards. Every True below is a live
        # bdtechjobs title the original filter silently threw away.
        ("DevOps Engineer", True),
        ("Senior Cloud Engineer", True),
        ("Mid to Senior Level Python Developer", True),
        ("Sr. Frontend Developer (React JS)", True),
        ("AI Systems Engineer", True),
        ("Senior SecOps Engineer", True),
        ("Cybersecurity Expert", True),
        ("Database Administrator", False),
    ],
)
def test_the_title_filter_is_code_not_a_model(title: str, wanted: bool) -> None:
    assert wanted_title(title) is wanted


def test_the_same_posting_twice_is_one_row_and_an_edit_updates_in_place(store: JobStore) -> None:
    first = _posting(title="Backend Engineer", description="Python")
    assert store.upsert(first) == ("greenhouse:1", 1)
    assert store.upsert(first) == ("greenhouse:1", 0)
    assert store.count() == 1

    edited = _posting(title="Backend Engineer", description="Python and Go")
    assert store.upsert(edited) == ("greenhouse:1", 0)
    assert store.count() == 1
    assert "Go" in (store.get("greenhouse:1") or first).description


def test_search_finds_the_body_not_the_title(store: JobStore) -> None:
    store.upsert(_posting(uid="greenhouse:1", title="Engineer", description="pgvector similarity"))
    store.upsert(_posting(uid="greenhouse:2", title="Cook", description="pancakes"))

    hits = store.search("pgvector")

    assert [j.uid for j in hits] == ["greenhouse:1"]


@pytest.mark.parametrize("query", ['"unbalanced', "a AND b", "site:(*)", "NOT", ":", "é"])
def test_a_hostile_query_cannot_break_the_search(store: JobStore, query: str) -> None:
    store.upsert(_posting())

    assert isinstance(store.search(query), list)


def test_fts_query_quotes_every_token() -> None:
    assert fts_query('pgvector "AND" escape') == '"pgvector" AND "AND" AND "escape"'
    assert fts_query("drop table jobs; --") == '"drop" AND "table" AND "jobs;" AND "--"'
    assert fts_query("") == '""'
    assert len(fts_query(" ".join(str(n) for n in range(50))).split(" AND ")) == 12


def test_scores_are_persisted_against_the_posting(store: JobStore, profile: Any) -> None:
    from careerpilot.match.extract import ExtractedRequirement, RequirementSet  # noqa: PLC0415
    from careerpilot.match.score import score_job  # noqa: PLC0415

    job = _posting()
    store.upsert(job)
    requirements = RequirementSet(
        requirements=[
            ExtractedRequirement(text="Python services", skill="Python", category="must_have")
        ],
        seniority="unspecified",
        location_constraint="unspecified",
    )

    store.save_score(score_job(job, requirements, profile))
    store.save_score(score_job(job, requirements, profile))

    row = sqlite3.connect(store.path).execute("SELECT COUNT(*) FROM scores").fetchone()
    assert row[0] == 1


def _posting(
    uid: str = "greenhouse:1", title: str = "Backend Engineer", description: str = "Python"
) -> JobPosting:
    source, _, source_id = uid.partition(":")
    return JobPosting(
        source=source,
        source_id=source_id,
        company="Acme",
        title=title,
        description=description,
        url=f"https://example.com/{uid}",
    )
