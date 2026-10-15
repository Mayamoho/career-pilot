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
    BdJobsAdapter,
    BdTechJobsAdapter,
    GreenhouseAdapter,
    JobicyAdapter,
    LeverAdapter,
)
from careerpilot.ingest.base import get_json
from careerpilot.ingest.boards import ADAPTERS, BOARDS, Board
from careerpilot.ingest.filter import wanted_title
from careerpilot.ingest.store import JobStore, fts_query
from careerpilot.pipeline import collect

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/acme"
LEVER_URL = "https://api.lever.co/v0/postings/acme"
BDTECHJOBS_URL = "https://www.bdtechjobs.com/api/jobs"
BDJOBS_LIST_URL = "https://gateway.bdjobs.com/recruitment-account-test/api/JobSearch/GetJobSearch"
BDJOBS_DETAIL_URL = "https://gateway.bdjobs.com/ActtivejobsTest/api/JobSubsystem/jobDetails"
JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"


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


BDJOBS_ROW = {
    "Jobid": "1542177",
    "jobTitle": "Python Automation Intern",
    "companyName": "Marive",
    "location": "Dhanmondi",
    "WorkPlace": "Home",
    "JobType": "Intern",
    "Salary": {
        "SalaryRange": None, "MinSalary": 12000, "MaxSalary": 24000,
        "SalaryType": None, "IsNegotiable": False, "HideSalary": False,
    },
    "experience": "NA",
    "eduRec": "Higher Secondary\n<ul><li>English Spoken and writing is must to have</li></ul>",
    "jobDescription": "Higher Secondary\n<ul><li>English Spoken and writing is a must</li></ul>",
    "jobContext": None,
    "deadline": "Oct 25, 2026",
    "deadlineDB": "2026-10-25T00:00:00Z",
    "publishDate": "2026-10-07T05:27:00Z",
    "Vacancies": 3,
    "OnlineJob": True,
}

# Trimmed from the live response, keys verbatim — including `CompnayName`, which their
# own API misspells, and `SkillsRequired`, which is a comma-separated string rather than
# the list the field name suggests.
BDJOBS_DETAIL_ROW = {
    "test": 1, "JobId": "1542177", "JobFound": "True", "error": "0",
    "CompnayName": "Marive", "JobTitle": "Python Automation Intern",
    "JobDescription": "<p><strong>About the Role</strong></p><p>We are looking for a "
                      "technically curious intern to test open-source tools, maintain "
                      "Python projects, and help automate business processes.</p>",
    "AdditionJobRequirements": "<ul><li>Basic Python skills</li><li>Familiarity with "
                               "GitHub</li><li>Bonus: Experience with APIs, automation, "
                               "or AI tools</li></ul>",
    "EducationRequirements": "<ul><li>Higher Secondary</li></ul>",
    "SkillsRequired": "Communication and interpersonal skill, Fluent in English, "
                      "Git and Github, JavaScript, Problem solving and decision "
                      "making, Python",
    "SuggestedSkills": "Python Django, Network Security, Automation",
    "JobLocation": "Anywhere in Bangladesh",
    "JobWorkPlace": "Work from home",
    "JobNature": "Full Time",
    "JobSalaryRangeText": "Tk. 12000 - 24000 (Monthly)",
    "Deadline": "Oct 25, 2026",
    "PostedOn": "Oct 7, 2026",
    "Age": "18 to 32 years",
    "Gender": "M,F,O",
    "JobVacancies": "3",
    "OnlineApply": "True",
}

# A live Jobicy row: `jobLevel` states the seniority as data, `jobType` is a real list,
# `salaryPeriod` is the word "monthly", and the whole body arrives as HTML.
JOBICY_ROW = {
    "id": 153136,
    "url": "https://jobicy.com/jobs/153136-compliance-engineer-intern",
    "jobSlug": "153136-compliance-engineer-intern",
    "jobTitle": "Compliance Engineer Intern",
    "companyName": "CertiK",
    "jobIndustry": ["Legal & Compliance"],
    "jobType": ["Internship"],
    "jobGeo": "APAC,  Europe",
    "jobLevel": "Entry-Level, Junior",
    "jobExcerpt": "CertiK is a leading Web3 security company&hellip;",
    "jobDescription": "<p>CertiK is a leading Web3 security company focused on securing "
                      "<a href=\"https://jobicy.com/company/certik\">blockchain "
                      "protocols</a> and smart contracts.</p>",
    "pubDate": "2026-10-13T02:48:33+00:00",
}

JOBICY_PAID_ROW = {
    "id": 155109,
    "url": "https://jobicy.com/jobs/155109-construction-estimating-career-coach",
    "jobTitle": "Backend Engineer",
    "companyName": "CourseCareers",
    "jobIndustry": ["Software Engineering"],
    "jobType": ["Part-Time"],
    "jobGeo": "Anywhere",
    "jobLevel": "Entry-Level, Junior",
    "jobDescription": "<p>Build services.</p>",
    "pubDate": "2026-10-12T11:51:33+00:00",
    "salaryMin": 50, "salaryMax": 1000, "salaryCurrency": "USD", "salaryPeriod": "monthly",
}


def _bdjobs_list(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"message": "Success", "statuscode": 200, "data": rows, "premiumData": []}


def _jobicy_page(rows: list[dict[str, Any]], cursor: str | None = None) -> dict[str, Any]:
    return {
        "apiVersion": "v2", "jobCount": len(rows), "hasMore": bool(cursor),
        "nextCursor": cursor, "appliedFilters": {}, "statusCode": 200, "success": True,
        "jobs": rows,
    }


def test_bdjobs_row_becomes_a_posting_and_buys_its_prose_from_the_detail_endpoint(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    """The list row is a card; the requirements text lives on a second host.

    Fit is scored on prose, so without the detail call every bdjobs posting would be
    judged on three lines of HTML — and the board would look confidently unsuitable.
    """
    respx_mock.get(BDJOBS_LIST_URL).mock(
        return_value=httpx.Response(200, json=_bdjobs_list([BDJOBS_ROW]))
    )
    respx_mock.get(BDJOBS_DETAIL_URL).mock(
        return_value=httpx.Response(200, json={"statuscode": 200, "data": [BDJOBS_DETAIL_ROW]})
    )

    with httpx.Client() as client:
        jobs = BdJobsAdapter().fetch_board(client, "1")

    assert len(jobs) == 1
    job = jobs[0]
    assert job.uid == "bdjobs:1542177"
    assert (job.title, job.company) == ("Python Automation Intern", "Marive")
    assert job.is_remote is True
    assert job.employment_type == "Intern"
    assert job.url == "https://jobs.bdjobs.com/jobdetails.asp?id=1542177&ln=0"
    assert job.compensation == "Tk. 12000 - 24000"
    # `deadline` is why this build can say "closes in 9 days": no Western feed publishes one.
    assert job.deadline is not None and job.deadline.strftime("%Y-%m-%d") == "2026-10-25"
    assert job.published_at is not None and job.published_at.month == 10
    # The detail's `JobLocation` outranks the list's branch city: "Dhanmondi" is where the
    # office is, "Anywhere in Bangladesh" is where the work happens.
    assert job.location == "Anywhere in Bangladesh"
    assert "<" not in job.description
    for needle in (
        "technically curious intern", "Basic Python skills", "Git and Github",
        "Python Django, Network Security, Automation", "Tk. 12000 - 24000 (Monthly)",
        "18 to 32 years", "Application deadline: Oct 25, 2026", "Vacancies: 3",
    ):
        assert needle in job.description


def test_bdjobs_spends_its_detail_budget_only_on_titles_worth_reading(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    """Twelve detail requests per board token is a hard ceiling, not a hint.

    bdjobs has no public API contract, so this is somebody's production server answering
    a personal job search. Rows the title filter will drop in `collect` never get a call.
    """
    wanted = [{**BDJOBS_ROW, "Jobid": str(1000 + i)} for i in range(14)]
    noise = [{**BDJOBS_ROW, "Jobid": "2001", "jobTitle": "Graduate Executive"}]
    respx_mock.get(BDJOBS_LIST_URL).mock(
        return_value=httpx.Response(200, json=_bdjobs_list(wanted + noise))
    )
    detail = respx_mock.get(BDJOBS_DETAIL_URL).mock(
        return_value=httpx.Response(200, json={"statuscode": 200, "data": [BDJOBS_DETAIL_ROW]})
    )

    with httpx.Client() as client:
        jobs = BdJobsAdapter().fetch_board(client, "1")

    assert len(jobs) == 15
    assert detail.call_count == 12
    assert _no_sleep == [0.6] * 12


def test_bdjobs_sends_the_parameters_it_measured_and_never_the_ignored_ones(
    respx_mock: respx.MockRouter,
) -> None:
    """`keyword`, `pg` and `rpp` change the rows; `category`/`location`/`jobType` answer
    200 with unfiltered data. Asking for a filter that is silently ignored would make the
    result set look narrower than it is, which is worse than not asking.
    """
    list_route = respx_mock.get(BDJOBS_LIST_URL).mock(
        return_value=httpx.Response(200, json=_bdjobs_list([]))
    )

    with httpx.Client() as client:
        assert BdJobsAdapter().fetch_board(client, "keyword:Python") == []
        assert BdJobsAdapter().fetch_board(client, "3") == []

    first = httpx.URL(list_route.calls[0].request.url)
    assert first.params["keyword"] == "python"
    assert first.params["pg"] == "1" and first.params["rpp"] == "50"
    assert not {"category", "location", "jobType", "sort"} & set(first.params.keys())
    assert httpx.URL(list_route.calls[1].request.url).params["pg"] == "3"


def test_bdjobs_a_bad_token_is_an_empty_board_not_an_exception(
    respx_mock: respx.MockRouter,
) -> None:
    """Their gateway returns 200 with `data: null` more often than it returns an error."""
    respx_mock.get(BDJOBS_LIST_URL).mock(
        return_value=httpx.Response(200, json={"message": "No record found", "data": None})
    )

    with httpx.Client() as client:
        assert BdJobsAdapter().fetch_board(client, "99") == []


def test_bdjobs_hybrid_work_is_not_claimed_as_remote(respx_mock: respx.MockRouter) -> None:
    """"Office,Home" is a real value on this board. Reading it as remote would put an
    office job in the workable-from-Dhaka pile, which is the one mistake `place_of`
    exists to prevent.
    """
    respx_mock.get(BDJOBS_LIST_URL).mock(
        return_value=httpx.Response(
            200, json=_bdjobs_list([{**BDJOBS_ROW, "WorkPlace": "Office,Home"}])
        )
    )
    respx_mock.get(BDJOBS_DETAIL_URL).mock(return_value=httpx.Response(404))

    with httpx.Client() as client:
        jobs = BdJobsAdapter().fetch_board(client, "1")

    assert jobs[0].is_remote is False
    # A missing detail body must not lose the posting: the list row is still a fact.
    assert "Higher Secondary" in jobs[0].description


def test_jobicy_row_labels_seniority_as_data_and_keeps_the_original_url(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(JOBICY_URL).mock(
        return_value=httpx.Response(200, json=_jobicy_page([JOBICY_ROW, JOBICY_PAID_ROW]))
    )

    with httpx.Client() as client:
        jobs = JobicyAdapter().fetch_board(client, "anywhere")

    assert [j.uid for j in jobs] == ["jobicy:153136", "jobicy:155109"]
    intern, paid = jobs
    assert intern.seniority == "Entry-Level, Junior"
    assert intern.is_remote is True
    assert intern.employment_type == "Internship"
    assert intern.department == "Legal & Compliance"
    assert "APAC" in (intern.location or "")
    assert intern.url.startswith("https://jobicy.com/jobs/")
    assert "<" not in intern.description and "blockchain protocols" in intern.description
    assert paid.compensation == "USD 50 - 1000 per month"


def test_jobicy_walks_the_cursor_and_stops_at_the_page_budget(
    respx_mock: respx.MockRouter,
) -> None:
    """`hasMore` stays true forever; the crawl is what bounds it."""
    route = respx_mock.get(JOBICY_URL).mock(
        side_effect=[
            httpx.Response(200, json=_jobicy_page([JOBICY_ROW], cursor="abc123")),
            httpx.Response(200, json=_jobicy_page([JOBICY_PAID_ROW], cursor="def456")),
            httpx.Response(200, json=_jobicy_page([])),
        ]
    )

    with httpx.Client() as client:
        jobs = JobicyAdapter().fetch_board(client, "apac")

    assert len(jobs) == 2
    assert route.call_count == 2
    assert httpx.URL(route.calls[1].request.url).params["cursor"] == "abc123"


def test_jobicy_a_rejected_parameter_is_a_dead_board_not_a_crash(
    respx_mock: respx.MockRouter, _no_sleep: list[float]
) -> None:
    """Jobicy enforces a parameter whitelist: an unknown key answers 400 with a message.

    That is the opposite of bdjobs, which answers 200 and ignores you. Both are handled by
    `get_json` returning None, so `collect` reports the token dead instead of the CLI
    raising in the middle of a scan.
    """
    respx_mock.get(JOBICY_URL).mock(
        return_value=httpx.Response(
            400, json={"success": False, "error": "Unexpected parameter 'limit'. See docs: https://jobi.cy/apidocs"}
        )
    )

    with httpx.Client() as client:
        assert JobicyAdapter().fetch_board(client, "anywhere") == []

    # Retried under the shared policy, then given up on — never raised to the caller.
    assert _no_sleep == [1, 2]


def test_bdtechjobs_publishes_its_deadline_too(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(BDTECHJOBS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 42,
                        "designation": "Backend Engineer",
                        "companyName": "Relisource",
                        "applicationDeadline": "2026-10-31",
                    }
                ]
            },
        )
    )

    with httpx.Client() as client:
        jobs = BdTechJobsAdapter().fetch_board(client, "1")

    assert jobs[0].deadline is not None and jobs[0].deadline.day == 31


def test_a_404_is_a_dead_board_not_a_retry(    respx_mock: respx.MockRouter, _no_sleep: list[float]
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
        # The postings this build is for. `intern` used to sit in the exclude list, so
        # every entry-level title was deleted before anything cost a call — and a
        # final-year student in Dhaka cannot apply to the senior roles that survived.
        # Live bdjobs titles, measured 2026-10-15.
        ("Python Automation Intern", True),
        ("Software Engineer Trainee", True),
        ("Junior Backend Developer (Laravel)", True),
        ("QA Intern", True),
        ("Graduate Executive", False),
        ("Management Trainee (Bank)", False),
        ("Marketing Intern", False),
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
