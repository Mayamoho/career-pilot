"""Adapters for the ATS providers with genuinely public, keyless job feeds.

Verified shapes (October 2026):
  Greenhouse  https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
  Ashby       https://api.ashbyhq.com/posting-api/job-board/{name}?includeCompensation=true
  Lever       https://api.lever.co/v0/postings/{site}?mode=json
  bdjobs      https://gateway.bdjobs.com/…/JobSearch/GetJobSearch  (+ jobDetails?jobId=)
  Jobicy      https://jobicy.com/api/v2/remote-jobs?geo={slug}

The first three are per-company; there is no cross-company search and no directory of
tokens, so the curated token list in `boards.py` — not the fetching — is the real asset.
Ashby is the highest-yield of the three because most AI companies moved there; Lever is
kept because it is cheap, but its install base has thinned considerably.

bdjobs and Jobicy are aggregated: one endpoint, thousands of employers. They also fail in
opposite directions, which is the reason both are written out rather than assumed.
bdjobs answers `200` with unfiltered rows for any parameter it does not know, so a filter
we believe is applied may not be. Jobicy answers `400 Unexpected parameter` for the same
mistake, so a typo fails loudly instead of quietly widening the result set.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from careerpilot.config import BDJOBS_DETAIL_LIMIT, BDJOBS_DETAIL_SLEEP_S
from careerpilot.domain.job import JobPosting, coerce_datetime, html_to_text
from careerpilot.ingest.base import get_json
from careerpilot.ingest.filter import wanted_title

_ONE_LINE = re.compile(r"\s+")

# Jobicy's `salaryPeriod` is prose, not an enum: live rows say "monthly" and "yearly".
_PERIODS = {
    "year": "per year", "yearly": "per year", "annum": "per year",
    "month": "per month", "monthly": "per month",
    "hour": "per hour", "hourly": "per hour",
}


class GreenhouseAdapter:
    name = "greenhouse"
    BASE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]:
        payload = get_json(client, self.BASE.format(token=token), params={"content": "true"})
        if not isinstance(payload, dict):
            return []
        org = company or token.replace("-", " ").title()
        out: list[JobPosting] = []
        for row in payload.get("jobs", []) or []:
            if not isinstance(row, dict):
                continue
            absolute = str(row.get("absolute_url") or "")
            location = row.get("location")
            out.append(
                JobPosting(
                    source="greenhouse",
                    source_id=str(row.get("id") or row.get("internal_job_id") or absolute),
                    company=org,
                    title=str(row.get("title") or "").strip(),
                    description=html_to_text(row.get("content")),
                    location=location.get("name") if isinstance(location, dict) else None,
                    is_remote=_gh_is_remote(row),
                    url=absolute,
                    apply_url=absolute,
                    published_at=coerce_datetime(
                        row.get("first_published") or row.get("updated_at")
                    ),
                    updated_at=coerce_datetime(row.get("updated_at")),
                    department=_first(row.get("departments")),
                )
            )
        return [j for j in out if j.title]


class AshbyAdapter:
    name = "ashby"
    BASE = "https://api.ashbyhq.com/posting-api/job-board/{name}"

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]:
        payload = get_json(
            client, self.BASE.format(name=token), params={"includeCompensation": "true"}
        )
        if not isinstance(payload, dict):
            return []
        org = company or token.replace("-", " ").title()
        out: list[JobPosting] = []
        for row in payload.get("jobs", []) or []:
            if not isinstance(row, dict) or not row.get("isListed", True):
                continue
            job_url = str(row.get("jobUrl") or "")
            out.append(
                JobPosting(
                    source="ashby",
                    source_id=str(row.get("jobId") or row.get("id") or job_url),
                    company=str(row.get("organizationName") or org),
                    title=str(row.get("title") or "").strip(),
                    description=html_to_text(
                        row.get("descriptionHtml") or row.get("descriptionPlain")
                    ),
                    location=row.get("location") or _ashby_country(row.get("address")),
                    is_remote=bool(row.get("isRemote")),
                    department=row.get("department"),
                    team=row.get("team"),
                    employment_type=row.get("employmentType"),
                    url=job_url,
                    apply_url=str(row.get("applyUrl") or job_url),
                    published_at=coerce_datetime(row.get("publishedAt")),
                    updated_at=coerce_datetime(row.get("publishedAt")),
                    compensation=_ashby_compensation(row.get("compensation")),
                )
            )
        return [j for j in out if j.title and j.source_id]


class LeverAdapter:
    name = "lever"
    BASE = "https://api.lever.co/v0/postings/{site}"

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]:
        payload = get_json(client, self.BASE.format(site=token), params={"mode": "json"})
        if not isinstance(payload, list):
            return []
        org = company or token.replace("-", " ").title()
        out: list[JobPosting] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            categories = _dict_field(row, "categories")
            lists = _dict_field(row, "lists")
            hosted = str(row.get("hostedUrl") or "")
            out.append(
                JobPosting(
                    source="lever",
                    source_id=str(row.get("id") or row.get("applyUrl") or ""),
                    company=org,
                    title=str(row.get("text") or "").strip(),
                    description=html_to_text(
                        row.get("description") or row.get("descriptionPlain")
                    ),
                    location=categories.get("location"),
                    is_remote=_lever_is_remote(categories, lists),
                    department=categories.get("department"),
                    team=categories.get("team"),
                    employment_type=row.get("workplaceType"),
                    url=hosted,
                    apply_url=str(row.get("applyUrl") or hosted),
                    published_at=coerce_datetime(row.get("createdAt")),
                    updated_at=coerce_datetime(row.get("lastApplied")),
                )
            )
        return [j for j in out if j.title and j.source_id]


class BdTechJobsAdapter:
    """The only genuinely public Bangladeshi tech board with a documented JSON API.

    Its field names are its own — `designation`, `applyLink`, `postedDate` — so they
    are read literally rather than guessed at. A wrong key here does not raise, it
    silently returns an empty board, which the collector then reports as dead.
    """

    name = "bdtechjobs"
    BASE = "https://www.bdtechjobs.com/api/jobs"

    def fetch_board(
        self, client: httpx.Client, token: str = "", company: str | None = None
    ) -> list[JobPosting]:
        payload = get_json(client, self.BASE, params={"page": _page(token), "limit": 50})
        if not isinstance(payload, dict):
            return []
        out: list[JobPosting] = []
        for row in payload.get("jobs", []) or []:
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("id") or "")
            title = _one_line(row.get("designation"))
            if not source_id or not title:
                continue
            url = str(row.get("applyLink") or "")
            deadline = str(row.get("applicationDeadline") or "")
            out.append(
                JobPosting(
                    source="bdtechjobs",
                    source_id=source_id,
                    company=str(row.get("companyName") or company or ""),
                    title=title,
                    description=html_to_text(_bd_body(row, deadline)),
                    location=row.get("location"),
                    is_remote=str(row.get("jobType") or "").lower()
                    in {"remote", "work from home", "wfh"},
                    department=_join(row.get("jobCategory")),
                    employment_type=str(row.get("jobType") or "").strip() or None,
                    seniority=str(row.get("seniorityLevel") or "").strip() or None,
                    url=url,
                    apply_url=url,
                    published_at=coerce_datetime(row.get("postedDate")),
                    deadline=coerce_datetime(deadline),
                )
            )
        return out


class BdJobsAdapter:
    """bdjobs.com — the Bangladeshi market, through the endpoints its own site calls.

    There is no documented API here, so two things are enforced rather than trusted.
    The detail endpoint is one request per job, so it is called only for rows that pass
    the title filter and only up to `BDJOBS_DETAIL_LIMIT` of them, with a pause between:
    this is somebody's production server answering a personal job search. And the field
    names are read exactly as measured, including `CompnayName`, which is misspelled in
    their own response — a guessed key here does not raise, it silently empties the board.
    """

    name = "bdjobs"
    LIST = "https://gateway.bdjobs.com/recruitment-account-test/api/JobSearch/GetJobSearch"
    DETAIL = "https://gateway.bdjobs.com/ActtivejobsTest/api/JobSubsystem/jobDetails"
    PAGE = "https://jobs.bdjobs.com/jobdetails.asp?id={id}&ln=0"

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]:
        payload = get_json(client, self.LIST, params=_bdjobs_params(token))
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []

        out: list[JobPosting] = []
        detailed = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            posting = _bdjobs_posting(row, company)
            if posting is None:
                continue
            # The list row carries a one-line description; fit is scored on prose, so the
            # body is fetched only for postings that would survive `collect` anyway.
            if detailed < BDJOBS_DETAIL_LIMIT and wanted_title(posting.title):
                detail = self._detail(client, posting.source_id)
                if detail is not None:
                    posting = posting.model_copy(update={"description": _bdjobs_full_body(
                        posting.description, detail
                    )})
                    if detail.get("JobLocation"):
                        posting = posting.model_copy(update={
                            "location": str(detail["JobLocation"]),
                        })
                detailed += 1
                time.sleep(BDJOBS_DETAIL_SLEEP_S)
            out.append(posting)
        return out

    def _detail(self, client: httpx.Client, job_id: str) -> dict[str, Any] | None:
        payload = get_json(client, self.DETAIL, params={"jobId": job_id}, retries=1)
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        return data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None


class JobicyAdapter:
    """Jobicy — remote-only, with the reachable region and the seniority stated as fields.

    Every row is remote, so `location` is the region the employer named and `is_remote`
    is never a claim that has to be checked against prose. `jobLevel` is the reason this
    source earns its place in a fresher's tool: it is the only feed here that states
    "Entry-Level, Junior" as data instead of hoping the title says so.

    The token is a `geo` slug. Unknown slugs answer 400 and `get_json` turns that into an
    empty board, which `collect` then reports as dead — a wrong token is visible, not
    silently broad.
    """

    name = "jobicy"
    BASE = "https://jobicy.com/api/v2/remote-jobs"
    PAGES = 2

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]:
        out: list[JobPosting] = []
        cursor: str | None = None
        for _ in range(self.PAGES):
            params: dict[str, Any] = {"geo": token}
            if cursor:
                params["cursor"] = cursor
            payload = get_json(client, self.BASE, params=params)
            if not isinstance(payload, dict):
                return out
            for row in payload.get("jobs") or []:
                if isinstance(row, dict):
                    posting = _jobicy_posting(row, company)
                    if posting is not None:
                        out.append(posting)
            cursor = payload.get("nextCursor")
            if not cursor or not payload.get("hasMore"):
                break
        return out


def _bdjobs_params(token: str) -> dict[str, Any]:
    """A page number, or `keyword:<stack>`.

    Measured against the live endpoint: `keyword`, `pg` and `rpp` change the rows that
    come back; `category`, `location`, `jobType` and `sort` are accepted and ignored.
    So narrowing by stack is the only server-side filter worth asking for, and everything
    else is done in code after the rows land here.
    """
    params: dict[str, Any] = {"isPro": "1", "rpp": 50, "pg": 1}
    if token.startswith("keyword:"):
        params["keyword"] = token.split(":", 1)[1].strip().lower()
    elif token.isdigit():
        params["pg"] = int(token)
    return params


def _bdjobs_posting(row: dict[str, Any], company: str | None = None) -> JobPosting | None:
    job_id = str(row.get("Jobid") or "").strip()
    title = _one_line(row.get("jobTitle"))
    if not job_id or not title:
        return None
    page = BdJobsAdapter.PAGE.format(id=job_id)
    workplace = str(row.get("WorkPlace") or "").strip()
    salary = row.get("Salary")
    return JobPosting(
        source="bdjobs",
        source_id=job_id,
        company=_one_line(row.get("companyName")) or company or "bdjobs employer",
        title=title,
        description=_bdjobs_list_body(row, salary),
        location=str(row.get("location") or "").strip() or None,
        is_remote=workplace.lower() in {"home", "remote", "work from home"},
        employment_type=str(row.get("JobType") or "").strip() or None,
        seniority=str(row.get("experience") or "").strip() or None,
        url=page,
        apply_url=page,
        published_at=coerce_datetime(row.get("publishDate")),
        deadline=coerce_datetime(row.get("deadlineDB")),
        compensation=_bdjobs_salary(salary),
    )


def _bdjobs_salary(salary: Any) -> str | None:
    if not isinstance(salary, dict):
        return None
    low, high = salary.get("MinSalary"), salary.get("MaxSalary")
    if not low and not high:
        return None
    if low and high and low != high:
        return f"Tk. {low} - {high}"
    return f"Tk. {low or high}"


def _fold(fields: dict[str, str]) -> str:
    """Render `label: value` lines, skipping blanks and exact repeats.

    bdjobs sends the same sentence three ways: the list row's `jobDescription` is a copy
    of its `eduRec`, and the detail endpoint restates the salary with the period attached.
    Duplicated prose is duplicated evidence — the scorer counts term hits, so a paragraph
    pasted twice reads as a requirement mentioned twice.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for label, value in fields.items():
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text if not label else f"{label}: {text}")
    return "\n".join(lines)


def _bdjobs_list_body(row: dict[str, Any], salary: Any) -> str:
    """Fold the structured list fields into prose, because fit is scored on the text."""
    return _fold({
        "": html_to_text(str(row.get("jobDescription") or "")),
        "Education": html_to_text(str(row.get("eduRec") or "")),
        "Additional requirements": html_to_text(str(row.get("jobContext") or "")),
        "Experience": str(row.get("experience") or ""),
        "Job type": str(row.get("JobType") or ""),
        "Work place": str(row.get("WorkPlace") or ""),
        "Location": str(row.get("location") or ""),
        "Salary": _bdjobs_salary(salary) or "",
        "Vacancies": str(row.get("Vacancies") or ""),
        "Application deadline": str(row.get("deadline") or ""),
    })


def _bdjobs_full_body(existing: str, detail: dict[str, Any]) -> str:
    """Append the detail endpoint's prose, which the list row does not carry.

    The list description is a card; the requirement text a scorer needs lives in
    `JobDescription`, `AdditionJobRequirements` and `SkillsRequired` on the other host.
    Both are kept: the first states the money and the deadline, the second states the job.
    """
    body = _fold({
        "": html_to_text(str(detail.get("JobDescription") or "")),
        "Additional requirements": html_to_text(
            str(detail.get("AdditionJobRequirements") or "")
        ),
        "Education": html_to_text(str(detail.get("EducationRequirements") or "")),
        "Skills required": _join(detail.get("SkillsRequired")),
        "Suggested skills": _join(detail.get("SuggestedSkills")),
        "Age": str(detail.get("Age") or ""),
        "Work place": str(detail.get("JobWorkPlace") or ""),
        "Salary": str(detail.get("JobSalaryRangeText") or ""),
    })
    return f"{existing}\n\n{body}" if existing and body else (body or existing)


def _jobicy_posting(row: dict[str, Any], company: str | None = None) -> JobPosting | None:
    job_id = str(row.get("id") or "").strip()
    title = _one_line(row.get("jobTitle"))
    url = str(row.get("url") or "")
    if not job_id or not title:
        return None
    return JobPosting(
        source="jobicy",
        source_id=job_id,
        company=_one_line(row.get("companyName")) or company or "Jobicy employer",
        title=title,
        description=html_to_text(
            str(row.get("jobDescription") or row.get("jobExcerpt") or "")
        ),
        location=str(row.get("jobGeo") or "").strip() or "Remote",
        is_remote=True,
        department=_join(row.get("jobIndustry")),
        employment_type=_join(row.get("jobType")) or None,
        seniority=str(row.get("jobLevel") or "").strip() or None,
        url=url,
        apply_url=url,
        published_at=coerce_datetime(row.get("pubDate")),
        compensation=_jobicy_salary(row),
    )


def _jobicy_salary(row: dict[str, Any]) -> str | None:
    low, high = _salary_amount(row.get("salaryMin")), _salary_amount(row.get("salaryMax"))
    if not low and not high:
        return None
    currency = str(row.get("salaryCurrency") or "USD")
    span = (
        f"{currency} {low} - {high}" if low and high and low != high
        else f"{currency} {low or high}"
    )
    # The field is `"monthly"`/`"yearly"`, not `"month"`/`"year"` — measured on live rows.
    period = _PERIODS.get(str(row.get("salaryPeriod") or "").strip().lower())
    return f"{span} {period}" if period else span


def _salary_amount(value: Any) -> str:
    """Jobicy writes the literal string `"None"` into `salaryMax` on open-ended pay."""
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in {"", "0", "none", "null"} else text


def _one_line(value: Any) -> str:
    """Collapse a `designation` to a single line.

    bdtechjobs sometimes returns the whole listing card in this field — title, work
    mode and seniority separated by blank lines. Left as-is it breaks the tab-separated
    CLI output and makes the content hash depend on the board's line wrapping.
    """
    return _ONE_LINE.sub(" ", str(value or "")).strip()


def _bd_body(row: dict[str, Any], deadline: str) -> str:
    """Fold the structured fields into prose, because fit is scored on the text."""
    fields = {
        "": str(row.get("description") or ""),
        "Required skills": _join(row.get("requiredSkills")),
        "Experience": str(row.get("experienceYears") or row.get("experience") or ""),
        "Salary": str(row.get("salary") or ""),
        "Application deadline": deadline,
    }
    return "\n".join(
        value if not label else f"{label}: {value}" for label, value in fields.items() if value
    )


def _join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _page(token: str) -> int:
    return int(token) if token.isdigit() else 1


def _dict_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _first(value: Any) -> str | None:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0].get("name")
    return None


def _ashby_country(address: Any) -> str | None:
    if not isinstance(address, dict):
        return None
    postal = address.get("postalAddress")
    return postal.get("country") if isinstance(postal, dict) else None


def _gh_is_remote(row: dict[str, Any]) -> bool:
    location = row.get("location")
    name = location.get("name") if isinstance(location, dict) else None
    if name and "remote" in str(name).lower():
        return True
    for item in row.get("metadata") or []:
        if isinstance(item, dict) and str(item.get("name", "")).lower() == "remote":
            return str(item.get("value", "")).lower() in {"yes", "true"}
    return False


def _lever_is_remote(categories: dict[str, Any], lists: dict[str, Any]) -> bool:
    if "remote" in str(categories.get("location", "")).lower():
        return True
    return "remote" in str(lists.get("allLocations", "")).lower()


def _ashby_compensation(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("compensationTierSummary") or value.get("summary")
    return str(summary) if summary else None
