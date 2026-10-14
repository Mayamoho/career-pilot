"""Adapters for the three ATS providers with genuinely public, keyless job feeds.

Verified shapes (October 2026):
  Greenhouse  https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
  Ashby       https://api.ashbyhq.com/posting-api/job-board/{name}?includeCompensation=true
  Lever       https://api.lever.co/v0/postings/{site}?mode=json

Each endpoint is per-company. There is no cross-company search and no directory of
tokens, so the curated token list in `boards.py` — not the fetching — is the real
asset. Ashby is the highest-yield of the three because most AI companies moved there;
Lever is kept because it is cheap, but its install base has thinned considerably.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from careerpilot.domain.job import JobPosting, coerce_datetime, html_to_text
from careerpilot.ingest.base import get_json

_ONE_LINE = re.compile(r"\s+")


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
                )
            )
        return out


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
