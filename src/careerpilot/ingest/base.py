"""HTTP plumbing shared by every ATS adapter.

Ingestion is deliberately model-free: a job posting is a fact, and facts are read
from an API, never generated.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import httpx

from careerpilot.domain.job import JobPosting

log = logging.getLogger("careerpilot.ingest")

HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
MAX_RETRY_SLEEP_S = 30.0

USER_AGENT = (
    "CareerPilot/0.1 "
    "(+https://github.com/Mayamoho/career-pilot; personal job search)"
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


class SourceAdapter(Protocol):
    name: str

    def fetch_board(
        self, client: httpx.Client, token: str, company: str | None = None
    ) -> list[JobPosting]: ...


def get_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 2,
) -> Any:
    """Fetch JSON with a small backoff. A 404 means the board token is wrong, which is
    expected noise in a curated token list, so it is not retried and not fatal."""
    last_error: str = ""
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code == HTTP_NOT_FOUND:
                return None
            if response.status_code == HTTP_TOO_MANY_REQUESTS:
                wait = float(response.headers.get("Retry-After", 2**attempt))
                log.warning("rate limited on %s, sleeping %.1fs", url, wait)
                time.sleep(min(wait, MAX_RETRY_SLEEP_S))
                continue
            if response.status_code >= HTTP_SERVER_ERROR:
                last_error = f"HTTP {response.status_code}"
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)
    log.warning("giving up on %s after %s attempt(s): %s", url, retries + 1, last_error)
    return None


def new_client(timeout: float = 25.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True)
