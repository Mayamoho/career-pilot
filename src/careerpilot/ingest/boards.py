"""Curated board tokens. This file is the actual moat.

Every public ATS feed is per-company and there is no directory of tokens, so
breadth comes from maintaining this list, not from writing fetchers. Tokens marked
verified were confirmed live against the provider API on 2026-10-13. The rest are
candidates: the collector records which ones 404 so the list self-corrects over time.
"""

from __future__ import annotations

from dataclasses import dataclass

from careerpilot.ingest.adapters import (
    AshbyAdapter,
    BdTechJobsAdapter,
    GreenhouseAdapter,
    LeverAdapter,
)
from careerpilot.ingest.base import SourceAdapter


@dataclass(frozen=True)
class Board:
    source: str
    token: str
    company: str
    verified: bool = False


ASHBY: tuple[Board, ...] = tuple(
    Board("ashby", t, c, v)
    for t, c, v in [
        ("openai", "OpenAI", True),
        ("cursor", "Cursor", True),
        ("perplexity", "Perplexity", True),
        ("decagon", "Decagon", True),
        ("ramp", "Ramp", True),
        ("elevenlabs", "ElevenLabs", True),
        ("notion", "Notion", True),
        ("baseten", "Baseten", True),
        ("deepgram", "Deepgram", True),
        ("linear", "Linear", False),
        ("synthflow", "Synthflow", False),
        ("harvey", "Harvey", False),
    ]
)

GREENHOUSE: tuple[Board, ...] = tuple(
    Board("greenhouse", t, c, v)
    for t, c, v in [
        ("stripe", "Stripe", True),
        ("datadoghq", "Datadog", False),
        ("cloudflare", "Cloudflare", False),
        ("mongodb", "MongoDB", False),
        ("hashicorp", "HashiCorp", False),
        ("twilio", "Twilio", False),
        ("elastic", "Elastic", False),
        ("grafana", "Grafana Labs", False),
        ("1password", "1Password", False),
        ("postman", "Postman", False),
    ]
)

LEVER: tuple[Board, ...] = tuple(
    Board("lever", t, c, v)
    for t, c, v in [
        ("spotify", "Spotify", True),
        ("hotjar", "Hotjar", False),
        ("netdata", "Netdata", False),
        ("klaviyo", "Klaviyo", False),
    ]
)

BD: tuple[Board, ...] = (
    Board("bdtechjobs", "1", "Bangladesh tech boards", True),
    Board("bdtechjobs", "2", "Bangladesh tech boards", True),
)

BOARDS: tuple[Board, ...] = (*ASHBY, *GREENHOUSE, *LEVER, *BD)

ADAPTERS: dict[str, type[SourceAdapter]] = {
    "greenhouse": GreenhouseAdapter,
    "ashby": AshbyAdapter,
    "lever": LeverAdapter,
    "bdtechjobs": BdTechJobsAdapter,
}

# Roles we actually want. Applied in code, after fetching, on the title.
#
# The second half of this list was written from evidence rather than taste: the first
# collect run kept 13 of bdtechjobs' 55 postings and dropped 42, and the ones it threw
# away were real engineering roles. Bangladeshi boards title those jobs "Developer" and
# "DevOps Engineer", not "Software Engineer", so a list written from how US boards name
# things silently filters out the entire local market.
TITLE_INCLUDE = (
    "ai engineer", "machine learning", "ml engineer", "backend", "software engineer",
    "platform engineer", "full stack", "fullstack", "applied ai", "llm", "agent",
    "data engineer", "security engineer", "sre", "infrastructure",
    "developer", "devops", "cloud", "secops", "cybersecurity", "site reliability",
    "ai systems", "information security",
)
TITLE_EXCLUDE = (
    "intern", "co-op", "principal", "staff ", "director", "vp ", "head of",
    "manager", "lead ", "senior staff", "distinguished", "circuit", "sales",
    "marketing", "designer", "recruiter", "legal", "finance", "content",
)
