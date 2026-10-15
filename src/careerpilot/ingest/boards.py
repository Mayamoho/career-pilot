"""Curated board tokens. This file is the actual moat.

The three global ATS providers are per-company and there is no directory of tokens, so
breadth there comes from maintaining this list, not from writing fetchers. Tokens marked
verified were confirmed live against the provider API — the original set on 2026-10-13,
Supabase and GitLab on 2026-10-15. The rest are candidates: the collector records which
ones 404 so the list self-corrects over time.

The two Bangladeshi entries and Jobicy break that shape and are better for it: they are
aggregated feeds, one request each covering thousands of employers, so their tokens are
a page number or a search term rather than a company. That is also why bdjobs is worth
the risk of depending on an endpoint it documents nowhere — the local market does not
publish itself on Greenhouse, and every Bangladeshi company token tried there 404'd.
"""

from __future__ import annotations

from dataclasses import dataclass

from careerpilot.ingest.adapters import (
    AshbyAdapter,
    BdJobsAdapter,
    BdTechJobsAdapter,
    GreenhouseAdapter,
    JobicyAdapter,
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
        # Supabase was found by measuring, not by guessing: it was the only board in a
        # probe of ~70 candidates whose postings are remote without being remote-for-the-US,
        # and it named a country in none of them. Eleven of its fifteen kept roles are
        # workable from Dhaka, which is more than the rest of this list manages together.
        ("supabase", "Supabase", True),
        ("linear", "Linear", False),
        ("synthflow", "Synthflow", False),
        ("harvey", "Harvey", False),
    ]
)

GREENHOUSE: tuple[Board, ...] = tuple(
    Board("greenhouse", t, c, v)
    for t, c, v in [
        ("stripe", "Stripe", True),
        # Fully distributed and it says so: the location is bare "Remote" with no
        # country attached, which is the shape that survives the place filter. Only one
        # of its 32 kept roles had that shape today, but the board is large enough that
        # the number moves every week.
        ("gitlab", "GitLab", True),
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

# bdjobs is the market: tens of thousands of active postings across every discipline in
# Bangladesh, with a deadline and a BDT salary on each row — facts no other source in
# this file publishes. Its gateway takes a page number or a `keyword:<term>` token,
# because the endpoint honours `keyword` but silently ignores `category` and `location`,
# so narrowing by stack is the one server-side filter worth using.
BDJOBS: tuple[Board, ...] = (
    Board("bdjobs", "1", "bdjobs.com", True),
    Board("bdjobs", "2", "bdjobs.com", True),
    *(Board("bdjobs", f"keyword:{term}", "bdjobs.com", True)
      for term in ("python", "laravel", "django", "react", "node", "java", "qa")),
)

# Jobicy labels the reachable region on every row and every job is remote, so
# `anywhere` is the only slice that is honestly open from Dhaka. `apac` is kept because
# it is the next widest set that includes Bangladesh.
JOBICY: tuple[Board, ...] = (
    Board("jobicy", "anywhere", "Jobicy", True),
    Board("jobicy", "apac", "Jobicy", True),
)

BOARDS: tuple[Board, ...] = (*ASHBY, *GREENHOUSE, *LEVER, *BD, *BDJOBS, *JOBICY)

ADAPTERS: dict[str, type[SourceAdapter]] = {
    "greenhouse": GreenhouseAdapter,
    "ashby": AshbyAdapter,
    "lever": LeverAdapter,
    "bdtechjobs": BdTechJobsAdapter,
    "bdjobs": BdJobsAdapter,
    "jobicy": JobicyAdapter,
}
