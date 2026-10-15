"""Process configuration. Every tunable lives here so behaviour is reproducible.

Model versions are pinned deliberately: an unpinned model is a silent dependency
that can change your eval numbers without a single line of code changing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CAREERPILOT_DATA_DIR", REPO_ROOT / "var"))
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@lru_cache
def load_env_file() -> bool:
    """Load `<repo>/.env` once, on first need.

    Lazily rather than at import, so pulling in this module changes nothing in the
    process — which is what lets the test suite run without a stray file on someone's
    machine silently supplying real credentials. Read from REPO_ROOT rather than the
    working directory, so the key is found whichever directory the CLI is invoked from.
    `override=False`: a key exported in the shell outranks a file that was forgotten.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    return True


# Two fact bases, deliberately. `profile.example.yaml` is committed and de-identified,
# so CI and the eval thresholds run everywhere. `profile.yaml` is the real one — same
# atoms, plus a phone number and a home address — and stays out of version control.
# The real file wins when it exists, because that is the one a CV has to be built from.
PROFILE_FILENAME = "profile.yaml"
PROFILE_EXAMPLE_FILENAME = "profile.example.yaml"


def profile_path() -> Path:
    """The fact base to draw from, preferring the uncommitted local one."""
    override = os.environ.get("CAREERPILOT_PROFILE")
    if override:
        return Path(override)
    real = REPO_ROOT / "data" / PROFILE_FILENAME
    return real if real.exists() else real.parent / PROFILE_EXAMPLE_FILENAME


class ModelSpec(BaseModel):
    """A pinned provider/model pair with pricing used for cost accounting."""

    provider: str
    model: str
    input_per_1m: float = Field(ge=0)
    output_per_1m: float = Field(ge=0)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_1m + completion_tokens * self.output_per_1m
        ) / 1_000_000


# Ordered fallback chain — every hop a free tier that needs no card. Groq first for latency,
# then the rest purely as quota, tried only when their key is configured (`credential_present`
# skips an unconfigured hop before it reaches the network).
# Prices are the per-token rates each provider's own endpoint reports, scaled to 1M. The free
# hops are 0.0 because on their free tiers they genuinely cost nothing; `traces` therefore
# under-reports those calls rather than inventing a price nobody is charged.
PRIMARY = ModelSpec(
    provider="groq", model="openai/gpt-oss-120b", input_per_1m=0.15, output_per_1m=0.60
)
# Re-pinned 2026-10-16 against the key itself, not from memory: `GET /v1beta/models` on this
# credential has no `gemini-2.0-flash-001` at all, so the second hop would have died on a 404
# exactly when it was needed most. `gemini-2.5-flash` is present, and it then answered 19 real
# calls the same evening while Groq's day was spent.
FALLBACK = ModelSpec(provider="gemini", model="gemini-2.5-flash", input_per_1m=0.0,
                     output_per_1m=0.0)

# The rest of the chain is quota, best to worst. Two facts decide the order.
#
# First, Groq and Gemini cap tokens per *model* per day, and the refusal names the model:
# `Rate limit reached for model openai/gpt-oss-120b … on tokens per day (TPD): Limit 200000`.
# A second model on a key that is already in `.env` is therefore a second daily window, not a
# copy of the one that just closed. Providers alternate below so consecutive hops draw on
# independent pools, instead of spending a whole day against one host before asking another.
#
# Second, what each key was seen to offer today. `GET /v1/models` on this Groq key lists 11
# ids, of which these do text at 131k context: gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard-20b,
# qwen/qwen3.8-27b. That endpoint returns no `supported_parameters` for this account, so the
# JSON-schema support of the second Groq hop is documented-but-not-reprobed; if it ignores
# `response_format` it answers prose, the validator rejects it as a contract failure, and the
# chain moves on — the failure is contained, it cannot corrupt a CV.
#
# Deliberately absent: `gemini-2.5-pro` and the `3.5`/`3.6`/`3.7`/`3.8` flash ids, which are on
# this key but whose free-tier status was not measured, and a billing-gated model would answer
# 403 or 429 on every hop. Also absent: models whose providers train on the inputs, because the
# payload carries a name, a phone number and an address.
QUOTA_HOPS: tuple[ModelSpec, ...] = (
    ModelSpec(provider="groq", model="openai/gpt-oss-20b", input_per_1m=0.0,
              output_per_1m=0.0),
    ModelSpec(provider="gemini", model="gemini-2.5-flash-lite", input_per_1m=0.0,
              output_per_1m=0.0),
    # Both ids read off OpenRouter's public /models with `structured_outputs` in
    # supported_parameters; its free tier is request-per-day capped, so this hop exists to
    # answer when the two token-metered providers above have both run out.
    ModelSpec(provider="openrouter", model="nvidia/nemotron-3-super-120b-a12b:free",
              input_per_1m=0.0, output_per_1m=0.0),
    ModelSpec(provider="openrouter", model="dots-studio/dots-3-note-preview:free",
              input_per_1m=0.0, output_per_1m=0.0),
    # No CEREBRAS_API_KEY or COHERE_API_KEY exists on this machine yet, so both are skipped
    # before the network today. Cohere's Chat V2 needs its own adapter: a different reply
    # shape and no JSON-schema body, so its answers are validated rather than enforced.
    ModelSpec(provider="cerebras", model="gpt-oss-120b", input_per_1m=0.0,
              output_per_1m=0.0),
    ModelSpec(provider="cohere", model="command-r7b-12-2024", input_per_1m=0.0,
              output_per_1m=0.0),
)

MODEL_CHAIN: tuple[ModelSpec, ...] = (PRIMARY, FALLBACK, *QUOTA_HOPS)

MAX_REPAIR_ATTEMPTS = 2
REQUEST_TIMEOUT_S = 45.0

# A connection that resets before the model answers is a property of the wire, not of
# the prompt or the provider, so it gets its own retry budget instead of being counted
# as a repair. Measured on a Dhaka broadband link in October 2026: POSTs to Groq reset
# outright about a third of the time, and a second try after a pause usually lands.
# An error the server actually responded with is not retried here — that moves down the
# fallback chain, which is the older and still correct rule.
NETWORK_ATTEMPTS = 3
NETWORK_BACKOFF_S = 1.5

# Groq's free tier limits *tokens per minute*, not requests, and one extraction call with
# a job description in it is a few thousand tokens. So a 429 naming a minute window is not
# a failure to route around, it is a schedule: the same request will succeed shortly.
# Waiting is the only correct response, and a second provider with no credential helps
# nobody. A 429 naming a *day* window is the opposite case and is not waited on at all —
# see `DAILY_QUOTA` in `ai/errors.py`.
RATE_LIMIT_WAIT_S = 45.0

# A CV bullet ships only if the entailment judge agrees it is supported.
# Anything below this is dropped rather than softened.
GROUNDING_MIN_CONFIDENCE = 0.75

# bdjobs has no documented public API: its own Angular frontend calls these two keyless
# JSON endpoints, and `robots.txt` on the gateway disallows nothing. That makes it fair
# game for a job seeker but not for a crawler, so the budget is a hard limit in code
# rather than a note in a docstring — measured 2026-10-15, one list page is 50 rows, and
# the detail endpoint is one request per job.
BDJOBS_DETAIL_LIMIT = 12
BDJOBS_DETAIL_SLEEP_S = 0.6

# ATS readiness. A one-page CV that is 350-800 words is the range the parsers and the
# human reader both tolerate; below it there is nothing to score, above it the second
# page gets skipped. The bands are deliberately not the fit bands: 60/100 on readiness
# means the machine will lose your CV, which is a different and more urgent message.
ATS_WORDS_MIN = 350
ATS_WORDS_MAX = 800
ATS_READY_MIN = 75.0
ATS_PARTIAL_MIN = 50.0


def require_secret(name: str) -> str:
    load_env_file()
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingCredentialError(name)
    return value


class MissingCredentialError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"{name} is not set. Export it in your environment or .env file. "
            "Tests do not need it: they run against the recorded FakeLLM."
        )
        self.secret_name = name


@lru_cache
def live_mode_enabled() -> bool:
    """Real provider calls are opt-in. CI never sets this.

    Read after the env file, so `.env` can carry `CAREERPILOT_LIVE=1` too: one file
    then holds the whole switch, identically on Linux, macOS and a Windows terminal
    where `export` does not exist. Cached because it is read on every model call.
    """
    load_env_file()
    return os.environ.get("CAREERPILOT_LIVE", "").strip() == "1"
