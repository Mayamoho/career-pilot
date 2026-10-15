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


# Ordered fallback chain. Groq first for latency, Gemini second for quota resilience.
# Both are already in use on CSxEDU, so this reuses credentials you actually have.
PRIMARY = ModelSpec(
    provider="groq", model="groq-llama-3.3-70b-specdec", input_per_1m=0.59, output_per_1m=0.79
)
FALLBACK = ModelSpec(
    provider="gemini", model="gemini-2.0-flash-001", input_per_1m=0.10, output_per_1m=0.40
)

MODEL_CHAIN: tuple[ModelSpec, ...] = (PRIMARY, FALLBACK)

MAX_REPAIR_ATTEMPTS = 2
REQUEST_TIMEOUT_S = 45.0

# A CV bullet ships only if the entailment judge agrees it is supported.
# Anything below this is dropped rather than softened.
GROUNDING_MIN_CONFIDENCE = 0.75


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
