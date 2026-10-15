"""Error taxonomy for the AI layer.

The distinction that matters: a *contract* failure means the model answered but not
in the shape we required, so retrying with the validator's complaint can genuinely
help. A *transport* failure means we got nothing, so retrying the same model is
usually wasted budget and we should move down the fallback chain instead.
"""

from __future__ import annotations

import re


class AIError(Exception):
    """Base class for everything raised by the AI layer."""


class TransportError(AIError):
    """We got no usable answer from that provider.

    `retryable` asks a narrower question than "did this fail": whether the *same*
    request to the *same* model would plausibly do any better on the second try. A rate
    limit would, because a minute passes. A 503, an unreadable body or a provider with
    no adapter would not, and those are the majority — so the flag is opt-in and the
    default is to move down the fallback chain instead of spending the retry budget.
    """

    def __init__(self, provider: str, detail: str, *, retryable: bool = False) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider
        self.retryable = retryable


class MissingProviderCredential(AIError):
    def __init__(self, provider: str, secret_name: str) -> None:
        super().__init__(f"{provider} is configured but {secret_name} is not set")
        self.provider = provider
        self.secret_name = secret_name


class NoCredentialError(AIError):
    """Live mode is on but not one provider has a key, so there is nothing to call.

    Distinct from ExhaustedRetriesError: that means providers were asked and refused, this
    means the network was never reached. Reporting "gave up after 0 attempts" would imply
    the models were the problem, when the fix is one line in `.env`.
    """

    def __init__(self, secret_names: str) -> None:
        super().__init__(
            "live mode is on but no provider has a key, so no model was asked anything. "
            f"Set at least one of these in .env (copy .env.example): {secret_names}. "
            "Groq gives the most tokens per day for free; any one of them is enough."
        )


class ContractError(AIError):
    """The model returned something that is not the requested structure."""

    def __init__(self, task: str, detail: str, raw: str = "") -> None:
        super().__init__(f"{task}: model output failed schema validation — {detail}")
        self.task = task
        self.detail = detail
        self.raw = raw


class NoRecordingError(AIError):
    """Nothing answered at all: no recorded reply, and live calls are switched off.

    This is deliberately *not* a ContractError. That class means the model did reply and
    the reply was malformed, which is retryable and is worth a repair attempt. Here no
    model was ever asked, so telling a user their output "failed schema validation"
    sends them to debug a prompt that was never run. It is a setup state, and the fix is
    in their environment, not in the code.
    """

    def __init__(self, task: str) -> None:
        super().__init__(
            f"{task}: no recorded answer for this prompt, and live model calls are off "
            "(CAREERPILOT_LIVE is not set). To run it for real, copy .env.example to .env "
            "in this folder, put your GROQ_API_KEY in it and leave CAREERPILOT_LIVE=1. "
            "To record answers for CI instead, run `careerpilot record-fixtures <uid>`. "
            "Tests never call a real model, which is the reason this switch exists."
        )
        self.task = task


class ExhaustedRetriesError(AIError):
    """Every model and every repair attempt failed. The caller must escalate."""

    def __init__(self, task: str, attempts: int, causes: list[str]) -> None:
        super().__init__(
            f"{task}: gave up after {attempts} attempt(s). "
            f"Escalate to a human. Causes: {' | '.join(causes)}{_hint(causes)}"
        )
        self.task = task
        self.attempts = attempts
        self.causes = causes


# The one definition of "this refusal is about a window, not a minute". `ai/providers.py`
# imports this to decide whether a 429 is worth sleeping through, and the hint below imports
# it to decide what to tell the human. Two regexes for one question is how a real Gemini quota
# 429 ended up labelled "not your daily quota" — the classifier grew a dialect and the hint
# did not.
DAILY_QUOTA = re.compile(
    r"per day|per month|per week|monthly|weekly|\(TPD\)|\(RPD\)|daily"
    r"|exceeded your current quota",
    re.I,
)
_MISSING_KEY = re.compile(r"MissingProviderCredential|is not set", re.I)
# A provider that answered 4xx/5xx on something that is not a quota refused the request
# itself. Telling that user to go set a key they already set sends them editing .env for
# nothing, when the fault is in what we sent.
_REFUSED = re.compile(r"HTTP [45]\d\d")


def _hint(causes: list[str]) -> str:
    """The one action the reader can actually take, derived from the causes, in code.

    A stack trace of provider JSON is not a next step. These failures each have exactly one
    answer: the window is closed, a provider refused the payload we built, or a hop has no
    key. Nothing here is invented — it is read off the causes that were raised. Groq and
    Gemini can both be out of tokens on the same evening, so the quota line names every
    provider that said so rather than blaming the first one.
    """
    blob = " | ".join(causes)
    refused = [c for c in causes if _REFUSED.search(c) and not DAILY_QUOTA.search(c)]
    if refused:
        return (
            "\nfix: a fallback provider was reached and refused the request itself, so this "
            f"is not your daily quota: {refused[-1][:180]}"
        )
    spent = ", ".join(
        dict.fromkeys(c.split(":", 1)[0].strip() for c in causes if DAILY_QUOTA.search(c))
    )
    if spent:
        return (
            f"\nfix: every provider that answered is out of quota today ({spent}); free "
            "tiers reset at midnight UTC. Work for free meanwhile — `careerpilot rank "
            "--max-new 0` reuses stored scores, and `jobs`, `closing`, `ats` and `start-"
            "here` never ask a model. `careerpilot keys` names the next hop worth a key."
        )
    if _MISSING_KEY.search(blob) and "HTTP" not in blob:
        return (
            "\nfix: a hop in the chain has no key, so a limit on the one you do have could not "
            "be routed around. `careerpilot keys` says which variable, and where the free key "
            "comes from."
        )
    return ""
