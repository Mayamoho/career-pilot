"""Error taxonomy for the AI layer.

The distinction that matters: a *contract* failure means the model answered but not
in the shape we required, so retrying with the validator's complaint can genuinely
help. A *transport* failure means we got nothing, so retrying the same model is
usually wasted budget and we should move down the fallback chain instead.
"""

from __future__ import annotations


class AIError(Exception):
    """Base class for everything raised by the AI layer."""


class TransportError(AIError):
    def __init__(self, provider: str, detail: str, *, retryable: bool = True) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider
        self.retryable = retryable


class MissingProviderCredential(AIError):
    def __init__(self, provider: str, secret_name: str) -> None:
        super().__init__(f"{provider} is configured but {secret_name} is not set")
        self.provider = provider
        self.secret_name = secret_name


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
            f"Escalate to a human. Causes: {' | '.join(causes)}"
        )
        self.task = task
        self.attempts = attempts
        self.causes = causes
