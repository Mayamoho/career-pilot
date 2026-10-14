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
