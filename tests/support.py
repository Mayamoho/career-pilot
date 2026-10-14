"""Test helpers.

`ScriptedGateway` answers calls in the order they arrive, which is fine for one module
but brittle for the whole pipeline, where the number of evidence-citation calls depends
on how many requirements the model extracted. `TaskFake` answers by task name instead,
so a pipeline test states what each model should say and stays correct as the number of
calls changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from careerpilot.ai.fake import ScriptedGateway
from careerpilot.ai.schemas import OutputT
from careerpilot.textutil import numbers

Payload = dict[str, Any] | str | BaseException | Callable[[str], Any]


class TaskFake(ScriptedGateway):
    """A scripted gateway that dispatches on the task name of each call."""

    def __init__(self, by_task: dict[str, Payload | list[Payload]]) -> None:
        super().__init__([])
        self.by_task = by_task
        self.used: list[str] = []

    def ask(self, *, task: str, system: str, user: str, schema: type[OutputT]) -> OutputT:
        self.used.append(task)
        if task not in self.by_task:
            msg = f"no scripted response for task {task!r} (schema {schema.__name__})"
            raise AssertionError(msg)
        payload = self._take(task)
        if callable(payload) and not isinstance(payload, BaseException):
            payload = payload(user)
        self.queue(payload)
        return super().ask(task=task, system=system, user=user, schema=schema)

    def _take(self, task: str) -> Any:
        entry = self.by_task[task]
        if isinstance(entry, list):
            if not entry:
                msg = f"scripted responses for task {task!r} ran out"
                raise AssertionError(msg)
            return entry.pop(0) if len(entry) > 1 else entry[0]
        return entry


def supported(user: str) -> dict[str, Any]:
    """An honest judge: supports a claim only if its digits appear in the evidence."""

    claim, _, evidence = user.partition("PERMITTED EVIDENCE:")
    claimed = numbers(claim)
    unsupported = sorted(claimed - numbers(evidence))
    if unsupported:
        return {
            "supported": False,
            "confidence": 0.9,
            "unsupported_span": f"numbers absent from evidence: {', '.join(unsupported)}",
        }
    return {"supported": True, "confidence": 0.9, "unsupported_span": ""}


def always_supported(user: str) -> dict[str, Any]:  # noqa: ARG001 - signature parity
    return {"supported": True, "confidence": 0.99, "unsupported_span": ""}


def always_rejected(user: str) -> dict[str, Any]:  # noqa: ARG001 - signature parity
    return {"supported": False, "confidence": 0.9, "unsupported_span": "whole claim"}
