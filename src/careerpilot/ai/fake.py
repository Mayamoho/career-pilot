"""A scripted stand-in for the AI gateway.

Used by unit tests where the point is the *logic around* the model call, not the
model call itself. Unlike the cassette store, this needs no recording: you hand it
the object you want returned, and it asserts that the call site asked for the schema
it claims to. That catches a class of bug where a refactor quietly changes which
prompt a module sends.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from pydantic import ValidationError

from careerpilot.ai.errors import ContractError
from careerpilot.ai.gateway import AIGateway
from careerpilot.ai.schemas import OutputT, StructuredOutput
from careerpilot.ai.trace import CallRecord, TraceSink


class ScriptedGateway(AIGateway):
    """Replays a queue of prepared payloads, validating them like a real response."""

    def __init__(self, script: list[Any] | None = None) -> None:
        super().__init__(providers={}, cassette=None, sink=TraceSink(quiet=True))
        self._script: deque[Any] = deque(script or [])
        self.calls: list[dict[str, str]] = []

    def queue(self, payload: Any) -> ScriptedGateway:
        self._script.append(payload)
        return self

    @property
    def pending(self) -> int:
        return len(self._script)

    def ask(self, *, task: str, system: str, user: str, schema: type[OutputT]) -> OutputT:
        self.calls.append({"task": task, "system": system, "user": user, "schema": schema.__name__})
        if not self._script:
            raise AssertionError(
                f"{task}: FakeLLM ran out of scripted responses. "
                f"{len(self.calls)} call(s) made. Add another queue() entry."
            )
        payload = self._script.popleft()
        if isinstance(payload, Exception):
            self._sink.record(CallRecord(
                task=task, provider="fake", model="scripted", attempt=0,
                duration_ms=0, ok=False, error_kind=type(payload).__name__, source="fake",
            ))
            raise payload
        text = payload if isinstance(payload, str) else _as_json_text(payload)
        try:
            result = schema.from_llm_json(text)
        except ValidationError as exc:
            self._sink.record(CallRecord(
                task=task, provider="fake", model="scripted", attempt=0,
                duration_ms=0, ok=False, error_kind="ContractError", source="fake",
            ))
            raise ContractError(task, str(exc), raw=text) from exc
        self._sink.record(CallRecord(
            task=task, provider="fake", model="scripted", attempt=0,
            duration_ms=0, ok=True, source="fake",
        ))
        return result


def _as_json_text(payload: Any) -> str:
    if isinstance(payload, StructuredOutput):
        return payload.model_dump_json()
    if isinstance(payload, dict):
        return json.dumps(payload)
    return str(payload)
