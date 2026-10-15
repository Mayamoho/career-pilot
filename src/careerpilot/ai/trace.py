"""Call tracing — the observability layer.

Every AI call appends one record. This is what makes "the agent is unreliable" a
queryable question instead of a vibe: you can group by task and see which prompt
fails schema validation, how often the fallback chain fires, and what it costs.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from careerpilot.config import DATA_DIR


class CallRecord(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    task: str
    provider: str
    model: str
    attempt: int
    duration_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    ok: bool
    error_kind: str | None = None
    input_hash: str = ""
    source: str = "live"

    def to_line(self) -> str:
        return self.model_dump_json(exclude_none=True)


class TraceSink:
    """Append-only JSONL, plus an in-memory copy for assertions in tests."""

    def __init__(self, path: Path | None = None, *, quiet: bool = False) -> None:
        self.path = path
        self.quiet = quiet
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, call: CallRecord) -> None:
        with self._lock:
            self._records.append(call)
        if self.path is not None and not self.quiet:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(call.to_line() + "\n")

    @property
    def records(self) -> list[CallRecord]:
        with self._lock:
            return list(self._records)

    def read_disk(self) -> list[CallRecord]:
        """Every call this machine has ever traced, not just the ones this process made.

        `careerpilot traces` runs in a process of its own, where the in-memory list is
        empty by definition. Reporting zeros there would be the one thing an
        observability layer must never do: look healthy because nothing was looked at.
        """
        if self.path is None or not self.path.exists():
            return []
        records: list[CallRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(CallRecord.model_validate_json(line))
            except ValidationError:
                continue  # a half-written last line from a killed process
        return records

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self, calls: list[CallRecord] | None = None) -> dict[str, Any]:
        records = self.records if calls is None else calls
        by_task: dict[str, dict[str, int | float]] = {}
        for call in records:
            bucket = by_task.setdefault(call.task, {"calls": 0, "failures": 0, "cost_usd": 0.0})
            bucket["calls"] = int(bucket["calls"]) + 1
            if not call.ok:
                bucket["failures"] = int(bucket["failures"]) + 1
            bucket["cost_usd"] = round(float(bucket["cost_usd"]) + call.cost_usd, 6)
        return {
            "total_calls": len(records),
            "total_failures": sum(1 for c in records if not c.ok),
            "total_cost_usd": round(sum(c.cost_usd for c in records), 6),
            "fallback_invocations": sum(1 for c in records if c.provider != "groq" and c.ok),
            "by_task": by_task,
        }


class Timer:
    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.duration_ms = int((time.perf_counter() - self._start) * 1000)

    @property
    def elapsed_ms(self) -> int:
        return getattr(self, "duration_ms", 0)


default_sink = TraceSink(DATA_DIR / "traces.jsonl")
