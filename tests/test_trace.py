"""The trace log, read back from disk.

`careerpilot traces` is a separate process from the one that made the calls, so
summarising only the in-memory list would report a healthy zero for a machine that has
spent real money. Both halves of that file — what is legible and what is not — are the
contract here.
"""

from __future__ import annotations

from pathlib import Path

from careerpilot.ai.trace import CallRecord, TraceSink


def test_a_traces_command_reports_the_history_not_only_this_process(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    writer = TraceSink(path)
    writer.record(
        CallRecord(
            task="tailor", provider="groq", model="m", attempt=0,
            duration_ms=120, ok=True, cost_usd=0.001, input_hash="a",
        )
    )
    writer.record(
        CallRecord(
            task="tailor", provider="groq", model="m", attempt=1,
            duration_ms=90, ok=False, error_kind="TransportError", input_hash="a",
        )
    )

    fresh = TraceSink(path)
    summary = fresh.summary(fresh.read_disk())

    assert fresh.records == []  # a new process remembers nothing
    assert summary["total_calls"] == 2
    assert summary["total_failures"] == 1
    assert summary["by_task"]["tailor"] == {"calls": 2, "failures": 1, "cost_usd": 0.001}


def test_a_half_written_line_is_skipped_rather_than_crashing_the_report(
    tmp_path: Path,
) -> None:
    """A killed pipeline leaves a truncated record. Reading the log back must not turn
    that into an error the operator has no way to fix."""
    path = tmp_path / "traces.jsonl"
    TraceSink(path).record(
        CallRecord(task="score", provider="groq", model="m", attempt=0, duration_ms=10, ok=True)
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"task": "score", "provi')

    records = TraceSink(path).read_disk()

    assert [r.task for r in records] == ["score"]
