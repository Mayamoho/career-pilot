"""What a person actually sees at the terminal.

These exist because the first run of this tool on a real machine produced a wall of
traceback for a state that is normal and fixable: no model key yet. Everything in here
asserts on the rendered output, not on the exception type — a correct exception that the
user reads as a crash is still a failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from careerpilot.cli import app
from careerpilot.domain.job import JobPosting
from careerpilot.ingest.store import JobStore

REPO = Path(__file__).resolve().parents[1]
PROFILE = str(REPO / "data" / "profile.example.yaml")

# `uid` is `source:source_id`, not the bare id — which is precisely the thing a person
# copying a value out of the wrong column gets wrong, so the tests spell it out.
OPEN_UID = "greenhouse:open1"
BLOCKED_UID = "greenhouse:sf1"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()



def _store_with(tmp_path: Path, *jobs: JobPosting) -> str:
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    store.upsert_many(list(jobs))
    store.close()
    return str(path)


def _job(uid: str, location: str | None, *, remote: bool) -> JobPosting:
    return JobPosting(
        source="greenhouse",
        source_id=uid,
        company="Example AI",
        title="Backend Engineer, RAG",
        location=location,
        is_remote=remote,
        description="Build retrieval services. Requirements: PostgreSQL, vector search.",
    )


def _output(result: Any) -> str:
    """stdout, plus stderr wherever this version of click keeps it."""
    stderr: str = str(getattr(result, "stderr", "") or "")
    return str(result.output) + stderr


def test_a_missing_live_switch_is_a_sentence_not_a_traceback(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path, _job("open1", "Remote", remote=True))

    result = runner.invoke(app, ["tailor", OPEN_UID, "--db", db, "--profile", PROFILE])

    assert result.exit_code == 1
    printed = _output(result)
    assert "Traceback" not in printed
    assert "CAREERPILOT_LIVE" in printed
    assert ".env" in printed
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_blocked_location_refuses_before_any_model_call(
    tmp_path: Path, runner: CliRunner
) -> None:
    """Exit 2, and nothing reaches the gateway: the refusal is a cost decision."""
    db = _store_with(tmp_path, _job("sf1", "San Francisco", remote=True))

    result = runner.invoke(app, ["tailor", BLOCKED_UID, "--db", db, "--profile", PROFILE])

    assert result.exit_code == 2
    assert "refusing:" in _output(result)
    assert "CAREERPILOT_LIVE" not in _output(result)


def test_scoring_reports_the_same_setup_state_in_one_line(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path, _job("open1", "Remote", remote=True))

    result = runner.invoke(app, ["score-job", OPEN_UID, "--db", db, "--profile", PROFILE])

    assert result.exit_code == 1
    assert "Traceback" not in _output(result)


def test_the_free_commands_run_with_no_credential_at_all(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path, _job("open1", "Remote", remote=True))

    result = runner.invoke(app, ["jobs", "--db", db, "--profile", PROFILE])

    assert result.exit_code == 0
    assert OPEN_UID in result.output
