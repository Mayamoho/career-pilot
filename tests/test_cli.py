"""What a person actually sees at the terminal.

These exist because the first run of this tool on a real machine produced a wall of
traceback for a state that is normal and fixable: no model key yet. Everything in here
asserts on the rendered output, not on the exception type — a correct exception that the
user reads as a crash is still a failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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

_EXTRACTED = {
    "requirements": [{"text": "Python services", "skill": "Python", "category": "must_have"}],
    "seniority": "junior",
    "location_constraint": "unspecified",
}


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


def _posting(
    uid: str,
    *,
    location: str | None = "Dhanmondi, Dhaka, Bangladesh",
    title: str = "Python Automation Intern",
    deadline: datetime | None = None,
    description: str = "Requirements: Python, Selenium, automation.",
) -> JobPosting:
    source, source_id = uid.split(":", 1)
    return JobPosting(
        source=source,
        source_id=source_id,
        company="Marive",
        title=title,
        location=location,
        url=f"https://jobs.bdjobs.com/jobdetails.asp?id={source_id}",
        description=description,
        deadline=deadline,
    )


# --- ats ------------------------------------------------------------------------------


def test_ats_runs_with_no_credential_and_no_network(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["ats", "--profile", PROFILE, "--db", str(tmp_path / "x.sqlite")])
    assert result.exit_code == 0
    assert "ATS readiness" in result.output
    assert "/100" in result.output


def test_ats_names_a_missing_keyword_only_when_the_fact_base_has_it(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The one promise this command exists to keep, tested through the terminal.

    A CV the tool itself printed cannot be missing a fact its own fact base holds, so
    the input here is a hand-written CV that leaves Docker out. Kubernetes is missing
    from both — and is the term that must never be offered as advice.
    """
    jd = tmp_path / "jd.txt"
    jd.write_text("Required: Docker, Kubernetes and Python", encoding="utf-8")
    cv = tmp_path / "cv.txt"
    cv.write_text(
        "AMINA KHATUN\n"
        "amina@example.com | +8801711223344 | Mirpur, Dhaka, Bangladesh\n"
        "github.com/amina\n"
        "SUMMARY\n"
        "Python developer\n"
        "SKILLS\n"
        "Python, SQL, Git\n"
        "PROJECTS\n"
        "- wrote a marksheet parser in Python that cut 40 hours of manual entry\n"
        "EDUCATION\n"
        "BSc in CSE, North South University\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["ats", "--profile", PROFILE, "--jd", str(jd), "--cv", str(cv)])

    assert result.exit_code == 0, result.output
    addable = result.output.split("YOU MAY ADD")[1].splitlines()[0]
    assert "docker" in addable
    assert "kubernetes" not in addable
    assert "NOT IN YOUR FACT BASE — DO NOT ADD: kubernetes" in result.output


def test_ats_scores_a_posting_from_the_store(tmp_path: Path, runner: CliRunner) -> None:
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    result = runner.invoke(app, ["ats", "bdjobs:1542177", "--db", db, "--profile", PROFILE])
    assert result.exit_code == 0
    assert "Marive — Python Automation Intern" in result.output
    assert "keyword_coverage" in result.output


def test_a_uid_that_is_not_in_the_store_is_an_error_the_user_can_act_on(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path)
    result = runner.invoke(app, ["ats", "bdjobs:999999", "--db", db, "--profile", PROFILE])
    assert result.exit_code == 2
    assert "no job with uid" in _output(result)


# --- rank -----------------------------------------------------------------------------


def test_rank_buys_each_score_once_and_reuses_it_on_the_next_run(
    tmp_path: Path, runner: CliRunner, use_task_fake: Any
) -> None:
    """The reason `--max-new` exists: a free tier throttles tokens per minute."""
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    gateway = use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
        }
    )

    first = runner.invoke(app, ["rank", "--db", db, "--profile", PROFILE, "--max-new", "5"])
    calls_after_first = len(gateway.calls)
    second = runner.invoke(app, ["rank", "--db", db, "--profile", PROFILE, "--max-new", "5"])

    assert "1 fresh score(s), 0 reused" in first.output
    assert "0 fresh score(s), 1 reused" in second.output
    assert len(gateway.calls) == calls_after_first, "the second run must cost nothing"
    assert OPEN_UID not in second.output


def test_rank_spends_nothing_when_the_budget_is_zero(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    result = runner.invoke(app, ["rank", "--db", db, "--profile", PROFILE, "--max-new", "0"])
    assert "0 fresh score(s)" in result.output
    assert "1 over the --max-new budget" in result.output


def test_rank_shows_it_is_working_without_dirtying_the_list(
    tmp_path: Path, runner: CliRunner, use_task_fake: Any
) -> None:
    """A fresh score is a model call, so it has to say so on stderr — and only there.

    stdout is what gets redirected into a shortlist file; progress mixed into it would
    make `careerpilot rank > jobs.txt` produce a file nobody can read.
    """
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    use_task_fake({"extract_requirements": _EXTRACTED, "cite_evidence": {
        "atom_ids": ["proj-csxedu-rag"]}})

    result = runner.invoke(app, ["rank", "--db", db, "--profile", PROFILE])

    assert "… scoring 1/5" in _output(result)
    assert "… scoring 1/5" not in result.stdout
    assert result.stdout.splitlines()[0].startswith("# workable from")


def test_tailor_names_each_stage_as_it_runs(
    tmp_path: Path, runner: CliRunner, use_task_fake: Any
) -> None:
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    use_task_fake({
        "extract_requirements": _EXTRACTED,
        "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
        "draft_bullets": {"bullets": []},
        "draft_letter": {"paragraphs": []},
    })

    result = runner.invoke(app, ["tailor", "bdjobs:1542177", "--db", db, "--profile", PROFILE])

    spoken = _output(result)
    assert "… drafting bullets" in spoken and "… gating every line" in spoken
    for prefix in ("fit ", "wrote "):
        assert any(line.startswith(prefix) for line in result.stdout.splitlines())


def test_rank_refuses_an_unknown_where_instead_of_returning_nothing(
    tmp_path: Path, runner: CliRunner
) -> None:
    db = _store_with(tmp_path, _posting("bdjobs:1542177"))
    result = runner.invoke(app, ["rank", "--db", db, "--profile", PROFILE, "--where", "dhaka"])
    assert result.exit_code == 2
    assert "choose one of: bd, open, all" in _output(result)


# --- closing --------------------------------------------------------------------------


def test_closing_lists_the_nearest_deadline_first_and_drops_the_expired_one(
    tmp_path: Path, runner: CliRunner
) -> None:
    soon = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=2)
    later = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=9)
    gone = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3)
    db = _store_with(
        tmp_path,
        _posting("bdjobs:1", deadline=later),
        _posting("bdjobs:2", deadline=soon),
        _posting("bdjobs:3", deadline=gone),
        _posting("bdjobs:4"),  # no deadline stated, so nothing to be closing about
    )

    result = runner.invoke(app, ["closing", "--db", db, "--profile", PROFILE, "--days", "21"])

    assert result.exit_code == 0
    assert "2 closing within 21 day(s); 1 already expired and dropped" in result.output
    body = result.output.splitlines()[1:]
    assert "bdjobs:2" in body[0] and "bdjobs:1" in body[1]


def test_a_deadline_that_passed_is_a_refusal_before_any_model_call(
    tmp_path: Path, runner: CliRunner
) -> None:
    """Exit 2 and no gateway touched: tailoring a closed posting costs a call for nothing."""
    gone = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3)
    db = _store_with(tmp_path, _posting("bdjobs:5", deadline=gone))

    result = runner.invoke(app, ["tailor", "bdjobs:5", "--db", db, "--profile", PROFILE])

    assert result.exit_code == 2
    assert "refusing:" in _output(result)
    assert "CAREERPILOT_LIVE" not in _output(result)


# --- start-here -----------------------------------------------------------------------


_ANSWERS = [
    "Amina Khatun",
    "Python developer — final year CSE student",
    "amina@example.com",
    "+8801711223344",
    "Mirpur, Dhaka, Bangladesh",
    "github.com/amina",
    "",
    "Python, SQL, Git",
    "Built a marksheet parser in Python | coursework | 2025",
    "",
    "Python Intern | bkash | 2024",
    "",
    "BSc in CSE | North South University | 2022-2026",
    "",
    "Data Structures | NPTEL | 2024",
    "",
    "1st runner up, NSU Hackathon | NSU | 2025",
    "",
    "Bangla (native)",
    "English (fluent)",
    "",
]


def test_start_here_writes_a_fact_base_that_the_rest_of_the_tool_accepts(
    tmp_path: Path, runner: CliRunner
) -> None:
    out = tmp_path / "profile.yaml"

    result = runner.invoke(app, ["start-here", "--out", str(out)], input="\n".join(_ANSWERS))

    assert result.exit_code == 0, result.output
    assert out.exists()
    # The file it just wrote must load, or the command has only looked successful.
    scored = runner.invoke(app, ["jobs", "--db", str(tmp_path / "e.sqlite"), "--profile", str(out)])
    assert scored.exit_code == 0
    assert "Amina Khatun" in out.read_text(encoding="utf-8")


def test_start_here_does_not_clobber_a_fact_base_it_was_not_asked_to_replace(
    tmp_path: Path, runner: CliRunner
) -> None:
    out = tmp_path / "profile.yaml"
    out.write_text("name: Keep me\nheadline: x\natoms: []\n", encoding="utf-8")

    result = runner.invoke(
        app, ["start-here", "--out", str(out)], input="\n".join(_ANSWERS)
    )

    assert result.exit_code == 2
    assert "--force" in result.output
    assert "Keep me" in out.read_text(encoding="utf-8")


def test_an_answer_stream_that_ends_at_eof_still_produces_a_fact_base(
    tmp_path: Path, runner: CliRunner
) -> None:
    """Ctrl+D is how people finish a piped file; it must not lose forty answers."""
    out = tmp_path / "profile.yaml"
    answers = [line for line in _ANSWERS if line != ""]

    result = runner.invoke(app, ["start-here", "--out", str(out)], input="\n".join(answers))

    assert result.exit_code == 0, result.output
    assert "Data Structures" in out.read_text(encoding="utf-8")


def test_start_here_can_be_read_on_screen_before_it_is_written(
    tmp_path: Path, runner: CliRunner
) -> None:
    out = tmp_path / "profile.yaml"

    result = runner.invoke(
        app, ["start-here", "--print", "--out", str(out)], input="\n".join(_ANSWERS)
    )

    assert result.exit_code == 0
    assert "name: Amina Khatun" in result.output
    assert not out.exists()


# --- tailor's new output forms --------------------------------------------------------


def test_tailor_writes_both_templates_and_the_plain_text_export(
    tmp_path: Path, runner: CliRunner, use_task_fake: Any
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": {
                "bullets": [
                    {
                        "text": "Built a retrieval service over pgvector and PostgreSQL "
                        "full-text search",
                        "atom_ids": ["proj-csxedu-rag"],
                    }
                ]
            },
            "draft_letter": {"paragraphs": []},
            "check_entailment": {"supported": True, "confidence": 0.9, "unsupported_span": ""},
        }
    )
    db = _store_with(tmp_path, _posting("bdjobs:77"))
    out = tmp_path / "artifacts"

    result = runner.invoke(
        app, ["tailor", "bdjobs:77", "--db", db, "--profile", PROFILE, "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert {"cv-ats.html", "cv-oxford.html", "cv.txt"} <= set(written)


def test_md_style_writes_no_html(
    tmp_path: Path, runner: CliRunner, use_task_fake: Any
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": {"bullets": []},
            "draft_letter": {"paragraphs": []},
        }
    )
    db = _store_with(tmp_path, _posting("bdjobs:78"))
    out = tmp_path / "md-only"

    result = runner.invoke(
        app, ["tailor", "bdjobs:78", "--db", db, "--profile", PROFILE, "--out", str(out),
              "--style", "md"]
    )

    assert result.exit_code == 0, result.output
    assert not list(out.rglob("*.html"))
