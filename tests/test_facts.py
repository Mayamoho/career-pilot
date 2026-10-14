"""The fact base and the text utilities it depends on.

Everything downstream compares technologies and numbers, so these small functions are
load-bearing: if `canonical` and `has_term` drift, the scorer and the gate disagree
about what a bullet is allowed to say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from careerpilot.ai.schemas import StructuredOutput
from careerpilot.domain.job import JobPosting, coerce_datetime, html_to_text
from careerpilot.profile.model import Atom, Profile, load_profile
from careerpilot.textutil import canonical, has_term, numbers


def test_the_shipped_profile_is_valid_and_addressable(profile: Profile) -> None:
    assert profile.name == "Abu Kawser"
    assert len(profile.atoms) == len(profile.by_id)
    assert {a.kind for a in profile.atoms} <= {
        "role", "project", "skill", "education", "award", "credential", "language"
    }
    assert "5.15" in profile.number_vocabulary
    assert "pgvector" in profile.skill_vocabulary


def test_the_metrics_a_generated_bullet_may_quote_are_exact(profile: Profile) -> None:
    compression = profile.resolve("proj-bornomala-compression")

    assert compression is not None
    assert compression.metrics["gain"] == "5.15x"
    assert "5.15" in profile.evidence_text(["proj-bornomala-compression"])


def test_a_duplicate_atom_id_is_refused_at_load(tmp_path: Path) -> None:
    broken = tmp_path / "profile.yaml"
    broken.write_text(
        "name: X\nheadline: Y\natoms:\n"
        "  - {id: dup, kind: skill, text: 'first atom text'}\n"
        "  - {id: dup, kind: skill, text: 'second atom text'}\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="duplicate atom ids"):
        load_profile(broken)


def test_an_untyped_atom_id_is_refused() -> None:
    with pytest.raises(Exception, match="match pattern"):
        Atom(id="Proj Named Like This", kind="project", text="some project text")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Node.js", "nodejs"), ("Node.js", "Node JS"), ("Golang", "Go"),
        ("PostgreSQL", "postgres"), ("pgvector", "PostgreSQL"), ("ReactJS", "React"),
        ("FastAPI", "fast api"), ("CI/CD", "cicd"),
    ],
)
def test_technology_names_fold_together(left: str, right: str) -> None:
    assert canonical(left) == canonical(right)


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("We use Google for auth", "go", False),
        ("Bangla text compression", "go", False),
        ("Written in Go, mostly", "go", True),
        ("a C compiler", "c", True),
        ("car rental", "c", False),
        ("PostgreSQL full-text search", "postgresql full text search", True),
        ("shadcn/ui components", "shadcn ui", True),
        ("Written in Go", "golang", False),
    ],
)
def test_term_matching_never_matches_inside_a_word(text: str, term: str, expected: bool) -> None:
    """`has_term` is literal; folding aliases like Golang→Go is `canonical`'s job."""
    assert has_term(text, term) is expected


def test_numbers_are_compared_as_written_not_as_rounded() -> None:
    assert numbers("5.15x faster, 100% of 5,000 messages") == {"5.15", "100", "5", "000"}
    assert numbers("no digits here") == set()


class _Answer(StructuredOutput):
    text: str
    score: int


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ('{"text": "hi", "score": 2}', "hi"),
        ('```json\n{"text": "hi", "score": 2}\n```', "hi"),
        ('```\n{"text": "hi", "score": 2}\n```', "hi"),
        ('Sure! Here is the answer:\n{"text": "hi", "score": 2}\nHope that helps.', "hi"),
        (' {"text": "a nested {brace} inside", "score": 2} ', "a nested {brace} inside"),
    ],
)
def test_a_fenced_or_padded_reply_still_parses(raw: str, text: str) -> None:
    assert _Answer.from_llm_json(raw) == _Answer(text=text, score=2)


@pytest.mark.parametrize("raw", ["", "not json at all", "[1, 2, 3]", '{"unterminated": '])
def test_a_reply_with_no_object_is_a_contract_error(raw: str) -> None:
    with pytest.raises(Exception, match="no JSON object found"):
        _Answer.from_llm_json(raw)


def test_an_extra_key_is_a_contract_error_not_something_to_ignore() -> None:
    with pytest.raises(Exception, match="Extra input"):
        _Answer.from_llm_json('{"text": "hi", "score": 2, "confident": true}')


def test_the_schema_travels_with_the_prompt() -> None:
    schema = _Answer.json_schema()

    assert "title" not in schema
    assert set(schema["properties"]) == {"text", "score"}
    assert schema["additionalProperties"] is False


def test_html_to_text_is_stable_and_block_aware() -> None:
    assert html_to_text("<p>Two</p><p>lines</p>") == "Two\nlines"
    assert html_to_text("<ul><li>a &amp; b</li><li>c</li></ul>") == "a & b\nc"
    assert html_to_text(None) == ""
    assert html_to_text("<div>  spaced   out  </div>") == "spaced out"


def test_content_hash_follows_the_text_not_the_crawl() -> None:
    base = _posting(description="Build retrieval services")
    same = _posting(description="Build retrieval services", published_at=None)
    changed = _posting(description="Build retrieval services and rerankers")

    assert base.content_hash == same.content_hash
    assert base.content_hash != changed.content_hash


@pytest.mark.parametrize(
    ("value", "year"),
        [(1759000000000, 2025), (1759000000, 2025), ("2026-10-01T09:00:00Z", 2026), ("junk", None)],
)
def test_every_timestamp_shape_the_apis_use_is_understood(value: Any, year: int | None) -> None:
    parsed = coerce_datetime(value)

    assert (parsed.year if parsed else None) == year
    assert parsed is None or parsed.tzinfo is None


def _posting(description: str, published_at: Any = 1759000000000) -> JobPosting:
    return JobPosting(
        source="greenhouse",
        source_id="1",
        company="Acme",
        title="Backend Engineer",
        description=description,
        published_at=coerce_datetime(published_at),
    )
