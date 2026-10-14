"""Requirement extraction and fit scoring.

The scoring tests exist to prove one property in particular: a model can only ever
lower a score, never raise one on its own say-so. Every point comes from a cited atom
that the profile actually contains.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from careerpilot.domain.job import JobPosting
from careerpilot.match.extract import ExtractedRequirement, RequirementSet, extract_requirements
from careerpilot.match.score import _years_of_experience, score_job
from careerpilot.profile.model import Atom, Profile
from support import TaskFake


def _req(text: str, skill: str, category: str = "must_have") -> dict[str, str]:
    return {"text": text, "skill": skill, "category": category}


def req(text: str, skill: str, category: str = "must_have") -> ExtractedRequirement:
    return ExtractedRequirement(text=text, skill=skill, category=category)


def test_extraction_merges_two_phrasings_of_one_requirement(use_task_fake: Any) -> None:
    use_task_fake(
        {
            "extract_requirements": {
                "requirements": [
                    _req("3+ years with PostgreSQL", "PostgreSQL"),
                    _req("Strong PostgreSQL tuning skills", "postgresql"),
                    _req("Nice to have: SSE streaming", "SSE", "nice_to_have"),
                ],
                "seniority": "mid",
                "location_constraint": "remote",
            }
        }
    )

    result = extract_requirements(_posting())

    assert [r.skill for r in result.requirements] == ["PostgreSQL", "SSE"]
    assert (result.seniority, result.location_constraint) == ("mid", "remote")


def test_experience_never_counts_time_that_has_not_happened() -> None:
    """A year in a period is a date, not a duration.

    The fact base dates Doctor Tracker "2026". Read in October, a period closing on 31
    December of its own year credited months of work that have not occurred, and a
    "5+ years" gate opened that no one had earned. Under-scoring is survivable; this
    is not.
    """
    future = Profile(
        name="Test",
        headline="Test",
        atoms=[
            Atom(
                id="proj-future",
                kind="project",
                text="A project that starts three years from now",
                period=str(date.today().year + 3),
            )
        ],
    )

    assert _years_of_experience(future) == 0.0


def test_a_hallucinated_citation_cannot_be_bought_past_a_years_gate(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    """The gate runs before the model, so a compliant answer is irrelevant."""
    use_task_fake({"cite_evidence": {"atom_ids": ["proj-doctortracker-api"]}})

    fit = score_job(
        _posting(),
        _set(req("5+ years of backend experience", "MongoDB")),
        profile,
    )

    assert fit.outcomes[0].matched is False
    assert fit.outcomes[0].via == "none"


def test_the_model_sees_the_requirements_section_not_the_whole_posting(use_task_fake: Any) -> None:
    gateway: TaskFake = use_task_fake(
        {"extract_requirements": {"requirements": [], "seniority": "x", "location_constraint": "y"}}
    )
    job = _posting()

    extract_requirements(job)

    sent = gateway.calls[-1]["user"]
    assert "Requirements:" in sent
    assert "About us. We were founded in 2019 and have grown a lot." not in sent


def test_years_of_experience_is_arithmetic_not_opinion(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake({"cite_evidence": {"atom_ids": []}})
    job = _posting()

    five = score_job(job, _set(req("5+ years of backend experience", "backend")), profile)
    two = score_job(job, _set(req("2+ years of backend experience", "Python")), profile)

    assert five.outcomes[0].matched is False
    assert "5+ years required" in five.outcomes[0].note
    assert two.outcomes[0].matched is True
    assert two.outcomes[0].via == "code"


def test_a_cited_atom_that_does_not_exist_cannot_add_weight(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake({"cite_evidence": {"atom_ids": ["proj-invented-by-the-model"]}})

    fit = score_job(_posting(), _set(req("Experience with Kubernetes", "Kubernetes")), profile)

    assert fit.total == 0.0
    assert fit.outcomes[0].matched is False
    assert fit.outcomes[0].evidence_atom_ids == []


def test_a_real_citation_does_count_as_evidence(profile: Profile, use_task_fake: Any) -> None:
    """The model's only job is to point at facts code could not name directly."""
    gateway: TaskFake = use_task_fake({"cite_evidence": {"atom_ids": ["proj-csxedu-rag"]}})

    fit = score_job(
        _posting(),
        _set(req("Answers must cite the passage they came from", "grounded question answering")),
        profile,
    )

    assert fit.outcomes[0].via == "model"
    assert fit.outcomes[0].evidence_atom_ids == ["proj-csxedu-rag"]
    assert fit.total == 100.0
    assert gateway.calls[-1]["task"] == "cite_evidence"


def test_code_matching_never_asks_the_model(profile: Profile, use_task_fake: Any) -> None:
    gateway: TaskFake = use_task_fake({"cite_evidence": {"atom_ids": []}})

    fit = score_job(_posting(), _set(req("pgvector similarity search", "pgvector")), profile)

    assert fit.outcomes[0].via == "code"
    assert gateway.calls == []


def test_an_empty_citation_leaves_the_requirement_unmatched(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake({"cite_evidence": {"atom_ids": []}})

    fit = score_job(
        _posting(),
        _set(req("Answers must cite the passage they came from", "grounded question answering")),
        profile,
    )

    assert fit.outcomes[0].matched is False
    assert "no evidence" in fit.outcomes[0].note


def test_band_is_dropped_when_a_must_have_is_missed(profile: Profile, use_task_fake: Any) -> None:
    use_task_fake({"cite_evidence": {"atom_ids": []}})

    fit = score_job(
        _posting(),
        _set(
            req("Production Go services", "Go"),
            req("Kubernetes at scale", "Kubernetes", "must_have"),
        ),
        profile,
    )

    assert fit.total == 50.0
    assert fit.band == "partial"
    assert fit.unmet_must_haves == ["Kubernetes at scale"]


def _set(*requirements: ExtractedRequirement) -> RequirementSet:
    return RequirementSet(
        requirements=list(requirements),
        seniority="unspecified",
        location_constraint="unspecified",
    )


def _posting() -> JobPosting:
    return JobPosting(
        source="greenhouse",
        source_id="1",
        company="Example",
        title="Backend Engineer",
        description=(
            "About us. We were founded in 2019 and have grown a lot. " + "filler " * 20
            + "Requirements:\n- Python\n- PostgreSQL\n"
        ),
    )
