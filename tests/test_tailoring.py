"""Tailoring, gating and rendering — the headline guarantee, tested end to end.

CareerPilot promises that every generated line comes from the candidate's own
recorded experience. These tests run the whole flow with a fake model that lies on
purpose, and assert that the lie never reaches the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from careerpilot.domain.job import JobPosting
from careerpilot.ground.gate import Claim, check_bullet, gate
from careerpilot.match.extract import ExtractedRequirement
from careerpilot.match.score import FitScore, LocationVerdict, RequirementOutcome
from careerpilot.pipeline import tailor
from careerpilot.profile.model import Atom, Profile
from careerpilot.report import render
from careerpilot.tailor.bullets import draft_bullets, select_targets, tailor_bullets
from careerpilot.tailor.letter import DraftedParagraph, check_paragraph, gate_letter
from careerpilot.textutil import canonical
from support import always_supported, supported

TRUTHY_BULLET = "Built a retrieval service over pgvector and PostgreSQL full-text search"
FABRICATED_BULLET = "Cut request latency by 40% with a pgvector cache in front of the assistant"
OPINION = (
    "I am applying because this role is the kind of systems work I want to keep doing, "
    "and I would rather grow in it than talk around it."
)


def test_targets_only_include_requirements_the_fact_base_can_back(profile: Profile) -> None:
    fit = _fit(
        _outcome("Production Python services", "Python", matched=True),
        _outcome("Kubernetes at scale", "Kubernetes", matched=False),
        _outcome("Distributed tracing", "OpenTelemetry", matched=False),
    )

    targets = select_targets(fit, profile)

    assert [t.skill for t in targets] == ["Python"]
    assert all(t.atoms for t in targets)


def test_targets_are_weighted_and_deduplicated_by_skill(profile: Profile) -> None:
    fit = _fit(
        _outcome("Nice: Python", "Python", matched=False, category="nice_to_have", weight=1.5),
        _outcome("Must: Python services", "python", matched=True, weight=3.0),
        _outcome("Must: pgvector search", "pgvector", matched=True, weight=3.0),
    )

    targets = select_targets(fit, profile)

    assert [t.requirement for t in targets] == ["Must: Python services", "Must: pgvector search"]
    assert len({canonical(t.skill) for t in targets}) == len(targets)


def test_the_model_is_shown_only_the_atoms_selected_for_its_targets(
    job: JobPosting, profile: Profile, use_task_fake: Any
) -> None:
    gateway = use_task_fake({"draft_bullets": {"bullets": []}})
    targets = select_targets(_fit(_outcome("pgvector search", "pgvector", matched=True)), profile)

    draft_bullets(job, targets)

    sent = gateway.calls[-1]["user"]
    cited = {a.id for t in targets for a in t.atoms}
    assert cited and all(f"id={atom_id}" in sent for atom_id in cited)
    unselected = [a.id for a in profile.atoms if a.id not in cited]
    assert not any(f"id={atom_id}" in sent for atom_id in unselected)


def test_a_fabricated_number_is_rejected_and_says_why(
    job: JobPosting,
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake(
        {
            "draft_bullets": _draft([TRUTHY_BULLET, FABRICATED_BULLET]),
            "check_entailment": supported,
        }
    )

    result = tailor_bullets(job, _pgvector_fit(), profile)

    assert [v.text for v in result.report.shipped] == [TRUTHY_BULLET]
    assert result.report.rejection_summary() == {"unsourced_number": 1}
    assert "40" in result.report.rejected[0].failures[0].detail


def test_a_citation_of_an_atom_that_does_not_exist_never_reaches_the_judge(
    job: JobPosting, profile: Profile, use_task_fake: Any
) -> None:
    gateway = use_task_fake(
        {
            "draft_bullets": {
                "bullets": [{"text": TRUTHY_BULLET, "atom_ids": ["proj-made-up-by-the-model"]}]
            },
            "check_entailment": always_supported,
        }
    )

    result = tailor_bullets(job, _pgvector_fit(), profile)

    assert result.report.shipped == []
    assert [f.kind for f in result.report.rejected[0].failures] == ["unknown_atom"]
    assert "check_entailment" not in gateway.used


def test_a_real_skill_attached_to_the_wrong_work_is_caught(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    """The subtle lie: not an invented skill, a real one bolted onto unrelated work."""
    use_task_fake({"check_entailment": always_supported})
    screens = _find_atom(profile, kind="project", containing="community screens")
    assert screens.id == "proj-learnloop-work"

    verdict = _check(
        "Built the Flutter community screens and autoscaled their Firebase service on Docker",
        [screens.id],
        profile,
    )

    assert verdict.shipped is False
    assert {f.kind for f in verdict.failures} == {"misattributed_skill"}
    assert {f.detail.split('"')[1] for f in verdict.failures} == {"Firebase", "Docker"}


def test_an_unsupported_but_sourced_claim_is_dropped_by_the_judge(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake(
        {"check_entailment": {"supported": False, "confidence": 0.9, "unsupported_span": "led"}}
    )

    verdict = _check("Led the company-wide migration to Rust", ["proj-csxedu-rag"], profile)

    assert verdict.shipped is False
    assert verdict.failures[0].kind == "not_entailed"


def test_a_hesitant_judge_is_not_believed(profile: Profile, use_task_fake: Any) -> None:
    use_task_fake(
        {"check_entailment": {"supported": True, "confidence": 0.4, "unsupported_span": ""}}
    )

    verdict = _check(TRUTHY_BULLET, ["proj-csxedu-rag"], profile)

    assert verdict.shipped is False
    assert "0.40" in verdict.failures[0].detail


def test_gate_never_softens_a_failure_it_only_removes_it(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake({"check_entailment": supported})

    claims = [
        Claim(text=text, atom_ids=["proj-csxedu-rag"])
        for text in (TRUTHY_BULLET, FABRICATED_BULLET)
    ]

    report = gate(claims, profile)

    assert report.total == 2
    assert len(report.shipped) == 1
    assert report.pass_rate == 0.5


def test_a_letter_paragraph_may_be_opinion_without_a_citation(
    profile: Profile,
    use_task_fake: Any,
) -> None:
    gateway = use_task_fake({"check_entailment": always_supported})

    verdict = check_paragraph(OPINION, [], profile)

    assert verdict.shipped is True
    assert gateway.used == []


def test_a_letter_paragraph_that_states_a_number_is_gated_like_a_bullet(
    profile: Profile, use_task_fake: Any
) -> None:
    use_task_fake({"check_entailment": supported})

    report = gate_letter([_paragraph(FABRICATED_BULLET, ["proj-csxedu-rag"])], profile)

    assert report.shipped == []
    assert report.rejection_summary()["unsourced_number"] == 1


def test_the_pipeline_drops_the_lie_before_the_cv_is_written(
    job: JobPosting,
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": _draft([TRUTHY_BULLET, FABRICATED_BULLET]),
            "draft_letter": {
                "paragraphs": [
                    _paragraph_payload(
                        TRUTHY_BULLET + " for the assistant the platform already ships"
                    ),
                    {"text": OPINION, "atom_ids": []},
                ]
            },
            "check_entailment": supported,
        }
    )

    tailored = tailor(job, profile)
    cv = render.render_cv(profile, tailored)
    letter = render.render_letter(profile, tailored)

    assert tailored.shipped_bullets == [TRUTHY_BULLET]
    assert TRUTHY_BULLET in cv
    assert "40%" not in cv
    assert FABRICATED_BULLET not in cv
    assert "40%" not in letter
    assert OPINION in letter


def test_the_audit_explains_every_decision(
    job: JobPosting,
    profile: Profile,
    use_task_fake: Any,
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": _draft([TRUTHY_BULLET, FABRICATED_BULLET]),
            "draft_letter": {"paragraphs": [{"text": OPINION, "atom_ids": []}]},
            "check_entailment": supported,
        }
    )

    record = render.audit(tailor(job, profile))

    shipped = [{"text": TRUTHY_BULLET, "atom_ids": ["proj-csxedu-rag"]}]
    assert record["bullets"]["shipped"] == shipped
    assert record["bullets"]["rejected_kinds"] == {"unsourced_number": 1}
    assert "40" in record["bullets"]["rejected"][0]["why"][0]
    assert every_point_is_attributed(record["fit"]["audit_trail"])


def test_nothing_shipped_leads_to_an_honest_cv_not_an_empty_one(
    job: JobPosting, profile: Profile, use_task_fake: Any
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": []},
            "draft_bullets": {"bullets": []},
            "draft_letter": {"paragraphs": []},
        }
    )

    tailored = tailor(job, profile)
    cv = render.render_cv(profile, tailored)

    assert "Nothing in the fact base could be phrased as a supported claim" in cv
    assert "rather say that than guess" in render.render_letter(profile, tailored)
    assert "Built the RAG AI assistant" in cv


def test_artifacts_are_written_under_the_requested_directory(
    job: JobPosting,
    profile: Profile,
    use_task_fake: Any,
    tmp_path: Path,
) -> None:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": _draft([TRUTHY_BULLET]),
            "draft_letter": {"paragraphs": [{"text": OPINION, "atom_ids": []}]},
            "check_entailment": supported,
        }
    )

    artifacts = render.write(tailor(job, profile), profile, out_dir=tmp_path / "out")

    assert artifacts.cv_path.exists() and artifacts.letter_path.exists()
    json.loads(artifacts.audit_path.read_text(encoding="utf-8"))
    assert "TODO" not in artifacts.cv_path.read_text(encoding="utf-8")


def _requirement(text: str, skill: str, category: str = "must_have") -> dict[str, str]:
    return {"text": text, "skill": skill, "category": category}


def _paragraph_payload(text: str) -> dict[str, Any]:
    return {"text": text, "atom_ids": ["proj-csxedu-rag"]}


def _pgvector_fit() -> FitScore:
    return _fit(_outcome("pgvector search", "pgvector", matched=True))


def _draft(texts: list[str]) -> dict[str, Any]:
    return {"bullets": [{"text": text, "atom_ids": ["proj-csxedu-rag"]} for text in texts]}


def _paragraph(text: str, atom_ids: list[str]) -> DraftedParagraph:
    return DraftedParagraph(text=text, atom_ids=atom_ids)


def _check(text: str, atom_ids: list[str], profile: Profile) -> Any:
    return check_bullet(text, atom_ids, profile)


def _find_atom(profile: Profile, *, kind: str, containing: str) -> Atom:
    matches = [a for a in profile.atoms if a.kind == kind and containing.lower() in a.text.lower()]
    assert len(matches) == 1, matches
    return matches[0]


def _outcome(
    text: str,
    skill: str,
    *,
    matched: bool,
    category: str = "must_have",
    weight: float = 3.0,
) -> RequirementOutcome:
    return RequirementOutcome(
        requirement=ExtractedRequirement(text=text, skill=skill, category=category),
        weight=weight,
        matched=matched,
        via="code" if matched else "none",
    )


def _fit(*outcomes: RequirementOutcome) -> FitScore:
    return FitScore(
        job_uid="greenhouse:1",
        company="Example AI",
        title="Backend Engineer, RAG",
        total=0.0,
        band="weak",
        matched_weight=0.0,
        total_weight=0.0,
        location=LocationVerdict(
            home="Bangladesh", mode="remote", workable=True, note="remote, no country named"
        ),
        outcomes=list(outcomes),
    )


def every_point_is_attributed(trail: list[dict[str, Any]]) -> bool:
    """A point is only worth something if the fact base can name where it came from."""
    return all(not row["matched"] or row["evidence_atoms"] for row in trail)


_EXTRACTED: dict[str, Any] = {
    "requirements": [
        _requirement("Experience with vector search or pgvector", "pgvector"),
        _requirement("2+ years building backend services in Python", "Python"),
    ],
    "seniority": "mid",
    "location_constraint": "remote",
}
