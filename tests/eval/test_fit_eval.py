"""Fit eval: what a model can and cannot do to a score.

The architecture's central promise is that the model handles language and code handles
facts. For scoring, that is a testable claim rather than a vibe: whatever the model
answers, the number cannot go up on its own say-so. This file enforces that in CI with
no provider, no cost and no cassette, by running one labelled corpus against gateways
of ascending dishonesty — and it states, in an assertion, the one hole that code does
not close.

The nightly live run is the other half: whether the real model cites evidence a human
would accept, which is a rate rather than an invariant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from careerpilot import pipeline
from careerpilot.ai.gateway import set_gateway
from careerpilot.config import live_mode_enabled, profile_path
from careerpilot.domain.job import JobPosting
from careerpilot.match.extract import RequirementSet
from careerpilot.match.score import STRONG_BAND_MIN, FitScore, score_job
from careerpilot.profile.model import Profile, load_profile

# An id that does not exist must be worth exactly nothing. Not "nearly nothing".
MAX_HALLUCINATED_POINTS = 0.0
BAND_AGREEMENT_MIN = 0.75

Requirement = dict[str, str]


def _req(text: str, skill: str, category: str = "must_have") -> Requirement:
    return {"text": text, "skill": skill, "category": category}


@dataclass(frozen=True)
class LabelledJob:
    """A posting, and the verdict this fact base genuinely deserves.

    `requirements` is the extraction a careful human reader would produce, held fixed
    so the only variable under test is how honest the citing model is.
    """

    title: str
    company: str
    location: str
    description: str
    requirements: tuple[Requirement, ...]
    expected_band: Literal["strong", "partial", "weak"]
    why: str = ""

    def posting(self) -> JobPosting:
        return JobPosting(
            source="greenhouse",
            source_id=self.company.lower().replace(" ", "-")[:40],
            company=self.company,
            title=self.title,
            location=self.location,
            description=self.description,
        )

    def requirement_set(self) -> RequirementSet:
        return RequirementSet.model_validate(
            {
                "requirements": list(self.requirements),
                "seniority": "unspecified",
                "location_constraint": "unspecified",
            }
        )


BACKEND_PLATFORM = LabelledJob(
    title="Backend Engineer, AI Platform",
    company="Northwind AI",
    location="Remote",
    description=(
        "Requirements:\n- MongoDB for the service register\n- Express or a comparable "
        "Node framework\n- TypeScript backend services\n- REST API design\n"
        "- PostgreSQL, and vector search on top of it\n- Server-sent event streaming\n"
        "- Dockerised deployment\n"
    ),
    requirements=(
        _req("MongoDB for the service register", "MongoDB"),
        _req("Express or a comparable Node framework", "Express"),
        _req("TypeScript backend services", "TypeScript"),
        _req("REST API design", "REST"),
        _req("PostgreSQL and vector search", "pgvector"),
        _req("Server-sent event streaming", "SSE", "nice_to_have"),
        _req("Dockerised deployment", "Docker", "nice_to_have"),
    ),
    expected_band="strong",
    why="every one of these is demonstrated by shipped work, not by a skills list",
)

STAFF_SRE = LabelledJob(
    title="Staff Site Reliability Engineer",
    company="Meridian Cloud",
    location="New York, NY",
    description=(
        "Requirements:\n- 8+ years operating production Kubernetes\n"
        "- Terraform modules for a large estate\n"
        "- Go services you have owned in production\n- A 24/7 on-call rotation\n"
    ),
    requirements=(
        _req("8+ years operating production Kubernetes", "Kubernetes"),
        _req("Terraform modules at scale", "Terraform"),
        _req("Go services owned in production", "Go"),
        _req("A 24/7 on-call rotation", "on-call"),
    ),
    expected_band="weak",
    why="one declared language skill is not an SRE estate",
)

PRODUCT_FULL_STACK = LabelledJob(
    title="Full-stack Engineer, Product",
    company="Lumen Health",
    location="Remote",
    description=(
        "Requirements:\n- Next.js in production\n- React with TypeScript\n- Tailwind\n"
        "- Authentication and session management\n- A relational database behind an API\n"
    ),
    requirements=(
        _req("Next.js in production", "Next.js"),
        _req("React with TypeScript", "React"),
        _req("Tailwind CSS", "Tailwind", "nice_to_have"),
        _req("Authentication and sessions", "authentication"),
        _req("A relational database behind an API", "PostgreSQL"),
    ),
    expected_band="strong",
    why="the portal work is this stack, deployed and reachable",
)

APPSEC = LabelledJob(
    title="Application Security Engineer",
    company="Cardamom Systems",
    location="Dhaka, Bangladesh",
    description=(
        "Requirements:\n- Web application security\n- Threat modelling\n"
        "- SAST or DAST in CI\n- CTF-level problem solving\n"
        "- Digital forensics or incident response\n- OAuth and JWT implementation\n"
    ),
    requirements=(
        _req("Web application security", "web security"),
        _req("Threat modelling", "threat modelling"),
        _req("SAST or DAST in CI", "SAST", "nice_to_have"),
        _req("CTF-level problem solving", "CTF", "nice_to_have"),
        _req("Digital forensics or incident response", "forensics"),
        _req("OAuth and JWT implementation", "OAuth"),
    ),
    expected_band="partial",
    why="real security work and real CTF results, no tooling or incident-response evidence",
)

CORPUS: tuple[LabelledJob, ...] = (BACKEND_PLATFORM, STAFF_SRE, PRODUCT_FULL_STACK, APPSEC)

# Two models, and only the second is trying to help.
NO_EVIDENCE: dict[str, Any] = {"atom_ids": []}
INVENTED_IDS: dict[str, Any] = {
    "atom_ids": ["proj-invented", "role-phantom", "award-nothing", "skill-fantasy"]
}
# Ids that exist, so the existence check waves them through, and demonstrate nothing.
BORROWED_IDS: dict[str, Any] = {
    "atom_ids": ["proj-bornomala-ppm", "proj-learnloop-work", "exp-eventify", "lang-bengali"]
}


@pytest.mark.eval
def test_an_invented_citation_is_worth_exactly_nothing(
    profile: Profile, use_task_fake: Any
) -> None:
    """Existence checking is the whole defence, so it gets a zero-tolerance threshold.

    If anyone widens `_model_match` — a fuzzy id match, an "closest atom" fallback, a
    tolerant normaliser — a hallucinated id starts buying points and this fails here
    instead of on a printed CV.
    """
    for job in CORPUS:
        use_task_fake({"cite_evidence": NO_EVIDENCE})
        baseline = score_job(job.posting(), job.requirement_set(), profile)

        use_task_fake({"cite_evidence": INVENTED_IDS})
        scored = score_job(job.posting(), job.requirement_set(), profile)

        delta = scored.total - baseline.total
        assert delta <= MAX_HALLUCINATED_POINTS, f"{job.company}: +{delta} from fake ids"
        assert scored.band == baseline.band


@pytest.mark.eval
def test_the_seniority_gate_holds_against_a_model_that_cites_everything(
    profile: Profile, use_task_fake: Any
) -> None:
    """The one structural backstop against a deceptive citation, and it is arithmetic.

    The model is handed real project and role ids — the largest citation
    `_model_match` will accept — and the "8+ years" requirement still comes back
    unmatched, because the gate returns before a model is ever consulted.
    """
    use_task_fake({"cite_evidence": _cite_all_work(profile)})

    fit = score_job(STAFF_SRE.posting(), STAFF_SRE.requirement_set(), profile)
    gated = [o for o in fit.outcomes if "8+ years" in o.requirement.text]

    assert gated, "the corpus lost its seniority requirement"
    assert all(o.matched is False and "years required" in o.note for o in gated)


@pytest.mark.eval
def test_a_borrowed_citation_lifts_a_bad_fit_to_partial_and_no_further(
    profile: Profile, use_task_fake: Any
) -> None:
    """The hole, pinned rather than denied.

    A real id that proves nothing does add weight, and no pure-code check can see it:
    the atom exists, so `_model_match` accepts it. The score reaches 75 — the raw
    arithmetic alone would have called this a strong fit. What stops it is the band
    rule, which refuses "strong" while any must-have is unmet, and the seniority gate
    behind that. Both are code, and the audit trail names the atoms, so the drift is a
    human-visible error rather than an invisible one.

    If a future change starts verifying that the cited atom mentions the skill, this
    total drops and the test fails on purpose: that is the signal to raise the bar, not
    to edit the number.
    """
    use_task_fake({"cite_evidence": BORROWED_IDS})

    fit = score_job(STAFF_SRE.posting(), STAFF_SRE.requirement_set(), profile)

    assert fit.total >= STRONG_BAND_MIN, "the deceptive-citation damage vanished; re-read me"
    assert fit.band == "partial", f"borrowed citations produced {fit.band} at {fit.total}"
    assert fit.unmet_must_haves == ["8+ years operating production Kubernetes"]


@pytest.mark.eval
def test_the_labels_are_not_wishful(profile: Profile, use_task_fake: Any) -> None:
    """Every labelled band must still be the band the code produces.

    Not a safety property; a hygiene one. A stale expectation turns the nightly eval
    into theatre, so the labels are re-checked on every push for free.
    """
    use_task_fake({"cite_evidence": NO_EVIDENCE})

    for job in CORPUS:
        fit = score_job(job.posting(), job.requirement_set(), profile)
        assert fit.band == job.expected_band, (
            f"{job.title}: labelled {job.expected_band}, scored {fit.band} at {fit.total}; "
            f"unmet must-haves: {fit.unmet_must_haves} — {job.why}"
        )


def _cite_all_work(profile: Profile) -> dict[str, Any]:
    """The largest citation the schema will accept — six real ids, four of them counted."""
    work = [a.id for a in profile.atoms if a.kind in {"project", "role"}]
    return {"atom_ids": work[:6]}


def _needs_credentials() -> bool:
    return not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@pytest.mark.eval
@pytest.mark.live
@pytest.mark.skipif(_needs_credentials(), reason="no provider credential in the environment")
def test_the_real_model_agrees_with_the_labels() -> None:
    """Nightly: extraction, citation and scoring against a real model, end to end.

    Everything above holds the model's output fixed. This is the half that cannot be
    deterministic, so it is a rate with a floor rather than an invariant — and one
    invariant it keeps: a job the profile does not fit is never called strong.
    """
    os.environ["CAREERPILOT_LIVE"] = "1"
    live_mode_enabled.cache_clear()
    try:
        candidate = load_profile(profile_path())
        agreed: list[FitScore] = []
        failures: list[str] = []
        for job in CORPUS:
            fit = pipeline.score(job.posting(), candidate).fit
            if fit.band == job.expected_band:
                agreed.append(fit)
            else:
                failures.append(f"{job.company}: wanted {job.expected_band}, got {fit.band}")
            if job.expected_band != "strong" and fit.band == "strong":
                failures.append(f"{job.company}: called a {job.expected_band} fit strong")

        rate = len(agreed) / len(CORPUS)
        assert not failures, failures
        assert rate >= BAND_AGREEMENT_MIN, f"only {rate:.0%} agreed: {failures}"
    finally:
        os.environ.pop("CAREERPILOT_LIVE", None)
        live_mode_enabled.cache_clear()
        set_gateway(None)
