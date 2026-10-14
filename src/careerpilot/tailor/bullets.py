"""Drafting tailored CV bullets.

The model writes prose; code decides what it is allowed to write about. Each target
requirement comes with the handful of profile facts that could support it, and the
model is shown nothing else. A bullet that needs a fact that is not in the listing
has nowhere to cite it from, so it fails the gate rather than shipping — which is the
outcome we want from a generator that cannot be trusted to be honest.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from careerpilot.ai.gateway import ask
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.domain.job import JobPosting
from careerpilot.ground.gate import GateReport, check_bullet
from careerpilot.match.score import FitScore
from careerpilot.profile.model import Atom, Profile
from careerpilot.textutil import canonical, has_term

MAX_BULLETS = 8
MAX_FACTS_PER_TARGET = 5
# Below this a canonical skill ("c", "go", "r") is a substring of almost everything, so
# partial matches are only allowed once the term is long enough to mean something.
MIN_PARTIAL_SKILL_MATCH = 3


class DraftedBullet(StructuredOutput):
    text: str = Field(min_length=20, max_length=320)
    atom_ids: list[str] = Field(default_factory=list, max_length=4)


class BulletDraft(StructuredOutput):
    bullets: list[DraftedBullet] = Field(default_factory=list, max_length=MAX_BULLETS)


class Target(BaseModel):
    requirement: str
    skill: str
    category: str
    atoms: list[Atom] = Field(default_factory=list)

    @property
    def atom_ids(self) -> list[str]:
        return [atom.id for atom in self.atoms]


class TailoredBullets(BaseModel):
    targets: list[Target] = Field(default_factory=list)
    draft: list[DraftedBullet] = Field(default_factory=list)
    report: GateReport = Field(default_factory=GateReport)


_SYSTEM_DRAFT = """Rewrite a candidate's own verified facts into CV bullets for one job.

You are given TARGET REQUIREMENTS and the ONLY FACTS you may use, each with an id.

- Every bullet must cite at least one fact id. Copy ids exactly as given.
- Use only what a fact states. Do not add a number, a technology, an outcome or a
  superlative that is not written in the facts you cite.
- Prefer the facts that carry concrete metrics.
- Do not combine unrelated facts into one sentence.
- Write in past tense, one clause per bullet, no more than 320 characters, no first
  person and no adjectives about how impressive the work was.
- If nothing in the facts supports a requirement, write no bullet for it. Fewer
  truthful bullets than absent bullets is the correct outcome.
- Return at most 8 bullets, most relevant first."""


def select_targets(fit: FitScore, profile: Profile, limit: int = 6) -> list[Target]:
    """Pick the requirements worth a bullet, then the facts that could back them.

    A target with no available evidence is dropped before the model is asked, so the
    generator is never placed in the position of having to invent something.
    """
    ordered = sorted(fit.outcomes, key=lambda o: (-o.weight, not o.matched))
    targets: list[Target] = []
    seen_skills: set[str] = set()
    for outcome in ordered:
        req = outcome.requirement
        key = canonical(req.skill)
        if key in seen_skills:
            continue
        atoms = _evidence_for(key, profile)
        if not atoms:
            continue
        seen_skills.add(key)
        targets.append(
            Target(
                requirement=req.text,
                skill=req.skill,
                category=req.category,
                atoms=atoms[:MAX_FACTS_PER_TARGET],
            )
        )
        if len(targets) >= limit:
            break
    return targets


def _evidence_for(skill: str, profile: Profile) -> list[Atom]:
    """Rank profile atoms against one requirement skill, without asking a model.

    An atom qualifies only if it actually names the skill. The kind bonus reorders the
    qualifying atoms; it never admits one, because admitting every project would hand
    the generator the entire profile and turn "cite your evidence" into "cite anything".
    """
    needle = canonical(skill)
    if not needle:
        return []
    ranked: list[tuple[int, int, Atom]] = []
    for position, atom in enumerate(profile.atoms):
        declared = {canonical(s) for s in atom.skills}
        score = 0
        if needle in declared:
            score += 4
        elif any(
            len(other) >= MIN_PARTIAL_SKILL_MATCH and (other in needle or needle in other)
            for other in declared
        ):
            score += 2
        if has_term(atom.text, skill):
            score += 3
        if not score:
            continue
        score += 2 if atom.kind in {"project", "role", "award"} else 1
        ranked.append((-score, position, atom))
    ranked.sort()
    return [atom for _, _, atom in ranked]


def draft_bullets(job: JobPosting, targets: list[Target]) -> list[DraftedBullet]:
    if not targets:
        return []
    facts = "\n".join(fact_lines(targets))
    wanted = "\n".join(
        f"- [{t.category}] {t.requirement} (skill: {t.skill})" for t in targets
    )
    user = (
        f"ROLE: {job.title} at {job.company}\n"
        f"LOCATION: {job.location or 'unspecified'}\n\n"
        f"TARGET REQUIREMENTS:\n{wanted}\n\n"
        f"FACTS (the only permitted source):\n{facts}"
    )
    draft = ask(task="draft_bullets", system=_SYSTEM_DRAFT, user=user, schema=BulletDraft)
    return draft.bullets


def fact_lines(targets: list[Target]) -> list[str]:
    lines: list[str] = []
    emitted: set[str] = set()
    for target in targets:
        for atom in target.atoms:
            if atom.id in emitted:
                continue
            emitted.add(atom.id)
            detail = " | ".join(filter(None, [atom.org, atom.period]))
            lines.append(f"- [id={atom.id}]{' ' + detail if detail else ''} {atom.text}")
            if atom.skills:
                lines.append(f"    skills: {', '.join(atom.skills)}")
            if atom.metrics:
                lines.append(
                    "    metrics: " + ", ".join(f"{k}={v}" for k, v in atom.metrics.items())
                )
    return lines


def tailor_bullets(job: JobPosting, fit: FitScore, profile: Profile) -> TailoredBullets:
    """Draft, then run every draft line through the groundedness gate."""
    targets = select_targets(fit, profile)
    result = TailoredBullets(targets=targets)
    result.draft = draft_bullets(job, targets)
    for bullet in result.draft:
        verdict = check_bullet(bullet.text, bullet.atom_ids, profile)
        (result.report.shipped if verdict.shipped else result.report.rejected).append(verdict)
    return result
