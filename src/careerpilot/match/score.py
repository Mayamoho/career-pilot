"""Fit scoring. Every point is computed here, in code, from cited evidence.

The model is allowed to do exactly one thing: point at atom ids. It cannot emit a
score, a percentage, or a judgement about fit. Two consequences worth stating plainly:

* a hallucinated atom id resolves to nothing and the requirement stays unmatched, so
  the failure mode is an under-score, never an inflated one;
* years-of-experience requirements are settled by arithmetic on the profile's own
  date ranges, because that is the one thing a language model reliably gets wrong.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from careerpilot.ai.gateway import ask
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.domain.job import JobPosting
from careerpilot.match.extract import CATEGORY_WEIGHTS, ExtractedRequirement, RequirementSet
from careerpilot.profile.model import Atom, Profile
from careerpilot.textutil import canonical, has_term

MatchVia = Literal["code", "model", "none"]

_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)

STRONG_BAND_MIN = 70.0
PARTIAL_BAND_MIN = 45.0


class RequirementOutcome(BaseModel):
    requirement: ExtractedRequirement
    weight: float
    matched: bool = False
    via: MatchVia = "none"
    evidence_atom_ids: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def evidence(self) -> list[str]:
        return [self.requirement.text, *self.evidence_atom_ids]


class FitScore(BaseModel):
    job_uid: str
    company: str
    title: str
    total: float
    band: Literal["strong", "partial", "weak"]
    matched_weight: float
    total_weight: float
    outcomes: list[RequirementOutcome]

    @property
    def met(self) -> int:
        return sum(1 for o in self.outcomes if o.matched)

    @property
    def unmet_must_haves(self) -> list[str]:
        return [
            o.requirement.text for o in self.outcomes
            if not o.matched and o.requirement.category == "must_have"
        ]

    def audit_trail(self) -> list[dict[str, object]]:
        return [
            {
                "requirement": o.requirement.text,
                "skill": o.requirement.skill,
                "category": o.requirement.category,
                "weight": o.weight,
                "matched": o.matched,
                "via": o.via,
                "evidence_atoms": o.evidence_atom_ids,
            }
            for o in self.outcomes
        ]


class EvidenceCitation(StructuredOutput):
    """The model may name evidence or say nothing. It may not say 'close enough'."""

    atom_ids: list[str] = Field(default_factory=list, max_length=6)


_SYSTEM_CITE = """Decide whether the candidate profile contains evidence for ONE requirement.

You are given the requirement and a numbered list of profile facts. Reply with the ids
of the facts that genuinely demonstrate the requirement.

- A fact demonstrates the requirement only if it states concrete work using that
  technology or capability. A generic skills list is not demonstration.
- If no fact demonstrates it, return an empty list. An empty list is the correct
  answer far more often than people expect.
- Never invent an id. Copy ids exactly as given.
- Do not comment on fit, strength, or percentages. Choose ids or return none."""


def score_job(job: JobPosting, requirements: RequirementSet, profile: Profile) -> FitScore:
    outcomes: list[RequirementOutcome] = []
    for req in requirements.requirements:
        weight = CATEGORY_WEIGHTS.get(req.category, 0.5)
        outcomes.append(_evaluate(req, weight, profile))

    total_weight = sum(o.weight for o in outcomes)
    matched_weight = sum(o.weight for o in outcomes if o.matched)
    total = round(100.0 * matched_weight / total_weight, 1) if total_weight > 0 else 0.0
    return FitScore(
        job_uid=job.uid,
        company=job.company,
        title=job.title,
        total=total,
        band=_band(total, outcomes),
        matched_weight=round(matched_weight, 3),
        total_weight=round(total_weight, 3),
        outcomes=outcomes,
    )


def _evaluate(req: ExtractedRequirement, weight: float, profile: Profile) -> RequirementOutcome:
    years_gate = _years_required(req.text)
    if years_gate is not None:
        available = _years_of_experience(profile)
        if available < years_gate:
            return RequirementOutcome(
                requirement=req, weight=weight, matched=False, via="none",
                note=f"code: {years_gate}+ years required, profile spans {available:.1f}",
            )

    direct = _code_match(req.skill, profile)
    if direct:
        return RequirementOutcome(
            requirement=req, weight=weight, matched=True, via="code",
            evidence_atom_ids=[a.id for a in direct[:4]],
            note="code: skill matched against declared profile vocabulary",
        )

    cited = _model_match(req, profile)
    if cited:
        return RequirementOutcome(
            requirement=req, weight=weight, matched=True, via="model",
            evidence_atom_ids=[a.id for a in cited],
            note="model cited existing evidence",
        )
    return RequirementOutcome(
        requirement=req, weight=weight, matched=False, via="none",
        note="no evidence found in profile",
    )


def _code_match(skill: str, profile: Profile) -> list[Atom]:
    """Match a requirement skill against the fact base without asking a model.

    Declared `skills` are compared canonically; free text is compared with boundary
    guards so that a requirement for "Go" cannot be satisfied by "GoDaddy" or by the
    "go" inside "Bangla". Ordering puts concrete work above a skills-list mention,
    which matters because the ids chosen here become the audit trail.
    """
    needle = canonical(skill)
    if len(needle) < 1:
        return []
    strong: list[Atom] = []
    weak: list[Atom] = []
    for atom in profile.atoms:
        if needle in {canonical(s) for s in atom.skills}:
            (strong if atom.kind in {"project", "role", "award"} else weak).append(atom)
        elif atom.kind in {"project", "role", "award"} and has_term(atom.text, skill):
            strong.append(atom)
    return strong + weak


def _model_match(req: ExtractedRequirement, profile: Profile) -> list[Atom]:
    candidates = [a for a in profile.atoms if a.kind in {"project", "role", "award", "skill"}]
    if not candidates:
        return []
    listing = "\n".join(f"{atom.id}: {atom.text}" for atom in candidates)
    user = f"REQUIREMENT: {req.text}\nSKILL: {req.skill}\n\nPROFILE FACTS:\n{listing}"
    try:
        cited = ask(task="cite_evidence", system=_SYSTEM_CITE, user=user, schema=EvidenceCitation)
    except Exception:
        return []
    by_id = profile.by_id
    # Existence check is the whole safety property: an id the profile does not have
    # cannot contribute weight.
    return [by_id[i] for i in cited.atom_ids if i in by_id][:4]


def _band(total: float, outcomes: list[RequirementOutcome]) -> str:
    must_have_misses = sum(
        1 for o in outcomes if o.requirement.category == "must_have" and not o.matched
    )
    if total >= STRONG_BAND_MIN and must_have_misses == 0:
        return "strong"
    if total >= PARTIAL_BAND_MIN:
        return "partial"
    return "weak"


def _years_required(text: str) -> int | None:
    match = _YEARS.search(text)
    return int(match.group(1)) if match else None


def _years_of_experience(profile: Profile) -> float:
    """Span of dated role/project atoms, computed from the fact base.

    An end is never later than today. "present" ends now, and a dated atom ends in
    December of its stated year — but a project dated "2026" read in October would
    otherwise close its period two and a half months in the future and hand the
    candidate work they have not done yet. That is the one place the arithmetic can
    inflate a score by itself, so it is the place that has to be conservative.
    """
    today = date.today()
    starts: list[date] = []
    ends: list[date] = []
    for atom in profile.atoms:
        if atom.kind not in {"role", "project"} or not atom.period:
            continue
        found = re.findall(r"(?:19|20)\d{2}", atom.period)
        if not found:
            continue
        starts.append(date(int(found[0]), 1, 1))
        ends.append(
            today
            if "present" in atom.period.lower()
            else min(today, date(int(found[-1]), 12, 31))
        )
    if not starts:
        return 0.0
    return max(0.0, (max(ends) - min(starts)).days / 365.25)
