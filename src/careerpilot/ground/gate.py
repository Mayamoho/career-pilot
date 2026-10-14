"""The groundedness gate.

This module is the reason the product is worth building. A tailored CV that contains
one invented metric is worse than no tailored CV at all, so nothing ships unless it
survives five checks, run cheapest first:

1. `no_citation`     — the bullet cites no profile atom.
2. `unknown_atom`    — it cites an id the profile does not contain.
3. `unsourced_number`— it states a number absent from the atoms it cites.
4. `misattributed_skill` — it names one of the candidate's technologies in a context
   that cites atoms which do not mention it.
5. `not_entailed`    — a judge model, shown only the bullet and its cited atoms, says
   the evidence does not support the claim.

Checks 1-4 are pure code and cost nothing. Only check 5 calls a model, and it calls
with a deliberately narrowed context: the judge never sees the full profile, so it
cannot fill gaps with knowledge the bullet did not cite.

A bullet that fails is dropped. It is not softened, hedged, or retried into
compliance. The report always says why.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from careerpilot.ai.gateway import ask
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.config import GROUNDING_MIN_CONFIDENCE
from careerpilot.profile.model import Atom, Profile
from careerpilot.textutil import canonical, has_term

FailureKind = Literal[
    "no_citation", "unknown_atom", "unsourced_number", "misattributed_skill", "not_entailed"
]

_YEAR_RANGE = re.compile(r"^(19\d{2}|20\d{2})$")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TECHISH = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[.#/+][A-Za-z0-9]+)+\b"
    r"|\b(?:Go|SQL|RAG|LLM|API|CI|CD|AWS|GCP|ML)\b"
)


class GroundingFailure(BaseModel):
    kind: FailureKind
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


class BulletVerdict(BaseModel):
    text: str
    atom_ids: list[str]
    shipped: bool
    failures: list[GroundingFailure] = Field(default_factory=list)


class GateReport(BaseModel):
    shipped: list[BulletVerdict] = Field(default_factory=list)
    rejected: list[BulletVerdict] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.shipped) + len(self.rejected)

    @property
    def pass_rate(self) -> float:
        return round(len(self.shipped) / self.total, 4) if self.total else 0.0

    def rejection_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for bullet in self.rejected:
            for failure in bullet.failures:
                counts[failure.kind] = counts.get(failure.kind, 0) + 1
        return counts


class Claim(StructuredOutput):
    text: str = Field(min_length=12, max_length=400)
    atom_ids: list[str] = Field(default_factory=list, max_length=5)


class ClaimList(StructuredOutput):
    claims: list[Claim] = Field(default_factory=list, max_length=12)


class EntailmentVerdict(StructuredOutput):
    supported: bool
    confidence: float = Field(ge=0, le=1)
    unsupported_span: str = Field(default="", max_length=200)


_JUDGE_SYSTEM = """You are an adversarial fact-checker for resume claims.

You receive ONE claim and the ONLY evidence permitted to support it. Judge whether the
evidence entails the claim.

- `supported` is true only if every factual element of the claim — every number, every
  technology, every superlative, every stated outcome — appears in the evidence.
- Being plausible is not being supported. Being the kind of thing a person with this
  background probably did is not supported.
- If the claim says "reduced", "improved", "increased" or any outcome, the evidence must
  state that outcome. A tool that exists is not proof it improved something.
- `unsupported_span`: quote the shortest fragment of the claim that the evidence does not
  support, or an empty string.
- `confidence`: how sure you are of your own verdict, 0 to 1.

Default to not supported when uncertain."""


def check_bullet(text: str, atom_ids: list[str], profile: Profile) -> BulletVerdict:
    failures: list[GroundingFailure] = []

    if not atom_ids:
        failures.append(GroundingFailure(
            kind="no_citation", detail="bullet cites no profile evidence"
        ))
    unknown = [i for i in atom_ids if i not in profile.by_id]
    if unknown:
        failures.append(GroundingFailure(
            kind="unknown_atom", detail=f"ids not present in fact base: {', '.join(unknown)}"
        ))

    atoms = [profile.by_id[i] for i in atom_ids if i in profile.by_id]
    surface = "\n".join(_surface(a) for a in atoms)

    if not unknown and atoms:
        for number in _numbers(text):
            if _YEAR_RANGE.match(number):
                continue
            if number not in _numbers(surface):
                failures.append(GroundingFailure(
                    kind="unsourced_number",
                    detail=f'"{number}" does not appear in the cited evidence',
                ))
        for failure in _misattributed_tech(text, atoms, profile):
            failures.append(failure)

    if not failures:
        verdict = _entailment(text, atoms)
        if verdict is None:
            failures.append(GroundingFailure(
                kind="not_entailed",
                detail="entailment check unavailable; refusing to ship an unverified claim",
            ))
        elif not verdict.supported or verdict.confidence < GROUNDING_MIN_CONFIDENCE:
            failures.append(GroundingFailure(
                kind="not_entailed",
                detail=(
                    f"judge confidence {verdict.confidence:.2f}; "
                    f"unsupported span: {verdict.unsupported_span or '(unspecified)'}"
                ),
            ))

    return BulletVerdict(text=text, atom_ids=atom_ids, shipped=not failures, failures=failures)


def gate(claims: list[Claim], profile: Profile) -> GateReport:
    report = GateReport()
    for claim in claims:
        verdict = check_bullet(claim.text, claim.atom_ids, profile)
        (report.shipped if verdict.shipped else report.rejected).append(verdict)
    return report


def _surface(atom: Atom) -> str:
    parts = [atom.text, *atom.skills, *[f"{k} {v}" for k, v in atom.metrics.items()]]
    return " | ".join(str(p) for p in parts if p)


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _misattributed_tech(text: str, atoms: list[Atom], profile: Profile) -> list[GroundingFailure]:
    """Flag a technology the candidate knows but that the *cited* atoms do not show.

    Restricted to the profile's own vocabulary on purpose. Ordinary English never
    matches, so there are no false positives on prose — but a bullet citing the
    Flutter screens while claiming Kubernetes autoscaling is caught exactly. That is
    the subtle failure mode: not inventing a skill, but attaching a real skill to
    work that never used it.
    """
    vocabulary = {canonical(s) for s in profile.skill_vocabulary if len(canonical(s)) > 1}
    cited_surfaces = [_surface(a) for a in atoms]
    failures: list[GroundingFailure] = []
    reported: set[str] = set()

    def consider(term: str) -> None:
        key = canonical(term)
        if key in reported or key not in vocabulary:
            return
        if not has_term(text, term):
            return
        if any(has_term(surface, term) for surface in cited_surfaces):
            return
        reported.add(key)
        failures.append(GroundingFailure(
            kind="misattributed_skill",
            detail=f'"{term}" is in the profile but not in the cited evidence',
        ))

    for token in _TECHISH.findall(text):
        consider(token)
    for atom in profile.atoms:
        for skill in atom.skills:
            consider(skill)
    return failures


def _entailment(text: str, atoms: list[Atom]) -> EntailmentVerdict | None:
    evidence = "\n".join(f"- {_surface(a)}  [id={a.id}]" for a in atoms)
    user = f"CLAIM:\n{text}\n\nPERMITTED EVIDENCE:\n{evidence}"
    try:
        return ask(
            task="check_entailment",
            system=_JUDGE_SYSTEM,
            user=user,
            schema=EntailmentVerdict,
        )
    except Exception:
        return None
