"""The candidate fact base: the only thing the generator is allowed to draw from.

Each `Atom` is one independently verifiable statement, with an id. The id is the
whole design. A generated bullet carries the ids it was built from, so "did the model
invent this?" becomes a lookup rather than a judgement call — and a bullet citing an
id that does not exist is a hard failure caught before any model is consulted.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

AtomKind = Literal["role", "project", "skill", "education", "award", "credential", "language"]


class Atom(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    kind: AtomKind
    text: str = Field(min_length=8)
    skills: list[str] = Field(default_factory=list)
    metrics: dict[str, str] = Field(default_factory=dict)
    source: str = "cv"
    org: str | None = None
    period: str | None = None
    until: date | None = None

    @field_validator("skills")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for skill in value:
            key = skill.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(skill.strip())
        return out


class Profile(BaseModel):
    name: str
    headline: str
    contact: dict[str, str] = Field(default_factory=dict)
    atoms: list[Atom]

    @field_validator("atoms")
    @classmethod
    def _unique_ids(cls, value: list[Atom]) -> list[Atom]:
        ids = [atom.id for atom in value]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate atom ids: {sorted(dupes)}")
        return value

    @property
    def by_id(self) -> dict[str, Atom]:
        return {atom.id: atom for atom in self.atoms}

    def resolve(self, atom_id: str) -> Atom | None:
        return self.by_id.get(atom_id)

    def evidence_text(self, atom_ids: list[str]) -> str:
        """Concatenated source text for the ids a bullet claims to be built from."""
        return "\n".join(
            _atom_surface(self.by_id[i]) for i in atom_ids if i in self.by_id
        )

    @property
    def skill_vocabulary(self) -> set[str]:
        """Every technology the profile actually declares.

        A `skill` atom's text is a comma-separated list, so it is split on commas and
        never into loose words: "Data structures and algorithms" is one technology,
        and "and" is never one. Everything downstream that asks "is this bullet a
        claim about the candidate's stack?" depends on this staying technology-only.
        """
        vocab: set[str] = set()
        for atom in self.atoms:
            vocab.update(s.lower() for s in atom.skills)
            if atom.kind == "skill":
                vocab.update(part.strip().lower() for part in atom.text.split(","))
        return {v for v in vocab if v}

    @property
    def number_vocabulary(self) -> set[str]:
        """Every number the profile states outright, as strings.

        The groundedness gate uses this to reject a bullet that says "40%" when no
        atom anywhere in the fact base contains 40. This is the single check that
        catches the most common and most damaging CV fabrication.
        """
        found: set[str] = set()
        for atom in self.atoms:
            found.update(_NUMBER.findall(atom.text))
            for value in atom.metrics.values():
                found.update(_NUMBER.findall(str(value)))
                found.update(_NUMBER.findall(value))
        return found

    def atoms_for_skills(self, skills: set[str]) -> list[Atom]:
        wanted = {s.lower() for s in skills if s}
        return [
            atom for atom in self.atoms
            if wanted & {s.lower() for s in atom.skills}
        ]


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _atom_surface(atom: Atom) -> str:
    parts = [atom.text]
    if atom.skills:
        parts.append("skills: " + ", ".join(atom.skills))
    if atom.metrics:
        parts.append("metrics: " + ", ".join(f"{k}={v}" for k, v in atom.metrics.items()))
    return " | ".join(parts)


def load_profile(path: Path) -> Profile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Profile.model_validate(raw)
