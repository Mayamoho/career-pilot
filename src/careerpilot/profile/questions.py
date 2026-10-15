"""`start-here`: the questionnaire for someone who believes they have nothing to record.

A final-year student in Dhaka usually has more facts than they think — a lab project,
an industrial attachment, an IEEE workshop, an NPTEL certificate, an HSC GPA — and the
thing they lack is the file format. So this asks for the evidence in the words a CV
needs and writes the YAML itself, then validates it before touching disk.

There is no model in this path. A questionnaire that invents stronger wording for your
projects is the exact failure this whole project is built to avoid.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import yaml
from pydantic import ValidationError

from careerpilot.profile.model import Atom, AtomKind, Profile

_MIN_FACT_CHARS = 8

# The one format every multi-line answer accepts, and the reason it is that format:
# `org` and `period` are what the fit scorer computes years-of-experience from, so a
# free-text line like "intern at X (2024)" would be one fact where the code needs three.
ENTRY_FORMAT = "one per line as `what you did | where | 2024-2025` — the last two are optional"


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    hint: str = ""
    answer: Literal["single", "lines", "contact"] = "single"
    kind: AtomKind | None = None
    contact_key: str = ""


ASK: tuple[Question, ...] = (
    Question("name", "Your full name (as it should appear on the CV)"),
    Question(
        "headline",
        "One line under your name — what you are aiming at",
        "e.g. 'Python & Django developer — final year CSE student, Dhaka'",
    ),
    Question("email", "Email a recruiter can reach you on", answer="contact", contact_key="email"),
    Question(
        "phone",
        "Phone with country code",
        "e.g. +8801XXXXXXXXX — the +880 is what makes it dialable from outside Bangladesh",
        answer="contact",
        contact_key="phone",
    ),
    Question(
        "location",
        "City, Country",
        "e.g. Mirpur, Dhaka, Bangladesh — this is what decides which jobs are workable",
        answer="contact",
        contact_key="location",
    ),
    Question(
        "github",
        "GitHub profile URL (blank if you have none)",
        answer="contact",
        contact_key="github",
    ),
    Question(
        "linkedin",
        "LinkedIn URL (blank if you have none)",
        answer="contact",
        contact_key="linkedin",
    ),
    Question(
        "skills",
        "Technologies you have actually used, comma-separated",
        "python, django, git, mysql — list what you have written code with, not what you "
        "have heard of",
    ),
    Question(
        "projects",
        "Projects: coursework, lab, personal, freelance or Upwork — anything you built",
        ENTRY_FORMAT,
        answer="lines",
        kind="project",
    ),
    Question(
        "work",
        "Internship, industrial training, attachment or part-time work",
        ENTRY_FORMAT,
        answer="lines",
        kind="role",
    ),
    Question(
        "education",
        "Education: SSC, HSC, diploma, university — with GPA or result if you have one",
        ENTRY_FORMAT,
        answer="lines",
        kind="education",
    ),
    Question(
        "certificates",
        "Courses and certificates: NPTEL, Coursera, a university tech club workshop",
        ENTRY_FORMAT,
        answer="lines",
        kind="credential",
    ),
    Question(
        "recognition",
        "Hackathons, contests, scholarships, club responsibility with a result",
        ENTRY_FORMAT,
        answer="lines",
        kind="award",
    ),
    Question(
        "languages",
        "Languages you can work in, with the level",
        "e.g. Bangla (native), English (fluent — IELTS 6.5)",
        answer="lines",
        kind="language",
    ),
)

_YEARS = re.compile(r"(?:19|20)\d{2}")
_SLUG = re.compile(r"[^a-z0-9]+")


def build(name: str, headline: str, contact: dict[str, str], facts: list[Atom]) -> Profile:
    """Assemble and validate. Nothing is written until this returns."""
    return Profile.model_validate({
        "name": name,
        "headline": headline,
        "contact": {k: v for k, v in contact.items() if v},
        "atoms": [atom.model_dump() for atom in facts],
    })


def parse_entry(line: str, kind: AtomKind) -> Atom | None:
    """One line of the questionnaire, in the documented `text | org | period` form."""
    padded = [part.strip() for part in line.split("|")] + ["", ""]
    text = padded[0]
    if len(text) < _MIN_FACT_CHARS:
        return None
    stated = padded[2]
    found = _YEARS.findall(text)
    period = stated or (
        f"{found[0]} - {found[-1]}" if len(found) > 1 else (found[0] if found else "")
    )
    return Atom(
        id=_atom_id(kind, text),
        kind=kind,
        text=text,
        org=padded[1] or None,
        period=period or None,
        source="start-here",
    )


def atoms_from(kind: AtomKind, lines: Iterable[str]) -> tuple[list[Atom], list[str]]:
    """Turn answered lines into atoms, and keep the ones too thin to be a fact.

    A line shorter than the model's own minimum is reported rather than silently
    dropped: "Python" as a whole project entry means the student had something to say
    and needs to hear what "enough" looks like.
    """
    kept: list[Atom] = []
    skipped: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        atom = parse_entry(text, kind)
        if atom is None:
            skipped.append(text)
        else:
            kept.append(atom)
    return kept, skipped


def skill_atom(line: str) -> Atom | None:
    skills = [part.strip() for part in line.split(",") if part.strip()]
    if not skills:
        return None
    text = ", ".join(skills)
    return Atom(
        id=_atom_id("skill", text), kind="skill", text=text, skills=skills, source="start-here"
    )


def to_yaml(profile: Profile) -> str:
    """The same shape a hand-written fact base has, so the next step is a text file you own."""
    payload: dict[str, object] = {
        "name": profile.name,
        "headline": profile.headline,
        "contact": profile.contact,
        "atoms": [
            {
                k: v
                for k, v in atom.model_dump(mode="json").items()
                if v not in (None, "", [], {})
            }
            for atom in profile.atoms
        ],
    }
    return _HEADER + yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100
    )


def collect(ask: Callable[[Question], list[str]]) -> tuple[Profile, list[str]]:
    """Run the questionnaire through `ask` and return the profile plus anything dropped.

    `ask` is a parameter rather than a `input()` call so the whole flow is testable and
    so the CLI can print the hint lines without this module knowing what a terminal is.
    """
    name = headline = ""
    contact: dict[str, str] = {}
    atoms: list[Atom] = []
    dropped: list[str] = []
    for question in ASK:
        answers = [a.strip() for a in ask(question) if a and a.strip()]
        if question.key == "name":
            name = answers[0] if answers else ""
        elif question.key == "headline":
            headline = answers[0] if answers else ""
        elif question.answer == "contact":
            contact[question.contact_key] = answers[0] if answers else ""
        elif question.key == "skills":
            atom = skill_atom(", ".join(answers))
            if atom:
                atoms.append(atom)
        elif question.kind is not None:
            kept, skipped = atoms_from(question.kind, answers)
            atoms.extend(kept)
            dropped.extend(skipped)
    if not atoms:
        atoms.append(Atom(
            id="skill-starter", kind="skill", text="No technologies recorded yet",
            source="start-here",
        ))
    profile = build(name or "Unnamed", headline, contact, _unique(atoms))
    return profile, dropped


def _unique(atoms: list[Atom]) -> list[Atom]:
    """`Atom.id` is the citation key the gate looks up, so two atoms cannot share one.

    Two entries that start with the same words slug to the same id; the model rejects
    duplicates outright, and a questionnaire that dies on a typographical coincidence
    teaches the student nothing.
    """
    seen: set[str] = set()
    out: list[Atom] = []
    for atom in atoms:
        candidate, suffix = atom.id, 1
        while candidate in seen:
            suffix += 1
            candidate = f"{atom.id}-{suffix}"
        seen.add(candidate)
        out.append(atom if candidate == atom.id else atom.model_copy(update={"id": candidate}))
    return out


_HEADER = """\
# Written by `careerpilot start-here` on the day you answered it.
#
# Every bullet in every tailored CV must be traceable to one of these atoms, so keep it
# factual and keep the numbers exact: the digits in `text` and in `metrics` are the only
# numbers the generator is ever allowed to claim. Add to it as you work — this file is
# the asset, the CVs are what it prints.
"""


def _atom_id(kind: str, text: str) -> str:
    """An id that satisfies the model's pattern without asking a beginner to write one."""
    slug = _SLUG.sub("-", text.lower()).strip("-")[:40].rstrip("-")
    return f"{kind}-{slug}" if slug else kind


def validate(payload: object) -> str | None:
    """A human-readable complaint instead of a pydantic traceback."""
    try:
        Profile.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - the CLI prints this
        first = exc.errors()[0]
        return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return None
