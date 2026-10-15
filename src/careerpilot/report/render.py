"""Building the artifacts a recruiter actually receives.

Only the "highlights" block is generated, and only lines that survived the gate appear
there. Everything else is copied verbatim from the fact base, which is grounded by
definition — so the CV is two things with different guarantees, stated plainly rather
than blurred together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from careerpilot.config import DATA_DIR
from careerpilot.pipeline import Tailored
from careerpilot.profile.model import Atom, Profile
from careerpilot.report import templates

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")
# `credential` was missing from this list, so a Coursera or NPTEL certificate — the kind
# of line a Bangladeshi student legitimately has — was silently dropped from every CV
# while still sitting in the fact base. A missing section is worse than a wrong one.
_SECTION_ORDER = [
    ("Tailored highlights", None),
    ("Experience", ("role",)),
    ("Projects", ("project",)),
    ("Certifications", ("credential",)),
    ("Recognition", ("award",)),
    ("Education", ("education",)),
    ("Skills", ("skill",)),
    ("Languages", ("language",)),
]


class Artifacts(BaseModel):
    cv_path: Path
    letter_path: Path
    audit_path: Path
    extra_paths: list[Path] = Field(default_factory=list)


def render_cv(profile: Profile, tailored: Tailored) -> str:
    # The headline is a section, not a decoration under the name: `careerpilot ats` looks
    # for a heading a parser can key on, and this file is one of its three inputs.
    lines = [f"# {profile.name}", ""]
    contact = _contact_line(profile)
    if contact:
        lines += [contact, ""]
    if profile.headline:
        lines += ["## Summary", "", profile.headline, ""]

    shipped = tailored.shipped_bullets
    lines += ["## Tailored highlights", ""]
    lines += [f"- {text}" for text in shipped] or [
        "- Nothing in the fact base could be phrased as a supported claim for this role."
    ]
    lines += [""]

    for title, kinds in _SECTION_ORDER:
        if kinds is None:
            continue
        atoms = [a for a in profile.atoms if a.kind in kinds]
        if not atoms:
            continue
        lines += [f"## {title}", ""]
        lines += [_atom_line(atom) for atom in atoms]
        lines += [""]
    return "\n".join(lines).rstrip() + "\n"


def render_letter(profile: Profile, tailored: Tailored) -> str:
    return "\n\n".join(letter_parts(profile, tailored)) + "\n"


def letter_parts(profile: Profile, tailored: Tailored) -> list[str]:
    """The letter's paragraphs, so the markdown and the HTML cannot drift apart."""
    job = tailored.scored.job
    paragraphs = tailored.letter or [
        (
            f"I am applying for {job.title}. My recorded experience does not support a "
            "stronger statement for this role, so I would rather say that than guess."
        )
    ]
    return [
        f"Dear hiring team at {job.company},",
        *paragraphs,
        f"Sincerely,\n{profile.name}",
    ]


def audit(tailored: Tailored) -> dict[str, Any]:
    job = tailored.scored.job
    bullet_report = tailored.bullets.report
    return {
        "job": {
            "uid": job.uid,
            "company": job.company,
            "title": job.title,
            "url": job.url,
            "source": job.source,
        },
        "fit": {
            "total": tailored.fit.total,
            "band": tailored.fit.band,
            "matched": tailored.fit.met,
            "requirements": len(tailored.fit.outcomes),
            "location": tailored.fit.location.model_dump(),
            "unmet_must_haves": tailored.fit.unmet_must_haves,
            "audit_trail": tailored.fit.audit_trail(),
        },
        "bullets": {
            "pass_rate": bullet_report.pass_rate,
            "rejected_kinds": bullet_report.rejection_summary(),
            "shipped": [
                {"text": v.text, "atom_ids": v.atom_ids} for v in bullet_report.shipped
            ],
            "rejected": [
                {"text": v.text, "atom_ids": v.atom_ids, "why": [str(f) for f in v.failures]}
                for v in bullet_report.rejected
            ],
        },
        "letter": {
            "pass_rate": tailored.letter_report.pass_rate,
            "rejected_kinds": tailored.letter_report.rejection_summary(),
            "shipped": [
                {"text": v.text, "atom_ids": v.atom_ids}
                for v in tailored.letter_report.shipped
            ],
            "rejected": [
                {"text": v.text, "atom_ids": v.atom_ids, "why": [str(f) for f in v.failures]}
                for v in tailored.letter_report.rejected
            ],
        },
        "targets": [
            {
                "requirement": t.requirement,
                "skill": t.skill,
                "category": t.category,
                "candidate_atoms": t.atom_ids,
            }
            for t in tailored.bullets.targets
        ],
    }


def write(
    tailored: Tailored,
    profile: Profile,
    out_dir: Path | None = None,
    styles: tuple[str, ...] = templates.STYLES,
) -> Artifacts:
    root = out_dir or (DATA_DIR / "out" / _safe(tailored.scored.job.uid))
    root.mkdir(parents=True, exist_ok=True)
    cv_path = root / "cv.md"
    letter_path = root / "cover-letter.md"
    audit_path = root / "audit.json"
    cv_path.write_text(render_cv(profile, tailored), encoding="utf-8")
    letter_path.write_text(render_letter(profile, tailored), encoding="utf-8")
    audit_path.write_text(json.dumps(audit(tailored), indent=2, default=str), encoding="utf-8")
    return Artifacts(
        cv_path=cv_path,
        letter_path=letter_path,
        audit_path=audit_path,
        extra_paths=write_cv(profile, root, tailored, styles),
    )


def write_cv(
    profile: Profile,
    out_dir: Path,
    tailored: Tailored | None = None,
    styles: tuple[str, ...] = templates.STYLES,
) -> list[Path]:
    """Every CV form this tool knows: one per print style, plus the plain text.

    `tailored` is optional because the CV a fresher needs before their first `tailor`
    run is still a CV. The plain-text file is what `ats` scores and what a portal's
    textarea takes; the HTML files are what becomes a PDF with `Ctrl+P`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for style in styles:
        path = out_dir / f"cv-{style}.html"
        path.write_text(templates.cv_html(profile, tailored, style), encoding="utf-8")
        written.append(path)
    text = out_dir / "cv.txt"
    text.write_text(templates.cv_txt(profile, tailored), encoding="utf-8")
    written.append(text)
    # The letter is HTML, so `--style md` asks for no letter file at all. Its markdown
    # twin is written by `write()` either way.
    if tailored is not None and styles:
        letter = out_dir / "cover-letter.html"
        letter.write_text(
            templates.letter_html(profile, letter_parts(profile, tailored)), encoding="utf-8"
        )
        written.append(letter)
    return written


def _safe(value: str) -> str:
    return _UNSAFE.sub("-", value)[:80]


def _contact_line(profile: Profile) -> str:
    wanted = ["email", "phone", "location", "github", "linkedin"]
    parts = [
        profile.contact[key]
        for key in wanted
        if key in profile.contact and "TODO" not in profile.contact[key]
    ]
    return " · ".join(parts)


def _atom_line(atom: Atom) -> str:
    stamp = " · ".join(filter(None, [atom.org, atom.period]))
    body = f"{atom.text} — {stamp}" if stamp else atom.text
    return f"- {body}"
