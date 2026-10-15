"""Will a machine be able to read this CV, and what is it missing?

No model and no network: the question has a deterministic answer, and "keywords you
should add" is precisely the place where a generated suggestion turns into an
instruction to lie. So the suggestions are a set intersection against the fact base.
A term the fact base does not already contain is reported under *not yours — do not
add*, never under *add these*. That single rule is why this is not another ATS grader.

The bands are deliberately not the fit bands. Scoring 60/100 on readiness means the
parser loses your CV before a person sees it, which is more urgent than being a weak
match for one role.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from careerpilot.config import ATS_PARTIAL_MIN, ATS_READY_MIN, ATS_WORDS_MAX, ATS_WORDS_MIN
from careerpilot.domain.job import html_to_text
from careerpilot.domain.place import named_countries
from careerpilot.ground.gate import TECHISH
from careerpilot.profile.model import Profile
from careerpilot.textutil import canonical, has_term

# What each check is worth, out of 100. `keyword_coverage` scores zero possible when
# there is no posting to compare against, and the total is then rescaled over what
# remains — a readiness score for the CV alone is still a real score.
WEIGHTS = {
    "contact_block": 15.0,
    "sections": 20.0,
    "keyword_coverage": 30.0,
    "proof_density": 15.0,
    "filler": 10.0,
    "length": 5.0,
    "parse_traps": 5.0,
}

# The vocabulary this checker knows how to look for. It is a list rather than a model
# because the alternative is asking a model which words in a job post are technologies,
# and then trusting its answer with the candidate's honesty. Deliberately weighted
# toward what Bangladeshi entry-level postings state (bdjobs/bdtechjobs titles and
# requirement lines, measured 2026-10-15) plus the remote-stack terms Jobicy carries.
SKILL_WORDS: tuple[str, ...] = (
    "python", "django", "flask", "fastapi", "laravel", "php", "wordpress", "javascript",
    "typescript", "node", "nodejs", "express", "react", "nextjs", "vue", "angular",
    "html", "css", "tailwind", "bootstrap", "java", "spring", "spring boot", "kotlin",
    "android", "ios", "flutter", "swift", "go", "c++", "c#", ".net", "asp.net", "rust",
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "database", "excel",
    "google sheets", "power bi", "tableau", "looker studio", "vba", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "machine learning", "deep learning",
    "nlp", "computer vision", "data analysis", "data cleaning", "statistics",
    "git", "github", "gitlab", "docker", "kubernetes", "linux", "bash", "shell",
    "aws", "azure", "gcp", "firebase", "vercel", "netlify", "rest", "graphql", "api",
    "web services", "microservices", "selenium", "cypress", "playwright", "junit",
    "pytest", "qa", "software testing", "test case", "testng", "automation",
    "jenkins", "github actions", "ci/cd", "agile", "scrum", "jira", "trello",
    "figma", "adobe photoshop", "corel draw", "networking", "ccna", "routing",
    "switch", "firewall", "windows server", "active directory", "vmware", "proxmox",
    "zabbix", "siem", "burp suite", "owasp", "penetration testing", "wordpress",
    "shopify", "woocommerce", "magento", "elementor", "laravel", "codeigniter",
    "jquery", "sass", "webpack", "postman", "swagger", "websocket", "socket.io",
    "raspberry pi", "arduino", "embedded c", "matlab", "simulink", "autocad",
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?=(?:\D*\d){8,})")
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u00b7\u2013\u2014]|\d+[.)])\s+")
_DIGIT = re.compile(r"\d")
# Years and date ranges, including the `2024-2025` and `01/02/2024` shapes a Cv stamp uses.
# Stripped before looking for proof, so a date never counts as a metric.
_DATE = re.compile(
    r"\b(?:19|20)\d{2}(?:\s*[-/]\s*(?:19|20)?\d{2,4})?\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
)
# A heading line, as opposed to a sentence: parsers key on short lines, so a section is
# only found if its name sits on a line of its own.
_HEADING_MAX_LEN = 70
# A term shorter than this is not a technology, it is a preposition that happened to be
# capitalised in the posting ("On", "In", "No SQL" aside).
_MIN_TERM_KEY = 3
# A posting writes "Git/GitHub" as one requirement when it means two, and `has_term` cannot
# reach either half through the slash — so a CV stating both was told to go and gain the
# experience it already had. Halves only count when both are real terms, which keeps "ci/cd"
# whole: `cd` on its own is a shell command, not a skill.
_COMPOUND = re.compile(r"\s*[/|]\s*|\s+and\s+")
_REAL_TERMS = frozenset(canonical(word) for word in SKILL_WORDS)


def _expand(token: str) -> tuple[str, ...]:
    parts = [part for part in _COMPOUND.split(token) if part]
    if len(parts) > 1 and all(canonical(part) in _REAL_TERMS for part in parts):
        return tuple(parts)
    return (token,)
# Four in ten bullets carrying a number is the bar, and it is deliberate: a fresher with
# three lab projects cannot honestly reach 100%, and demanding it is how CVs end up with
# invented percentages.
_PROOF_TARGET = 0.4
# Typographic punctuation a CV may legitimately contain. Anything else non-ASCII is
# either Bangla script or a decorative glyph, and both are dropped by some portals.
_PLAIN_PUNCTUATION = "\u2013\u2014\u00b7\u2018\u2019\u201c\u201d\u2026"
# A markdown table row: `| a | b |`, or the `|---|---|` rule below a header.
_TABLE_ROW = re.compile(r"^\s*\|.*\|.*\||\|\s*-{3,}")
_SECTION_HEADS = {
    "Summary": ("summary", "objective", "career objective", "profile", "about me"),
    "Skills": ("skill", "technical skills", "technolog", "tools"),
    "Experience or Projects": (
        "experience", "employment", "work history", "project", "internship", "portfolio",
    ),
    "Education": ("education", "academic", "degree", "university", "result of", "ssc", "hsc"),
}

# Words that cost a fresher's CV credibility on sight. This is the list Bangladeshi
# students are taught in school coaching centres, which is exactly why it says nothing.
FILLER: tuple[str, ...] = (
    "hard working", "hardworking", "team player", "good communication skills",
    "excellent communication skills", "quick learner", "fast learner", "self motivated",
    "self-motivated", "go getter", "honest and sincere", "well organized",
    "capability to work", "dynamic individual", "keep on learning",
)

# Things a parser eats instead of your content. "Photo" and "Marital status" are not
# hypothetical here: they are on most Bangladeshi CV templates by default.
_TRAPS: tuple[tuple[str, str], ...] = (
    ("photo", "a photo/line — some parsers read it as a field and shift everything after it"),
    ("marital status", "marital status — not a hiring fact in most markets, and a parse trap"),
    ("date of birth", "date of birth — belongs in the application form, not the CV"),
    ("father's name", "parent's name — a legacy field, drop it"),
    ("nationality", "nationality — your address and work permit already say this"),
    ("references available", "'references available upon request' — one line of dead weight"),
)

_NON_ASCII = re.compile(r"[^\x00-\x7f]")


class Check(BaseModel):
    """One measurement. `fix` is what to do about it, in the candidate's words."""

    id: str
    points: float
    possible: float
    why: str
    fix: str = ""


class AtsReport(BaseModel):
    checks: list[Check]
    words: int = 0
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    addable_terms: list[str] = Field(default_factory=list)
    not_in_fact_base: list[str] = Field(default_factory=list)
    job_label: str = ""

    @property
    def possible(self) -> float:
        return sum(c.possible for c in self.checks)

    @property
    def total(self) -> float:
        available = self.possible
        if available <= 0:
            return 0.0
        return round(100.0 * sum(c.points for c in self.checks) / available, 1)

    @property
    def band(self) -> Literal["ready", "partial", "weak"]:
        score = self.total
        if score >= ATS_READY_MIN:
            return "ready"
        if score >= ATS_PARTIAL_MIN:
            return "partial"
        return "weak"

    def check(self, check_id: str) -> Check | None:
        return next((c for c in self.checks if c.id == check_id), None)


def read_cv(path: Path) -> str:
    """Read a CV the way a parser would: markup off, text only."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return html_to_text(raw) if path.suffix.lower() in {".html", ".htm"} else raw


def score_cv(text: str, profile: Profile, job_text: str = "") -> AtsReport:
    """Score `text` for machine-readability, optionally against one posting's prose."""
    matched, missing = job_terms(job_text, text) if job_text else ([], [])
    addable = _addable(profile, missing)
    return AtsReport(
        words=_word_count(text),
        matched_terms=matched,
        missing_terms=missing,
        addable_terms=addable,
        not_in_fact_base=[term for term in missing if term not in set(addable)],
        job_label="",
        checks=[
            _contact_block(text),
            _sections(text),
            _keyword_coverage(job_text, matched, missing, profile),
            _proof_density(text),
            _filler(text),
            _length(text),
            _parse_traps(text),
        ],
    )


def job_terms(job_text: str, cv_text: str) -> tuple[list[str], list[str]]:
    """Terms the posting asks for, split into the ones your CV says and the ones it does not.

    Matching goes through `canonical` and `has_term`, the same two functions the fit
    scorer uses, so "Node.js", "node js" and "nodejs" cannot be counted as three
    different gaps in the same document.
    """
    wanted: dict[str, str] = {}
    for term in SKILL_WORDS:
        if has_term(job_text, term):
            wanted.setdefault(canonical(term), term)
    for token in TECHISH.findall(job_text):
        for part in _expand(token):
            key = canonical(part)
            if len(key) >= _MIN_TERM_KEY:
                wanted.setdefault(key, part)
    found: list[str] = []
    absent: list[str] = []
    for display in sorted(wanted.values(), key=str.lower):
        (found if has_term(cv_text, display) else absent).append(display)
    return found, absent


def _contact_block(text: str) -> Check:
    weight = WEIGHTS["contact_block"]
    parts = {
        "email": bool(_EMAIL.search(text)),
        "phone (at least 8 digits)": bool(_PHONE.search(text)),
        "a location a parser can read": bool(named_countries(text)),
        "github or linkedin URL": bool(re.search(r"(?:github\.com|linkedin\.com)/", text, re.I)),
    }
    points = weight * sum(parts.values()) / len(parts)
    missing = [label for label, present in parts.items() if not present]
    return Check(
        id="contact_block",
        points=round(points, 2),
        possible=weight,
        why="all four present" if not missing else "missing " + ", ".join(missing),
        fix="" if not missing else "put them on one line under your name, no table, no icons",
    )


def _sections(text: str) -> Check:
    weight = WEIGHTS["sections"]
    lines = [
        line.strip().lower() for line in text.splitlines()
        if line.strip() and len(line.strip()) <= _HEADING_MAX_LEN
    ]
    found, absent = [], []
    for label, heads in _SECTION_HEADS.items():
        if any(any(head in line for head in heads) for line in lines):
            found.append(label)
        else:
            absent.append(label)
    points = weight * len(found) / len(_SECTION_HEADS)
    return Check(
        id="sections",
        points=round(points, 2),
        possible=weight,
        why=f"found: {', '.join(found) or 'none'}",
        fix="" if not absent else "add a heading named " + ", ".join(absent)
        + " — parsers key on the heading text, not on the paragraph under it",
    )


def _keyword_coverage(
    job_text: str, matched: list[str], missing: list[str], profile: Profile
) -> Check:
    weight = WEIGHTS["keyword_coverage"]
    if not job_text:
        return Check(
            id="keyword_coverage", points=0.0, possible=0.0,
            why="no posting given — pass a job uid or --jd to check this",
        )
    total_terms = len(matched) + len(missing)
    if not total_terms:
        return Check(
            id="keyword_coverage", points=weight, possible=weight,
            why="this posting names no technology this checker recognises",
        )
    points = weight * len(matched) / total_terms
    addable = _addable(profile, missing)
    return Check(
        id="keyword_coverage",
        points=round(points, 2),
        possible=weight,
        why=f"{len(matched)}/{total_terms} of the posting's terms appear in the CV",
        fix=(
            f"use the exact wording for: {', '.join(addable)}" if addable
            else "nothing you already have is missing — see 'not in your fact base'"
        ),
    )


def _addable(profile: Profile, missing: list[str]) -> list[str]:
    vocab = _surface(profile)
    return [term for term in missing if has_term(vocab, term)]


def _proof_density(text: str) -> Check:
    weight = WEIGHTS["proof_density"]
    bullets = [line for line in text.splitlines() if _BULLET.match(line)]
    if not bullets:
        return Check(
            id="proof_density", points=0.0, possible=weight,
            why="no bullet lines found",
            fix="write your work as bullets, one line per claim",
        )
    # A date is not proof. Rows carry their stamp on the same line, so `2025` standing in for
    # "this bullet is quantified" would let a CV score itself by listing years.
    proven = [line for line in bullets if _DIGIT.search(_DATE.sub(" ", line))]
    # Four in ten is the bar: a fresher with three lab projects cannot honestly reach
    # 100% numeric proof, and demanding it is how CVs end up with invented percentages.
    ratio = len(proven) / len(bullets)
    points = weight * min(1.0, ratio / _PROOF_TARGET)
    return Check(
        id="proof_density",
        points=round(points, 2),
        possible=weight,
        why=f"{len(proven)}/{len(bullets)} bullets contain a number",
        fix=(
            "" if ratio >= _PROOF_TARGET else
            "add the count, the time or the money to more bullets — only numbers you "
            "can point at in your fact base"
        ),
    )


def _filler(text: str) -> Check:
    weight = WEIGHTS["filler"]
    lowered = text.lower()
    hits = [phrase for phrase in FILLER if phrase in lowered]
    points = weight * max(0.0, 1.0 - len(hits) / 3.0)
    return Check(
        id="filler",
        points=round(points, 2),
        possible=weight,
        why=f"{len(hits)} filler phrase(s)" if hits else "no filler phrases",
        fix="" if not hits else "delete and replace with what you did: " + "; ".join(hits),
    )


def _length(text: str) -> Check:
    weight = WEIGHTS["length"]
    words = _word_count(text)
    if words >= ATS_WORDS_MIN and words <= ATS_WORDS_MAX:
        points, why, fix = weight, f"{words} words — one page", ""
    elif words < ATS_WORDS_MIN:
        points = weight * words / ATS_WORDS_MIN
        points = min(points, weight * 0.99)
        why, fix = f"{words} words — thin", "add the projects and coursework you actually did"
    else:
        points = weight * max(0.4, 1.0 - (words - ATS_WORDS_MAX) / ATS_WORDS_MAX)
        why = f"{words} words — over one page"
        fix = "cut to the last two years: one page is read, two are skimmed"
    return Check(id="length", points=round(points, 2), possible=weight, why=why, fix=fix)


def _parse_traps(text: str) -> Check:
    weight = WEIGHTS["parse_traps"]
    lowered = text.lower()
    tripped = [note for needle, note in _TRAPS if needle in lowered]
    if _is_table(text):
        tripped.append("a table — parsers read its columns in the wrong order")
    if "\t" in text:
        tripped.append("tab characters — some parsers drop everything after one")
    if _stray_glyphs(text):
        tripped.append(
            "characters outside plain Latin (Bangla, stars, arrows) — many portals strip them"
        )
    points = weight * max(0.0, 1.0 - len(tripped) / 3.0)
    return Check(
        id="parse_traps",
        points=round(points, 2),
        possible=weight,
        why=f"{len(tripped)} trap(s)" if tripped else "no known parse traps",
        fix="" if not tripped else "; ".join(tripped),
    )


def _is_table(text: str) -> bool:
    """A grid, not a contact line written with pipes as separators.

    "amina@example.com | +8801711223344 | Dhaka" is one plain line that every parser
    reads correctly; calling it a table would penalise the very layout this tool emits.
    """
    return any(_TABLE_ROW.match(line) for line in text.splitlines())


def _stray_glyphs(text: str) -> str:
    """Non-ASCII characters that are not ordinary English punctuation."""
    return "".join({c for c in _NON_ASCII.findall(text)} - set(_PLAIN_PUNCTUATION))


def _surface(profile: Profile) -> str:
    """Everything the fact base states, as one searchable string.

    The intersection that produces "keywords you may add" is taken from this, not from
    the skills atoms alone: a Django project is evidence of Django even when the profile
    never lists it as a skill.
    """
    parts: list[str] = [profile.name, profile.headline]
    for atom in profile.atoms:
        parts.append(atom.text)
        parts.extend(atom.skills)
    return "\n".join(parts)


def _word_count(text: str) -> int:
    return len(text.split())
