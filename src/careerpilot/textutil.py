"""Shared text normalisation.

Skill names arrive from four places — a hand-written YAML profile, an ATS job title,
model-extracted requirements, and generated prose — and they must compare equal across
all of them. One canonical form, one alias table, so matching rules cannot drift
between the scorer and the groundedness gate.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9+# ]")
_MULTI_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

ALIASES: dict[str, str] = {
    "js": "javascript", "ts": "typescript", "golang": "go",
    "postgres": "postgresql", "pg": "postgresql", "pgvector": "postgresql",
    "ml": "machine learning", "dl": "deep learning",
    "next": "nextjs", "nextjs": "nextjs",
    "reactjs": "react", "node": "nodejs", "nodejs": "nodejs",
    "expressjs": "express", "fastapi": "fastapi",
    "genai": "generativeai", "llms": "llm",
    "ci": "cicd", "cd": "cicd", "cicd": "cicd",
    "reedsolomon": "reed-solomon",
    "typescript": "typescript",
}


def canonical(text: str) -> str:
    """Fold a technology name to a comparable key.

    Separators are removed rather than preserved so "Node.js", "node js" and
    "nodejs" all resolve to the same thing.
    """
    lowered = text.lower().replace(".", "").replace("/", " ").replace("-", " ")
    key = _MULTI_WS.sub(" ", _NON_ALNUM.sub("", lowered)).strip()
    single = key.replace(" ", "")
    return ALIASES.get(single, ALIASES.get(key, single))


def has_term(text: str, term: str) -> bool:
    """Word-boundary search tolerant of separators inside multi-word terms.

    Boundary guards stop "go" matching inside "google" and "c" inside "car", which is
    the failure mode that would make the groundedness gate useless.
    """
    pattern = re.escape(term.lower()).replace(r"\ ", r"[\s./-]+")
    return re.search(rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])", text.lower()) is not None


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))
