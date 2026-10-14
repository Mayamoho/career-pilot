"""Requirement extraction. The model's *only* job here is to read prose and emit
structure. It is never asked for a fit score, because a number from a language model
is a mood, not a measurement.
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from careerpilot.ai.gateway import ask
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.domain.job import JobPosting

CATEGORY_WEIGHTS = {"must_have": 3.0, "nice_to_have": 1.5, "preferred": 1.0, "unknown": 0.5}


class ExtractedRequirement(StructuredOutput):
    text: str = Field(min_length=4, max_length=300)
    skill: str = Field(min_length=1, max_length=60)
    category: str = Field(pattern="^(must_have|nice_to_have|preferred|unknown)$")


class RequirementSet(StructuredOutput):
    requirements: list[ExtractedRequirement] = Field(default_factory=list, max_length=25)
    seniority: str = Field(default="unspecified", max_length=40)
    location_constraint: str = Field(default="unspecified", max_length=60)

    @field_validator("requirements", mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        return value if isinstance(value, list) else []


SYSTEM = """Extract the concrete requirements a hiring manager would actually check.

Read the job text and list distinct, testable requirements. For each one:
- `text`: the requirement, phrased close to the source, under 300 characters.
- `skill`: a single canonical technology or capability name (e.g. "PostgreSQL",
  "distributed systems", "RAG"), not a sentence.
- `category`: "must_have" for hard gates and "X+ years" statements, "preferred" for
  "nice to have"/"bonus", "nice_to_have" for anything else that is clearly wanted,
  "unknown" when the text does not say.

Rules:
- Only requirements present in the text. Never add an industry-standard requirement
  the posting does not state.
- Merge duplicates. Do not split one requirement into two.
- At most 25. Prefer the ones that actually decide the shortlist.
- If the posting is empty or unreadable, return an empty list."""


def extract_requirements(job: JobPosting) -> RequirementSet:
    excerpt = job.requirements_excerpt()
    user = (
        f"ROLE: {job.title} at {job.company}\n"
        f"LOCATION: {job.location or 'unspecified'}\n\n"
        f"JOB TEXT:\n{excerpt}"
    )
    result = ask(task="extract_requirements", system=SYSTEM, user=user, schema=RequirementSet)
    return _normalise(result)


def _normalise(result: RequirementSet) -> RequirementSet:
    """Code-side hygiene the model cannot be trusted to do consistently.

    Dedupe is on the normalized skill, not the prose, because two phrasings of the
    same requirement would otherwise be double-counted and inflate the score.
    """
    seen: dict[str, ExtractedRequirement] = {}
    for req in result.requirements:
        key = re.sub(r"[^a-z0-9+#. ]", "", req.skill.lower()).strip()
        if not key or key in seen:
            continue
        seen[key] = req
    kept = list(seen.values())[:25]
    return RequirementSet(
        requirements=kept,
        seniority=result.seniority.strip() or "unspecified",
        location_constraint=result.location_constraint.strip() or "unspecified",
    )
