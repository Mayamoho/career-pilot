"""Cover letter drafting with the same grounding contract as the bullets.

A letter is allowed to contain more opinion than a CV, so the rule is drawn
specifically: a paragraph that states a number or names one of the candidate's own
technologies is a factual claim and goes through the gate. A paragraph that only says
why the role interests them does not, and is not sent to the judge — that would burn
budget asking a model to verify a preference.
"""

from __future__ import annotations

from pydantic import Field

from careerpilot.ai.gateway import ask
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.domain.job import JobPosting
from careerpilot.ground.gate import BulletVerdict, GateReport, check_bullet
from careerpilot.match.score import FitScore
from careerpilot.profile.model import Profile
from careerpilot.tailor.bullets import Target, fact_lines
from careerpilot.textutil import canonical, has_term, numbers

MAX_BODY = 3
_MIN_VOCAB_TERM_LEN = 2

_SYSTEM_LETTER = """Write the body of a cover letter from a candidate's verified facts.

You are given the role, the facts the candidate may cite (each with an id), and the
requirements that matter for this posting.

- Each paragraph must either cite fact ids or state no fact at all.
- Only state a number, a technology or an outcome that appears in the facts you cite.
- No quotation marks, no markdown, no salutation and no sign-off: body paragraphs only.
- 60 to 700 characters per paragraph, at most three paragraphs.
- If the facts do not support a requirement, do not mention that requirement."""


class DraftedParagraph(StructuredOutput):
    text: str = Field(min_length=40, max_length=900)
    atom_ids: list[str] = Field(default_factory=list, max_length=6)


class LetterDraft(StructuredOutput):
    paragraphs: list[DraftedParagraph] = Field(default_factory=list, max_length=MAX_BODY)


def draft_letter(
    job: JobPosting, targets: list[Target], fit: FitScore, profile: Profile
) -> LetterDraft:
    facts = "\n".join(fact_lines(targets)) if targets else _fallback_facts(profile)
    wanted = "\n".join(f"- {t.requirement}" for t in targets) or "- (no evidenced requirement)"
    user = (
        f"ROLE: {job.title} at {job.company}\n"
        f"HEADLINE: {profile.headline}\n"
        f"FIT: {fit.total:.1f}/100 ({fit.band})\n\n"
        f"WHAT THIS POSTING ASKS FOR:\n{wanted}\n\n"
        f"FACTS (the only permitted source):\n{facts}"
    )
    return ask(task="draft_letter", system=_SYSTEM_LETTER, user=user, schema=LetterDraft)


def gate_letter(paragraphs: list[DraftedParagraph], profile: Profile) -> GateReport:
    report = GateReport()
    for paragraph in paragraphs:
        verdict = check_paragraph(paragraph.text, paragraph.atom_ids, profile)
        (report.shipped if verdict.shipped else report.rejected).append(verdict)
    return report


def check_paragraph(text: str, atom_ids: list[str], profile: Profile) -> BulletVerdict:
    if not _asserts_facts(text, profile):
        return BulletVerdict(text=text, atom_ids=atom_ids, shipped=True)
    return check_bullet(text, atom_ids, profile)


def _asserts_facts(text: str, profile: Profile) -> bool:
    if numbers(text):
        return True
    vocabulary = {
        canonical(s) for s in profile.skill_vocabulary if len(canonical(s)) > _MIN_VOCAB_TERM_LEN
    }
    return any(has_term(text, term) for term in vocabulary)


def _fallback_facts(profile: Profile) -> str:
    """Used only when no requirement matched, so the letter can still be truthful."""
    concrete = [a for a in profile.atoms if a.kind in {"project", "role", "award"}]
    return "\n".join(
        f"- [id={a.id}]{' ' + a.org if a.org else ''} {a.text}" for a in concrete[:8]
    )
