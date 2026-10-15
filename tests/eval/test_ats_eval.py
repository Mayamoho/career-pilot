"""The eval for the ATS checker: a suggestion must never outrun the fact base.

`tests/test_ats.py` shows the behaviour on hand-picked CVs. This enforces it as a rate
over a corpus, because the failure it is guarding against is statistical: a checker that
offers an unsupportable keyword in 1 case in 20 is still a checker that tells people to
lie, and no single example would have caught it.

Three rates, only one of which may move:

* `UNSUPPORTED_SUGGESTION_RATE` must be exactly zero. Not a target — a definition. The
  oracle is a plain set membership test written independently of `has_term`, so the
  checker cannot pass by agreeing with itself.
* `SUPPORTED_SUGGESTION_RECALL` has a floor: an honest checker that names nothing is
  useless, and this is the number that punishes a checker gone too timid.
* `TRAP_CATCH_RATE` has a floor on the CVs that are known to be unparseable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from careerpilot.ats.reader import score_cv
from careerpilot.profile.model import Atom, Profile
from careerpilot.textutil import canonical, has_term

pytestmark = pytest.mark.eval

UNSUPPORTED_SUGGESTION_RATE = 0.0
SUPPORTED_SUGGESTION_RECALL = 0.8
TRAP_CATCH_RATE = 0.8

# The candidate's recorded experience. Anything outside this set is, by construction, a
# keyword this tool is forbidden to suggest.
FACT_BASE = (
    "Python, Django, PostgreSQL, Git, Docker, REST APIs, pandas, SQL, "
    "built a Django attendance system used by 300 students, "
    "Python Intern at a Dhaka software company, automated 40 regression tests"
)
PROVEN_KEYWORDS = {"python", "django", "postgresql", "docker", "pandas", "git", "sql", "rest"}

# Terms a Bangladeshi entry-level posting routinely asks for that this candidate has no
# evidence for. The tempting ones are all here.
UNPROVEN_KEYWORDS = {
    "kubernetes", "aws", "selenium", "laravel", "react", "tensorflow", "spark", "c++",
}

_CLEAN_CV = (
    "AMINA KHATUN\n"
    "amina@example.com | +8801711223344 | Mirpur, Dhaka, Bangladesh | github.com/amina\n\n"
    "SUMMARY\nFinal year CSE student looking for backend work.\n\n"
    "SKILLS\nPython, Git\n\n"
    "PROJECTS\n- Built a Django attendance system used by 300 students\n"
    "- Cut report prep time by 40% with a pandas script\n\n"
    "EDUCATION\nBSc in CSE, North South University, 2022-2026\n"
)

_BAD_CVS = (
    "CURRICULUM VITA\n"
    "Photograph: [attached]\nMarital Status: Single\nDate of Birth: 12/2001\n"
    "Father's Name: Abdul Khatun\nNationality: Bangladeshi\n"
    "| Skill | Level |\n|---|---|\n| Python | Good |\n"
    "I am a hard working team player and quick learner with good communication skills.\n"
    "References available upon request.\n"
)


@dataclass(frozen=True)
class Case:
    """One posting, and the keywords the fact base genuinely does or does not back."""

    job_text: str
    proven: tuple[str, ...] = ()
    unproven: tuple[str, ...] = ()
    cv: str = field(default=_CLEAN_CV)
    why: str = ""


CORPUS = (
    Case(
        "Python Backend Developer. Required: Python, Django, PostgreSQL, Docker. "
        "Nice to have: Kubernetes.",
        proven=("python", "django", "postgresql", "docker"),
        unproven=("kubernetes",),
        why="the ordinary Bd backend posting, with one cloud term the candidate lacks",
    ),
    Case(
        "QA Trainee. Must have Selenium, JUnit, test case design and SQL.",
        unproven=("selenium", "junit"),
        why="a trainee posting whose whole toolchain is missing from the fact base",
    ),
    Case(
        "Data Analyst (Junior). Python, pandas, SQL, Power BI, Tableau.",
        proven=("python", "pandas", "sql"),
        unproven=("power bi", "tableau"),
        why="data role: three of the five terms are provable, two are not",
    ),
    Case(
        "Full Stack Engineer. React, Next.js, Node.js, REST, AWS, Docker.",
        proven=("docker", "rest"),
        unproven=("react", "nextjs", "nodejs", "aws"),
        why="the highest-pressure case: a popular stack, most of it unprovable",
    ),
    Case(
        "ML Intern. TensorFlow, PyTorch, Python, scikit-learn, statistics.",
        proven=("python",),
        unproven=("tensorflow", "pytorch", "scikit-learn"),
        why="every framework named is one the candidate cannot back",
    ),
    Case(
        "DevOps Trainee. Linux, AWS, Kubernetes, Terraform, CI/CD pipelines.",
        unproven=("linux", "aws", "kubernetes"),
        why="nothing in this posting is in the fact base except general engineering",
    ),
    Case(
        "Lab Instructor. Python, C++, database, SQL, good communication.",
        proven=("python", "sql"),
        unproven=("c++",),
        why="a common university job, and the one C++ the candidate never wrote",
    ),
    Case(
        "Backend Intern. Python, Django, REST APIs, Git, PostgreSQL.",
        proven=("python", "django", "git", "postgresql"),
        why="a posting the fact base covers completely",
    ),
)


def _profile() -> Profile:
    return Profile(
        name="Amina Khatun",
        headline="Final year CSE student",
        contact={"email": "amina@example.com", "location": "Mirpur, Dhaka, Bangladesh"},
        atoms=[
            Atom(id="skill-stack", kind="skill", text=FACT_BASE, source="start-here"),
            Atom(
                id="proj-attendance",
                kind="project",
                text="Built a Django attendance system used by 300 students",
                org="coursework",
                period="2025",
                source="start-here",
            ),
            Atom(
                id="role-intern",
                kind="role",
                text="Python Intern at a Dhaka software company, automated 40 regression tests",
                org="Marive",
                period="2024",
                source="start-here",
            ),
        ],
    )


def _oracle_provable(term: str) -> bool:
    """Independent of the checker: a suggestion is allowed only if it is in this set."""
    return term in PROVEN_KEYWORDS


def test_not_one_suggestion_in_the_corpus_asks_for_a_keyword_the_fact_base_lacks() -> None:
    profile = _profile()
    suggestions = [
        term
        for case in CORPUS
        for term in score_cv(case.cv, profile, case.job_text).addable_terms
    ]

    rate = sum(1 for term in suggestions if not _oracle_provable(term)) / max(1, len(suggestions))
    assert rate == UNSUPPORTED_SUGGESTION_RATE, [
        f"suggested a keyword nobody has evidence for: {term!r} ({case.why})"
        for case in CORPUS
        for term in score_cv(case.cv, profile, case.job_text).addable_terms
        if not _oracle_provable(term)
    ]


def test_every_unprovable_keyword_lands_in_the_do_not_add_list() -> None:
    """Compared through `canonical`, because the checker reports the posting's own spelling."""
    profile = _profile()
    for case in CORPUS:
        report = score_cv(case.cv, profile, case.job_text)
        named = {canonical(t) for t in report.matched_terms + report.not_in_fact_base}
        suggested = {canonical(t) for t in report.addable_terms}
        for term in case.unproven:
            assert canonical(term) in named, f"{term} vanished without a trace: {case.why}"
            assert canonical(term) not in suggested, f"{term} was suggested: {case.why}"


def test_a_keyword_the_fact_base_does_back_is_offered_when_the_cv_leaves_it_out() -> None:
    """The other half of the promise: honest does not mean silent."""
    profile = _profile()
    offered = {
        canonical(term)
        for case in CORPUS
        for term in score_cv(case.cv, profile, case.job_text).addable_terms
    }
    # Provable, asked for by at least one posting, and genuinely absent from the CV.
    expected = {
        canonical(term)
        for term in PROVEN_KEYWORDS
        if not has_term(_CLEAN_CV, term)
    } & {canonical(term) for case in CORPUS for term in case.proven}

    recall = len(offered & expected) / max(1, len(expected))
    assert recall >= SUPPORTED_SUGGESTION_RECALL, f"expected {expected}, offered {offered}"


def test_the_cv_that_is_a_real_bdjobs_template_gets_no_credibility_score() -> None:
    profile = _profile()
    report = score_cv(_BAD_CVS, profile)
    assert report.check("parse_traps").points == 0.0  # type: ignore[union-attr]
    assert report.check("filler").points == 0.0  # type: ignore[union-attr]
    assert report.total < 55.0


def test_the_trap_catch_rate_holds_across_every_bad_line_at_once() -> None:
    profile = _profile()
    traps = (_BAD_CVS, "Marital status: single\n", "Photograph: attached\n", "| a | b |\n|---|---|")
    caught = sum(
        1
        for text in traps
        if (check := score_cv(text, profile).check("parse_traps")) is not None
        and check.points < 5.0
    )
    assert caught / len(traps) >= TRAP_CATCH_RATE
