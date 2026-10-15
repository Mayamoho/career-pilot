"""Which postings are worth a candidate's time, decided before anything costs a call.

This filter lives apart from `boards.py` rather than in it, because `boards.py` imports
the adapters to build `ADAPTERS` and one adapter needs this filter: an adapter asking
`boards` for the predicate would be a cycle.

It is a pure substring test on the title, deliberately dumb. A model deciding whether a
posting "looks like" this candidate's kind of job would be a model guessing at a fact that
is printed on the row.

Seniority words are not in the include list on purpose. `"intern"` used to be in
`TITLE_EXCLUDE`, which silently deleted the entire entry-level market — the postings a
final-year student is actually allowed to apply for. Removing it is enough: a title needs
a technical term either way, so "Python Automation Intern" is kept and "Management Trainee
(Bank)" is not, without a second rule to maintain.
"""

from __future__ import annotations

# Roles we actually want.
#
# The second half of this list was written from evidence rather than taste: the first
# collect run kept 13 of bdtechjobs' 55 postings and dropped 42, and the ones it threw
# away were real engineering roles. Bangladeshi boards title those jobs "Developer" and
# "DevOps Engineer", not "Software Engineer", so a list written from how US boards name
# things silently filters out the entire local market.
TITLE_INCLUDE = (
    "ai engineer", "machine learning", "ml engineer", "backend", "software engineer",
    "platform engineer", "full stack", "fullstack", "applied ai", "llm", "agent",
    "data engineer", "security engineer", "sre", "infrastructure",
    "developer", "devops", "cloud", "secops", "cybersecurity", "site reliability",
    "ai systems", "information security", "frontend", "python", "django", "laravel",
    "react", "node", "java", "golang", "qa", "quality assurance", "testing",
    "network", "system administrator", "it officer", "it support", "it engineer",
    "information technology", "software development",
)
TITLE_EXCLUDE = (
    "co-op", "principal", "staff ", "director", "vp ", "head of",
    "manager", "lead ", "senior staff", "distinguished", "circuit", "sales",
    "marketing", "designer", "recruiter", "legal", "finance", "content",
)


def wanted_title(title: str) -> bool:
    """Keep the posting only if its title names technical work this candidate can do."""
    lowered = title.lower()
    if any(term in lowered for term in TITLE_EXCLUDE):
        return False
    return any(term in lowered for term in TITLE_INCLUDE)
