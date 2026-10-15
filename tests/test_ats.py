"""The ATS checker, and the one property that makes it worth shipping.

Every other CV grader answers "this job wants Kubernetes" with "then say Kubernetes".
This one answers with a set intersection against the fact base, so the only keywords it
can suggest are keywords the candidate already has evidence for. Tests 1-11 measure the
scoring; the last group pins the refusal, because a scorer that can be talked into
recommending a lie is worse than no scorer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from careerpilot.ats.reader import AtsReport, job_terms, read_cv, score_cv
from careerpilot.profile.model import Profile, load_profile
from careerpilot.report import templates

REPO = Path(__file__).resolve().parents[1]

# Deliberately not `data/profile.yaml`: these assertions are about which words a fact
# base happens to contain, so the fixture is the checked-in example, not a person.
@pytest.fixture(scope="module")
def example() -> Profile:
    return load_profile(REPO / "data" / "profile.example.yaml")


def _report(text: str, profile: Profile, job_text: str = "") -> AtsReport:
    return score_cv(text, profile, job_text)


def _points(report: AtsReport, check_id: str) -> tuple[float, float]:
    check = report.check(check_id)
    assert check is not None, f"{check_id} missing from {[c.id for c in report.checks]}"
    return check.points, check.possible


# --- the contact block and the sections parsers key on -------------------------------


def test_a_headline_contact_line_scores_full_weight(example: Profile) -> None:
    text = (
        "Amina Khatun\n"
        "Backend engineer\n"
        "amina@example.com | +8801711223344 | Mirpur, Dhaka, Bangladesh\n"
        "github.com/amina\n"
    )
    points, possible = _points(_report(text, example), "contact_block")
    assert points == possible


def test_a_cv_without_a_phone_loses_part_of_the_contact_block(example: Profile) -> None:
    text = "Amina\namina@example.com | Dhaka, Bangladesh | github.com/amina\n"
    points, possible = _points(_report(text, example), "contact_block")
    assert 0 < points < possible


def test_a_section_must_be_a_heading_not_a_sentence(example: Profile) -> None:
    """Parsers key on the heading line. A paragraph that mentions skills is not a section."""
    shallow = "I have skills in many areas and my education was good and I have experience.\n"
    assert _report(shallow, example).check("sections").points == 0.0  # type: ignore[union-attr]
    headed = "SKILLS\nPython\nEXPERIENCE\nBuilt X\nEDUCATION\nBSc\nSUMMARY\nEngineer\n"
    assert _report(headed, example).check("sections").points == 20.0  # type: ignore[union-attr]


# --- the honesty guarantee -----------------------------------------------------------


def test_a_term_the_fact_base_does_not_contain_is_never_suggested(example: Profile) -> None:
    cv = (
        "Backend engineer. Built services in Python.\n"
        "github.com/amina | Mirpur, Dhaka, Bangladesh"
    )
    report = _report(cv, example, "Required: Python, Kubernetes, Docker")

    assert "python" in report.matched_terms
    # Docker is in the fact base, so it may be named. Kubernetes is not, so it may not.
    assert report.addable_terms == ["docker"]
    assert report.not_in_fact_base == ["kubernetes"]
    # The suggestion line is the only place a tool can instruct a candidate to lie.
    fix = report.check("keyword_coverage").fix  # type: ignore[union-attr]
    assert "kubernetes" not in fix
    assert "docker" in fix


def test_a_term_the_fact_base_does_contain_is_suggested_by_name(example: Profile) -> None:
    """Docker is in the fact base but absent from this CV — so, and only so, it is addable."""
    cv = "Backend engineer. Built services in Python. Mirpur, Dhaka, Bangladesh"
    report = _report(cv, example, "Required: Python and Docker")

    assert "docker" in report.addable_terms
    assert report.not_in_fact_base == []
    assert "docker" in report.check("keyword_coverage").fix  # type: ignore[union-attr]


def test_the_suggestion_lists_partition_every_missing_term(example: Profile) -> None:
    report = _report(
        "Python developer in Dhaka, Bangladesh",
        example,
        "Python, Docker, Kubernetes, PHP, React, Laravel",
    )
    assert set(report.addable_terms) | set(report.not_in_fact_base) == set(report.missing_terms)
    assert not set(report.addable_terms) & set(report.not_in_fact_base)


@pytest.mark.parametrize("term", ["kubernetes", "django", "php", "laravel"])
def test_the_ordinary_bangladeshi_acronyms_a_fresher_is_tempted_by_stay_out(
    example: Profile, term: str
) -> None:
    assert term not in templates.cv_txt(example, None).lower()
    report = _report("Python. Dhaka, Bangladesh", example, f"Required: {term}")
    assert report.addable_terms == []
    assert report.not_in_fact_base == [term]


# --- keyword matching goes through the one canonicalisation path ---------------------


def test_nodejs_node_and_node_dot_js_are_one_gap_not_three() -> None:
    """Three spellings of one runtime must not become three missing keywords."""
    matched, missing = job_terms("Node.js services. node js APIs. express", "Built with Node.js")
    node_terms = [term for term in matched + missing if "node" in term.lower()]
    assert len(node_terms) == 1
    assert "express" in missing


def test_a_slash_written_pair_is_two_terms_and_not_one_unknown() -> None:
    """A bdjobs posting wrote 'Git/GitHub', and matching the whole string finds neither half.

    The live run put it under "not in your fact base" for a CV stating both — which tells a
    candidate to go and gain the experience they already have. That is the wrong direction of
    error for a tool whose whole job is to not lie at you.
    """
    matched, missing = job_terms("Must know Git/GitHub and mysql", "Used git and GitHub daily")
    assert matched == ["git", "github"]
    assert missing == ["mysql"]


def test_ci_slash_cd_stays_one_because_cd_alone_is_not_a_skill() -> None:
    """The expansion is gated on both halves being real terms, or 'ci/cd' becomes 'cd'."""
    matched, missing = job_terms("Owned ci/cd pipelines", "set up ci/cd with GitHub Actions")
    assert matched == ["ci/cd"]
    assert missing == []


def test_a_short_capitalised_word_is_not_read_as_a_technology() -> None:
    matched, missing = job_terms("On the team, In office, No travel", "Nothing relevant here")
    assert missing == []
    assert matched == []


# --- proof, filler, length, traps ----------------------------------------------------


def test_a_bullet_with_no_number_anywhere_costs_the_proof_weight(example: Profile) -> None:
    text = "\n".join("- built a thing" for _ in range(6))
    points, possible = _points(_report(text, example), "proof_density")
    assert points == 0.0 and possible == 15.0
    assert "fact base" in _report(text, example).check("proof_density").fix  # type: ignore[union-attr]


def test_a_date_in_the_stamp_is_not_proof(example: Profile) -> None:
    """`(Brain Station 23 | 2025)` is where and when, not a number that proves anything."""
    text = (
        "- Built a doctor-tracker PWA (personal | 2024-2025)\n"
        "- Styled a portal with Tailwind CSS (freelance | 2024)\n"
        "- Wrote a marksheet parser that cut 40 hours of entry (university lab | 2025)\n"
    )
    report = _report(text, example)
    check = report.check("proof_density")
    assert check is not None and check.why == "1/3 bullets contain a number"


def test_the_proof_bar_is_four_in_ten_not_every_bullet(example: Profile) -> None:
    """Demanding 100% numeric proof is how a CV ends up with invented percentages."""
    text = "- Cut load time by 40%\n" * 4 + "- Wrote a parser\n" * 6
    points, _ = _points(_report(text, example), "proof_density")
    assert points == 15.0


def test_a_cv_of_pure_adjectives_scores_zero_on_filler(example: Profile) -> None:
    text = "Hard working, team player, quick learner, good communication skills."
    points, possible = _points(_report(text, example), "filler")
    assert points == 0.0 and possible == 10.0
    assert "delete" in _report(text, example).check("filler").fix  # type: ignore[union-attr]


def test_an_empty_cv_is_weak_rather_than_favourable(example: Profile) -> None:
    report = _report("", example)
    assert report.total < 50
    assert report.band == "weak"


def test_the_photo_and_marital_status_fields_bd_templates_ship_are_traps(example: Profile) -> None:
    text = "Photograph: attached\nMarital status: single\nDate of birth: 2001-01-01\n"
    points, possible = _points(_report(text, example), "parse_traps")
    assert points == 0.0 and possible == 5.0


def test_a_contact_line_written_with_pipes_is_not_a_table(example: Profile) -> None:
    """This tool's own header line must not be flagged by this tool's own checker."""
    text = "amina@example.com | +8801711223344 | Mirpur, Dhaka, Bangladesh\n"
    assert _report(text, example).check("parse_traps").points == 5.0  # type: ignore[union-attr]


def test_a_markdown_table_is_a_trap(example: Profile) -> None:
    report = _report("| Skill | Level |\n|---|---|\n| Python | Good |\n", example)
    check = report.check("parse_traps")
    assert check.points < check.possible  # type: ignore[union-attr]
    assert "table" in check.fix  # type: ignore[union-attr]


def test_the_plain_text_cv_this_tool_emits_survives_this_tools_checker(
    example: Profile, tmp_path: Path
) -> None:
    """Round trip: what we print must not be what we flag."""
    out = tmp_path / "cv.txt"
    out.write_text(templates.cv_txt(example, None), encoding="utf-8")
    report = _report(read_cv(out), example)
    assert report.check("parse_traps").points == 5.0  # type: ignore[union-attr]
    assert report.check("sections").points == 20.0  # type: ignore[union-attr]
    assert report.band == "ready"


def test_reading_an_html_cv_measures_its_text_not_its_markup(
    example: Profile, tmp_path: Path
) -> None:
    out = tmp_path / "cv.html"
    out.write_text(templates.cv_html(example, None, "ats"), encoding="utf-8")
    text = read_cv(out)
    assert "<h1>" not in text
    assert example.name in text
    assert _report(text, example).check("sections").points == 20.0  # type: ignore[union-attr]


# --- the total, when a posting is not in play ----------------------------------------


def test_a_score_without_a_posting_is_rescaled_rather_than_capped_at_seventy(
    example: Profile,
) -> None:
    """`keyword_coverage` has nothing to compare against, so it is not scored at all."""
    alone = _report(templates.cv_txt(example, None), example)
    coverage = alone.check("keyword_coverage")
    assert coverage is not None
    assert coverage.possible == 0.0
    assert alone.total > 70.0
    assert sum(c.possible for c in alone.checks) == 70.0


def test_bands_are_read_off_the_score_not_off_a_vibe(example: Profile) -> None:
    report = _report(templates.cv_txt(example, None), example)
    band: str = report.band
    assert band == "ready"
    for check in report.checks:
        check.points = 0.0
    assert report.total == 0.0
    assert report.band == "weak"
