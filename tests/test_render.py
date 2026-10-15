"""The three CV outputs — markdown, two HTML templates, plain text — from one set of facts.

Templates are string, not jinja2, so the guarantees that matter have to be asserted:
nothing unclosed, nothing unescaped, no table a parser reads in the wrong order, and
`Certifications` present because a fresher's Coursera and NPTEL certificates were being
silently dropped from every CV this tool ever printed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from careerpilot.domain.job import JobPosting
from careerpilot.pipeline import tailor
from careerpilot.profile.model import Atom, Profile, load_profile
from careerpilot.report import render, templates

REPO = Path(__file__).resolve().parents[1]
_EXTRACTED = {
    "requirements": [{"text": "Python services", "skill": "Python", "category": "must_have"}],
    "seniority": "junior",
    "location_constraint": "unspecified",
}
# A claim the example fact base genuinely supports, so the gate lets it through.
TRUTHY_BULLET = "Built a retrieval service over pgvector and PostgreSQL full-text search"
CITED = ["proj-csxedu-rag"]


@pytest.fixture(scope="module")
def example() -> Profile:
    return load_profile(REPO / "data" / "profile.example.yaml")


def _credential_profile() -> Profile:
    return Profile(
        name="Amina Khatun",
        headline="Python developer — final year CSE student",
        contact={"email": "amina@example.com", "location": "Mirpur, Dhaka, Bangladesh"},
        atoms=[
            Atom(
                id="credential-dsa",
                kind="credential",
                text="Data Structures and Algorithms",
                org="NPTEL",
                period="2024",
                source="start-here",
            ),
            Atom(
                id="project-parser",
                kind="project",
                text="Built a marksheet parser that reads scanned PDFs",
                org="coursework",
                period="2025",
                source="start-here",
            ),
        ],
    )


def _tailored(job: JobPosting, profile: Profile, use_task_fake: Any) -> Any:
    use_task_fake(
        {
            "extract_requirements": _EXTRACTED,
            "cite_evidence": {"atom_ids": ["proj-csxedu-rag"]},
            "draft_bullets": {"bullets": [{"text": TRUTHY_BULLET, "atom_ids": CITED}]},
            "draft_letter": {"paragraphs": []},
            "check_entailment": {"supported": True, "confidence": 0.9, "unsupported_span": ""},
        }
    )
    return tailor(job, profile)


# --- what gets printed ---------------------------------------------------------------


@pytest.mark.parametrize("style", templates.STYLES)
def test_both_templates_carry_the_certifications_section(style: str) -> None:
    """The gap this fixes: a `credential` atom was in no section list, so it never printed."""
    html = templates.cv_html(_credential_profile(), None, style)

    assert "CERTIFICATIONS" in html.upper()
    assert "Data Structures and Algorithms" in html
    assert "NPTEL" in html


def test_certifications_reach_the_plain_text_cv_too() -> None:
    text = templates.cv_txt(_credential_profile(), None)
    assert "CERTIFICATIONS" in text
    assert "Data Structures and Algorithms" in text


def test_a_stamped_row_still_opens_with_the_bullet_marker() -> None:
    """The stamp trails the claim, or this tool's own checker stops seeing the row as a bullet.

    A live run scored a fresher's fact base at 0/3 proven bullets: `university lab | 2025 -
    Wrote a parser` opens with an employer name, which nothing keys on, so the "40 hours"
    inside it was invisible to the very command that had just emitted the line.
    """
    text = templates.cv_txt(_credential_profile(), None)
    assert "- Built a marksheet parser that reads scanned PDFs (coursework | 2025)" in text
    assert "coursework | 2025 - Built" not in text


def test_a_headline_becomes_a_summary_section_rather_than_a_decorative_line() -> None:
    """`ats` looks for a heading a parser can key on, so the headline has to be a section."""
    fresh = _credential_profile()
    assert templates.build_blocks(fresh, None)[0].heading == "Summary"
    assert "SUMMARY" in templates.cv_txt(fresh, None)
    for style in templates.STYLES:
        assert "summary" in templates.cv_html(fresh, None, style).lower()


def test_a_profile_with_no_headline_prints_no_empty_summary() -> None:
    bare = _credential_profile().model_copy(update={"headline": ""})
    assert "Summary" not in [b.heading for b in templates.build_blocks(bare, None)]
    assert "<h2></h2>" not in templates.cv_html(bare, None, "ats")


def test_a_cv_without_a_run_of_tailor_still_has_every_fact_in_it() -> None:
    """`tailored=None` is the normal state of a fresher, not an error path."""
    blocks = templates.build_blocks(_credential_profile(), None)
    assert "Tailored highlights" not in [b.heading for b in blocks]
    assert len(blocks) == 3  # Summary, Projects, Certifications


def test_shipped_bullets_are_copied_verbatim_into_both_templates(
    job: JobPosting, example: Profile, use_task_fake: Any
) -> None:
    tailored = _tailored(job, example, use_task_fake)
    assert tailored.shipped_bullets, "the fake must ship something for this to mean anything"
    for bullet in tailored.shipped_bullets:
        for style in templates.STYLES:
            assert bullet in templates.cv_html(example, tailored, style)
        assert bullet in templates.cv_txt(example, tailored)


# --- the shapes an ATS and a recruiter both need -------------------------------------


@pytest.mark.parametrize("style", templates.STYLES)
def test_neither_template_uses_a_table_or_a_column_grid(style: str) -> None:
    """A two-column layout is the single most common reason a CV parses to garbage."""
    html = templates.cv_html(example_profile(), None, style)
    assert "<table" not in html
    assert "column-count" not in html
    assert "<img" not in html


def test_the_ats_style_is_plain_single_column_with_a4_margins() -> None:
    html = templates.cv_html(example_profile(), None, "ats")
    assert "@page { size: A4; margin: 18mm; }" in html
    assert "display: grid" not in html


def test_the_oxford_style_right_aligns_dates_without_a_table() -> None:
    """Dates sit on the right visually and stay on one line for a parser."""
    html = templates.cv_html(example_profile(), None, "oxford")
    assert "display: grid" in html
    assert "grid-template-columns: 1fr auto" in html


def test_the_plain_text_cv_is_ascii_only() -> None:
    """Everything non-ASCII is dropped by some portals, so our own export may not carry it."""
    text = templates.cv_txt(_non_ascii_profile(), None)
    assert text.isascii()
    assert "AMINA KHATUN" in text
    assert "English (fluent)" in text
    assert "\u2014" not in text and "\u2726" not in text and "বাংলা" not in text


def test_the_oxford_letter_shares_the_cv_header(
    job: JobPosting, example: Profile, use_task_fake: Any
) -> None:
    tailored = _tailored(job, example, use_task_fake)
    html = templates.letter_html(example, render.letter_parts(example, tailored))
    assert example.name in html
    assert "<!doctype html>" in html


# --- escaping, which is the whole attack surface of a string template ----------------


def test_every_field_is_escaped_in_every_output() -> None:
    """`start-here` takes typed input, so a name may legitimately contain markup."""
    profile = _non_ascii_profile().model_copy(
        update={
            "name": "Amina <script>alert(1)</script>",
            "headline": "Python & <b>Django</b> developer",
        }
    )
    for style in templates.STYLES:
        html = templates.cv_html(profile, None, style)
        assert "<script>alert(1)</script>" not in html
        assert "<b>Django</b>" not in html
        assert "&lt;script&gt;" in html
        assert "&amp;" in html
    assert "<script>" not in templates.cv_txt(profile, None)


def test_a_skill_that_contains_angle_brackets_cannot_break_the_list() -> None:
    profile = Profile(
        name="Amina",
        headline="x",
        contact={},
        atoms=[
            Atom(
                id="skill-c",
                kind="skill",
                text="C++, <C#",
                skills=["C++", "<C#"],
                source="test",
            )
        ],
    )
    html = templates.cv_html(profile, None, "ats")
    assert "<C#" not in html
    assert "&lt;C#" in html


def test_the_written_files_are_the_whole_set_and_parse_as_html(
    job: JobPosting, example: Profile, use_task_fake: Any, tmp_path: Path
) -> None:
    tailored = _tailored(job, example, use_task_fake)
    artifacts = render.write(tailored, example, out_dir=tmp_path / "out")

    names = sorted(p.name for p in [artifacts.cv_path, *artifacts.extra_paths])
    assert names == [
        "cover-letter.html",
        "cv-ats.html",
        "cv-oxford.html",
        "cv.md",
        "cv.txt",
    ]
    for path in artifacts.extra_paths:
        body = path.read_text(encoding="utf-8")
        if path.suffix == ".html":
            assert body.startswith("<!doctype html>")
            assert body.rstrip().endswith("</html>")
    json.loads(artifacts.audit_path.read_text(encoding="utf-8"))


def test_md_style_writes_no_html_and_both_writes_all_of_it(
    job: JobPosting, example: Profile, use_task_fake: Any, tmp_path: Path
) -> None:
    tailored = _tailored(job, example, use_task_fake)
    # cv.txt and the letter do not depend on which template was asked for, so they are
    # written either way: the plain text is the one artifact a portal box always accepts.
    only_one_style = render.write(tailored, example, out_dir=tmp_path / "a", styles=("ats",))
    assert [p.name for p in only_one_style.extra_paths] == [
        "cv-ats.html",
        "cv.txt",
        "cover-letter.html",
    ]

    everything = render.write(tailored, example, out_dir=tmp_path / "b", styles=templates.STYLES)
    assert len(everything.extra_paths) == 4


def _non_ascii_profile() -> Profile:
    return Profile(
        name="Amina Khatun",
        headline="Python developer — final year CSE student ✦",
        contact={"email": "amina@example.com", "location": "Mirpur, Dhaka, Bangladesh"},
        atoms=[
            Atom(id="language-bangla", kind="language", text="বাংলা (native)", source="test"),
            Atom(id="language-english", kind="language", text="English (fluent)", source="test"),
        ],
    )


def example_profile() -> Profile:
    return load_profile(REPO / "data" / "profile.example.yaml")
