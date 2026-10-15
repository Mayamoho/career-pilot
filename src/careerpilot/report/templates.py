"""The CV as a document: two house styles, built as strings, no template engine.

Both styles render the *same* content — the gated bullets plus the fact base verbatim —
and differ only in markup, because a candidate should be able to pick a look without
changing a word of what is claimed about them.

HTML rather than PDF: `Ctrl+P` on this file produces a PDF in any browser, the text in
that PDF still selects and copies, and the dependency list stays at five packages.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from html import escape

from careerpilot.pipeline import Tailored
from careerpilot.profile.model import Atom, Profile

STYLES = ("ats", "oxford")

# The same order `render.py` uses for the markdown, so the three outputs of one run
# cannot disagree about what a recruiter reads first.
_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Experience", ("role",)),
    ("Projects", ("project",)),
    ("Certifications", ("credential",)),
    ("Recognition", ("award",)),
    ("Education", ("education",)),
    ("Skills", ("skill",)),
    ("Languages", ("language",)),
)

_CONTACT_KEYS = ("email", "phone", "location", "github", "linkedin")

_BASE_CSS = """
@page { size: A4; margin: 18mm; }
body { margin: 0; }
p, li, h1, h2 { margin: 0; }
a { color: inherit; text-decoration: none; }
"""

# Single column, Arial, no table, no icons, no columns: the layout an ATS parser is
# least likely to scramble. Bold headings and hyphen bullets, because those survive
# being copied into a portal's textarea.
_ATS_CSS = """
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.35; color: #000; }
h1 { font-size: 15pt; text-transform: uppercase; letter-spacing: 0.6pt; }
.headline { margin-top: 3pt; }
.contact { margin-top: 3pt; font-size: 10pt; }
h2 {
  font-size: 11pt; text-transform: uppercase; letter-spacing: 0.6pt;
  margin: 14pt 0 4pt; border-bottom: 1px solid #000; padding-bottom: 2pt;
}
.lead { font-weight: bold; }
li { margin: 0 0 4pt; }
ul { padding-left: 16pt; margin: 0; }
"""

# The Oxford shape: centred name, contact line under it, a thin rule, serif section
# headings with a hairline below, dates set against the right margin. The dates are
# positioned with CSS grid, never with a two-column table — a table is what breaks
# parsing, and it is the one thing this template is meant to demonstrate.
_OXFORD_CSS = """
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.45; color: #111;
}
.name { text-align: center; font-size: 21pt; letter-spacing: 0.4pt; }
.headline { text-align: center; font-style: italic; margin-top: 2pt; }
.contact { text-align: center; font-size: 9.5pt; margin-top: 4pt; }
.rule { border-bottom: 1px solid #444; margin-top: 9pt; }
h2 {
  font-size: 10.5pt; letter-spacing: 1pt; text-transform: uppercase;
  margin: 13pt 0 4pt; border-bottom: 0.5pt solid #999; padding-bottom: 1pt;
}
.row { display: grid; grid-template-columns: 1fr auto; column-gap: 10pt; }
.where { font-weight: bold; }
.when { text-align: right; white-space: nowrap; color: #333; }
.detail { margin: 1pt 0 6pt; }
ul { list-style: disc; margin: 0; padding-left: 16pt; }
li { margin: 0 0 3pt; }
"""


@dataclass
class Row:
    """One line of the CV: the claim, and where and when it happened."""

    text: str
    where: str = ""
    when: str = ""


@dataclass
class Block:
    heading: str
    rows: list[Row]


def build_blocks(profile: Profile, tailored: Tailored | None) -> list[Block]:
    """The CV's content, decided once and then styled by every output format.

    `tailored` may be None: a candidate who has not run `tailor` yet still has a CV, it
    simply has no highlights section. That is the normal state of a fresher.
    """
    blocks: list[Block] = []
    if tailored is not None:
        shipped = tailored.shipped_bullets or [
            "Nothing in the fact base could be phrased as a supported claim for this role."
        ]
        blocks.append(Block("Tailored highlights", [Row(text=text) for text in shipped]))
    for heading, kinds in _SECTIONS:
        atoms = [atom for atom in profile.atoms if atom.kind in kinds]
        if not atoms:
            continue
        if heading in {"Skills", "Languages"}:
            blocks.append(Block(heading, [Row(text=_skills_line(atoms))]))
        else:
            blocks.append(Block(heading, [
                Row(text=atom.text, where=atom.org or "", when=atom.period or "") for atom in atoms
            ]))
    if profile.headline:
        blocks.insert(0, Block("Summary", [Row(text=profile.headline)]))
    return blocks


def cv_html(profile: Profile, tailored: Tailored | None, style: str) -> str:
    if style == "ats":
        head = [f"<h1>{escape(profile.name)}</h1>"]
    else:
        head = [f'<div class="name">{escape(profile.name)}</div>']
    contact = _contact(profile)
    if contact:
        head.append(f'<div class="contact">{escape(contact)}</div>')
    if style == "oxford":
        head.append('<div class="rule"></div>')

    body: list[str] = list(head)
    for block in build_blocks(profile, tailored):
        body.append(f"<h2>{escape(block.heading)}</h2>")
        body.append(_block_html(block, style))

    css = _ATS_CSS if style == "ats" else _OXFORD_CSS
    newline = "\n"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{escape(profile.name)} — CV</title>\n"
        f"<style>{_BASE_CSS}{css}</style>\n</head>\n<body>\n{newline.join(body)}\n</body>\n</html>\n"
    )


def letter_html(profile: Profile, paragraphs: list[str]) -> str:
    """The cover letter under the Oxford header: the two documents arrive as one set."""
    body = [
        f'<div class="name">{escape(profile.name)}</div>',
        f'<div class="contact">{escape(_contact(profile))}</div>',
        '<div class="rule"></div>',
        *(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs),
    ]
    newline = "\n"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{escape(profile.name)} — cover letter</title>\n"
        f"<style>{_BASE_CSS}{_OXFORD_CSS}\n"
        "p { margin: 0 0 10pt; text-align: justify; }\n</style>\n</head>\n"
        f"<body>\n{newline.join(body)}\n</body>\n</html>\n"
    )


def cv_txt(profile: Profile, tailored: Tailored | None) -> str:
    """ASCII-only plain text — the one input every parser survives and every portal box takes.

    Deliberately free of the characters the ATS checker flags: no em dash, no middot, no
    Bangla. `careerpilot ats --cv cv.txt` therefore measures the content, not the
    typography, and this is what a student pastes into jobs.bdjobs.com's application box.
    """
    lines = [_ascii(profile.name).upper(), ""]
    contact = _ascii(_contact(profile))
    if contact:
        lines += [contact, ""]
    for block in build_blocks(profile, tailored):
        heading = _ascii(block.heading).upper()
        lines += [heading, "-" * len(heading)]
        for row in block.rows:
            stamp = " | ".join(filter(None, [_ascii(row.where), _ascii(row.when)]))
            text = _ascii(row.text)
            # The stamp trails, so every row still opens with the bullet marker. Written the
            # other way round, `university lab | 2025 - Wrote a parser` is a line a parser
            # splits on nothing, and this tool's own checker stopped seeing the bullet at all.
            lines.append(f"- {text} ({stamp})" if stamp else f"- {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _block_html(block: Block, style: str) -> str:
    if style == "ats":
        items = []
        for row in block.rows:
            lead = " | ".join(filter(None, [row.where, row.when]))
            prefix = f'<span class="lead">{escape(lead)}</span> - ' if lead else ""
            items.append(f"<li>{prefix}{escape(row.text)}</li>")
        return "<ul>" + "".join(items) + "</ul>"
    parts = []
    for row in block.rows:
        if row.where or row.when:
            parts.append(
                '<div class="row">'
                f'<span class="where">{escape(row.where)}</span>'
                f'<span class="when">{escape(row.when)}</span></div>'
                f'<div class="detail">{escape(row.text)}</div>'
            )
        else:
            parts.append(f'<div class="detail">{escape(row.text)}</div>')
    return "".join(parts)


def _contact(profile: Profile) -> str:
    parts = [
        profile.contact[key]
        for key in _CONTACT_KEYS
        if key in profile.contact and "TODO" not in profile.contact[key]
    ]
    return " | ".join(parts)


def _skills_line(atoms: list[Atom]) -> str:
    values = (
        part.strip() for atom in atoms for part in atom.text.split(",") if part.strip()
    )
    return ", ".join(dict.fromkeys(values))


def _ascii(text: str) -> str:
    """Fold typographic punctuation to ASCII and drop what cannot be folded.

    Dropping is the honest choice here: the surviving text is exactly what a parser will
    see after its own normalisation, and a Bangla line that matters belongs in the HTML.
    """
    folded = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in folded if ch.isascii() and (ch.isprintable() or ch == " ")).strip()
