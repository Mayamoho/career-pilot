"""The normalized job record. Every adapter maps into this and nothing else."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Source = Literal["greenhouse", "ashby", "lever", "bdtechjobs", "bdjobs", "remotive", "manual"]

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"(?i)</(p|div|li|br|h[1-6]|tr)[^>]*>")
_WS = re.compile(r"[ \t\f\v]+")
# Timestamps above this are milliseconds; the ATS feeds mix both.
_EPOCH_MS_CUTOFF = 1e11
_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&nbsp;": " ", "&rsquo;": "'", "&lsquo;": "'",
    "&ldquo;": '"', "&rdquo;": '"', "&hellip;": "...",
}


def html_to_text(raw: str | None) -> str:
    """Deterministic HTML-to-text.

    Deterministic matters more than pretty here: the content hash that dedupes
    postings is computed on this output, so two runs must agree byte for byte.
    """
    if not raw:
        return ""
    text = _BLOCK.sub("\n", raw)
    text = _TAG.sub(" ", text)
    for entity, replacement in _ENTITIES.items():
        text = text.replace(entity, replacement)
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class JobPosting(BaseModel):
    source: Source
    source_id: str
    company: str
    title: str
    description: str = ""
    location: str | None = None
    is_remote: bool = False
    department: str | None = None
    team: str | None = None
    employment_type: str | None = None
    seniority: str | None = None
    url: str = ""
    apply_url: str = ""
    published_at: datetime | None = None
    updated_at: datetime | None = None
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    compensation: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def content_hash(self) -> str:
        """Stable identity across re-fetches: what changed is the text, not the crawl."""
        blob = "\x1f".join(
            [self.company.lower().strip(), self.title.lower().strip(), self.description]
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def searchable(self) -> str:
        return f"{self.title}\n{self.company}\n{self.department or ''}\n{self.description}"

    def requirements_excerpt(self, limit: int = 6000) -> str:
        """Slice the section that actually states requirements.

        Feeding a whole 12k-character posting to a small model is how you get a
        confident summary of the wrong paragraph.
        """
        lowered = self.description.lower()
        markers = (
            "qualifications", "requirements", "requirement", "what you'll bring",
            "what you will bring", "we're looking for", "we are looking for",
            "desired skills", "skills & experience", "about you", "minimum qualifications",
        )
        best = -1
        for marker in markers:
            idx = lowered.find(marker)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
        start = best if best != -1 else 0
        return self.description[start : start + limit]


def coerce_datetime(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > _EPOCH_MS_CUTOFF else value
        return datetime.fromtimestamp(seconds, UTC).replace(tzinfo=None)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
