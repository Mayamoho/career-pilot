"""SQLite persistence with FTS5 keyword search.

Chosen over a vector store for v1 on purpose: FTS5 gives real BM25 ranking with zero
infrastructure, and the hybrid-retrieval pattern is already proven on CSxEDU where it
matters. Embeddings are an additive second signal, not a prerequisite.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from careerpilot.config import DATA_DIR
from careerpilot.domain.job import JobPosting
from careerpilot.match.score import FitScore

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    uid             TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    location        TEXT,
    is_remote       INTEGER NOT NULL DEFAULT 0,
    department      TEXT,
    url             TEXT NOT NULL,
    apply_url       TEXT,
    content_hash    TEXT NOT NULL,
    description     TEXT NOT NULL,
    published_at    TEXT,
    updated_at      TEXT,
    collected_at    TEXT NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_hash    ON jobs(content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_pub     ON jobs(published_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    title, company, description, content='jobs', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS board_health (
    source   TEXT NOT NULL,
    token    TEXT NOT NULL,
    ok       INTEGER NOT NULL,
    count    INTEGER NOT NULL DEFAULT 0,
    probed_at TEXT NOT NULL,
    PRIMARY KEY (source, token)
);

CREATE TABLE IF NOT EXISTS scores (
    uid          TEXT PRIMARY KEY REFERENCES jobs(uid),
    total        REAL NOT NULL,
    band         TEXT NOT NULL,
    matched      INTEGER NOT NULL,
    required     INTEGER NOT NULL,
    detail       TEXT NOT NULL,
    scored_at    TEXT NOT NULL
);
"""

FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
  INSERT INTO jobs_fts(rowid, title, company, description)
  VALUES (new.rowid, new.title, new.company, new.description);
END;
CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, description)
  VALUES ('delete', old.rowid, old.title, old.company, old.description);
END;
CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, description)
  VALUES ('delete', old.rowid, old.title, old.company, old.description);
  INSERT INTO jobs_fts(rowid, title, company, description)
  VALUES (new.rowid, new.title, new.company, new.description);
END;
"""


class JobStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DATA_DIR / "jobs.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        try:
            self._conn.executescript(FTS_TRIGGERS)
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, job: JobPosting) -> tuple[str, int]:
        """Returns (uid, 1 if newly inserted else 0). Dedupe is on content, so an
        edited posting updates in place instead of creating a phantom second job."""
        row = self._conn.execute(
            "SELECT content_hash FROM jobs WHERE uid = ?", (job.uid,)
        ).fetchone()
        payload = job.model_dump_json()
        if row is None:
            self._conn.execute(
                """INSERT INTO jobs (uid, source, source_id, company, title, location, is_remote,
                       department, url, apply_url, content_hash, description, published_at,
                       updated_at, collected_at, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job.uid, job.source, job.source_id, job.company, job.title, job.location,
                    int(job.is_remote), job.department, job.url, job.apply_url, job.content_hash,
                    job.description,
                    job.published_at.isoformat() if job.published_at else None,
                    job.updated_at.isoformat() if job.updated_at else None,
                    job.collected_at.isoformat(), payload,
                ),
            )
            self._conn.commit()
            return job.uid, 1
        if row["content_hash"] != job.content_hash:
            self._conn.execute(
                """UPDATE jobs SET title=?, company=?, location=?, is_remote=?, department=?,
                       url=?, apply_url=?, content_hash=?, description=?, updated_at=?,
                       collected_at=?, payload=? WHERE uid=?""",
                (
                    job.title, job.company, job.location, int(job.is_remote), job.department,
                    job.url, job.apply_url, job.content_hash, job.description,
                    job.updated_at.isoformat() if job.updated_at else None,
                    job.collected_at.isoformat(), payload, job.uid,
                ),
            )
            self._conn.commit()
        return job.uid, 0

    def upsert_many(self, jobs: Iterable[JobPosting]) -> dict[str, int]:
        inserted = updated = 0
        for job in jobs:
            _, flag = self.upsert(job)
            inserted += flag
            updated += 1 - flag
        return {"inserted": inserted, "unchanged_or_updated": updated}

    def get(self, uid: str) -> JobPosting | None:
        row = self._conn.execute("SELECT payload FROM jobs WHERE uid = ?", (uid,)).fetchone()
        return JobPosting.model_validate(json.loads(row["payload"])) if row else None

    def all(self, limit: int = 1000) -> list[JobPosting]:
        rows = self._conn.execute(
            "SELECT payload FROM jobs ORDER BY published_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [JobPosting.model_validate(json.loads(r["payload"])) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def search(self, query: str, limit: int = 50) -> list[JobPosting]:
        if self.fts_enabled and query.strip():
            try:
                rows = self._conn.execute(
                    """SELECT j.payload FROM jobs_fts f
                       JOIN jobs j ON j.rowid = f.rowid
                       WHERE jobs_fts MATCH ? ORDER BY bm25(jobs_fts) LIMIT ?""",
                    (fts_query(query), limit),
                ).fetchall()
                return [JobPosting.model_validate(json.loads(r["payload"])) for r in rows]
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT payload FROM jobs WHERE title LIKE ? OR description LIKE ? LIMIT ?",
            (like, like, limit),
        ).fetchall()
        return [JobPosting.model_validate(json.loads(r["payload"])) for r in rows]

    def record_board_health(self, source: str, token: str, ok: bool, count: int) -> None:
        self._conn.execute(
            """INSERT INTO board_health (source, token, ok, count, probed_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(source, token) DO UPDATE SET ok=excluded.ok,
                   count=excluded.count, probed_at=excluded.probed_at""",
            (source, token, int(ok), count, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def dead_boards(self) -> list[tuple[str, str]]:
        rows = self._conn.execute("SELECT source, token FROM board_health WHERE ok = 0").fetchall()
        return [(r["source"], r["token"]) for r in rows]

    def save_score(self, fit: FitScore) -> None:
        """Persist the score plus its full audit trail, keyed by the posting."""
        detail = json.dumps(
            {
                "total": fit.total,
                "band": fit.band,
                "matched": fit.met,
                "required": len(fit.outcomes),
                "audit": fit.audit_trail(),
            }
        )
        self._conn.execute(
            """INSERT INTO scores (uid, total, band, matched, required, detail, scored_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(uid) DO UPDATE SET total=excluded.total, band=excluded.band,
                 matched=excluded.matched, required=excluded.required,
                 detail=excluded.detail, scored_at=excluded.scored_at""",
            (
                fit.job_uid,
                fit.total,
                fit.band,
                fit.met,
                len(fit.outcomes),
                detail,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()


def fts_query(raw: str) -> str:
    """Quote each token so a stray `:` or `"` in a search cannot break out of FTS."""
    tokens = [t for t in raw.replace('"', " ").split() if t]
    return " AND ".join(f'"{t}"' for t in tokens[:12]) or '""'
