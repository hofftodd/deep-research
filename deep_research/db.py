"""SQLite persistence for topics, research questions, sources, notes and the activity log."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'queued',
    phase         TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,
    model         TEXT NOT NULL DEFAULT '',
    options       TEXT NOT NULL DEFAULT '{}',
    report_md     TEXT,
    report_refs   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT
);
CREATE TABLE IF NOT EXISTS questions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id   INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'open',
    iteration  INTEGER NOT NULL DEFAULT 0,
    queries    TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id    INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    question_id INTEGER,
    url         TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    snippet     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    note_count  INTEGER NOT NULL DEFAULT 0,
    chars       INTEGER NOT NULL DEFAULT 0,
    content     TEXT,
    error       TEXT,
    fetched_at  TEXT,
    UNIQUE(topic_id, url)
);
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id    INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    question_id INTEGER,
    claim       TEXT NOT NULL,
    quote       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id  INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    ts        TEXT NOT NULL,
    level     TEXT NOT NULL DEFAULT 'info',
    message   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic_id, id);
CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_id);
CREATE INDEX IF NOT EXISTS idx_sources_topic ON sources(topic_id);
"""

ACTIVE_STATUSES = ("planning", "researching", "writing")
FINAL_STATUSES = ("completed", "failed", "cancelled")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            return [dict(r) for r in cur.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        with self._cursor() as cur:
            row = cur.execute(sql, args).fetchone()
            return dict(row) if row else None

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._cursor() as cur:
            cur.execute(sql, args)
            return cur.lastrowid

    # ---- topics -------------------------------------------------------------------------
    def create_topic(self, title: str, description: str, model: str, options: dict) -> int:
        ts = now()
        return self._exec(
            "INSERT INTO topics (title, description, model, options, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (title, description, model, json.dumps(options), ts, ts),
        )

    def get_topic(self, topic_id: int) -> dict[str, Any] | None:
        t = self._one("SELECT * FROM topics WHERE id = ?", (topic_id,))
        if t:
            t["options"] = json.loads(t["options"] or "{}")
            t["report_refs"] = json.loads(t["report_refs"]) if t["report_refs"] else []
        return t

    def list_topics(self) -> list[dict[str, Any]]:
        return self._all(
            "SELECT t.id, t.title, t.status, t.phase, t.progress, t.model, t.created_at,"
            " t.updated_at, t.completed_at, t.error,"
            " (SELECT COUNT(*) FROM sources s WHERE s.topic_id = t.id AND s.status = 'used') AS source_count,"
            " (SELECT COUNT(*) FROM notes n WHERE n.topic_id = t.id) AS note_count"
            " FROM topics t ORDER BY t.id DESC"
        )

    def update_topic(self, topic_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now()
        for k in ("options", "report_refs"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE topics SET {cols} WHERE id = ?", (*fields.values(), topic_id))

    def delete_topic(self, topic_id: int) -> None:
        self._exec("DELETE FROM topics WHERE id = ?", (topic_id,))

    def reset_topic(self, topic_id: int) -> None:
        """Clear all research artefacts so the topic can be researched again from scratch."""
        for table in ("events", "notes", "sources", "questions"):
            self._exec(f"DELETE FROM {table} WHERE topic_id = ?", (topic_id,))
        self.update_topic(
            topic_id, status="queued", phase="", progress=0, report_md=None, report_refs=None,
            error=None, started_at=None, completed_at=None,
        )

    def queued_topic_ids(self) -> list[int]:
        return [r["id"] for r in self._all("SELECT id FROM topics WHERE status = 'queued' ORDER BY id")]

    def mark_interrupted(self) -> int:
        """Topics left mid-run by a previous process can't be resumed; mark them failed."""
        placeholders = ",".join("?" * len(ACTIVE_STATUSES))
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE topics SET status = 'failed', error = 'Interrupted by server restart; use Retry.',"
                f" updated_at = ? WHERE status IN ({placeholders})",
                (now(), *ACTIVE_STATUSES),
            )
            return cur.rowcount

    # ---- questions ------------------------------------------------------------------------
    def add_question(self, topic_id: int, text: str, iteration: int) -> int:
        return self._exec(
            "INSERT INTO questions (topic_id, text, iteration) VALUES (?, ?, ?)",
            (topic_id, text, iteration),
        )

    def questions(self, topic_id: int) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT q.*, (SELECT COUNT(*) FROM notes n WHERE n.question_id = q.id) AS note_count"
            " FROM questions q WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        )
        for r in rows:
            r["queries"] = json.loads(r["queries"])
        return rows

    def update_question(self, question_id: int, status: str | None = None, queries: list[str] | None = None) -> None:
        if status is not None:
            self._exec("UPDATE questions SET status = ? WHERE id = ?", (status, question_id))
        if queries is not None:
            self._exec("UPDATE questions SET queries = ? WHERE id = ?", (json.dumps(queries), question_id))

    # ---- sources ------------------------------------------------------------------------
    def add_source(self, topic_id: int, question_id: int, url: str, title: str, snippet: str) -> int | None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO sources (topic_id, question_id, url, title, snippet)"
                " VALUES (?, ?, ?, ?, ?)",
                (topic_id, question_id, url, title, snippet),
            )
            return cur.lastrowid if cur.rowcount else None

    def update_source(self, source_id: int, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE sources SET {cols} WHERE id = ?", (*fields.values(), source_id))

    def sources(self, topic_id: int) -> list[dict[str, Any]]:
        """Source metadata (without the cached page text)."""
        return self._all(
            "SELECT id, topic_id, question_id, url, title, snippet, status, note_count, chars, error, fetched_at"
            " FROM sources WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        )

    def source_content(self, source_id: int) -> str:
        row = self._one("SELECT content FROM sources WHERE id = ?", (source_id,))
        return (row or {}).get("content") or ""

    def count_sources(self, topic_id: int, status: str | None = None) -> int:
        if status:
            row = self._one("SELECT COUNT(*) c FROM sources WHERE topic_id = ? AND status = ?", (topic_id, status))
        else:
            row = self._one("SELECT COUNT(*) c FROM sources WHERE topic_id = ?", (topic_id,))
        return row["c"] if row else 0

    # ---- notes --------------------------------------------------------------------------
    def add_note(self, topic_id: int, source_id: int, question_id: int, claim: str, quote: str) -> int:
        return self._exec(
            "INSERT INTO notes (topic_id, source_id, question_id, claim, quote) VALUES (?, ?, ?, ?, ?)",
            (topic_id, source_id, question_id, claim, quote),
        )

    def notes(self, topic_id: int, question_id: int | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT n.*, s.url, s.title AS source_title FROM notes n JOIN sources s ON s.id = n.source_id"
            " WHERE n.topic_id = ?"
        )
        args: tuple = (topic_id,)
        if question_id is not None:
            sql += " AND n.question_id = ?"
            args += (question_id,)
        return self._all(sql + " ORDER BY n.id", args)

    # ---- events -------------------------------------------------------------------------
    def log(self, topic_id: int, message: str, level: str = "info") -> None:
        self._exec(
            "INSERT INTO events (topic_id, ts, level, message) VALUES (?, ?, ?, ?)",
            (topic_id, now(), level, message),
        )

    def events(self, topic_id: int, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return self._all(
            "SELECT * FROM events WHERE topic_id = ? AND id > ? ORDER BY id LIMIT ?",
            (topic_id, after, limit),
        )
