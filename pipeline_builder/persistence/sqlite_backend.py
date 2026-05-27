"""SQLite-based storage backend — zero extra dependencies (sqlite3 is stdlib).

Key advantage over FileBackend: snapshots are stored in the same database
as session state, so a single-file backup captures everything needed for
crash-safe rollback.

All operations use WAL mode for concurrent-read safety and transactions
for atomicity.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    pipeline_name TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS states (
    session_id   TEXT PRIMARY KEY,
    data         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    session_id   TEXT NOT NULL,
    stage_name   TEXT NOT NULL,
    nodes_json   TEXT NOT NULL,
    artifacts_json TEXT NOT NULL,
    PRIMARY KEY (session_id, stage_name)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteBackend:
    """Persistent storage in a single SQLite file.

    Usage::

        pipe = Pipeline("my_pipe", hierarchy=[...],
                        storage=SQLiteBackend("./runs.db"))
    """

    def __init__(self, db_path: str | Path, pipeline_name: str = "") -> None:
        self._path = str(Path(db_path))
        self._pipeline_name = pipeline_name
        self._local = threading.local()  # per-thread connection
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection management (per-thread to avoid sqlite3 thread restrictions)
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------ #

    def save(self, session_id: str, state_dict: dict[str, Any]) -> None:
        now = _now()
        data = json.dumps(state_dict, ensure_ascii=False)
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO sessions (session_id, pipeline_name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET updated_at=?",
                (session_id, self._pipeline_name, now, now, now),
            )
            conn.execute(
                "INSERT INTO states (session_id, data) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET data=?",
                (session_id, data, data),
            )

    def load(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT data FROM states WHERE session_id=?", (session_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def save_snapshot(
        self,
        session_id: str,
        stage_name: str,
        nodes: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> None:
        nodes_json = json.dumps(nodes, ensure_ascii=False)
        arts_json = json.dumps(artifacts, ensure_ascii=False)
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO snapshots (session_id, stage_name, nodes_json, artifacts_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, stage_name) DO UPDATE SET "
                "nodes_json=?, artifacts_json=?",
                (session_id, stage_name, nodes_json, arts_json, nodes_json, arts_json),
            )

    def load_snapshot(
        self, session_id: str, stage_name: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        row = self._conn().execute(
            "SELECT nodes_json, artifacts_json FROM snapshots "
            "WHERE session_id=? AND stage_name=?",
            (session_id, stage_name),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["nodes_json"]), json.loads(row["artifacts_json"])

    def list_sessions(self) -> list[str]:
        rows = self._conn().execute(
            "SELECT session_id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def list_sessions_detail(self) -> list[dict[str, Any]]:
        """Extended listing with metadata (for tooling / dashboards)."""
        rows = self._conn().execute(
            "SELECT session_id, pipeline_name, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM states WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM snapshots WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
