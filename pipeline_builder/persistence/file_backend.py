"""File-based storage backend (JSON files, existing baton behaviour)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class FileBackend:
    """One JSON file per session, one JSON file per snapshot.

    State file:    {dir}/{session_id}.json
    Snapshot file: {dir}/{session_id}.snap_{stage_name}.json

    All writes are atomic (write to .tmp, then os.replace).
    Thread-safe via a per-instance lock.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def save(self, session_id: str, state_dict: dict[str, Any]) -> None:
        target = self._dir / f"{session_id}.json"
        tmp = target.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(state_dict, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(target)

    def load(self, session_id: str) -> dict[str, Any] | None:
        target = self._dir / f"{session_id}.json"
        if not target.exists():
            return None
        with self._lock:
            return json.loads(target.read_text(encoding="utf-8"))

    def save_snapshot(
        self,
        session_id: str,
        stage_name: str,
        nodes: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> None:
        target = self._dir / f"{session_id}.snap_{stage_name}.json"
        tmp = target.with_suffix(".tmp")
        data = {"nodes": nodes, "artifacts": artifacts}
        with self._lock:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(target)

    def load_snapshot(
        self, session_id: str, stage_name: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        target = self._dir / f"{session_id}.snap_{stage_name}.json"
        if not target.exists():
            return None
        with self._lock:
            data = json.loads(target.read_text(encoding="utf-8"))
        return data["nodes"], data["artifacts"]

    def list_sessions(self) -> list[str]:
        with self._lock:
            files = sorted(
                self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        return [
            f.stem for f in files
            if ".snap_" not in f.stem and not f.stem.endswith(".tmp")
        ]

    def delete(self, session_id: str) -> None:
        with self._lock:
            for f in self._dir.glob(f"{session_id}*"):
                f.unlink(missing_ok=True)
