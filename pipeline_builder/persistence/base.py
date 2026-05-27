"""StorageBackend protocol — pluggable persistence for baton pipelines."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Interface for pipeline state persistence.

    Implementations must be thread-safe: save/load can be called from the
    heartbeat thread concurrently with the main execution thread.

    Minimal implementation needs only save/load; snapshot methods can be
    no-ops (snapshots then live only in memory and are lost on crash).
    """

    def save(self, session_id: str, state_dict: dict[str, Any]) -> None:
        """Persist full pipeline state."""

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Return persisted state, or None if session not found."""

    def save_snapshot(
        self,
        session_id: str,
        stage_name: str,
        nodes: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> None:
        """Persist a pre-stage snapshot so rollback survives crashes."""

    def load_snapshot(
        self, session_id: str, stage_name: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return (nodes, artifacts) snapshot, or None if not found."""

    def list_sessions(self) -> list[str]:
        """Return all known session IDs, most recent first."""

    def delete(self, session_id: str) -> None:
        """Remove all persisted data for a session."""
