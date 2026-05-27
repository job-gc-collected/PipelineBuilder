from __future__ import annotations

import copy
import json
import threading
import uuid
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .agent import AgentMessage

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class _AnyState(BaseModel):
    """Default typed state when no state_schema is provided.

    Accepts arbitrary extra fields so that pipelines without an explicit
    schema can still use ``state.data.x = y`` as a typed-ish escape hatch.
    """
    model_config = ConfigDict(extra="allow")


class ArtifactStore:
    """Thread-safe key-value store for user-owned pipeline data.

    This is what stages access via ``state.artifacts``.  It is completely
    isolated from pipeline_builder's own internal bookkeeping.
    """
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


class InternalStore:
    """Thread-safe store for pipeline_builder's own runtime bookkeeping.

    Completely separate from :class:`ArtifactStore` — user code never
    touches this.  Stores: skip flags, loop round counters, goal-check
    counters, pipeline fingerprint, stored hierarchy.

    Transient keys (skip flags — ``__pb_skip_*``) are NOT persisted
    across session saves; all other keys are.
    """
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def to_persist(self) -> dict[str, Any]:
        """Return only the keys that should survive a session save/resume.

        Skip flags (``__pb_skip_*``) are transient and excluded.
        """
        with self._lock:
            return {k: v for k, v in self._data.items()
                    if not k.startswith("__pb_skip_")}


class StageRecord:
    def __init__(self, name: str) -> None:
        self.name = name
        self.started_at: datetime = datetime.now()
        self.completed_at: datetime | None = None
        self.status: str = "running"

    def complete(self) -> None:
        self.completed_at = datetime.now()
        self.status = "completed"

    def fail(self) -> None:
        self.completed_at = datetime.now()
        self.status = "failed"

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration of this stage in milliseconds, or None if still running."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000


class State:
    def __init__(
        self,
        session_id: str | None = None,
        state_schema: type[BaseModel] | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.artifacts = ArtifactStore()
        self._internal = InternalStore()   # pipeline_builder bookkeeping
        self._nodes: dict[str, list[BaseModel]] = {}
        self._history: list[StageRecord] = []
        # P2: per-stage snapshots taken before execution
        # Each snapshot stores {"nodes": ..., "artifacts": ..., "data": ...} so
        # rollback restores the complete pipeline state.
        self._snapshots: dict[str, dict] = {}
        # P1: ordered list of completed stage names (used for resume + rollback trimming)
        self._completed_stages: list[str] = []
        # Agent messages: inter-agent communication bus
        self._messages: list[Any] = []   # list[AgentMessage]; Any to avoid import cycle
        self._lock = threading.Lock()
        # Typed pipeline context (state.data)
        self._state_schema: type[BaseModel] = state_schema or _AnyState
        self._data: BaseModel = self._state_schema()

    # ------------------------------------------------------------------ #
    # Node access                                                         #
    # ------------------------------------------------------------------ #

    def set_nodes(self, level: str, nodes: list[BaseModel]) -> None:
        with self._lock:
            self._nodes[level] = nodes

    def extend_nodes(self, level: str, nodes: list[BaseModel]) -> None:
        with self._lock:
            self._nodes.setdefault(level, []).extend(nodes)

    def clear_nodes(self, level: str) -> None:
        """Remove all nodes at a level (called before a stage group runs)."""
        with self._lock:
            self._nodes.pop(level, None)

    def get_nodes(self, level: str) -> list[BaseModel]:
        with self._lock:
            return list(self._nodes.get(level, []))

    def add_history(self, record: StageRecord) -> None:
        with self._lock:
            self._history.append(record)

    @property
    def history(self) -> list[StageRecord]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------ #
    # Typed pipeline context (state.data)                                 #
    # ------------------------------------------------------------------ #

    @property
    def data(self) -> BaseModel:
        """Typed pipeline context — a Pydantic model instance.

        Set ``state_schema`` on the Pipeline to get a fully typed model with
        IDE autocompletion and runtime validation.  Without a schema, returns
        an ``_AnyState`` instance that accepts arbitrary extra fields.

        Thread-safety note: reading ``state.data.field`` is safe from any
        thread.  Writing ``state.data.field = value`` is safe for sequential
        stages; use :meth:`update_data` for batch writes from parallel stages.
        """
        return self._data

    def update_data(self, **kwargs: Any) -> None:
        """Thread-safe batch update to typed state fields.

        Preferred for writes from parallel stages (workers > 1)::

            state.update_data(mental_model=model, mode="explore_only")
        """
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._data, key, value)

    # ------------------------------------------------------------------ #
    # P1: completion tracking                                             #
    # ------------------------------------------------------------------ #

    def mark_completed(self, stage_name: str) -> None:
        with self._lock:
            if stage_name not in self._completed_stages:
                self._completed_stages.append(stage_name)

    def is_completed(self, stage_name: str) -> bool:
        with self._lock:
            return stage_name in self._completed_stages

    def trim_completed(self, stages_to_remove: set[str]) -> None:
        """Remove specific stage names from the completed list (graph-aware rollback)."""
        with self._lock:
            self._completed_stages = [
                s for s in self._completed_stages if s not in stages_to_remove
            ]

    # ------------------------------------------------------------------ #
    # Agent messages                                                      #
    # ------------------------------------------------------------------ #

    def post_message(
        self,
        from_agent: str,
        content: str,
        to_agent: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Post a message from one agent, optionally addressed to another.

        ``to_agent=None`` means broadcast — readable by all agents.
        """
        from .agent import AgentMessage
        with self._lock:
            self._messages.append(
                AgentMessage(
                    from_agent=from_agent,
                    content=content,
                    to_agent=to_agent,
                    metadata=metadata or {},
                )
            )

    def get_messages(
        self,
        to_agent: str | None = ...,   # type: ignore[assignment]
        from_agent: str | None = None,
    ) -> list[Any]:
        """Return messages matching the given filters.

        ``to_agent`` default (Ellipsis sentinel) returns messages addressed to
        that agent PLUS broadcasts.  Pass ``to_agent=None`` explicitly to get
        only broadcasts.
        """
        with self._lock:
            msgs = list(self._messages)

        result = []
        for m in msgs:
            if from_agent is not None and m.from_agent != from_agent:
                continue
            if to_agent is ...:
                # Ellipsis sentinel: caller didn't filter by recipient, return all
                result.append(m)
            elif to_agent is None:
                # Explicit None: only broadcasts
                if m.to_agent is None:
                    result.append(m)
            else:
                # Specific agent: addressed to them, or broadcast
                if m.to_agent is None or m.to_agent == to_agent:
                    result.append(m)
        return result

    @property
    def messages(self) -> list[Any]:
        """All messages in chronological order."""
        with self._lock:
            return list(self._messages)

    # ------------------------------------------------------------------ #
    # P2: snapshot / restore for checkpoint rollback                     #
    # ------------------------------------------------------------------ #

    def snapshot_before(self, stage_name: str) -> None:
        """Deep-copy nodes, user artifacts, and typed data before a stage runs.

        ``_internal`` (pipeline_builder bookkeeping) is NOT snapshotted — its values
        survive rollback unchanged, which is the correct behaviour for skip
        flags, loop round counters, etc.
        """
        with self._lock:
            self._snapshots[stage_name] = {
                "nodes": copy.deepcopy(self._nodes),
                "artifacts": copy.deepcopy(self.artifacts._data),
                "data": copy.deepcopy(self._data),
            }

    def restore_to(self, stage_name: str) -> None:
        """Restore nodes and user artifacts to the state before stage_name ran.

        Internal __pb_* artifact keys are preserved from the current state
        so the scheduler's own tracking (skip flags, loop rounds, etc.) is not
        disturbed by the rollback.
        """
        with self._lock:
            if stage_name not in self._snapshots:
                warnings.warn(
                    f"[baton] restore_to('{stage_name}'): no snapshot found — "
                    "stage may not have run yet, or snapshots were lost across a "
                    "session resume. Rollback skipped.",
                    stacklevel=3,
                )
                return
            snap = self._snapshots[stage_name]
            self._nodes = copy.deepcopy(snap["nodes"])
            # Restore user artifacts (pure user data — no __pb_* keys).
            self.artifacts._data = copy.deepcopy(snap["artifacts"])
            # _internal is NOT restored: skip flags, loop counters, etc.
            # must survive rollback so the scheduler stays coherent.

            # Restore typed state data.
            if "data" in snap:
                self._data = copy.deepcopy(snap["data"])

            # Remove stage_name and all later stages from completed list
            try:
                idx = self._completed_stages.index(stage_name)
                self._completed_stages = self._completed_stages[:idx]
            except ValueError:
                pass  # stage never completed — nothing to trim

    # ------------------------------------------------------------------ #
    # P1: serialization for disk persistence                             #
    # ------------------------------------------------------------------ #

    def to_dict(self, schemas: dict[str, type[BaseModel]]) -> dict:
        """Serialize resumable state.

        - Nodes: only levels with a schema entry are serialized.
        - Artifacts: pure user data — serialized as-is; non-JSON-serializable
          values are dropped with a warning.
        - Internal: pipeline_builder bookkeeping (loop counters, gc counters, fingerprint)
          serialized separately; transient skip flags are excluded.
        """
        with self._lock:
            nodes_data: dict[str, list] = {}
            for level, nodes in self._nodes.items():
                if level in schemas:
                    nodes_data[level] = [n.model_dump(mode="json") for n in nodes]

            artifacts_data: dict[str, Any] = {}
            for key, value in self.artifacts.all().items():
                try:
                    json.dumps(value)
                    artifacts_data[key] = value
                except (TypeError, ValueError):
                    warnings.warn(
                        f"[baton] artifact '{key}' is not JSON-serializable "
                        "and will not be persisted to disk.",
                        stacklevel=2,
                    )

            return {
                "session_id": self.session_id,
                "nodes": nodes_data,
                "completed_stages": list(self._completed_stages),
                "artifacts": artifacts_data,
                "internal": self._internal.to_persist(),
                "messages": [m.to_dict() for m in self._messages],
                "state_data": self._data.model_dump(mode="json"),
            }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        schemas: dict[str, type[BaseModel]],
        state_schema: type[BaseModel] | None = None,
    ) -> "State":
        """Reconstruct state from a persisted snapshot."""
        from .agent import AgentMessage
        state = cls(session_id=data["session_id"], state_schema=state_schema)
        for level, nodes_data in data.get("nodes", {}).items():
            schema = schemas.get(level)
            if schema:
                state._nodes[level] = [schema.model_validate(d) for d in nodes_data]
        state._completed_stages = data.get("completed_stages", [])
        for key, value in data.get("artifacts", {}).items():
            state.artifacts.set(key, value)
        # Restore internal bookkeeping (loop counters, gc counters, fingerprint…)
        for key, value in data.get("internal", {}).items():
            state._internal.set(key, value)
        # Backwards compatibility: old sessions stored __pb_* in artifacts
        for key in list(data.get("artifacts", {}).keys()):
            if key.startswith("__pb_"):
                state._internal.set(key, state.artifacts._data.pop(key))
        state._messages = [AgentMessage.from_dict(m) for m in data.get("messages", [])]
        # Restore typed state data if present in snapshot
        if "state_data" in data:
            schema = state_schema or _AnyState
            try:
                state._data = schema.model_validate(data["state_data"])
            except Exception:
                warnings.warn(
                    "[baton] state_data could not be restored from snapshot "
                    "(schema may have changed). Starting with fresh defaults.",
                    stacklevel=2,
                )
        return state
