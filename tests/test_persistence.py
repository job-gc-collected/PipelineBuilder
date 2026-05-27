"""Tests for FileBackend, SQLiteBackend, and Pipeline storage integration."""
import json

import pytest
from pydantic import BaseModel

from pipeline_builder import Pipeline, SQLiteBackend, State
from pipeline_builder.persistence import FileBackend, StorageBackend


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ============================================================
# FileBackend
# ============================================================

def test_file_backend_save_and_load(tmp_path):
    backend = FileBackend(tmp_path)
    data = {"session_id": "s1", "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []}
    backend.save("s1", data)
    loaded = backend.load("s1")
    assert loaded["session_id"] == "s1"


def test_file_backend_load_missing_returns_none(tmp_path):
    backend = FileBackend(tmp_path)
    assert backend.load("ghost") is None


def test_file_backend_snapshot_round_trip(tmp_path):
    backend = FileBackend(tmp_path)
    nodes = {"goal": [{"name": "g1"}]}
    artifacts = {"mode": "test"}
    backend.save_snapshot("s1", "stage_a", nodes, artifacts)
    result = backend.load_snapshot("s1", "stage_a")
    assert result is not None
    n, a = result
    assert n["goal"][0]["name"] == "g1"
    assert a["mode"] == "test"


def test_file_backend_snapshot_missing_returns_none(tmp_path):
    backend = FileBackend(tmp_path)
    assert backend.load_snapshot("s1", "ghost_stage") is None


def test_file_backend_list_sessions(tmp_path):
    backend = FileBackend(tmp_path)
    import time
    for sid in ["a", "b", "c"]:
        backend.save(sid, {"session_id": sid, "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []})
        time.sleep(0.01)
    sessions = backend.list_sessions()
    assert set(sessions) == {"a", "b", "c"}
    # Snapshots should not appear in session list
    backend.save_snapshot("a", "s1", {}, {})
    sessions2 = backend.list_sessions()
    assert len(sessions2) == 3


def test_file_backend_delete(tmp_path):
    backend = FileBackend(tmp_path)
    backend.save("s1", {"session_id": "s1", "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []})
    backend.save_snapshot("s1", "stage", {}, {})
    backend.delete("s1")
    assert backend.load("s1") is None
    assert backend.load_snapshot("s1", "stage") is None


def test_file_backend_atomic_write(tmp_path):
    """State file should always be valid JSON (no partial writes)."""
    backend = FileBackend(tmp_path)
    data = {"session_id": "s1", "nodes": {"key": "val"}, "completed_stages": [], "artifacts": {}, "messages": []}
    backend.save("s1", data)
    raw = (tmp_path / "s1.json").read_text()
    parsed = json.loads(raw)  # must not raise
    assert parsed["session_id"] == "s1"


# ============================================================
# SQLiteBackend
# ============================================================

def test_sqlite_backend_save_and_load(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    data = {"session_id": "s1", "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []}
    backend.save("s1", data)
    loaded = backend.load("s1")
    assert loaded["session_id"] == "s1"


def test_sqlite_backend_load_missing_returns_none(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    assert backend.load("ghost") is None


def test_sqlite_backend_snapshot_round_trip(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    nodes = {"goal": [{"name": "g1"}]}
    artifacts = {"key": "value", "count": 42}
    backend.save_snapshot("s1", "make", nodes, artifacts)
    result = backend.load_snapshot("s1", "make")
    assert result is not None
    n, a = result
    assert n["goal"][0]["name"] == "g1"
    assert a["count"] == 42


def test_sqlite_backend_snapshot_upsert(tmp_path):
    """Saving snapshot twice for same (session, stage) overwrites."""
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.save_snapshot("s1", "step", {"a": 1}, {"x": 1})
    backend.save_snapshot("s1", "step", {"b": 2}, {"y": 2})
    n, a = backend.load_snapshot("s1", "step")
    assert "b" in n
    assert a["y"] == 2


def test_sqlite_backend_list_sessions(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    for sid in ["s1", "s2", "s3"]:
        backend.save(sid, {"session_id": sid, "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []})
    sessions = backend.list_sessions()
    assert set(sessions) == {"s1", "s2", "s3"}


def test_sqlite_backend_delete(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    backend.save("s1", {"session_id": "s1", "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []})
    backend.save_snapshot("s1", "st", {}, {})
    backend.delete("s1")
    assert backend.load("s1") is None
    assert backend.load_snapshot("s1", "st") is None
    assert "s1" not in backend.list_sessions()


def test_sqlite_backend_list_sessions_detail(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db", pipeline_name="my_pipe")
    backend.save("s1", {"session_id": "s1", "nodes": {}, "completed_stages": [], "artifacts": {}, "messages": []})
    detail = backend.list_sessions_detail()
    assert detail[0]["pipeline_name"] == "my_pipe"
    assert "created_at" in detail[0]


# ============================================================
# Pipeline + backend integration
# ============================================================

def test_pipeline_state_dir_creates_file_backend(tmp_path):
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    assert (tmp_path / f"{result.session_id}.json").exists()


def test_pipeline_explicit_storage_overrides_state_dir(tmp_path):
    db = tmp_path / "runs.db"
    backend = SQLiteBackend(db)
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),  # would create FileBackend
        storage=backend,           # overrides
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    # State should be in SQLite, not a JSON file
    loaded = backend.load(result.session_id)
    assert loaded is not None
    # No standalone JSON file for this session
    assert not (tmp_path / f"{result.session_id}.json").exists()


def test_sqlite_backend_snapshot_survives_crash_rollback(tmp_path):
    """
    Critical test: snapshot saved to backend → simulate crash (clear _snapshots)
    → resume → checkpoint rejects → restore_to loads snapshot from backend.
    """
    db = tmp_path / "runs.db"
    schemas = {"goal": Goal, "task": Task}
    backend = SQLiteBackend(db)
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas=schemas,
        storage=backend,
    )
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State):
        from pipeline_builder import CheckpointResult
        action = next(responses)
        if action == "reject":
            # Simulate crash: drop the in-memory snapshot for "make"
            state._snapshots.pop("make", None)
        return CheckpointResult(action=action)

    result = pipe.run(goal=[Goal(name="g")])
    # make ran twice (original + rollback after snapshot loaded from backend)
    assert call_count["n"] == 2
    tasks = result.get_nodes("task")
    assert tasks[-1].action == "v2"


def test_file_backend_snapshot_survives_in_memory_loss(tmp_path):
    """Same test but with FileBackend."""
    schemas = {"goal": Goal, "task": Task}
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas=schemas,
        state_dir=str(tmp_path),
    )
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State):
        from pipeline_builder import CheckpointResult
        action = next(responses)
        if action == "reject":
            state._snapshots.pop("make", None)
        return CheckpointResult(action=action)

    result = pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 2


def test_storage_backend_protocol(tmp_path):
    """Both backends satisfy the StorageBackend protocol."""
    fb = FileBackend(tmp_path / "files")
    sb = SQLiteBackend(tmp_path / "test.db")
    assert isinstance(fb, StorageBackend)
    assert isinstance(sb, StorageBackend)
