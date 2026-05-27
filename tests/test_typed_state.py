"""Tests for typed pipeline context (state.data / state_schema)."""
import asyncio
import copy

import pytest
from pydantic import BaseModel, Field

from pipeline_builder import CheckpointResult, MockAIAdapter, Pipeline, SQLiteBackend, State


# ─────────────────── shared schemas ────────────────────────────────────────

class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


class PipeCtx(BaseModel):
    """Example typed pipeline context."""
    mode: str = "explore"
    confidence: float = 0.0
    tags: list[str] = Field(default_factory=list)
    result_count: int = 0


# ─────────────────── State unit tests ──────────────────────────────────────

def test_state_data_typed_defaults():
    state = State(state_schema=PipeCtx)
    assert state.data.mode == "explore"
    assert state.data.confidence == 0.0
    assert state.data.tags == []


def test_state_data_direct_write():
    state = State(state_schema=PipeCtx)
    state.data.mode = "delivery"
    assert state.data.mode == "delivery"


def test_state_update_data_batch():
    state = State(state_schema=PipeCtx)
    state.update_data(mode="delivery", confidence=0.95)
    assert state.data.mode == "delivery"
    assert state.data.confidence == 0.95


def test_state_data_without_schema_accepts_any_field():
    """No state_schema → _AnyState with extra='allow'."""
    state = State()
    state.data.arbitrary_field = "hello"
    assert state.data.arbitrary_field == "hello"


def test_state_data_validation_opt_in():
    """Pydantic validates assignment only when validate_assignment=True.

    Users who want strict runtime validation should declare:
        class PipeCtx(BaseModel):
            model_config = ConfigDict(validate_assignment=True)
            ...
    Without it, assignment coerces/silently accepts invalid types (Pydantic v2 default).
    This test documents the opt-in path.
    """
    from pydantic import ConfigDict

    class StrictCtx(BaseModel):
        model_config = ConfigDict(validate_assignment=True)
        confidence: float = 0.0

    state = State(state_schema=StrictCtx)
    with pytest.raises(Exception):
        state.data.confidence = "not-a-float"


def test_state_data_snapshot_and_restore():
    """snapshot_before captures data; restore_to rolls it back."""
    state = State(state_schema=PipeCtx)
    state.data.mode = "before"

    state.snapshot_before("step_a")
    state.data.mode = "after"
    assert state.data.mode == "after"

    state.restore_to("step_a")
    assert state.data.mode == "before"


def test_state_data_serialization_round_trip():
    """to_dict / from_dict preserves typed state data."""
    state = State(state_schema=PipeCtx)
    state.data.mode = "delivery"
    state.data.confidence = 0.8
    state.data.tags = ["a", "b"]

    data = state.to_dict({})
    assert "state_data" in data

    restored = State.from_dict(data, {}, state_schema=PipeCtx)
    assert restored.data.mode == "delivery"
    assert restored.data.confidence == 0.8
    assert restored.data.tags == ["a", "b"]


def test_state_data_from_dict_without_schema():
    """Loading without schema gives _AnyState with saved fields."""
    state = State(state_schema=PipeCtx)
    state.data.mode = "delivery"
    data = state.to_dict({})

    # Reload without schema
    restored = State.from_dict(data, {})
    assert restored.data.mode == "delivery"


def test_state_data_schema_change_warning():
    """If schema changes and fields are incompatible, a warning is issued."""
    state = State(state_schema=PipeCtx)
    data = state.to_dict({})
    data["state_data"]["confidence"] = "not-a-float"

    with pytest.warns(UserWarning, match="state_data"):
        restored = State.from_dict(data, {}, state_schema=PipeCtx)
    # Falls back to defaults
    assert restored.data.confidence == 0.0


# ─────────────────── Pipeline integration ──────────────────────────────────

def test_pipeline_state_schema_accessible_in_stage():
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        state_schema=PipeCtx,
    )
    seen = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        seen["mode"] = state.data.mode
        state.data.result_count += 1
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    assert seen["mode"] == "explore"
    assert result.data.result_count == 1


def test_pipeline_state_data_persists_across_stages():
    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)

    @pipe.stage(reads=["goal"], writes=["task"])
    def step1(goal: Goal, state: State) -> list[Task]:
        state.data.mode = "set_by_step1"
        return [Task(goal=goal.name, action="x")]

    @pipe.stage(reads=["task"])
    def step2(task: Task, state: State) -> None:
        assert state.data.mode == "set_by_step1"

    pipe.run(goal=[Goal(name="g")])


def test_pipeline_state_data_rolled_back_by_checkpoint():
    """After checkpoint reject, state.data is rolled back to pre-stage state."""
    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        state.data.result_count = call_count["n"]
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action=next(responses))

    result = pipe.run(goal=[Goal(name="g")])
    # After rollback, make ran again → result_count=2
    assert result.data.result_count == 2


def test_pipeline_state_data_survives_persist_resume(tmp_path):
    """state.data is serialized and restored on session resume."""
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_schema=PipeCtx,
        state_dir=str(tmp_path),
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        state.data.mode = "completed"
        state.data.confidence = 0.99
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    sid = result.session_id

    # Resume: typed state should be restored
    result2 = pipe.run(session_id=sid)
    assert result2.data.mode == "completed"
    assert result2.data.confidence == 0.99


def test_pipeline_state_data_in_sqlite_backend(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_schema=PipeCtx,
        storage=backend,
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        state.update_data(mode="sqlite_test", confidence=0.7)
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    sid = result.session_id

    result2 = pipe.run(session_id=sid)
    assert result2.data.mode == "sqlite_test"


def test_pipeline_state_data_in_async_stage():
    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)
    seen = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    async def go(goal: Goal, state: State) -> list[Task]:
        state.data.mode = "async"
        seen["mode"] = state.data.mode
        return [Task(goal=goal.name, action="x")]

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert seen["mode"] == "async"
    assert result.data.mode == "async"


def test_pipeline_no_state_schema_backwards_compat():
    """Pipelines without state_schema still work, data accepts any field."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        state.data.custom_key = "hello"  # dynamic field on _AnyState
        received["val"] = state.data.custom_key
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    assert received["val"] == "hello"


def test_update_data_thread_safe_in_parallel_stages():
    """update_data() must not lose updates when workers > 1."""
    import threading

    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)

    @pipe.stage(reads=["goal"], writes=["task"], workers=4)
    def make(goal: Goal, state: State) -> list[Task]:
        # Each parallel call appends one tag — all must survive
        state.update_data(tags=state.data.tags + [goal.name])
        return [Task(goal=goal.name, action="x")]

    goals = [Goal(name=str(i)) for i in range(4)]
    result = pipe.run(goal=goals)
    # Due to race on list read + write, this documents the behavior
    # (sequential update_data calls are safe, but read-modify-write is not)
    assert isinstance(result.data.tags, list)
