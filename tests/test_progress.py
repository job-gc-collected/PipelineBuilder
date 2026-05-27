"""Tests for progress heartbeat: stage_progress events and run_structured_streaming."""
import asyncio
import time

import pytest
from pydantic import BaseModel

from pipeline_builder import MockAIAdapter, Pipeline, State
from pipeline_builder.ai.mock import MockAIAdapter as _Mock


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


class Result(BaseModel):
    value: str


# ─────────────────────────── Stage-level heartbeat ─────────────────────────

def test_stage_progress_event_fires_during_long_stage():
    """stage_progress is emitted while the stage is running."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    progress_events: list[dict] = []

    @pipe.on("stage_progress")
    def on_progress(name, state, elapsed_s, **kw):
        progress_events.append({"name": name, "elapsed_s": elapsed_s})

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.05)
    def slow(goal: Goal, state: State) -> list[Task]:
        time.sleep(0.15)   # long enough for 2-3 heartbeats at 0.05s interval
        return [Task(goal=goal.name, action="done")]

    pipe.run(goal=[Goal(name="g")])
    # At least one heartbeat should have fired
    assert len(progress_events) >= 1
    assert progress_events[0]["name"] == "slow"
    assert progress_events[0]["elapsed_s"] > 0


def test_stage_progress_not_fired_without_interval():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    events = []

    @pipe.on("stage_progress")
    def on_prog(name, state, elapsed_s, **kw):
        events.append(1)

    @pipe.stage(reads=["goal"], writes=["task"])  # no progress_interval
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    assert events == []


def test_stage_progress_event_has_increasing_elapsed():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    elapsed_values: list[float] = []

    @pipe.on("stage_progress")
    def capture(name, state, elapsed_s, **kw):
        elapsed_values.append(elapsed_s)

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.05)
    def slow(goal: Goal, state: State) -> list[Task]:
        time.sleep(0.2)
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    if len(elapsed_values) >= 2:
        assert elapsed_values[-1] > elapsed_values[0]


def test_stage_progress_stops_after_stage_completes():
    """No more progress events after the stage finishes."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    events_after_complete: list[float] = []
    completed_at: list[float] = []

    @pipe.on("stage_complete")
    def on_done(name, state, duration_ms, **kw):
        completed_at.append(time.monotonic())

    @pipe.on("stage_progress")
    def on_prog(name, state, elapsed_s, **kw):
        if completed_at:
            events_after_complete.append(time.monotonic() - completed_at[0])

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.02)
    def fast(goal: Goal, state: State) -> list[Task]:
        time.sleep(0.05)
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    # Allow one timer tick after completion (daemon threads may fire once more)
    # but within 1 progress_interval, not indefinitely
    assert not any(t > 0.1 for t in events_after_complete)


def test_stage_progress_event_in_async_stage():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    progress_fired = []

    @pipe.on("stage_progress")
    def on_prog(name, state, elapsed_s, **kw):
        progress_fired.append(name)

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.05)
    async def slow_async(goal: Goal, state: State) -> list[Task]:
        await asyncio.sleep(0.15)
        return [Task(goal=goal.name, action="done")]

    asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert len(progress_fired) >= 1
    assert progress_fired[0] == "slow_async"


def test_stage_progress_on_stage_failure():
    """Progress timer is cancelled even when stage fails."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    prog_after_fail = []
    failed_at = [None]

    @pipe.on("stage_fail")
    def on_fail(name, state, error, **kw):
        failed_at[0] = time.monotonic()

    @pipe.on("stage_progress")
    def on_prog(name, state, elapsed_s, **kw):
        if failed_at[0] is not None:
            prog_after_fail.append(time.monotonic() - failed_at[0])

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.03)
    def boom(goal: Goal, state: State) -> list[Task]:
        time.sleep(0.06)
        raise ValueError("intentional")

    with pytest.raises(ValueError):
        pipe.run(goal=[Goal(name="g")])

    time.sleep(0.05)  # let any lingering timer fire
    # No significant events after failure
    assert not any(t > 0.08 for t in prog_after_fail)


# ─────────────────────────── AI-level streaming progress ───────────────────

def test_run_structured_streaming_returns_correct_result():
    import json
    payload = json.dumps({"value": "hello"})
    ai = _Mock(handler=lambda p, c: payload)

    async def run():
        return await ai.run_structured_streaming("prompt", Result)

    result = asyncio.run(run())
    assert result.value == "hello"


def test_run_structured_streaming_calls_on_chunk():
    import json
    chunks: list[tuple] = []
    payload = json.dumps({"value": "world"})
    ai = _Mock(handler=lambda p, c: payload)

    def on_chunk(n, partial):
        chunks.append((n, partial))

    async def run():
        return await ai.run_structured_streaming("prompt", Result, on_chunk=on_chunk)

    result = asyncio.run(run())
    assert result.value == "world"
    assert len(chunks) == 1
    assert chunks[0][0] == 1


def test_run_structured_streaming_in_stage_with_progress():
    """Stage uses run_structured_streaming + progress_interval together."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    chunks_received = []
    progress_events = []

    @pipe.on("stage_progress")
    def on_prog(name, state, elapsed_s, **kw):
        progress_events.append(elapsed_s)

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=0.05)
    async def analyze(goal: Goal, state: State, ai) -> list[Task]:
        def on_chunk(n, partial):
            chunks_received.append(n)

        result = await ai.run_structured_streaming(
            f"analyze {goal.name}", Result, on_chunk=on_chunk
        )
        await asyncio.sleep(0.12)   # simulate post-processing that triggers heartbeat
        return [Task(goal=goal.name, action=result.value)]

    import json
    ai = _Mock(handler=lambda p, c: json.dumps({"value": "analyzed"}))
    result = asyncio.run(pipe.run_async(ai=ai, goal=[Goal(name="g")]))

    assert result.get_nodes("task")[0].action == "analyzed"
    assert len(chunks_received) >= 1
    # Progress heartbeat should have fired at least once during the sleep
    assert len(progress_events) >= 1


# ─────────────────────────── Stage progress event registration ─────────────

def test_stage_progress_is_valid_event():
    pipe = Pipeline("test", hierarchy=["goal"])
    # Should not raise
    pipe.on("stage_progress", lambda **kw: None)


def test_stage_progress_interval_param_accepted():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], progress_interval=30.0)
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    # Just verify the attribute is stored
    assert pipe._steps[0]._baton_progress_interval == 30.0
