"""Tests for async pipeline execution (run_async, async stages, DAGSpec.run_async)."""
import asyncio

import pytest
from pydantic import BaseModel

from pipeline_builder import MockAIAdapter, Pipeline, State
from pipeline_builder.core.dag import DAGNode, DAGSpec


# --- Schemas ---

class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ============================================================
# Async stage execution
# ============================================================

def test_async_stage_runs_in_run_async():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    ran = []

    @pipe.stage(reads=["goal"], writes=["task"])
    async def make(goal: Goal, state: State) -> list[Task]:
        ran.append(goal.name)
        return [Task(goal=goal.name, action="async")]

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert ran == ["g"]
    assert result.get_nodes("task")[0].action == "async"


def test_sync_stage_runs_in_run_async():
    """Sync stages are also valid in run_async (run via asyncio.to_thread)."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    ran = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        ran.append(goal.name)
        return [Task(goal=goal.name, action="sync")]

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert ran == ["g"]
    assert result.get_nodes("task")[0].action == "sync"


def test_mixed_sync_async_stages():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    order = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def sync_step(goal: Goal, state: State) -> list[Task]:
        order.append("sync")
        return [Task(goal=goal.name, action="sync")]

    @pipe.stage(reads=["task"], writes=["task"])
    async def async_step(task: Task, state: State) -> list[Task]:
        order.append("async")
        return [Task(goal=task.goal, action="async")]

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert order == ["sync", "async"]
    assert result.get_nodes("task")[0].action == "async"


def test_async_parallel_workers():
    """Async stages with workers=N use asyncio.gather for concurrency."""
    import time

    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], workers=4)
    async def slow(goal: Goal, state: State) -> list[Task]:
        await asyncio.sleep(0.05)
        return [Task(goal=goal.name, action=goal.name)]

    goals = [Goal(name=str(i)) for i in range(4)]
    t0 = time.monotonic()
    result = asyncio.run(pipe.run_async(goal=goals))
    elapsed = time.monotonic() - t0

    assert len(result.get_nodes("task")) == 4
    assert elapsed < 0.18  # parallel: ~50ms, not 4 × 50ms = 200ms


def test_async_parallel_results_preserve_order():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], workers=4)
    async def process(goal: Goal, state: State) -> list[Task]:
        await asyncio.sleep(0.01)
        return [Task(goal=goal.name, action=goal.name)]

    goals = [Goal(name=str(i)) for i in range(5)]
    result = asyncio.run(pipe.run_async(goal=goals))
    names = [t.action for t in result.get_nodes("task")]
    assert names == [str(i) for i in range(5)]


def test_run_async_with_initial_artifacts():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    seen = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    async def check(goal: Goal, state: State) -> list[Task]:
        seen["cfg"] = state.artifacts.get("cfg")
        return [Task(goal=goal.name, action="x")]

    asyncio.run(pipe.run_async(goal=[Goal(name="g")], artifacts={"cfg": "v1"}))
    assert seen["cfg"] == "v1"


# ============================================================
# Stage timeout
# ============================================================

def test_sync_stage_timeout_raises():
    """Sync stage that exceeds timeout raises TimeoutError.

    The sleep is intentionally short (0.2s > 0.05s timeout) so the background
    thread finishes quickly after the timeout fires.  Python threads cannot be
    forcibly killed, so asyncio.run() blocks until the thread exits when the
    stage uses asyncio.to_thread() internally.  Using a short sleep avoids
    adding 10+ seconds of cleanup time to the test suite.
    """
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], timeout=0.05)
    def hang(goal: Goal, state: State) -> list[Task]:
        import time
        time.sleep(0.2)   # longer than timeout but short enough to not block suite
        return [Task(goal=goal.name, action="x")]

    with pytest.raises(TimeoutError):
        pipe.run(goal=[Goal(name="g")])


def test_async_stage_timeout_raises():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], timeout=0.05)
    async def hang(goal: Goal, state: State) -> list[Task]:
        await asyncio.sleep(10)
        return [Task(goal=goal.name, action="x")]

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        asyncio.run(pipe.run_async(goal=[Goal(name="g")]))


def test_stage_that_finishes_before_timeout_succeeds():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], timeout=5.0)
    async def fast(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="ok")]

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert result.get_nodes("task")[0].action == "ok"


# ============================================================
# DAGSpec.run_async
# ============================================================

def test_dag_spec_run_async_basic():
    order = []

    dag = DAGSpec()
    dag.add(DAGNode("a", fn=lambda: order.append("a") or "result_a"))
    dag.add(DAGNode("b", fn=lambda: order.append("b") or "result_b", depends_on=["a"]))

    result = asyncio.run(dag.run_async())
    assert result.ok("a")
    assert result.ok("b")
    assert order.index("a") < order.index("b")


def test_dag_spec_run_async_parallel():
    """Independent nodes run concurrently."""
    import time

    async def slow_node() -> str:
        await asyncio.sleep(0.05)
        return "done"

    dag = DAGSpec()
    dag.add(DAGNode("x", fn=slow_node))
    dag.add(DAGNode("y", fn=slow_node))

    t0 = time.monotonic()
    result = asyncio.run(dag.run_async(workers=2))
    elapsed = time.monotonic() - t0

    assert result.ok("x") and result.ok("y")
    assert elapsed < 0.09


def test_dag_spec_run_async_error_propagates():
    def boom():
        raise ValueError("intentional")

    dag = DAGSpec()
    dag.add(DAGNode("bad", fn=boom))
    dag.add(DAGNode("dep", fn=lambda: "ok", depends_on=["bad"]))

    result = asyncio.run(dag.run_async())
    assert "bad" in result.failed()
    assert "dep" in result.skipped()


# ============================================================
# Event hooks with run_async
# ============================================================

def test_event_hooks_fire_in_run_async():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    events = []

    @pipe.on("stage_start")
    def on_start(name, state, **kw):
        events.append(("start", name))

    @pipe.on("stage_complete")
    def on_complete(name, state, duration_ms, **kw):
        events.append(("complete", name))

    @pipe.stage(reads=["goal"], writes=["task"])
    async def go(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert ("start", "go") in events
    assert ("complete", "go") in events
