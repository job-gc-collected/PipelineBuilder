"""Tests for @pipe.goal_check — periodic AI self-correction."""
import asyncio

import pytest
from pydantic import BaseModel

from pipeline_builder import GoalCheckResult, MockAIAdapter, Pipeline, State


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


class PipeCtx(BaseModel):
    goal_text: str = "default goal"
    confidence: float = 1.0
    extra: str = ""


# ─────────────────── basic firing ──────────────────────────────────────────

def test_goal_check_fires_after_interval():
    """Goal check fires once after `interval` stages complete."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def s1(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="s1")]

    @pipe.stage(reads=["task"], writes=["task"])
    def s2(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="s2")]

    @pipe.stage(reads=["task"], writes=["task"])
    def s3(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="s3")]

    @pipe.goal_check(interval=3)
    def check(state: State) -> GoalCheckResult:
        fired.append(1)
        return GoalCheckResult(verdict="continue", note="ok")

    pipe.run(goal=[Goal(name="g")])
    assert len(fired) == 1


def test_goal_check_does_not_fire_before_interval():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def s1(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=5)   # pipeline only has 1 stage
    def check(state: State) -> GoalCheckResult:
        fired.append(1)
        return GoalCheckResult(verdict="continue")

    pipe.run(goal=[Goal(name="g")])
    assert len(fired) == 0


def test_goal_check_fires_multiple_times():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    for i in range(1, 7):
        @pipe.stage(reads=["goal"] if i == 1 else ["task"], writes=["task"])
        def stage_fn(x, state: State) -> list[Task]:
            return [Task(goal=getattr(x, "name", getattr(x, "goal", "")), action="x")]
        stage_fn.__name__ = f"s{i}"
        stage_fn._baton_reads = ["goal" if i == 1 else "task"]

    # Simpler: just make 6 real stages
    pipe2 = Pipeline("test", hierarchy=["goal", "task"])
    fired2 = []

    @pipe2.stage(reads=["goal"], writes=["task"])
    def a1(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def a2(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def a3(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def a4(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def a5(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def a6(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="x")]

    @pipe2.goal_check(interval=2, max_checks=3)
    def gc(state: State) -> GoalCheckResult:
        fired2.append(1)
        return GoalCheckResult(verdict="continue")

    pipe2.run(goal=[Goal(name="g")])
    # 6 stages / interval=2 = 3 fires (limited by max_checks=3)
    assert len(fired2) == 3


def test_goal_check_respects_max_checks():
    """After max_checks, the goal check stops firing."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    stages = ["s1", "s2", "s3", "s4", "s5", "s6"]
    prev_write = "goal"
    for name in stages:
        reads = [prev_write]

        def make_stage(n, r, w):
            @pipe.stage(reads=[r], writes=[w])
            def fn(x, state: State) -> list[Task]:
                return [Task(goal=getattr(x, "name", getattr(x, "goal", "")), action=n)]
            fn.__name__ = n
            return fn
        make_stage(name, prev_write, "task")
        prev_write = "task"

    # Rebuild cleanly
    pipe3 = Pipeline("test", hierarchy=["goal", "task"])
    fired3 = []

    @pipe3.stage(reads=["goal"], writes=["task"])
    def b1(g: Goal, state: State) -> list[Task]: return [Task(goal=g.name, action="x")]
    @pipe3.stage(reads=["task"], writes=["task"])
    def b2(t: Task, state: State) -> list[Task]: return [Task(goal=t.goal, action="x")]
    @pipe3.stage(reads=["task"], writes=["task"])
    def b3(t: Task, state: State) -> list[Task]: return [Task(goal=t.goal, action="x")]
    @pipe3.stage(reads=["task"], writes=["task"])
    def b4(t: Task, state: State) -> list[Task]: return [Task(goal=t.goal, action="x")]
    @pipe3.stage(reads=["task"], writes=["task"])
    def b5(t: Task, state: State) -> list[Task]: return [Task(goal=t.goal, action="x")]
    @pipe3.stage(reads=["task"], writes=["task"])
    def b6(t: Task, state: State) -> list[Task]: return [Task(goal=t.goal, action="x")]

    @pipe3.goal_check(interval=1, max_checks=2)
    def gc3(state: State) -> GoalCheckResult:
        fired3.append(1)
        return GoalCheckResult(verdict="continue")

    pipe3.run(goal=[Goal(name="g")])
    assert len(fired3) == 2  # capped at max_checks=2


# ─────────────────── adjust verdict ────────────────────────────────────────

def test_goal_check_adjust_updates_state_data():
    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)
    updates_applied = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1)
    def check(state: State) -> GoalCheckResult:
        return GoalCheckResult(
            verdict="adjust",
            note="correcting confidence",
            data_updates={"confidence": 0.5, "extra": "adjusted"},
        )

    result = pipe.run(goal=[Goal(name="g")])
    assert result.data.confidence == 0.5
    assert result.data.extra == "adjusted"


# ─────────────────── rollback verdict ──────────────────────────────────────

def test_goal_check_rollback_reruns_stages():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"make": 0, "refine": 0, "check": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["make"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['make']}")]

    @pipe.stage(reads=["task"], writes=["task"])
    def refine(task: Task, state: State) -> list[Task]:
        call_count["refine"] += 1
        return [Task(goal=task.goal, action=task.action + "_refined")]

    # Check fires after both stages; first time rollback to make, second time continue
    check_calls = [0]

    @pipe.goal_check(interval=2, rollback_to="make")
    def check(state: State) -> GoalCheckResult:
        call_count["check"] += 1
        check_calls[0] += 1
        if check_calls[0] == 1:
            return GoalCheckResult(verdict="rollback", note="need to redo")
        return GoalCheckResult(verdict="continue", note="ok now")

    result = pipe.run(goal=[Goal(name="g")])
    # make ran twice (original + after rollback)
    assert call_count["make"] == 2
    assert call_count["check"] == 2
    tasks = result.get_nodes("task")
    assert tasks[0].action == "v2_refined"


def test_goal_check_rollback_to_override_per_result():
    """result.rollback_to overrides the decorator's rollback_to."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def step_a(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"a{call_count['n']}")]

    @pipe.stage(reads=["task"], writes=["task"])
    def step_b(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action=task.action + "b")]

    checks = [0]

    @pipe.goal_check(interval=2, rollback_to="step_b")  # default: rollback to step_b
    def gc(state: State) -> GoalCheckResult:
        checks[0] += 1
        if checks[0] == 1:
            # Override: rollback further back to step_a
            return GoalCheckResult(verdict="rollback", rollback_to="step_a")
        return GoalCheckResult(verdict="continue")

    result = pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 2  # step_a ran twice


# ─────────────────── AI injection ──────────────────────────────────────────

def test_goal_check_receives_ai_when_declared():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received_ai = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1)
    def check(state: State, ai) -> GoalCheckResult:
        received_ai.append(type(ai).__name__)
        return GoalCheckResult(verdict="continue")

    pipe.run(ai=MockAIAdapter(), goal=[Goal(name="g")])
    assert received_ai == ["MockAIAdapter"]


def test_goal_check_no_ai_param_works_without_ai():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1)
    def check(state: State) -> GoalCheckResult:
        fired.append(1)
        return GoalCheckResult(verdict="continue")

    pipe.run(goal=[Goal(name="g")])  # no ai= passed
    assert len(fired) == 1


# ─────────────────── async support ─────────────────────────────────────────

def test_goal_check_fires_in_run_async():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.stage(reads=["goal"], writes=["task"])
    async def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1)
    async def check(state: State) -> GoalCheckResult:
        fired.append(1)
        return GoalCheckResult(verdict="continue")

    asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert len(fired) == 1


# ─────────────────── validation ─────────────────────────────────────────────

def test_goal_check_rollback_to_validation():
    """rollback_to pointing to nonexistent stage raises at run time."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1, rollback_to="ghost_stage")
    def gc(state: State) -> GoalCheckResult:
        return GoalCheckResult(verdict="continue")

    with pytest.raises(ValueError, match="ghost_stage"):
        pipe.run(goal=[Goal(name="g")])


def test_goal_check_interval_zero_raises():
    from pipeline_builder.core.goal_check import goal_check
    with pytest.raises(ValueError, match="interval"):
        @goal_check(interval=0)
        def bad(state): ...


# ─────────────────── event hook ────────────────────────────────────────────

def test_goal_check_emits_event():
    pipe = Pipeline("test", hierarchy=["goal", "task"], state_schema=PipeCtx)
    events = []

    @pipe.on("goal_check")
    def handler(name, state, verdict, note, **kw):
        events.append((name, verdict, note))

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.goal_check(interval=1)
    def gc(state: State) -> GoalCheckResult:
        return GoalCheckResult(verdict="adjust", note="tweaking confidence",
                               data_updates={"confidence": 0.9})

    pipe.run(goal=[Goal(name="g")])
    assert ("gc", "adjust", "tweaking confidence") in events
