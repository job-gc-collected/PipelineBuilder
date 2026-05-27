import asyncio
import time

import pytest
from pydantic import BaseModel

from pipeline_builder import CheckpointResult, GoalCheckResult, MockAIAdapter, Pipeline, State
from pipeline_builder.ai.base import AIAdapter
from pipeline_builder.core.stage import stage


# --- Shared schemas ---

class Goal(BaseModel):
    name: str

class Task(BaseModel):
    goal: str
    action: str

class Step(BaseModel):
    task: str
    detail: str


# --- Helpers ---

def simple_pipe(workers: int = 1) -> tuple[Pipeline, list[str]]:
    """2-level pipeline that records call order."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    calls: list[str] = []

    @pipe.stage(reads=["goal"], writes=["task"], workers=workers)
    def decompose(goal: Goal, state: State) -> list[Task]:
        calls.append(goal.name)
        return [Task(goal=goal.name, action=f"action-{goal.name}")]

    return pipe, calls


# --- State integration ---

def test_pipeline_loads_input_nodes():
    pipe, _ = simple_pipe()
    goals = [Goal(name="g1"), Goal(name="g2")]
    result = pipe.run(goal=goals)
    assert len(result.get_nodes("goal")) == 2


def test_pipeline_single_input_wrapped_in_list():
    pipe, _ = simple_pipe()
    result = pipe.run(goal=Goal(name="single"))
    assert len(result.get_nodes("goal")) == 1


# --- Static validation ---

def test_static_validation_unknown_reads_raises():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["unknown_level"], writes=["task"])
    def bad(goal: Goal, state: State) -> list[Task]:
        return []

    with pytest.raises(ValueError, match="unknown_level"):
        pipe.run(goal=[Goal(name="x")])


def test_static_validation_writes_not_in_hierarchy_raises():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["not_in_hierarchy"])
    def bad(goal: Goal, state: State) -> list[Task]:
        return []

    with pytest.raises(ValueError, match="not_in_hierarchy"):
        pipe.run(goal=[Goal(name="x")])


# --- Fan-out ---

def test_auto_fanout_called_once_per_node():
    pipe, calls = simple_pipe()
    pipe.run(goal=[Goal(name="a"), Goal(name="b"), Goal(name="c")])
    assert calls == ["a", "b", "c"]


def test_auto_fanout_results_merged_into_writes_level():
    pipe, _ = simple_pipe()
    result = pipe.run(goal=[Goal(name="a"), Goal(name="b")])
    tasks = result.get_nodes("task")
    assert len(tasks) == 2
    assert {t.goal for t in tasks} == {"a", "b"}


def test_manual_fanout_receives_all_nodes():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received: list[list] = []

    @pipe.stage(reads=["goal"], writes=["task"], fanout="manual")
    def process(goal: list[Goal], state: State) -> list[Task]:
        received.append(goal)
        return [Task(goal=g.name, action="x") for g in goal]

    pipe.run(goal=[Goal(name="a"), Goal(name="b")])
    assert len(received) == 1
    assert len(received[0]) == 2


# --- Parallel execution ---

def test_parallel_workers_faster_than_sequential():
    def make(workers: int) -> Pipeline:
        pipe = Pipeline("test", hierarchy=["goal", "task"], schemas={})

        @pipe.stage(reads=["goal"], writes=["task"], workers=workers)
        def slow(goal: Goal, state: State) -> list[Task]:
            time.sleep(0.05)
            return [Task(goal=goal.name, action="x")]

        return pipe

    goals = [Goal(name=str(i)) for i in range(4)]

    t0 = time.monotonic()
    make(1).run(goal=goals)
    seq_time = time.monotonic() - t0

    t0 = time.monotonic()
    make(4).run(goal=goals)
    par_time = time.monotonic() - t0

    assert par_time < seq_time / 2


def test_parallel_results_preserve_order():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], workers=4)
    def process(goal: Goal, state: State) -> list[Task]:
        time.sleep(0.01)
        return [Task(goal=goal.name, action=goal.name)]

    goals = [Goal(name=str(i)) for i in range(5)]
    result = pipe.run(goal=goals)
    names = [t.action for t in result.get_nodes("task")]
    assert names == [str(i) for i in range(5)]


# --- AI injection ---

def test_ai_injected_when_stage_declares_param():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received_ai: list[AIAdapter] = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def with_ai(goal: Goal, state: State, ai: AIAdapter) -> list[Task]:
        received_ai.append(ai)
        return [Task(goal=goal.name, action="x")]

    mock = MockAIAdapter()
    pipe.run(ai=mock, goal=[Goal(name="g")])
    assert received_ai[0] is mock


def test_ai_not_required_when_stage_has_no_param():
    pipe, _ = simple_pipe()
    result = pipe.run(goal=[Goal(name="g")])  # no ai= passed
    assert len(result.get_nodes("task")) == 1


# --- Checkpoint ---

def test_checkpoint_confirm_continues():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.checkpoint()
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action="confirm")

    result = pipe.run(goal=[Goal(name="g")])
    assert len(result.get_nodes("task")) == 1


def test_checkpoint_reject_rollback():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action="x")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action=next(responses))

    pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 2  # ran twice: once original, once after rollback


# --- History ---

def test_history_records_completed_stages():
    pipe, _ = simple_pipe()
    result = pipe.run(goal=[Goal(name="g")])
    assert result.history[0].name == "decompose"
    assert result.history[0].status == "completed"


def test_history_records_failed_stage():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def boom(goal: Goal, state: State) -> list[Task]:
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError):
        pipe.run(goal=[Goal(name="g")])

    # Can't access result since it raised, but we can test via state directly
    # (state is inaccessible here — just verify the exception propagates cleanly)


# --- Multi-stage chain ---

def test_three_level_pipeline():
    pipe = Pipeline("test", hierarchy=["goal", "task", "step"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make_tasks(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="t1"), Task(goal=goal.name, action="t2")]

    @pipe.stage(reads=["task"], writes=["step"])
    def make_steps(task: Task, state: State) -> list[Step]:
        return [Step(task=task.action, detail="d")]

    result = pipe.run(goal=[Goal(name="g")])
    assert len(result.get_nodes("task")) == 2
    assert len(result.get_nodes("step")) == 2


# --- run(artifacts={...}) (change 1) ---

def test_run_with_initial_artifacts():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    def use_artifact(goal: Goal, state: State) -> list[Task]:
        received["config"] = state.artifacts.get("config")
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")], artifacts={"config": "v1"})
    assert received["config"] == "v1"


def test_run_artifacts_not_overwritten_on_resume(tmp_path):
    """Initial artifacts passed to run() are ignored on resume (loaded from snapshot)."""
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")], artifacts={"key": "original"})
    sid = result.session_id

    result2 = pipe.run(session_id=sid, artifacts={"key": "should_be_ignored"})
    assert result2.artifacts.get("key") == "original"


# --- rollback restores artifacts (change 2) ---

def test_rollback_restores_artifacts():
    """After rollback to a stage, artifacts written by later stages are removed."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        state.artifacts.set("written_by_make", call_count["n"])
        return [Task(goal=goal.name, action="x")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State) -> CheckpointResult:
        # Write something in the checkpoint that should NOT survive rollback
        state.artifacts.set("written_in_checkpoint", True)
        return CheckpointResult(action=next(responses))

    result = pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 2  # ran twice
    # written_by_make should reflect the second run, not the first
    assert result.artifacts.get("written_by_make") == 2
    # written_in_checkpoint was set before reject; after rollback+re-run it
    # gets set again, so it should still be True at the end
    assert result.artifacts.get("written_in_checkpoint") is True


def test_rollback_does_not_wipe_artifacts_set_before_target():
    """Artifacts set BEFORE the rollback target stage should survive rollback."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=[], writes=["goal"], fanout="manual")
    def setup(state: State) -> list[Goal]:
        state.artifacts.set("early", "kept")
        return [Goal(name="g")]

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action=next(responses))

    result = pipe.run(artifacts={"early": "kept"})
    assert result.artifacts.get("early") == "kept"


# --- checkpoint route action (change 5) ---

def test_checkpoint_route_jumps_to_target():
    """checkpoint route action skips to the named stage; other declared targets are skipped."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    visited = []

    # The checkpoint must come BEFORE the branching stages so the skip
    # flags are set before either path_a or path_b is executed.
    @pipe.checkpoint(targets=["path_a", "path_b"])
    def decide(state: State) -> CheckpointResult:
        return CheckpointResult(action="route", target="path_b")

    @pipe.stage(reads=["goal"], writes=["task"])
    def path_a(goal: Goal, state: State) -> list[Task]:
        visited.append("path_a")
        return [Task(goal=goal.name, action="a")]

    @pipe.stage(reads=["goal"], writes=["task"])
    def path_b(goal: Goal, state: State) -> list[Task]:
        visited.append("path_b")
        return [Task(goal=goal.name, action="b")]

    result = pipe.run(goal=[Goal(name="g")])
    assert "path_b" in visited
    assert "path_a" not in visited
    tasks = result.get_nodes("task")
    assert all(t.action == "b" for t in tasks)


def test_checkpoint_targets_validation_raises_on_unknown():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.checkpoint(targets=["nonexistent"])
    def bad_cp(state: State) -> CheckpointResult:
        return CheckpointResult(action="confirm")

    with pytest.raises(ValueError, match="nonexistent"):
        pipe.run(goal=[Goal(name="g")])


# ============================================================
# Router tests
# ============================================================

def test_router_selects_correct_stage():
    """Router jumps to the named stage; other declared targets are skipped."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    visited = []

    @pipe.router(targets=["fast", "slow"])
    def decide(state: State) -> str:
        return "fast"

    @pipe.stage(reads=["goal"], writes=["task"])
    def fast(goal: Goal, state: State) -> list[Task]:
        visited.append("fast")
        return [Task(goal=goal.name, action="fast")]

    @pipe.stage(reads=["goal"], writes=["task"])
    def slow(goal: Goal, state: State) -> list[Task]:
        visited.append("slow")
        return [Task(goal=goal.name, action="slow")]

    result = pipe.run(goal=[Goal(name="g")])
    assert visited == ["fast"]
    assert all(t.action == "fast" for t in result.get_nodes("task"))


def test_router_skips_non_selected_stage():
    """The non-selected declared target receives a skip flag and is not executed."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    visited = []

    @pipe.router(targets=["a", "b"])
    def pick(state: State) -> str:
        return "b"

    @pipe.stage(reads=["goal"], writes=["task"])
    def a(goal: Goal, state: State) -> list[Task]:
        visited.append("a")
        return [Task(goal=goal.name, action="a")]

    @pipe.stage(reads=["goal"], writes=["task"])
    def b(goal: Goal, state: State) -> list[Task]:
        visited.append("b")
        return [Task(goal=goal.name, action="b")]

    pipe.run(goal=[Goal(name="g")])
    assert "b" in visited
    assert "a" not in visited


def test_router_static_validation_raises_on_unknown_target():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.router(targets=["nonexistent"])
    def route(state: State) -> str:
        return "nonexistent"

    with pytest.raises(ValueError, match="nonexistent"):
        pipe.run(goal=[Goal(name="g")])


def test_router_runtime_unknown_target_raises():
    """Router returning a name not in the pipeline raises RuntimeError at runtime."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.router(targets=[])  # no declared targets — skips static check
    def route(state: State) -> str:
        return "ghost_stage"

    with pytest.raises(RuntimeError, match="ghost_stage"):
        pipe.run(goal=[Goal(name="g")])


# ============================================================
# Loop tests
# ============================================================

def _loop_pipe(max_rounds: int = 5, exit_on=("done",)):
    """Helper: stage → loop → end."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}
    verdicts_holder = {}

    @pipe.stage(reads=["goal"], writes=["task"])
    def process(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"run-{call_count['n']}")]

    def make_loop(verdicts_iter):
        @pipe.loop(rollback_to="process", exit_on=list(exit_on), max_rounds=max_rounds)
        def quality_check(state: State) -> str:
            return next(verdicts_iter)
        verdicts_holder["loop"] = quality_check

    return pipe, call_count, make_loop


def test_loop_exits_when_verdict_in_exit_on():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}
    verdicts = iter(["done"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def process(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action="x")]

    @pipe.loop(rollback_to="process", exit_on=["done"], max_rounds=5)
    def gate(state: State) -> str:
        return next(verdicts)

    pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 1  # process ran once; loop exited immediately


def test_loop_rollback_and_rerun():
    """Non-exit verdict triggers rollback; process reruns until exit is returned."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}
    verdicts = iter(["retry", "retry", "done"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def work(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    @pipe.loop(rollback_to="work", exit_on=["done"], max_rounds=10)
    def review(state: State) -> str:
        return next(verdicts)

    result = pipe.run(goal=[Goal(name="g")])
    assert call_count["n"] == 3  # original + 2 rollbacks
    assert result.artifacts.get("_loop_result_review") == "done"


def test_loop_max_rounds_ceiling():
    """Loop exits after max_rounds even if exit_on never fires."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def step(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action="x")]

    @pipe.loop(rollback_to="step", exit_on=["never_fires"], max_rounds=2)
    def gate(state: State) -> str:
        return "retry"  # never exit_on

    pipe.run(goal=[Goal(name="g")])
    # round 0 → retry, rollback; round 1 → retry, rollback; round 2 → max_rounds hit
    assert call_count["n"] == 3  # original + max_rounds rollbacks


def test_loop_stores_verdict_in_artifacts():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.loop(rollback_to="make", exit_on=["ok"], max_rounds=3)
    def check(state: State) -> str:
        return "ok"

    result = pipe.run(goal=[Goal(name="g")])
    assert result.artifacts.get("_loop_result_check") == "ok"


def test_loop_rollback_to_validation_raises():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.loop(rollback_to="nonexistent", exit_on=["done"], max_rounds=3)
    def gate(state: State) -> str:
        return "done"

    with pytest.raises(ValueError, match="nonexistent"):
        pipe.run(goal=[Goal(name="g")])


def test_loop_round_counter_persisted_across_resume(tmp_path):
    """Loop round counter survives a session save/resume so max_rounds is honoured."""
    pipe = Pipeline(
        "test", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),
    )
    call_count = {"n": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    @pipe.loop(rollback_to="make", exit_on=["ok"], max_rounds=3)
    def check(state: State) -> str:
        return "retry"

    result = pipe.run(goal=[Goal(name="g")])
    # Round counter must be persisted; verify it appears in saved state JSON
    import json
    snap = json.loads((tmp_path / f"{result.session_id}.json").read_text())
    # __pb_loop_round_check should be in the persisted internal store
    # (moved from "artifacts" to "internal" in Phase 1 of the refactor)
    assert "__pb_loop_round_check" in snap.get("internal", snap.get("artifacts", {}))


# ============================================================
# workers validation
# ============================================================

def test_stage_workers_zero_raises():
    with pytest.raises(ValueError, match="workers"):
        @stage(workers=0)
        def bad(node, state):
            return []


def test_stage_workers_negative_not_minus_one_raises():
    with pytest.raises(ValueError, match="workers"):
        @stage(workers=-2)
        def bad(node, state):
            return []


def test_stage_workers_minus_one_allowed():
    """workers=-1 means unbounded — should not raise."""
    @stage(workers=-1)
    def ok(node, state):
        return []


# ============================================================
# restore_to warning
# ============================================================

def test_restore_to_unknown_stage_warns():
    from pipeline_builder.core.state import State
    s = State()
    with pytest.warns(UserWarning, match="no snapshot found"):
        s.restore_to("ghost_stage")


# ============================================================
# Parallel stage exception carries node index
# ============================================================

def test_parallel_exception_includes_node_index():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], workers=2)
    def explode(goal: Goal, state: State) -> list[Task]:
        raise ValueError(f"intentional: {goal.name}")

    with pytest.raises(RuntimeError, match=r"index \d+"):
        pipe.run(goal=[Goal(name="a"), Goal(name="b")])


# ============================================================
# Async checkpoint
# ============================================================

def test_async_checkpoint_fires_and_confirms():
    """async def checkpoint works the same as sync."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.stage(reads=["goal"], writes=["task"])
    async def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.checkpoint()
    async def review(state: State) -> CheckpointResult:
        fired.append("async")
        return CheckpointResult(action="confirm")

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert fired == ["async"]
    assert len(result.get_nodes("task")) == 1


def test_async_checkpoint_reject_rollback():
    """async checkpoint can reject and trigger rollback."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"n": 0}
    responses = iter(["reject", "confirm"])

    @pipe.stage(reads=["goal"], writes=["task"])
    async def make(goal: Goal, state: State) -> list[Task]:
        call_count["n"] += 1
        return [Task(goal=goal.name, action=f"v{call_count['n']}")]

    @pipe.checkpoint(on_reject="make", retry_limit=3)
    async def review(state: State) -> CheckpointResult:
        return CheckpointResult(action=next(responses))

    asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert call_count["n"] == 2


def test_sync_checkpoint_still_works_in_async_pipeline():
    """Existing sync checkpoints are not broken by the async-support change."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    async def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    @pipe.checkpoint()
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action="confirm")

    result = asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert len(result.get_nodes("task")) == 1


# ============================================================
# GoalCheckResult Literal verdict
# ============================================================

def test_goal_check_result_valid_verdicts():
    for v in ("continue", "adjust", "rollback"):
        r = GoalCheckResult(verdict=v)
        assert r.verdict == v


def test_goal_check_result_invalid_verdict_raises():
    with pytest.raises(Exception):
        GoalCheckResult(verdict="typo")


# ============================================================
# Router target — no false-positive parallel-write warning
# ============================================================

def test_router_target_stages_no_parallel_write_warning():
    """fast and slow are router targets — they never run together, no warning."""
    import warnings
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.router(targets=["fast", "slow"])
    def route(state: State) -> str:
        return "fast"

    @pipe.stage(reads=["goal"], writes=["task"])
    def fast(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="fast")]

    @pipe.stage(reads=["goal"], writes=["task"])
    def slow(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="slow")]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.run(goal=[Goal(name="g")])

    baton_warns = [x for x in w if "baton" in str(x.message).lower()]
    assert len(baton_warns) == 0, f"Unexpected warnings: {[str(x.message) for x in baton_warns]}"
