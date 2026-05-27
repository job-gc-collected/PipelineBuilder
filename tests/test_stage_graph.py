"""Tests for stage-level DAG execution: parallelism, fan-in, depends_on, rollback."""
import asyncio
import time

import pytest
from pydantic import BaseModel

from pipeline_builder import CheckpointResult, MockAIAdapter, Pipeline, State


class PR(BaseModel):
    title: str


class File(BaseModel):
    path: str


class BugIssue(BaseModel):
    file: str
    desc: str


class StyleIssue(BaseModel):
    file: str
    desc: str


class Comment(BaseModel):
    content: str


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ─────────────────────────── Dependency inference ──────────────────────────

def test_infer_deps_sequential_chain():
    """A reads goal, B reads task (A writes), B should depend on A."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def step_a(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="a")]

    @pipe.stage(reads=["task"], writes=["task"])
    def step_b(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="b")]

    group = [step_a, step_b]
    deps = pipe._infer_group_deps(group)
    assert "step_a" in deps["step_b"]   # b depends on a
    assert deps["step_a"] == []          # a has no deps (reads "goal" = input)


def test_infer_deps_parallel_independent():
    """Two stages both reading "file" but writing different levels → no deps between them."""
    pipe = Pipeline("test", hierarchy=["pr", "file", "bug_issue", "style_issue"])

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    def bugs(file: File, state: State) -> list[BugIssue]:
        return [BugIssue(file=file.path, desc="bug")]

    @pipe.stage(reads=["file"], writes=["style_issue"])
    def style(file: File, state: State) -> list[StyleIssue]:
        return [StyleIssue(file=file.path, desc="style")]

    deps = pipe._infer_group_deps([bugs, style])
    assert deps["bugs"] == []   # neither depends on the other
    assert deps["style"] == []


def test_infer_deps_fan_in():
    """Stage needing two upstream outputs depends on both."""
    pipe = Pipeline("test", hierarchy=["pr", "file", "bug_issue", "style_issue", "comment"])

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    def bugs(file: File, state: State) -> list[BugIssue]: return []

    @pipe.stage(reads=["file"], writes=["style_issue"])
    def style(file: File, state: State) -> list[StyleIssue]: return []

    @pipe.stage(reads=["bug_issue", "style_issue"], writes=["comment"], fanout="manual")
    def synthesize(bug_issue: list, style_issue: list, state: State) -> list[Comment]: return []

    deps = pipe._infer_group_deps([bugs, style, synthesize])
    assert "bugs" in deps["synthesize"]
    assert "style" in deps["synthesize"]
    assert deps["bugs"] == []
    assert deps["style"] == []


def test_explicit_depends_on_adds_cross_level_dep():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def init(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="init")]

    @pipe.stage(reads=["goal"], writes=["task"], depends_on=["init"])
    def finalize(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="final")]

    deps = pipe._infer_group_deps([init, finalize])
    assert "init" in deps["finalize"]


# ─────────────────────────── Parallel execution ────────────────────────────

def test_parallel_stages_run_concurrently():
    """Two independent stages run in parallel: total ≈ max(time), not sum."""
    pipe = Pipeline("test", hierarchy=["pr", "file", "bug_issue", "style_issue"])

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    def bugs(file: File, state: State) -> list[BugIssue]:
        time.sleep(0.1)
        return [BugIssue(file=file.path, desc="bug")]

    @pipe.stage(reads=["file"], writes=["style_issue"])
    def style(file: File, state: State) -> list[StyleIssue]:
        time.sleep(0.1)
        return [StyleIssue(file=file.path, desc="style")]

    t0 = time.monotonic()
    result = pipe.run(file=[File(path="auth.py")])
    elapsed = time.monotonic() - t0

    assert len(result.get_nodes("bug_issue")) == 1
    assert len(result.get_nodes("style_issue")) == 1
    assert elapsed < 0.18   # parallel: ~0.10s, not ~0.20s


def test_parallel_stages_results_not_merged_different_levels():
    """Parallel stages writing DIFFERENT levels each get their own set_nodes."""
    pipe = Pipeline("test", hierarchy=["file", "bug_issue", "style_issue"])

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    def bugs(file: File, state: State) -> list[BugIssue]:
        return [BugIssue(file=file.path, desc="bug1"),
                BugIssue(file=file.path, desc="bug2")]

    @pipe.stage(reads=["file"], writes=["style_issue"])
    def style(file: File, state: State) -> list[StyleIssue]:
        return [StyleIssue(file=file.path, desc="style1")]

    result = pipe.run(file=[File(path="x.py")])
    assert len(result.get_nodes("bug_issue")) == 2
    assert len(result.get_nodes("style_issue")) == 1


def test_parallel_stages_same_level_merged():
    """Genuinely parallel stages writing to the SAME level → results merged."""
    pipe = Pipeline("test", hierarchy=["file", "issue"], schemas={})

    class Issue(BaseModel):
        kind: str

    @pipe.stage(reads=["file"], writes=["issue"])
    def find_bugs(file: File, state: State) -> list[Issue]:
        return [Issue(kind="bug")]

    @pipe.stage(reads=["file"], writes=["issue"])
    def find_smells(file: File, state: State) -> list[Issue]:
        return [Issue(kind="smell")]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress the expected parallel-write warning
        result = pipe.run(file=[File(path="x.py")])

    issues = result.get_nodes("issue")
    kinds = {i.kind for i in issues}
    assert "bug" in kinds
    assert "smell" in kinds


# ─────────────────────────── Fan-in ────────────────────────────────────────

def test_fan_in_waits_for_all_inputs():
    """synthesize runs only after BOTH bugs and style complete."""
    pipe = Pipeline("test", hierarchy=["file", "bug_issue", "style_issue", "comment"])
    order = []

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    def bugs(file: File, state: State) -> list[BugIssue]:
        order.append("bugs")
        return [BugIssue(file=file.path, desc="b")]

    @pipe.stage(reads=["file"], writes=["style_issue"])
    def style(file: File, state: State) -> list[StyleIssue]:
        order.append("style")
        return [StyleIssue(file=file.path, desc="s")]

    @pipe.stage(reads=["bug_issue", "style_issue"], writes=["comment"], fanout="manual")
    def synthesize(bug_issue: list, style_issue: list, state: State) -> list[Comment]:
        order.append("synthesize")
        return [Comment(content=f"bugs={len(bug_issue)} style={len(style_issue)}")]

    result = pipe.run(file=[File(path="x.py")])
    # synthesize must come last
    assert order.index("synthesize") > order.index("bugs")
    assert order.index("synthesize") > order.index("style")
    # synthesize sees BOTH inputs
    comment = result.get_nodes("comment")[0]
    assert "bugs=1" in comment.content
    assert "style=1" in comment.content


# ─────────────────────────── Sequential fallback ───────────────────────────

def test_sequential_chain_correct_output():
    """Sequential A→B→C produces correct final output despite group execution."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def step1(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="v1")]

    @pipe.stage(reads=["task"], writes=["task"])
    def step2(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action=task.action + "_v2")]

    @pipe.stage(reads=["task"], writes=["task"])
    def step3(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action=task.action + "_v3")]

    result = pipe.run(goal=[Goal(name="g")])
    tasks = result.get_nodes("task")
    assert len(tasks) == 1
    assert tasks[0].action == "v1_v2_v3"


# ─────────────────────────── Graph-aware rollback ──────────────────────────

def test_graph_rollback_reruns_descendants():
    """Rolling back to stage A also marks B and C (which depend on A) as not-completed."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    call_count = {"a": 0, "b": 0, "c": 0}

    @pipe.stage(reads=["goal"], writes=["task"])
    def stage_a(goal: Goal, state: State) -> list[Task]:
        call_count["a"] += 1
        return [Task(goal=goal.name, action=f"a{call_count['a']}")]

    @pipe.stage(reads=["task"], writes=["task"])
    def stage_b(task: Task, state: State) -> list[Task]:
        call_count["b"] += 1
        return [Task(goal=task.goal, action=task.action + f"_b{call_count['b']}")]

    @pipe.stage(reads=["task"], writes=["task"])
    def stage_c(task: Task, state: State) -> list[Task]:
        call_count["c"] += 1
        return [Task(goal=task.goal, action=task.action + f"_c{call_count['c']}")]

    responses = iter(["reject", "confirm"])

    @pipe.checkpoint(on_reject="stage_a", retry_limit=3)
    def review(state: State) -> CheckpointResult:
        return CheckpointResult(action=next(responses))

    result = pipe.run(goal=[Goal(name="g")])
    # All three ran twice (original + after rollback)
    assert call_count["a"] == 2
    assert call_count["b"] == 2
    assert call_count["c"] == 2


def test_depends_on_static_validation_unknown_name():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], depends_on=["ghost"])
    def make(goal: Goal, state: State) -> list[Task]:
        return []

    with pytest.raises(ValueError, match="ghost"):
        pipe.run(goal=[Goal(name="g")])


# ─────────────────────────── Async parallel ────────────────────────────────

def test_async_parallel_stages():
    pipe = Pipeline("test", hierarchy=["file", "bug_issue", "style_issue"])

    @pipe.stage(reads=["file"], writes=["bug_issue"])
    async def async_bugs(file: File, state: State) -> list[BugIssue]:
        await asyncio.sleep(0.05)
        return [BugIssue(file=file.path, desc="bug")]

    @pipe.stage(reads=["file"], writes=["style_issue"])
    async def async_style(file: File, state: State) -> list[StyleIssue]:
        await asyncio.sleep(0.05)
        return [StyleIssue(file=file.path, desc="style")]

    t0 = time.monotonic()
    result = asyncio.run(pipe.run_async(file=[File(path="x.py")]))
    elapsed = time.monotonic() - t0

    assert len(result.get_nodes("bug_issue")) == 1
    assert len(result.get_nodes("style_issue")) == 1
    assert elapsed < 0.09   # parallel: ~0.05s


def test_barrier_separates_groups():
    """Stages on either side of a checkpoint form separate groups."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def before(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="before")]

    @pipe.checkpoint()
    def gate(state: State) -> CheckpointResult:
        return CheckpointResult(action="confirm")

    @pipe.stage(reads=["task"], writes=["task"])
    def after(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="after")]

    result = pipe.run(goal=[Goal(name="g")])
    assert result.get_nodes("task")[0].action == "after"

    # Verify two groups were formed
    groups = pipe._collect_stage_groups()
    assert len(groups) == 2
    assert groups[0][0].__name__ == "before"
    assert groups[1][0].__name__ == "after"
