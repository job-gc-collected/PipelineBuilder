"""Tests for pipeline versioning — fingerprint, compatibility checks, aliases."""
import json

import pytest
from pydantic import BaseModel

from pipeline_builder import CompatibilityError, Pipeline, SQLiteBackend, State
from pipeline_builder.core.versioning import check_resume_compatibility, compute_fingerprint


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ─────────────────────────── compute_fingerprint ────────────────────────────

def test_same_pipeline_same_fingerprint():
    pipe = Pipeline("p", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return []

    fp1 = pipe._pipeline_fingerprint()
    fp2 = pipe._pipeline_fingerprint()
    assert fp1 == fp2


def test_different_hierarchy_different_fingerprint():
    p1 = Pipeline("p", hierarchy=["goal", "task"])
    p2 = Pipeline("p", hierarchy=["goal", "task", "step"])
    assert p1._pipeline_fingerprint() != p2._pipeline_fingerprint()


def test_added_stage_changes_fingerprint():
    pipe = Pipeline("p", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def step1(goal: Goal, state: State) -> list[Task]:
        return []

    fp1 = pipe._pipeline_fingerprint()

    @pipe.stage(reads=["task"], writes=["task"])
    def step2(task: Task, state: State) -> list[Task]:
        return []

    fp2 = pipe._pipeline_fingerprint()
    assert fp1 != fp2


def test_logic_change_does_not_change_fingerprint():
    """workers, timeout, agent changes don't affect fingerprint — only structure does."""
    pipe1 = Pipeline("p", hierarchy=["goal", "task"])
    pipe2 = Pipeline("p", hierarchy=["goal", "task"])

    @pipe1.stage(reads=["goal"], writes=["task"], workers=1)
    def make1(goal: Goal, state: State) -> list[Task]:
        return []

    @pipe2.stage(reads=["goal"], writes=["task"], workers=4, timeout=30.0)
    def make1(goal: Goal, state: State) -> list[Task]:  # noqa: F811
        return []

    assert pipe1._pipeline_fingerprint() == pipe2._pipeline_fingerprint()


# ─────────────────────────── check_resume_compatibility ─────────────────────

def test_identical_fingerprint_no_check():
    result = check_resume_compatibility(
        "s1", ["make"], "fp1", "fp1",
        {"make"}, {}, ["goal", "task"], ["goal", "task"],
    )
    assert result == ["make"]


def test_safe_change_warns_and_continues(recwarn):
    result = check_resume_compatibility(
        "s1", ["make"], "old_fp", "new_fp",
        {"make"},  # make still exists → safe
        {}, ["goal", "task"], ["goal", "task"],
    )
    assert result == ["make"]
    assert any("resuming" in str(w.message).lower() or "fingerprint" in str(w.message).lower()
               for w in recwarn.list)


def test_removed_stage_raises():
    with pytest.raises(CompatibilityError, match="old_stage"):
        check_resume_compatibility(
            "s1", ["old_stage"], "old_fp", "new_fp",
            {"new_stage"},   # old_stage gone, no alias
            {}, ["goal", "task"], ["goal", "task"],
        )


def test_renamed_with_alias_remaps(recwarn):
    result = check_resume_compatibility(
        "s1", ["parse_doc"], "old_fp", "new_fp",
        {"parse_prd"},                          # new canonical name
        {"parse_doc": "parse_prd"},             # alias maps old → new
        ["goal", "task"], ["goal", "task"],
    )
    assert result == ["parse_prd"]


def test_hierarchy_change_raises():
    with pytest.raises(CompatibilityError, match="hierarchy"):
        check_resume_compatibility(
            "s1", [], "old_fp", "new_fp",
            {"make"}, {},
            ["goal", "task", "step"],   # current (longer)
            ["goal", "task"],           # stored (shorter)
        )


def test_error_message_suggests_alias():
    with pytest.raises(CompatibilityError) as exc:
        check_resume_compatibility(
            "s1", ["old_name"], "x", "y", {"new_name"}, {},
            ["a"], ["a"],
        )
    assert "aliases" in str(exc.value)
    assert "old_name" in str(exc.value)


# ─────────────────────────── Pipeline integration ────────────────────────────

def test_fingerprint_stored_in_new_session(tmp_path):
    pipe = Pipeline(
        "p", hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    stored_fp = result._internal.get("__baton_pipeline_fingerprint")
    assert stored_fp is not None
    assert stored_fp == pipe._pipeline_fingerprint()


def test_identical_pipeline_resumes_silently(tmp_path):
    schemas = {"goal": Goal, "task": Task}
    pipe = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=str(tmp_path))

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    sid = result.session_id

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning = failure
        result2 = pipe.run(session_id=sid)  # must be silent

    assert len(result2.get_nodes("task")) == 1


def test_safe_change_resume_warns_not_fails(tmp_path):
    schemas = {"goal": Goal, "task": Task}
    state_dir = str(tmp_path)

    # First run: pipeline with one stage
    pipe1 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe1.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe1.run(goal=[Goal(name="g")])
    sid = result.session_id

    # Second run: same pipeline PLUS a new stage added at the end
    pipe2 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe2.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:  # noqa: F811
        return [Task(goal=goal.name, action="x")]

    @pipe2.stage(reads=["task"], writes=["task"])
    def extra(task: Task, state: State) -> list[Task]:
        return [Task(goal=task.goal, action="extra")]

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe2.run(session_id=sid)

    assert any("compatible" in str(x.message).lower() or "changed" in str(x.message).lower()
               for x in w)


def test_breaking_change_raises_on_resume(tmp_path):
    schemas = {"goal": Goal, "task": Task}
    state_dir = str(tmp_path)

    pipe1 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe1.stage(reads=["goal"], writes=["task"])
    def important(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe1.run(goal=[Goal(name="g")])
    sid = result.session_id

    # New pipeline: "important" stage REMOVED
    pipe2 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe2.stage(reads=["goal"], writes=["task"])
    def something_else(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="y")]

    with pytest.raises(CompatibilityError, match="important"):
        pipe2.run(session_id=sid)


def test_rename_with_alias_resumes(tmp_path):
    schemas = {"goal": Goal, "task": Task}
    state_dir = str(tmp_path)

    # First run: stage called "old_parse"
    pipe1 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe1.stage(reads=["goal"], writes=["task"])
    def old_parse(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe1.run(goal=[Goal(name="g")])
    sid = result.session_id

    # Second run: stage renamed to "new_parse" with alias
    pipe2 = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, state_dir=state_dir)

    @pipe2.stage(reads=["goal"], writes=["task"], aliases=["old_parse"])
    def new_parse(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="y")]

    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result2 = pipe2.run(session_id=sid)

    # new_parse was already completed (via alias mapping), session resumed fully
    assert "new_parse" in result2._completed_stages


def test_can_resume_method(tmp_path):
    schemas = {"goal": Goal, "task": Task}
    backend = SQLiteBackend(tmp_path / "runs.db")
    pipe = Pipeline("p", hierarchy=["goal", "task"], schemas=schemas, storage=backend)

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    sid = result.session_id

    ok, reason = pipe.can_resume(sid)
    assert ok
    assert "match" in reason.lower() or "compatible" in reason.lower()


def test_can_resume_returns_false_for_unknown_session(tmp_path):
    backend = SQLiteBackend(tmp_path / "runs.db")
    pipe = Pipeline("p", hierarchy=["goal"], storage=backend)
    ok, reason = pipe.can_resume("nonexistent")
    assert not ok


def test_stage_aliases_collected(tmp_path):
    pipe = Pipeline("p", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], aliases=["old_a", "old_b"])
    def new_a(goal: Goal, state: State) -> list[Task]:
        return []

    aliases = pipe._stage_aliases()
    assert aliases["old_a"] == "new_a"
    assert aliases["old_b"] == "new_a"
