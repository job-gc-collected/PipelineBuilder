from pydantic import BaseModel
from pipeline_builder import Pipeline, State, CheckpointResult


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    action: str


def _pipe() -> Pipeline:
    pipe = Pipeline("demo", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], workers=3)
    def expand(goal: Goal, state: State) -> list[Task]: ...

    @pipe.checkpoint(on_reject="expand")
    def review(state: State) -> CheckpointResult: ...

    return pipe


def test_show_contains_pipeline_name(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "demo" in out


def test_show_contains_hierarchy(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "goal" in out
    assert "task" in out


def test_show_contains_stage_name(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "expand" in out


def test_show_contains_parallel_marker(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "×3" in out


def test_show_contains_checkpoint(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "CHECK" in out
    assert "review" in out


def test_show_contains_rollback(capsys):
    _pipe().show()
    out = capsys.readouterr().out
    assert "↩" in out
    assert "expand" in out


def test_mermaid_starts_with_code_fence():
    md = _pipe().to_mermaid()
    assert md.startswith("```mermaid")
    assert md.endswith("```")


def test_mermaid_contains_flowchart():
    md = _pipe().to_mermaid()
    assert "flowchart TD" in md


def test_mermaid_contains_all_nodes():
    md = _pipe().to_mermaid()
    assert "expand" in md
    assert "review" in md
    assert "checkpoint" in md


def test_mermaid_contains_reject_edge():
    md = _pipe().to_mermaid()
    assert "reject" in md
    assert "expand" in md


# --- Loop visualization ---

def _pipe_with_loop():
    from pipeline_builder import Pipeline, State
    from pydantic import BaseModel

    class Item(BaseModel):
        value: str

    pipe = Pipeline("test", hierarchy=["item"], schemas={})

    @pipe.stage(reads=["item"], writes=["item"])
    def process(item: Item, state: State) -> list[Item]:
        return [item]

    @pipe.loop(rollback_to="process", exit_on=["ok"], max_rounds=3)
    def quality(state: State) -> str:
        return "ok"

    return pipe


def test_ascii_contains_loop():
    out = _pipe_with_loop().show.__func__  # just ensure it doesn't crash
    diag = _pipe_with_loop()
    import io, sys
    buf = io.StringIO()
    sys.stdout = buf
    diag.show()
    sys.stdout = sys.__stdout__
    output = buf.getvalue()
    assert "LOOP" in output
    assert "quality" in output
    assert "process" in output


def test_ascii_loop_shows_rollback_target():
    import io, sys
    buf = io.StringIO()
    sys.stdout = buf
    _pipe_with_loop().show()
    sys.stdout = sys.__stdout__
    assert "rollback:process" in buf.getvalue()


def test_mermaid_contains_loop_node():
    md = _pipe_with_loop().to_mermaid()
    assert "quality" in md
    assert "loop" in md.lower()


def test_mermaid_loop_has_rollback_edge():
    md = _pipe_with_loop().to_mermaid()
    assert "rollback" in md
    assert "process" in md
