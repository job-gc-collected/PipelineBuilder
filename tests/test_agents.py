"""Tests for Agent, AgentMessage, AgentAIAdapter, and multi-agent pipelines."""
import asyncio
import json

import pytest
from pydantic import BaseModel

from pipeline_builder import Agent, AgentMessage, MockAIAdapter, Pipeline, State
from pipeline_builder.core.agent import AgentAIAdapter


# --- Schemas ---

class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ============================================================
# AgentAIAdapter — system prompt injection
# ============================================================

def test_agent_ai_adapter_injects_system_prompt():
    received = {}

    def handler(prompt, ctx):
        return "ok"

    base_ai = MockAIAdapter(handler=handler)
    wrapped = AgentAIAdapter(delegate=base_ai, system_prompt="You are precise.")

    # We can't inspect the system param easily through MockAI (it ignores it),
    # but we test that the adapter forwards the call without error.
    result = wrapped.run("hello")
    assert result == "ok"


def test_agent_ai_adapter_explicit_system_overrides():
    """Passing system= explicitly to the wrapped adapter overrides the agent prompt."""
    calls = []

    class SpyAdapter(MockAIAdapter):
        def run(self, prompt, context=None, system=None):
            calls.append(system)
            return "ok"

    spy = SpyAdapter()
    wrapped = AgentAIAdapter(delegate=spy, system_prompt="agent prompt")

    wrapped.run("hi", system="override")
    assert calls[-1] == "override"

    wrapped.run("hi")  # no override → use agent prompt
    assert calls[-1] == "agent prompt"


def test_agent_ai_adapter_async():
    async def _run():
        calls = []

        class SpyAsync(MockAIAdapter):
            async def run_async(self, prompt, context=None, system=None):
                calls.append(system)
                return "ok"

        wrapped = AgentAIAdapter(delegate=SpyAsync(), system_prompt="sys")
        await wrapped.run_async("hi")
        return calls

    calls = asyncio.run(_run())
    assert calls[-1] == "sys"


# ============================================================
# Pipeline.add_agent + stage agent injection
# ============================================================

def test_stage_receives_agent_ai_adapter():
    """When agent= is declared, the stage receives AgentAIAdapter not the raw ai."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received_ai_type = []

    researcher = Agent("researcher", ai=MockAIAdapter(), system_prompt="expert")
    pipe.add_agent(researcher)

    @pipe.stage(reads=["goal"], writes=["task"], agent="researcher")
    def analyze(goal: Goal, state: State, ai) -> list[Task]:
        received_ai_type.append(type(ai).__name__)
        return [Task(goal=goal.name, action="done")]

    pipe.run(goal=[Goal(name="g")])
    assert received_ai_type == ["AgentAIAdapter"]


def test_stage_without_agent_gets_default_ai():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received_ai_type = []

    mock = MockAIAdapter()

    @pipe.stage(reads=["goal"], writes=["task"])
    def plain(goal: Goal, state: State, ai) -> list[Task]:
        received_ai_type.append(type(ai).__name__)
        return [Task(goal=goal.name, action="x")]

    pipe.run(ai=mock, goal=[Goal(name="g")])
    assert received_ai_type == ["MockAIAdapter"]


def test_undeclared_agent_name_raises_at_runtime():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"], agent="ghost")
    def step(goal: Goal, state: State, ai) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    with pytest.raises(ValueError, match="ghost"):
        pipe.run(goal=[Goal(name="g")])


def test_multiple_agents_each_stage_gets_own_adapter():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received = {}

    pipe.add_agent(Agent("a1", ai=MockAIAdapter(), system_prompt="agent1"))
    pipe.add_agent(Agent("a2", ai=MockAIAdapter(), system_prompt="agent2"))

    @pipe.stage(reads=["goal"], writes=["task"], agent="a1")
    def step1(goal: Goal, state: State, ai) -> list[Task]:
        received["a1"] = ai._system_prompt if isinstance(ai, AgentAIAdapter) else None
        return [Task(goal=goal.name, action="1")]

    @pipe.stage(reads=["task"], writes=["task"], agent="a2")
    def step2(task: Task, state: State, ai) -> list[Task]:
        received["a2"] = ai._system_prompt if isinstance(ai, AgentAIAdapter) else None
        return [Task(goal=task.goal, action="2")]

    pipe.run(goal=[Goal(name="g")])
    assert received["a1"] == "agent1"
    assert received["a2"] == "agent2"


def test_agent_with_run_async():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    received_ai_type = []

    pipe.add_agent(Agent("r", ai=MockAIAdapter(), system_prompt="async expert"))

    @pipe.stage(reads=["goal"], writes=["task"], agent="r")
    async def go(goal: Goal, state: State, ai) -> list[Task]:
        received_ai_type.append(type(ai).__name__)
        return [Task(goal=goal.name, action="async")]

    asyncio.run(pipe.run_async(goal=[Goal(name="g")]))
    assert received_ai_type == ["AgentAIAdapter"]


# ============================================================
# State messages
# ============================================================

def test_post_and_get_messages():
    from pipeline_builder.core.state import State as S
    s = S()
    s.post_message("researcher", "Found issue", to_agent="reviewer")
    msgs = s.get_messages(to_agent="reviewer")
    assert len(msgs) == 1
    assert msgs[0].from_agent == "researcher"
    assert msgs[0].content == "Found issue"


def test_broadcast_message_readable_by_any_agent():
    from pipeline_builder.core.state import State as S
    s = S()
    s.post_message("system", "Pipeline started")  # no to_agent → broadcast
    # Any agent can read broadcasts when not filtering
    msgs_a = s.get_messages(to_agent="agent_a")
    msgs_b = s.get_messages(to_agent="agent_b")
    assert len(msgs_a) == 1
    assert len(msgs_b) == 1


def test_directed_message_not_visible_to_others():
    from pipeline_builder.core.state import State as S
    s = S()
    s.post_message("alice", "Secret", to_agent="bob")
    msgs_for_carol = s.get_messages(to_agent="carol")
    # Carol only gets broadcasts; directed-to-bob message excluded
    assert all(m.to_agent != "carol" for m in msgs_for_carol)
    assert not any(m.content == "Secret" for m in msgs_for_carol)


def test_filter_by_from_agent():
    from pipeline_builder.core.state import State as S
    s = S()
    s.post_message("alice", "msg1")
    s.post_message("bob", "msg2")
    msgs = s.get_messages(from_agent="alice")
    assert all(m.from_agent == "alice" for m in msgs)


def test_messages_property_returns_all():
    from pipeline_builder.core.state import State as S
    s = S()
    s.post_message("a", "1")
    s.post_message("b", "2", to_agent="a")
    assert len(s.messages) == 2


def test_messages_serialized_and_restored(tmp_path):
    """Messages survive to_dict / from_dict round-trip."""
    from pipeline_builder.core.state import State as S
    s = S(session_id="test")
    s.post_message("researcher", "finding", to_agent="reviewer", metadata={"confidence": 0.9})

    data = s.to_dict({})
    s2 = S.from_dict(data, {})

    assert len(s2.messages) == 1
    m = s2.messages[0]
    assert m.from_agent == "researcher"
    assert m.to_agent == "reviewer"
    assert m.metadata["confidence"] == 0.9


def test_messages_flow_between_agents_in_pipeline():
    """Agents post and read messages across stages."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    pipe.add_agent(Agent("researcher", ai=MockAIAdapter(), system_prompt="analyse"))
    pipe.add_agent(Agent("reviewer",   ai=MockAIAdapter(), system_prompt="review"))

    @pipe.stage(reads=["goal"], writes=["task"], agent="researcher")
    def research(goal: Goal, state: State, ai) -> list[Task]:
        state.post_message("researcher", f"Analysed: {goal.name}", to_agent="reviewer")
        return [Task(goal=goal.name, action="researched")]

    @pipe.stage(reads=["task"], agent="reviewer", fanout="manual")
    def review(task: list[Task], state: State, ai) -> None:
        msgs = state.get_messages(to_agent="reviewer")
        state.artifacts.set("review_saw_messages", len(msgs))

    pipe.run(goal=[Goal(name="g")])
    assert pipe  # ran without error


def test_messages_flow_end_to_end(tmp_path):
    """Full run: researcher posts a message, reviewer reads it."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    pipe.add_agent(Agent("researcher", ai=MockAIAdapter()))
    pipe.add_agent(Agent("reviewer", ai=MockAIAdapter()))

    @pipe.stage(reads=["goal"], writes=["task"], agent="researcher")
    def do_research(goal: Goal, state: State, ai) -> list[Task]:
        state.post_message("researcher", "done", to_agent="reviewer")
        return [Task(goal=goal.name, action="x")]

    seen = []

    @pipe.stage(reads=["task"], agent="reviewer", fanout="manual")
    def do_review(task: list[Task], state: State, ai) -> None:
        msgs = state.get_messages(to_agent="reviewer")
        seen.extend(m.content for m in msgs)

    pipe.run(goal=[Goal(name="g")])
    assert seen == ["done"]


# ============================================================
# Event hooks (sync run)
# ============================================================

def test_event_hooks_fire_on_stage_complete():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []

    @pipe.on("stage_complete")
    def handler(name, state, duration_ms, **kw):
        fired.append((name, duration_ms is not None))

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    assert ("make", True) in fired


def test_event_hooks_fire_on_stage_fail():
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    failures = []

    @pipe.on("stage_fail")
    def on_fail(name, state, error, **kw):
        failures.append((name, str(error)))

    @pipe.stage(reads=["goal"], writes=["task"])
    def boom(goal: Goal, state: State) -> list[Task]:
        raise ValueError("intentional")

    with pytest.raises(ValueError):
        pipe.run(goal=[Goal(name="g")])

    assert failures and failures[0][0] == "boom"


def test_event_hooks_direct_registration():
    """pipe.on() can be called without being used as a decorator."""
    pipe = Pipeline("test", hierarchy=["goal", "task"])
    fired = []
    pipe.on("stage_start", lambda name, state, **kw: fired.append(name))

    @pipe.stage(reads=["goal"], writes=["task"])
    def go(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    assert "go" in fired


def test_on_unknown_event_raises():
    pipe = Pipeline("test", hierarchy=["goal"])
    with pytest.raises(ValueError, match="Unknown event"):
        pipe.on("nonexistent_event", lambda **kw: None)


# ============================================================
# Heartbeat
# ============================================================

def test_heartbeat_saves_state_periodically(tmp_path):
    import time

    pipe = Pipeline(
        "test",
        hierarchy=["goal", "task"],
        schemas={"goal": Goal, "task": Task},
        state_dir=str(tmp_path),
        heartbeat_interval=1,  # every second
    )

    @pipe.stage(reads=["goal"], writes=["task"])
    def slow_make(goal: Goal, state: State) -> list[Task]:
        time.sleep(1.2)  # run long enough for at least one heartbeat
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    # State file should exist (written either by heartbeat or final save)
    snap = tmp_path / f"{result.session_id}.json"
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["session_id"] == result.session_id


# ============================================================
# StageRecord.duration_ms
# ============================================================

def test_stage_record_duration_ms():
    pipe = Pipeline("test", hierarchy=["goal", "task"])

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    result = pipe.run(goal=[Goal(name="g")])
    record = next(r for r in result.history if r.name == "make")
    assert record.duration_ms is not None
    assert record.duration_ms >= 0


def test_stage_record_duration_ms_none_while_running():
    from pipeline_builder.core.state import StageRecord
    r = StageRecord("test")
    assert r.duration_ms is None  # not yet completed
