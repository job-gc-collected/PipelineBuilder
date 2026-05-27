import json

import pytest
from pydantic import BaseModel

from pipeline_builder.ai.mock import MockAIAdapter


class Task(BaseModel):
    action: str
    priority: int = 1


class TaskList(BaseModel):
    tasks: list[Task]


def test_mock_run_returns_handler_result():
    ai = MockAIAdapter(handler=lambda p, c: "hello")
    assert ai.run("any prompt") == "hello"


def test_mock_run_default_handler():
    ai = MockAIAdapter()
    result = ai.run("some prompt")
    assert "some prompt" in result or isinstance(result, str)


def test_mock_run_structured_valid_json():
    payload = json.dumps({"tasks": [{"action": "write tests", "priority": 2}]})
    ai = MockAIAdapter(handler=lambda p, c: payload)
    result = ai.run_structured("decompose", TaskList)
    assert isinstance(result, TaskList)
    assert result.tasks[0].action == "write tests"
    assert result.tasks[0].priority == 2


def test_mock_run_structured_passes_context():
    received = {}

    def handler(prompt: str, ctx: dict | None) -> str:
        received["ctx"] = ctx
        return json.dumps({"tasks": [{"action": "x"}]})

    ai = MockAIAdapter(handler=handler)
    ai.run_structured("prompt", TaskList, context={"key": "value"})
    assert received["ctx"] == {"key": "value"}


def test_mock_run_structured_invalid_json_returns_defaults():
    ai = MockAIAdapter(handler=lambda p, c: "not json at all")
    result = ai.run_structured("prompt", Task)
    assert isinstance(result, Task)


# --- system param (change 3) ---

def test_mock_run_accepts_system_param():
    ai = MockAIAdapter(handler=lambda p, c: "ok")
    result = ai.run("prompt", system="some instructions")
    assert result == "ok"


def test_mock_run_structured_accepts_system_param():
    payload = '{"action": "x"}'
    ai = MockAIAdapter(handler=lambda p, c: payload)
    result = ai.run_structured("prompt", Task, system="some instructions")
    assert result.action == "x"


# --- _type_default fixes (change 4) ---

class DictModel(BaseModel):
    mapping: dict[str, int]
    nested: dict


class OptionalModel(BaseModel):
    maybe: str | None = None
    count: int | None = None


def test_mock_run_structured_dict_field_fallback():
    """dict fields must fall back to {} not None."""
    ai = MockAIAdapter(handler=lambda p, c: "not json")
    result = ai.run_structured("prompt", DictModel)
    assert isinstance(result, DictModel)
    assert result.mapping == {}
    assert result.nested == {}


def test_mock_run_structured_optional_field_fallback():
    """Optional[X] (anyOf) should produce a valid default (None is fine)."""
    ai = MockAIAdapter(handler=lambda p, c: "not json")
    result = ai.run_structured("prompt", OptionalModel)
    assert isinstance(result, OptionalModel)
