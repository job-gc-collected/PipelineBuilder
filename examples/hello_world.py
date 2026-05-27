"""Hello World — goal → task pipeline with a human review checkpoint.

Demonstrates:
- 2-level hierarchy: goal → task
- Async stage (async def)
- Auto fan-out: decompose() called once per Goal
- Checkpoint with rollback
- Artifact store for cross-stage data

Run from the project root:
    python examples/hello_world.py
"""
from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel
from pipeline_builder import CheckpointResult, MockAIAdapter, Pipeline, State


class Goal(BaseModel):
    description: str
    priority: int = 1


class Task(BaseModel):
    goal_id: str
    action: str


pipe = Pipeline(
    name="hello_world",
    hierarchy=["goal", "task"],
    schemas={"goal": Goal, "task": Task},
)


@pipe.stage(reads=["goal"], writes=["task"])
async def decompose(goal: Goal, state: State, ai: MockAIAdapter) -> list[Task]:
    """Ask AI to break a goal into tasks."""
    class Result(BaseModel):
        tasks: list[Task]

    r = await ai.run_structured_async(
        "Break this goal into concrete tasks",
        Result,
        context={"description": goal.description, "priority": goal.priority},
    )
    run_count = state.artifacts.get("run_count", 0) + 1
    state.artifacts.set("run_count", run_count)
    return r.tasks


@pipe.checkpoint(on_reject="decompose", retry_limit=3)
def review(state: State) -> CheckpointResult:
    tasks = state.get_nodes("task")
    runs  = state.artifacts.get("run_count", 1)
    print(f"\n⏸  Review (run #{runs}) — {len(tasks)} tasks generated:")
    for t in tasks:
        print(f"  [{t.goal_id}] {t.action}")

    if not sys.stdin.isatty():
        print("  (non-interactive: auto-confirming)")
        return CheckpointResult(action="confirm")

    ans = input("\nConfirm? [y=yes / n=reject] ").strip().lower()
    return CheckpointResult(action="confirm" if ans == "y" else "reject")


import json

def _make_mock() -> MockAIAdapter:
    def handler(prompt: str, ctx: dict | None) -> str:
        desc = (ctx or {}).get("description", "task")
        return json.dumps({"tasks": [
            {"goal_id": desc[:8], "action": f"Analyse: {desc}"},
            {"goal_id": desc[:8], "action": f"Implement: {desc}"},
        ]})
    return MockAIAdapter(handler=handler)


async def main() -> None:
    goals = [
        Goal(description="Build auth system", priority=1),
        Goal(description="Add user profiles",  priority=2),
    ]

    result = await pipe.run_async(ai=_make_mock(), goal=goals)

    print(f"\n✅ Done  session={result.session_id}")
    print(f"   {len(result.get_nodes('task'))} tasks total")
    print(f"   stages: {[r.name + '/' + r.status for r in result.history]}")


if __name__ == "__main__":
    asyncio.run(main())
