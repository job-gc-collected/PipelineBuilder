"""
GoalCheck — periodic self-correction for multi-stage AI pipelines.

Addresses the "goal drift" problem: after many stages, the pipeline may
have drifted from the original objective.  A goal check fires every N
completed stages, lets an AI evaluate the current state, and can either
continue, adjust state.data fields, or roll back to an earlier stage.

Usage::

    class MyCtx(BaseModel):
        original_goal: str = ""
        findings: list[str] = Field(default_factory=list)

    pipe = Pipeline("...", hierarchy=[...], state_schema=MyCtx)

    @pipe.goal_check(interval=3, rollback_to="extract_probes")
    def check_on_track(state: State, ai: AIAdapter) -> GoalCheckResult:
        return ai.run_structured(
            "Are we still aligned with the original goal?",
            GoalCheckResult,
            context={
                "original_goal": state.data.original_goal,
                "completed": [r.name for r in state.history],
                "findings_so_far": state.data.findings,
            },
        )
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class GoalCheckResult(BaseModel):
    """Return value for a goal-check function.

    The ``verdict`` controls what the pipeline does next:

    * ``"continue"``  — everything on track; proceed normally.
    * ``"adjust"``    — still on track but some state.data fields need
                        correction; apply ``data_updates`` and continue.
    * ``"rollback"``  — goal drift detected; restore state to
                        ``rollback_to`` stage and re-run from there.

    Designed to be returned directly from ``ai.run_structured(...)`` so
    the AI can express its self-evaluation in structured form.

    Using ``Literal`` for ``verdict`` means Pydantic validates the value
    on construction — a typo like ``"contineu"`` raises immediately
    rather than silently falling through to the "continue" branch.
    """
    verdict: Literal["continue", "adjust", "rollback"] = "continue"
    note: str = ""                     # AI's explanation (shown in logs/events)
    rollback_to: str | None = None     # stage name; overrides decorator default
    data_updates: dict[str, Any] = Field(default_factory=dict)


def goal_check(
    interval: int = 3,
    rollback_to: str | None = None,
    max_checks: int = 10,
) -> Callable:
    """Decorator that marks a function as a periodic goal-check.

    interval:    run after every ``interval`` completed stages.
    rollback_to: default rollback target when verdict="rollback" and
                 the function doesn't specify ``result.rollback_to``.
    max_checks:  hard ceiling on how many times this check may fire
                 (guards against infinite rollback loops).

    The decorated function may accept ``state`` and optionally ``ai``
    (injected if declared in its signature).
    """
    if interval < 1:
        raise ValueError(f"goal_check interval must be >= 1, got {interval}")

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper._baton_type = "goal_check"
        wrapper._baton_gc_interval = interval
        wrapper._baton_gc_rollback_to = rollback_to
        wrapper._baton_gc_max_checks = max_checks
        return wrapper

    return decorator
