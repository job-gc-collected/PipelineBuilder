from __future__ import annotations

import functools
from typing import Callable


def loop(
    rollback_to: str,
    exit_on: list[str] | str | None = None,
    max_rounds: int = 3,
) -> Callable:
    """
    A code-driven conditional re-run gate.

    The decorated function inspects state and returns a string verdict.
    - If verdict is in exit_on  → advance to next step normally
    - If max_rounds exhausted   → advance regardless (soft ceiling)
    - Otherwise                 → restore state to rollback_to and re-run

    The verdict is stored in state.artifacts["_loop_result_<name>"] so
    downstream stages can branch on it (e.g. via a router).

    Usage::

        @pipe.loop(rollback_to="run_probes", exit_on=["complete", "human_confirm"], max_rounds=2)
        def completeness_gate(state: State) -> str:
            report = check_completeness(state)
            return report.overall_route   # "complete" | "re_probe" | "human_confirm"

        # After the loop, read the final verdict:
        @pipe.router(targets=["human_review", "build_techplan"])
        def after_loop(state: State) -> str:
            verdict = state.artifacts.get("_loop_result_completeness_gate")
            return "human_review" if verdict == "human_confirm" else "build_techplan"
    """
    if isinstance(exit_on, str):
        exit_on = [exit_on]

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._baton_type = "loop"
        wrapper._baton_rollback_to = rollback_to
        wrapper._baton_exit_on = exit_on or []
        wrapper._baton_max_rounds = max_rounds
        return wrapper

    return decorator
