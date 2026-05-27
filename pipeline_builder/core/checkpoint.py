from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class CheckpointResult:
    action: Literal["confirm", "reject", "comment", "route"]
    comment: str | None = None
    # For action="route": name of the stage to jump to (no rollback).
    # For static validation, declare reachable targets via @pipe.checkpoint(targets=[...]).
    target: str | None = None


def checkpoint(
    on_reject: str | None = None,
    retry_limit: int = 3,
    targets: list[str] | None = None,
) -> Callable:
    """
    targets: optional list of stage names this checkpoint may route to via
             CheckpointResult(action="route", target="...").
             Used for static validation and visualization only.
    """
    def decorator(fn: Callable) -> Callable:
        # Preserve coroutine nature so inspect.iscoroutinefunction(wrapper)
        # is correct — the executor uses this to decide whether to await.
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper._baton_type = "checkpoint"
        wrapper._baton_on_reject = on_reject
        wrapper._baton_retry_limit = retry_limit
        wrapper._baton_targets = targets or []
        return wrapper

    return decorator
