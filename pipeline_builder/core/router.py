from __future__ import annotations

import functools
from typing import Callable


def router(targets: list[str] | None = None) -> Callable:
    """
    Declare a routing step that returns the name of the next stage to jump to.

    targets: optional list of possible return values — used for static validation
             and visualization only.  At runtime any valid stage name is accepted.

    Usage::

        @pipe.router(targets=["fast_path", "full_path"])
        def decide(state: State) -> str:
            return "fast_path" if state.artifacts.get("skip") else "full_path"
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._baton_type = "router"
        wrapper._baton_targets = targets or []
        return wrapper

    return decorator
