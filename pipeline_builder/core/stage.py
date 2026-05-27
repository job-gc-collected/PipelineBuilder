from __future__ import annotations

import functools
import inspect
from typing import Callable, Literal


def stage(
    reads: list[str] | None = None,
    writes: list[str] | None = None,
    fanout: Literal["auto", "manual"] = "auto",
    workers: int = 1,
    retry: int = 0,
    retry_delay: float = 1.0,
    agent: str | None = None,
    timeout: float | None = None,
    progress_interval: float = 0.0,
    depends_on: list[str] | None = None,
    aliases: list[str] | None = None,
) -> Callable:
    """
    workers=1  → sequential (default)
    workers=N  → N concurrent threads
    workers=-1 → unbounded (one thread per node)

    retry=N           → retry up to N times on exception (default 0 = no retry)
    retry_delay=T     → base delay in seconds; actual delay = T * 2^attempt (exponential backoff)

    agent=NAME        → name of the Agent registered on the pipeline that should
                        execute this stage.  The pipeline injects that agent's
                        AgentAIAdapter (carrying its system_prompt) as ``ai``.
    timeout=T         → abort the stage if it runs longer than T seconds.
                        In sync mode uses a thread-pool future; in async mode uses
                        asyncio.wait_for().  Raises TimeoutError on breach.
    progress_interval → emit a ``stage_progress`` event every N seconds while the
                        stage is executing.  Use with ``@pipe.on("stage_progress")``
                        to surface heartbeats for long-running AI calls.  0 = disabled.
    depends_on        → explicit list of stage names this stage depends on, in addition
                        to the dependency automatically inferred from reads/writes.
                        Useful when a stage depends on another stage that writes to a
                        different level (cross-level dependency).
    aliases           → list of previous names this stage was known by.  When a session
                        is resumed after a rename, baton maps the old name to this stage
                        so the completed-stage record survives the rename without error.
                        Example: after renaming ``parse_doc`` to ``parse_prd``, add
                        ``aliases=["parse_doc"]`` to prevent CompatibilityError on resume.
    """
    if workers != -1 and workers < 1:
        raise ValueError(
            f"workers must be a positive integer or -1 (unbounded), got {workers}"
        )

    def decorator(fn: Callable) -> Callable:
        # Preserve coroutine nature so inspect.iscoroutinefunction(wrapper) is
        # correct — the pipeline uses this to decide whether to await or thread.
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper._baton_type = "stage"
        wrapper._baton_reads = reads or []
        wrapper._baton_writes = writes or []
        wrapper._baton_fanout = fanout
        wrapper._baton_workers = workers
        wrapper._baton_retry = retry
        wrapper._baton_retry_delay = retry_delay
        wrapper._baton_agent = agent
        wrapper._baton_timeout = timeout
        wrapper._baton_progress_interval = progress_interval
        wrapper._baton_depends_on = depends_on or []
        wrapper._baton_aliases = aliases or []
        return wrapper

    return decorator
