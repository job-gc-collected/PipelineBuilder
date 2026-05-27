"""PipelineExecutor — the async execution engine for baton pipelines.

All runtime logic lives here.  ``Pipeline`` is a thin registration/config
object; it creates a ``PipelineExecutor`` and calls ``execute()`` for each
``run_async()`` invocation.

Design notes
------------
* The executor holds references to ``pipe`` (Pipeline config), ``state``
  (the mutable session state), and ``ai`` (default AI adapter).
* Pipeline config is accessed via ``self.pipe.*``; execution context via
  ``self.state`` / ``self.ai``.  This eliminates the pattern of passing
  ``state`` and ``ai`` through every private method.
* All stage execution is async-native; sync stages run via
  ``asyncio.to_thread()`` so they don't block the event loop.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

from contextlib import contextmanager

from .checkpoint import CheckpointResult
from .goal_check import GoalCheckResult
from .group import batch_extend_mode, can_reach, infer_group_deps, levels_to_clear
from .state import State, StageRecord

if TYPE_CHECKING:
    from ..ai.base import AIAdapter
    from .pipeline import Pipeline

logger = logging.getLogger("baton")


@contextmanager
def _noop_ctx():
    """No-op context manager used when no tracer is configured."""
    yield None


def _retry_sleep(exc: Exception, default_sleep: float) -> float:
    """Return how long to sleep before the next retry.

    Honours ``Retry-After`` on HTTP 429 responses; falls back to
    ``default_sleep`` otherwise.
    """
    try:
        if getattr(exc, "status_code", None) == 429:
            response = getattr(exc, "response", None)
            if response is not None:
                header = response.headers.get("retry-after")
                if header is not None:
                    wait = float(header) + 0.5
                    logger.info("Rate limited — honouring Retry-After: %.1fs", wait)
                    return wait
    except Exception:
        pass
    return default_sleep


class PipelineExecutor:
    """Async execution engine for a single pipeline run.

    Instantiated once per ``run_async()`` call and discarded when execution
    completes (or fails).  Not thread-safe across multiple concurrent runs
    of the same pipeline — create separate instances for each run.
    """

    def __init__(
        self,
        pipe: "Pipeline",
        state: State,
        ai: "AIAdapter | None" = None,
    ) -> None:
        self.pipe = pipe
        self.state = state
        self.ai = ai

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def execute(self) -> State:
        """Run the pipeline to completion and return the final state."""
        pipe = self.pipe
        state = self.state

        pipe._start_heartbeat(state)
        tracer = pipe._tracer
        try:
            with (tracer.pipeline_span(pipe.name, state.session_id) if tracer else _noop_ctx()):
                index = 0
                while index < len(pipe._steps):
                    step = pipe._steps[index]
                    btype = getattr(step, "_baton_type", None)
                    if btype == "stage":
                        group: list[Callable] = []
                        while (
                            index < len(pipe._steps)
                            and getattr(pipe._steps[index], "_baton_type", None) == "stage"
                        ):
                            group.append(pipe._steps[index])
                            index += 1
                        index = await self._run_stage_group(group, index)
                        if pipe._goal_checks:
                            for _ in group:
                                new_idx = await self._run_goal_checks_if_due(index)
                                if new_idx != index:
                                    index = new_idx
                                    break
                    elif btype == "checkpoint":
                        index = await self._run_checkpoint(step, index)
                    elif btype == "router":
                        index = self._run_router(step, index)
                    elif btype == "loop":
                        index = self._run_loop(step, index)
                    else:
                        index += 1
        finally:
            pipe._stop_heartbeat()
            pipe._save_state(state)

        return state

    # ------------------------------------------------------------------ #
    # Stage group DAG                                                      #
    # ------------------------------------------------------------------ #

    async def _run_stage_group(self, group: list[Callable], index: int) -> int:
        """Execute a group of consecutive stages as a dependency DAG.

        Uses the pure functions from ``baton.core.group`` for all DAG
        reasoning — no scheduling logic lives here.
        """
        deps = infer_group_deps(group)
        state = self.state

        # On a fresh run, clear only the levels that genuinely parallel stages
        # both write (determined by group.levels_to_clear).
        if not any(state.is_completed(s.__name__) for s in group):
            for lv in levels_to_clear(group, deps):
                state.clear_nodes(lv)

        # Snapshot each not-yet-completed stage before any of them runs.
        for s in group:
            if not state.is_completed(s.__name__):
                state.snapshot_before(s.__name__)
                self.pipe._save_snapshot_to_backend(state, s.__name__)

        completed_in_group: set[str] = {
            s.__name__ for s in group if state.is_completed(s.__name__)
        }

        while len(completed_in_group) < len(group):
            remaining = [s for s in group if s.__name__ not in completed_in_group]
            ready = [
                s for s in remaining
                if all(dep in completed_in_group for dep in deps[s.__name__])
            ]
            if not ready:
                break

            # Handle router-set skip flags before deciding what to run.
            skipped = [s for s in ready if state._internal.get(f"__baton_skip_{s.__name__}")]
            to_run  = [s for s in ready if s not in skipped]

            for s in skipped:
                state._internal.set(f"__baton_skip_{s.__name__}", False)
                completed_in_group.add(s.__name__)

            # group.batch_extend_mode() tells us which stages need extend_nodes
            # (those writing to a level that another stage in this batch also writes).
            extend_mode = batch_extend_mode(to_run)

            async def _run_one(s: Callable) -> None:
                await self._run_stage(s, 0, take_snapshot=False,
                                      use_extend=extend_mode[s.__name__],
                                      check_skip=False)

            if len(to_run) == 1:
                await _run_one(to_run[0])
                completed_in_group.add(to_run[0].__name__)
            elif len(to_run) > 1:
                tasks = [asyncio.create_task(_run_one(s)) for s in to_run]
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for s, res in zip(to_run, gathered):
                    if isinstance(res, Exception):
                        raise RuntimeError(
                            f"Stage '{s.__name__}' failed in parallel group: {res}"
                        ) from res
                    completed_in_group.add(s.__name__)

        return index

    # ------------------------------------------------------------------ #
    # Stage execution                                                      #
    # ------------------------------------------------------------------ #

    async def _run_stage(
        self,
        step: Callable,
        index: int,
        take_snapshot: bool = True,
        use_extend: bool = False,
        check_skip: bool = True,
    ) -> int:
        state = self.state
        pipe = self.pipe

        if state.is_completed(step.__name__):
            return index + 1

        if check_skip:
            skip_key = f"__baton_skip_{step.__name__}"
            if state._internal.get(skip_key):
                state._internal.set(skip_key, False)
                return index + 1

        if take_snapshot:
            state.snapshot_before(step.__name__)
            pipe._save_snapshot_to_backend(state, step.__name__)

        effective_ai = pipe._resolve_ai(step, self.ai)
        agent_name = getattr(step, "_baton_agent", None)
        record = StageRecord(step.__name__)
        inject_ai = self._wants_ai(step)
        retry = getattr(step, "_baton_retry", 0)
        retry_delay = getattr(step, "_baton_retry_delay", 1.0)
        timeout = getattr(step, "_baton_timeout", None)
        progress_interval = getattr(step, "_baton_progress_interval", 0.0)

        pipe._emit("stage_start", name=step.__name__, state=state)
        logger.debug("Stage '%s' starting", step.__name__)

        _progress_task: "asyncio.Task | None" = (
            asyncio.create_task(self._progress_coro(step.__name__, progress_interval))
            if progress_interval > 0 else None
        )
        try:
            reads = step._baton_reads
            writes = step._baton_writes
            workers = getattr(step, "_baton_workers", 1)
            all_results: list = []

            async def _call_one(node: Any) -> list:
                kw: dict[str, Any] = {"state": state}
                if inject_ai:
                    kw["ai"] = effective_ai
                result = await self._call_with_retry(step, retry, retry_delay, node, **kw)
                if result is None:
                    return []
                return result if isinstance(result, list) else [result]

            async def _call_one_with_timeout(node: Any) -> list:
                if timeout is None:
                    return await _call_one(node)
                return await asyncio.wait_for(_call_one(node), timeout=timeout)

            if step._baton_fanout == "auto" and reads:
                nodes = state.get_nodes(reads[0])
                if workers == 1 or len(nodes) <= 1:
                    for node in nodes:
                        all_results.extend(await _call_one_with_timeout(node))
                else:
                    max_concurrent = len(nodes) if workers == -1 else min(workers, len(nodes))
                    sem = asyncio.Semaphore(max_concurrent)

                    async def _guarded(node: Any, idx: int) -> tuple[int, list]:
                        async with sem:
                            try:
                                return idx, await _call_one_with_timeout(node)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Stage '{step.__name__}' failed on node at index {idx}: {exc}"
                                ) from exc

                    pairs = await asyncio.gather(*[_guarded(n, i) for i, n in enumerate(nodes)])
                    for chunk in [c for _, c in sorted(pairs, key=lambda p: p[0])]:
                        all_results.extend(chunk)
            else:
                kwargs: dict[str, Any] = {level: state.get_nodes(level) for level in reads}
                kwargs["state"] = state
                if inject_ai:
                    kwargs["ai"] = effective_ai

                async def _manual_call() -> Any:
                    return await self._call_with_retry(step, retry, retry_delay, **kwargs)

                if timeout is not None:
                    result = await asyncio.wait_for(_manual_call(), timeout=timeout)
                else:
                    result = await _manual_call()

                if result:
                    all_results = result if isinstance(result, list) else [result]

            if writes and all_results:
                if use_extend:
                    state.extend_nodes(writes[0], all_results)
                else:
                    state.set_nodes(writes[0], all_results)

            record.complete()
            state.mark_completed(step.__name__)
            if len(state._completed_stages) % pipe.persist_every == 0:
                pipe._save_state(state)

            pipe._emit("stage_complete", name=step.__name__, state=state,
                       duration_ms=record.duration_ms)
            logger.debug("Stage '%s' completed in %.0fms", step.__name__, record.duration_ms or 0)

        except Exception as exc:
            record.fail()
            pipe._emit("stage_fail", name=step.__name__, state=state, error=exc)
            logger.error("Stage '%s' failed: %s", step.__name__, exc)
            raise
        finally:
            if _progress_task is not None:
                _progress_task.cancel()
                try:
                    await _progress_task
                except asyncio.CancelledError:
                    pass
            state.add_history(record)

        return index + 1

    # ------------------------------------------------------------------ #
    # Barriers: loop / router / checkpoint                                 #
    # ------------------------------------------------------------------ #

    def _run_loop(self, step: Callable, index: int) -> int:
        state = self.state
        rollback_to = step._baton_rollback_to
        max_rounds = step._baton_max_rounds
        exit_on = step._baton_exit_on

        round_key = f"__baton_loop_round_{step.__name__}"
        current_round = state._internal.get(round_key, 0)
        verdict: str = step(state)
        state.artifacts.set(f"_loop_result_{step.__name__}", verdict)

        self.pipe._emit("loop", name=step.__name__, state=state,
                        verdict=verdict, round=current_round)

        should_exit = verdict in exit_on or current_round >= max_rounds
        if should_exit:
            state._internal.set(round_key, 0)
            return index + 1

        state._internal.set(round_key, current_round + 1)
        return self._graph_rollback(rollback_to)

    def _run_router(self, step: Callable, index: int) -> int:
        state = self.state
        target_name: str = step(state)

        for t in step._baton_targets:
            if t != target_name:
                state._internal.set(f"__baton_skip_{t}", True)

        for i, s in enumerate(self.pipe._steps):
            if s.__name__ == target_name:
                return i
        raise RuntimeError(
            f"Router '{step.__name__}' returned '{target_name}' "
            f"but no step with that name exists in the pipeline"
        )

    async def _run_checkpoint(self, step: Callable, index: int) -> int:
        """Run a checkpoint, supporting both sync and ``async def`` functions.

        Making checkpoints async enables patterns like:
        - Sending a Slack message and awaiting a response
        - Posting to a webhook and waiting for callback
        - Any I/O-bound human-interaction flow
        """
        state = self.state
        on_reject = step._baton_on_reject
        retry_limit = step._baton_retry_limit

        for _ in range(retry_limit):
            if inspect.iscoroutinefunction(step):
                result: CheckpointResult = await step(state)
            else:
                result = await asyncio.to_thread(step, state)
            self.pipe._emit("checkpoint", name=step.__name__, state=state, action=result.action)

            if result.action == "confirm":
                return index + 1

            if result.action == "route" and result.target:
                for t in getattr(step, "_baton_targets", []):
                    if t != result.target:
                        state._internal.set(f"__baton_skip_{t}", True)
                for i, s in enumerate(self.pipe._steps):
                    if s.__name__ == result.target:
                        return i
                raise RuntimeError(
                    f"Checkpoint '{step.__name__}' routed to '{result.target}' "
                    f"but no step with that name exists in the pipeline"
                )

            if result.action == "reject" and on_reject:
                return self._graph_rollback(on_reject)

        return index + 1

    # ------------------------------------------------------------------ #
    # Graph-aware rollback                                                 #
    # ------------------------------------------------------------------ #

    def _graph_rollback(self, target_name: str) -> int:
        """Restore state to target's snapshot AND mark target + dependents as not-completed."""
        state = self.state
        all_stages = [s for s in self.pipe._steps if getattr(s, "_baton_type", None) == "stage"]
        full_deps = infer_group_deps(all_stages)

        descendants: set[str] = set()
        queue = [target_name]
        while queue:
            cur = queue.pop()
            for name, d_list in full_deps.items():
                if cur in d_list and name not in descendants:
                    descendants.add(name)
                    queue.append(name)

        state.trim_completed(descendants | {target_name})
        self.pipe._ensure_snapshot_loaded(state, target_name)
        state.restore_to(target_name)

        for i, s in enumerate(self.pipe._steps):
            if s.__name__ == target_name:
                return i
        raise RuntimeError(f"Rollback target '{target_name}' not found in pipeline")

    # ------------------------------------------------------------------ #
    # Goal checks                                                          #
    # ------------------------------------------------------------------ #

    async def _run_goal_checks_if_due(self, current_index: int) -> int:
        state = self.state
        for gc in self.pipe._goal_checks:
            interval = gc._baton_gc_interval
            max_checks = gc._baton_gc_max_checks
            count_key = f"__baton_gc_stage_{gc.__name__}"
            fired_key = f"__baton_gc_fired_{gc.__name__}"

            stage_count = state._internal.get(count_key, 0) + 1
            state._internal.set(count_key, stage_count)

            fired = state._internal.get(fired_key, 0)
            if stage_count < interval or fired >= max_checks:
                continue

            state._internal.set(count_key, 0)
            state._internal.set(fired_key, fired + 1)

            effective_ai = self.pipe._resolve_ai(gc, self.ai)
            inject_ai = self._wants_ai(gc)
            kw: dict[str, Any] = {"state": state}
            if inject_ai:
                kw["ai"] = effective_ai

            if inspect.iscoroutinefunction(gc):
                result: GoalCheckResult = await gc(**kw)
            else:
                result = await asyncio.to_thread(gc, **kw)

            self.pipe._emit("goal_check", name=gc.__name__, state=state,
                            verdict=result.verdict, note=result.note)
            logger.info("GoalCheck '%s' verdict=%s (check %d/%d): %s",
                        gc.__name__, result.verdict, fired + 1, max_checks, result.note)

            if result.verdict == "adjust" and result.data_updates:
                state.update_data(**result.data_updates)
            elif result.verdict == "rollback":
                target = result.rollback_to or gc._baton_gc_rollback_to
                if not target:
                    logger.warning("GoalCheck '%s' rollback but no rollback_to — ignoring",
                                   gc.__name__)
                    continue
                state._internal.set(count_key, 0)
                return self._graph_rollback(target)

        return current_index

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _wants_ai(self, fn: Callable) -> bool:
        return "ai" in inspect.signature(fn).parameters

    async def _call_with_retry(
        self, fn: Callable, retry: int, retry_delay: float, *args, **kwargs
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retry + 1):
            try:
                if inspect.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < retry:
                    sleep = _retry_sleep(exc, retry_delay * (2 ** attempt))
                    logger.warning("%s failed (attempt %d/%d), retrying in %.1fs — %s",
                                   fn.__name__, attempt + 1, retry + 1, sleep, exc)
                    await asyncio.sleep(sleep)
        raise last_exc  # type: ignore[misc]

    async def _progress_coro(self, step_name: str, interval: float) -> None:
        elapsed = 0.0
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            self.pipe._emit("stage_progress", name=step_name, state=self.state,
                            elapsed_s=elapsed)
            logger.debug("Stage '%s' still running (%.0fs elapsed)", step_name, elapsed)
