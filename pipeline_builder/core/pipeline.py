from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from .checkpoint import checkpoint as cp_decorator
from .executor import _noop_ctx, _retry_sleep  # noqa: F401  (re-exported for back-compat)
from .goal_check import GoalCheckResult, goal_check as gc_decorator
from .group import can_reach, collect_stage_groups, infer_group_deps
from .loop import loop as loop_decorator
from .router import router as router_decorator
from .stage import stage as stage_decorator
from .state import State

if TYPE_CHECKING:
    from ..ai.base import AIAdapter
    from ..persistence.base import StorageBackend
    from ..tracing.tracer import BatonTracer
    from .agent import Agent

logger = logging.getLogger("baton")

_VALID_EVENTS = frozenset(
    {"stage_start", "stage_complete", "stage_fail", "checkpoint", "loop",
     "goal_check", "stage_progress"}
)


class Pipeline:
    def __init__(
        self,
        name: str,
        hierarchy: list[str],
        schemas: dict[str, type[BaseModel]] | None = None,
        state_dir: str | None = None,           # shorthand → FileBackend
        storage: "StorageBackend | None" = None, # explicit backend (takes priority)
        persist_every: int = 1,
        heartbeat_interval: int = 0,
        tracer: "BatonTracer | None" = None,
        state_schema: type[BaseModel] | None = None,  # typed pipeline context
    ) -> None:
        self.name = name
        self.hierarchy = hierarchy
        self.schemas = schemas or {}
        self.persist_every = persist_every
        self.heartbeat_interval = heartbeat_interval
        self._tracer = tracer
        self._state_schema = state_schema
        self._steps: list[Callable] = []
        self._agents: dict[str, "Agent"] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._hb_stop: threading.Event | None = None
        # Goal checks: list of registered check functions (separate from steps)
        self._goal_checks: list[Callable] = []

        # Resolve storage backend
        if storage is not None:
            self._backend: "StorageBackend | None" = storage
        elif state_dir is not None:
            from ..persistence.file_backend import FileBackend
            self._backend = FileBackend(state_dir)
        else:
            self._backend = None

        # Keep state_dir for backward-compat attribute access
        self.state_dir = state_dir

    # ------------------------------------------------------------------ #
    # Step registration                                                    #
    # ------------------------------------------------------------------ #

    def stage(
        self,
        reads: list[str] | None = None,
        writes: list[str] | None = None,
        fanout: str = "auto",
        workers: int = 1,
        retry: int = 0,
        retry_delay: float = 1.0,
        agent: str | None = None,
        timeout: float | None = None,
        progress_interval: float = 0.0,
        depends_on: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            wrapped = stage_decorator(
                reads=reads, writes=writes, fanout=fanout,
                workers=workers, retry=retry, retry_delay=retry_delay,
                agent=agent, timeout=timeout,
                progress_interval=progress_interval,
                depends_on=depends_on,
                aliases=aliases,
            )(fn)
            self._steps.append(wrapped)
            return wrapped
        return decorator

    def checkpoint(
        self,
        on_reject: str | None = None,
        retry_limit: int = 3,
        targets: list[str] | None = None,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            wrapped = cp_decorator(
                on_reject=on_reject, retry_limit=retry_limit, targets=targets
            )(fn)
            self._steps.append(wrapped)
            return wrapped
        return decorator

    def router(self, targets: list[str] | None = None) -> Callable:
        def decorator(fn: Callable) -> Callable:
            wrapped = router_decorator(targets=targets)(fn)
            self._steps.append(wrapped)
            return wrapped
        return decorator

    def loop(
        self,
        rollback_to: str,
        exit_on: list[str] | str | None = None,
        max_rounds: int = 3,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            wrapped = loop_decorator(rollback_to=rollback_to, exit_on=exit_on, max_rounds=max_rounds)(fn)
            self._steps.append(wrapped)
            return wrapped
        return decorator

    # ------------------------------------------------------------------ #
    # Agent registry                                                       #
    # ------------------------------------------------------------------ #

    def add_agent(self, agent: "Agent") -> None:
        """Register an Agent so stages can reference it by name."""
        self._agents[agent.name] = agent

    def _resolve_ai(self, step: Callable, default_ai: "AIAdapter | None") -> "AIAdapter | None":
        """Return the AI adapter for a stage, wrapping in Agent/TracedAI adapters as needed."""
        ai = default_ai

        # Agent wrapping
        agent_name = getattr(step, "_baton_agent", None)
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(
                    f"Stage '{step.__name__}' declares agent='{agent_name}' "
                    f"but no such agent was registered on the pipeline. "
                    f"Call pipe.add_agent(Agent('{agent_name}', ...)) first."
                )
            from .agent import AgentAIAdapter
            agent = self._agents[agent_name]
            ai = AgentAIAdapter(delegate=agent.ai, system_prompt=agent.system_prompt)

        # Tracer wrapping (outermost layer so it records the final prompt+system)
        if ai is not None and self._tracer is not None:
            from ..tracing.traced_adapter import TracedAIAdapter
            ai = TracedAIAdapter(delegate=ai, tracer=self._tracer)

        return ai

    # ------------------------------------------------------------------ #
    # Event hooks                                                          #
    # ------------------------------------------------------------------ #

    def on(self, event: str, handler: Callable | None = None) -> Callable:
        """Register an event handler, usable as a decorator or direct call.

        Events: stage_start | stage_complete | stage_fail | checkpoint | loop

        Handler signatures:
          stage_start(name, state)
          stage_complete(name, state, duration_ms)
          stage_fail(name, state, error)
          checkpoint(name, state, action)
          loop(name, state, verdict, round)

        Unknown kwargs are silently ignored — handlers may accept **kw.

        Usage (decorator)::

            @pipe.on("stage_complete")
            def log(name, state, duration_ms, **kw):
                print(f"{name} took {duration_ms:.0f}ms")

        Usage (direct)::

            pipe.on("stage_fail", lambda name, state, error, **kw: ...)
        """
        if event not in _VALID_EVENTS:
            raise ValueError(f"Unknown event '{event}'. Valid events: {sorted(_VALID_EVENTS)}")

        def _register(fn: Callable) -> Callable:
            self._event_handlers.setdefault(event, []).append(fn)
            return fn

        if handler is not None:
            return _register(handler)
        return _register  # used as decorator

    def _emit(self, event: str, **kwargs: Any) -> None:
        for handler in self._event_handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Event handler for '%s' raised: %s", event, exc)

    # ------------------------------------------------------------------ #
    # Static validation                                                    #
    # ------------------------------------------------------------------ #

    def goal_check(
        self,
        interval: int = 3,
        rollback_to: str | None = None,
        max_checks: int = 10,
    ) -> Callable:
        """Register a periodic goal-check that fires every ``interval`` stages.

        The decorated function receives ``state`` and optionally ``ai``, and must
        return a :class:`GoalCheckResult`.  The check is independent of the step
        list — it fires as a post-stage hook rather than in the linear sequence.

        Usage::

            @pipe.goal_check(interval=3, rollback_to="extract_probes")
            def check_on_track(state: State, ai) -> GoalCheckResult:
                return ai.run_structured(
                    "Still aligned with original goal?",
                    GoalCheckResult,
                    context={"goal": state.data.original_goal, ...},
                )
        """
        def decorator(fn: Callable) -> Callable:
            wrapped = gc_decorator(
                interval=interval, rollback_to=rollback_to, max_checks=max_checks
            )(fn)
            self._goal_checks.append(wrapped)
            return wrapped
        return decorator

    def _validate(self) -> None:
        """Static dependency check: fail before run, not mid-run."""
        # In the graph model, any hierarchy level can be provided as initial
        # input to run().  We still check that reads/writes use declared levels.
        hierarchy_set = set(self.hierarchy)

        for step in self._steps:
            btype = getattr(step, "_baton_type", None)
            if btype != "stage":
                continue

            for r in step._baton_reads:
                if r not in hierarchy_set:
                    raise ValueError(
                        f"Stage '{step.__name__}' reads '{r}' "
                        f"which is not in hierarchy {self.hierarchy}"
                    )
            for w in step._baton_writes:
                if w not in hierarchy_set:
                    raise ValueError(
                        f"Stage '{step.__name__}' writes '{w}' "
                        f"which is not in hierarchy {self.hierarchy}"
                    )

        step_names = {s.__name__ for s in self._steps}
        for step in self._steps:
            btype = getattr(step, "_baton_type", None)

            if btype == "router":
                for target in step._baton_targets:
                    if target not in step_names:
                        raise ValueError(
                            f"Router '{step.__name__}' declares target '{target}' "
                            f"but no step with that name exists in the pipeline"
                        )

            if btype == "checkpoint":
                for target in getattr(step, "_baton_targets", []):
                    if target not in step_names:
                        raise ValueError(
                            f"Checkpoint '{step.__name__}' declares target '{target}' "
                            f"but no step with that name exists in the pipeline"
                        )

            if btype == "loop":
                target = step._baton_rollback_to
                if target not in step_names:
                    raise ValueError(
                        f"Loop '{step.__name__}' rollback_to '{target}' "
                        f"but no step with that name exists in the pipeline"
                    )

        for gc in self._goal_checks:
            target = gc._baton_gc_rollback_to
            if target is not None and target not in step_names:
                raise ValueError(
                    f"GoalCheck '{gc.__name__}' rollback_to '{target}' "
                    f"but no step with that name exists in the pipeline"
                )

        # Check explicit depends_on references are valid, and detect cycles
        for step in self._steps:
            if getattr(step, "_baton_type", None) != "stage":
                continue
            for dep in getattr(step, "_baton_depends_on", []):
                if dep not in step_names:
                    raise ValueError(
                        f"Stage '{step.__name__}' depends_on '{dep}' "
                        f"but no stage with that name exists in the pipeline"
                    )

        # Warn if genuinely parallel stages write to the same level.
        # Uses can_reach() from group.py so the check is consistent with execution.
        #
        # Stages declared as targets of the same router or checkpoint-route are
        # mutually exclusive — only one runs, so they will never write concurrently.
        # Build that exclusion set first to suppress false-positive warnings.
        import warnings
        exclusive_pairs: set[frozenset] = set()
        for step in self._steps:
            btype = getattr(step, "_baton_type", None)
            targets: list[str] = []
            if btype == "router":
                targets = step._baton_targets
            elif btype == "checkpoint":
                targets = getattr(step, "_baton_targets", [])
            for i_t, t_a in enumerate(targets):
                for t_b in targets[i_t + 1:]:
                    exclusive_pairs.add(frozenset({t_a, t_b}))

        for group in collect_stage_groups(self._steps):
            group_deps = infer_group_deps(group)
            level_writers: dict[str, list[str]] = {}
            for s in group:
                for lv in s._baton_writes:
                    level_writers.setdefault(lv, []).append(s.__name__)
            warned: set[frozenset] = set()
            for lv, names in level_writers.items():
                if len(names) > 1:
                    for i_a, a in enumerate(names):
                        for b in names[i_a + 1:]:
                            pair = frozenset({a, b})
                            if (
                                pair not in warned
                                and pair not in exclusive_pairs
                                and not (
                                    can_reach(group_deps, a, b)
                                    or can_reach(group_deps, b, a)
                                )
                            ):
                                warnings.warn(
                                    f"[baton] Stages '{a}' and '{b}' both write level '{lv}' "
                                    "without a dependency between them — their results will be "
                                    "merged (extend_nodes). Add depends_on to serialize if "
                                    "that is not intended.",
                                    stacklevel=3,
                                )
                                warned.add(pair)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save_state(self, state: State) -> None:
        if self._backend is None:
            return
        self._backend.save(state.session_id, state.to_dict(self.schemas))

    def _load_state(self, session_id: str) -> State | None:
        if self._backend is None:
            return None
        data = self._backend.load(session_id)
        return (
            State.from_dict(data, self.schemas, state_schema=self._state_schema)
            if data else None
        )

    # ------------------------------------------------------------------ #
    # Pipeline versioning                                                  #
    # ------------------------------------------------------------------ #

    def _pipeline_fingerprint(self) -> str:
        """Compute a stable 16-char hash of the pipeline's structural identity."""
        from .versioning import compute_fingerprint
        return compute_fingerprint(self.name, self.hierarchy, self._steps)

    def _stage_aliases(self) -> dict[str, str]:
        """Return a mapping of {alias_name: canonical_name} for all stages."""
        result: dict[str, str] = {}
        for s in self._steps:
            if getattr(s, "_baton_type", None) == "stage":
                for alias in getattr(s, "_baton_aliases", []):
                    result[alias] = s.__name__
        return result

    def can_resume(self, session_id: str) -> tuple[bool, str]:
        """Check without loading state whether a session is compatible.

        Returns ``(can_resume, reason)`` — safe to call before ``run()``.

        Usage::

            ok, reason = pipe.can_resume("abc123")
            if not ok:
                print(f"Cannot resume: {reason}")
        """
        if self._backend is None:
            return False, "No storage backend configured"
        data = self._backend.load(session_id)
        if data is None:
            return False, f"Session '{session_id}' not found"

        # Support both new ("internal") and old ("artifacts") storage layout
        internal = {**data.get("artifacts", {}), **data.get("internal", {})}
        stored_fp = internal.get("__pb_pipeline_fingerprint", "")
        current_fp = self._pipeline_fingerprint()
        if stored_fp == current_fp:
            return True, "Pipeline fingerprint matches"

        current_stage_names = {
            s.__name__ for s in self._steps if getattr(s, "_baton_type", None) == "stage"
        }
        stored_hierarchy = internal.get("__pb_stored_hierarchy")

        from .versioning import check_resume_compatibility, CompatibilityError
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_resume_compatibility(
                    session_id,
                    data.get("completed_stages", []),
                    stored_fp, current_fp,
                    current_stage_names,
                    self._stage_aliases(),
                    self.hierarchy,
                    stored_hierarchy,
                )
            return True, f"Pipeline changed but compatible (fingerprint {stored_fp!r} → {current_fp!r})"
        except CompatibilityError as exc:
            return False, str(exc)

    def _init_session_versioning(self, state: State) -> None:
        """Store fingerprint and hierarchy in a new session's artifacts."""
        state._internal.set("__pb_pipeline_fingerprint", self._pipeline_fingerprint())
        state._internal.set("__pb_stored_hierarchy", self.hierarchy)

    def _check_resume_versioning(self, state: State) -> None:
        """On resume: compare fingerprints and update completed_stages if needed."""
        from .versioning import check_resume_compatibility, CompatibilityError
        stored_fp = state._internal.get("__pb_pipeline_fingerprint", "")
        current_fp = self._pipeline_fingerprint()
        if stored_fp == current_fp:
            return  # identical — nothing to do

        current_stage_names = {
            s.__name__ for s in self._steps if getattr(s, "_baton_type", None) == "stage"
        }
        stored_hierarchy = state._internal.get("__pb_stored_hierarchy")

        updated = check_resume_compatibility(
            state.session_id,
            state._completed_stages,
            stored_fp, current_fp,
            current_stage_names,
            self._stage_aliases(),
            self.hierarchy,
            stored_hierarchy,
        )
        # Apply remapped completed stages and update stored fingerprint
        state._completed_stages = updated
        state._internal.set("__pb_pipeline_fingerprint", current_fp)
        state._internal.set("__pb_stored_hierarchy", self.hierarchy)

    def _save_snapshot_to_backend(self, state: State, stage_name: str) -> None:
        """After snapshot_before(), persist it to the backend so rollback survives crashes."""
        if self._backend is None:
            return
        snap = state._snapshots.get(stage_name)
        if not snap:
            return
        # Nodes contain Pydantic model instances — serialize to JSON-safe dicts.
        nodes_serialized: dict[str, list] = {}
        for level, nodes in snap["nodes"].items():
            if level in self.schemas:
                nodes_serialized[level] = [n.model_dump(mode="json") for n in nodes]
        # Embed typed state data in the artifacts dict under a reserved key.
        artifacts = dict(snap["artifacts"])
        if "data" in snap:
            artifacts["__pb_typed_state_data__"] = snap["data"].model_dump(mode="json")
        self._backend.save_snapshot(
            state.session_id, stage_name, nodes_serialized, artifacts
        )

    def _ensure_snapshot_loaded(self, state: State, stage_name: str) -> None:
        """Before restore_to(), populate in-memory snapshot from backend if missing."""
        if stage_name in state._snapshots:
            return
        if self._backend is None:
            return
        result = self._backend.load_snapshot(state.session_id, stage_name)
        if result is None:
            return
        nodes_serialized, artifacts_with_data = result
        # Deserialize back to Pydantic model instances.
        nodes: dict[str, list] = {}
        for level, nodes_data in nodes_serialized.items():
            schema = self.schemas.get(level)
            if schema:
                nodes[level] = [schema.model_validate(d) for d in nodes_data]
        # Extract typed state data from the reserved key.
        artifacts = {k: v for k, v in artifacts_with_data.items()
                     if k != "__pb_typed_state_data__"}
        snap: dict = {"nodes": nodes, "artifacts": artifacts}
        data_raw = artifacts_with_data.get("__pb_typed_state_data__")
        if data_raw is not None and self._state_schema:
            try:
                snap["data"] = self._state_schema.model_validate(data_raw)
            except Exception:
                pass  # If schema changed, omit — restore_to will use current _data
        state._snapshots[stage_name] = snap
        logger.debug("Loaded snapshot for '%s' from backend", stage_name)

    # ------------------------------------------------------------------ #
    # Goal checks                                                          #
    # ------------------------------------------------------------------ #

    # Heartbeat                                                            #
    # ------------------------------------------------------------------ #

    def _start_heartbeat(self, state: State) -> None:
        if not self.heartbeat_interval or not self._backend:
            return
        self._hb_stop = threading.Event()

        def _beat() -> None:
            while not self._hb_stop.wait(self.heartbeat_interval):  # type: ignore[union-attr]
                self._save_state(state)
                logger.debug("Heartbeat: state saved (session=%s)", state.session_id)

        threading.Thread(target=_beat, daemon=True, name="baton-heartbeat").start()

    def _stop_heartbeat(self) -> None:
        if self._hb_stop is not None:
            self._hb_stop.set()
            self._hb_stop = None

    # ------------------------------------------------------------------ #
    # Sync run                                                             #
    # ------------------------------------------------------------------ #

    def run(
        self,
        session_id: str | None = None,
        ai: "AIAdapter | None" = None,
        artifacts: dict[str, Any] | None = None,
        **input_nodes: Any,
    ) -> State:
        """Sync entry point.  Delegates to :meth:`run_async` via ``asyncio.run()``.

        If called from within a running event loop (e.g. Jupyter notebooks),
        raises ``RuntimeError`` — use ``await pipe.run_async(...)`` instead.
        """
        try:
            return asyncio.run(
                self.run_async(
                    session_id=session_id,
                    ai=ai,
                    artifacts=artifacts,
                    **input_nodes,
                )
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "cannot run" in msg or "already running" in msg or "event loop" in msg:
                raise RuntimeError(
                    "pipe.run() cannot be called from within a running async context "
                    "(e.g. Jupyter notebooks).  Use 'await pipe.run_async()' instead."
                ) from exc
            raise

    # ------------------------------------------------------------------ #
    # Async run                                                            #
    # ------------------------------------------------------------------ #

    async def run_async(
        self,
        session_id: str | None = None,
        ai: "AIAdapter | None" = None,
        artifacts: dict[str, Any] | None = None,
        **input_nodes: Any,
    ) -> State:
        """Async entry point.  Creates a :class:`PipelineExecutor` and runs it."""
        self._validate()
        state = self._init_or_load_state(session_id, artifacts, input_nodes)
        from .executor import PipelineExecutor
        return await PipelineExecutor(self, state, ai).execute()

    def _init_or_load_state(
        self,
        session_id: str | None,
        artifacts: dict[str, Any] | None,
        input_nodes: dict[str, Any],
    ) -> State:
        """Load an existing session or create a fresh one."""
        state: State | None = None
        if session_id:
            state = self._load_state(session_id)
            if state:
                self._check_resume_versioning(state)
                completed = state._completed_stages
                logger.info(
                    "Resuming session %s — skipping %d completed stage(s): %s",
                    session_id, len(completed), completed,
                )

        if state is None:
            state = State(session_id=session_id, state_schema=self._state_schema)
            if artifacts:
                for k, v in artifacts.items():
                    state.artifacts.set(k, v)
            for level, nodes in input_nodes.items():
                state.set_nodes(level, nodes if isinstance(nodes, list) else [nodes])
            self._init_session_versioning(state)

        return state

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Stage-group utilities — thin wrappers that bind self._steps        #
    # The underlying pure logic lives in baton.core.group                #
    # ------------------------------------------------------------------ #

    def _collect_stage_groups(self) -> list[list[Callable]]:
        """Convenience wrapper: split self._steps into stage groups."""
        return collect_stage_groups(self._steps)

    def _infer_group_deps(self, group: list[Callable]) -> dict[str, list[str]]:
        """Convenience wrapper: infer deps for a group of steps."""
        return infer_group_deps(group)

    # ------------------------------------------------------------------ #
    # Visualization                                                        #
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        from .viz import ascii_diagram
        print(ascii_diagram(self))

    def to_mermaid(self) -> str:
        from .viz import mermaid_diagram
        return mermaid_diagram(self)
