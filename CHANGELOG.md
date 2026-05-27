# Changelog

All notable changes to baton are documented here.
Format: [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-22

First stable release.

### Architecture

- **Async-first execution**: `pipe.run()` now delegates to `pipe.run_async()` via
  `asyncio.run()`.  All stages, retries, timeouts, and progress heartbeats are
  async-native.  Sync stages run transparently via `asyncio.to_thread()`.
- **PipelineExecutor**: execution logic extracted from `Pipeline` into a dedicated
  `PipelineExecutor` class (`baton/core/executor.py`).  `Pipeline` is now a thin
  registration + configuration object.
- **InternalStore**: baton's own bookkeeping (`__baton_*` keys) moved to a private
  `State._internal` store, completely separate from user-visible `state.artifacts`.

### New features

- **Stage graph** (`depends_on`, inter-stage DAG): stages between two barriers
  run as a dependency DAG.  Dependencies inferred from `reads/writes`; override
  with `@pipe.stage(..., depends_on=["other_stage"])`.
- **Typed pipeline state** (`state_schema`): declare a Pydantic model for the
  pipeline context.  `state.data.field` is typed, IDE-complete, and persisted
  with sessions.
- **Goal check** (`@pipe.goal_check`): periodic AI self-evaluation.  Fires every
  N completed stages; can `continue`, `adjust` state.data, or `rollback`.
- **Multi-agent** (`Agent`, `AgentMessage`): register agents with roles and system
  prompts.  Stages declare `agent=` to receive the agent's AI adapter.  Agents
  communicate via `state.post_message` / `state.get_messages`.
- **Pipeline versioning** (`aliases`, `can_resume`): stage renames survive
  in-flight sessions via `aliases=["old_name"]`.  `pipe.can_resume(session_id)`
  classifies changes as safe or breaking before resuming.
- **SQLiteBackend**: zero-dependency SQLite persistence with snapshot storage,
  `list_sessions()`, and atomic writes.
- **Prompt caching** (`ClaudeAdapter`): system prompts wrapped with
  `cache_control: ephemeral` by default.
- **`run_structured_streaming`**: stream `input_json_delta` events from Claude
  for progress visibility on long structured calls.
- **`run_streaming`**: async generator for token-level text streaming.
- **Rate-limit aware retry**: honours `Retry-After` header on HTTP 429 responses.
- **Stage-level timeout**: `@pipe.stage(..., timeout=60.0)` raises `TimeoutError`
  after the given seconds.
- **Progress heartbeat**: `@pipe.stage(..., progress_interval=10.0)` emits
  `stage_progress` events while a stage runs.
- **OTel tracing** (`BatonTracer`): optional `opentelemetry-sdk` integration.
  Stage spans and AI call spans with `gen_ai.*` semantic conventions.
- **Checkpoint routing**: `CheckpointResult(action="route", target="stage")` jumps
  to a named stage.
- **`@pipe.checkpoint(targets=[...])`**: static validation of routing targets.

### Breaking changes from pre-1.0 snapshots

- `pipe.run()` now calls `asyncio.run(pipe.run_async(...))`.  If called from an
  already-running event loop, raises `RuntimeError` with a helpful message.
- `state.artifacts` no longer contains `__baton_*` internal keys.  Internal state
  lives in `state._internal` (not user-accessible).
- `to_dict()` serialization format: internal keys now live under `"internal"` key
  (backwards-compatible load path for old sessions that stored them in `"artifacts"`).

---

## [0.1.0] — initial prototype

- `Pipeline`, `@stage`, `@checkpoint`, `@loop`, `@router`
- `DAGSpec` / `DAGNode` for intra-stage parallelism
- `MockAIAdapter`, `ClaudeAdapter`, `OpenAIAdapter`
- File-based state persistence
- `pipe.show()` / `pipe.to_mermaid()` visualization
