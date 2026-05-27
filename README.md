# Baton

**Code schedules. AI executes. State is truth.**

Most AI workflow frameworks let the AI decide what to do next. That works for demos. In production, it means unpredictable execution paths, hard-to-test pipelines, and human review bolted on as an afterthought.

Baton flips this: **the scheduler is deterministic Python code** and **AI only ever executes one assigned task**. Human checkpoints, goal self-correction, and structured decomposition are first-class — not callbacks.

---

## Install

```bash
pip install pipeline-builder                         # core (pydantic only)
pip install "pipeline-builder[anthropic]"            # + Anthropic Claude
pip install "pipeline-builder[openai]"              # + OpenAI
pip install "pipeline-builder[otel]"               # + OpenTelemetry tracing
pip install "pipeline-builder[anthropic,otel]"     # bundle
```

---

## Quick start

```python
import asyncio
from pydantic import BaseModel
from baton import Pipeline, State, CheckpointResult, ClaudeAdapter

class PR(BaseModel):
    title: str
    diff: str

class Issue(BaseModel):
    file: str
    severity: str
    message: str

pipe = Pipeline("pr_review", hierarchy=["pr", "issue"])

@pipe.stage(reads=["pr"], writes=["issue"], workers=4)   # 4 PRs reviewed in parallel
async def review(pr: PR, state: State, ai: ClaudeAdapter) -> list[Issue]:
    class Result(BaseModel):
        issues: list[Issue]
    return (await ai.run_structured_async("Review this diff", Result,
                                          context={"diff": pr.diff})).issues

@pipe.checkpoint(on_reject="review")                      # human confirms before posting
def confirm(state: State) -> CheckpointResult:
    issues = state.get_nodes("issue")
    print(f"{len(issues)} issues found. Post? (y/n)")
    return CheckpointResult(action="confirm" if input() == "y" else "reject")

result = asyncio.run(pipe.run_async(
    ai=ClaudeAdapter(),
    pr=[PR(title="feat: auth", diff="...")],
))
```

---

## Core concepts

### Pipeline hierarchy

Every pipeline processes a typed tree of nodes. You define the levels:

```python
pipe = Pipeline("explore", hierarchy=["requirement", "change", "probe"])
```

### Stage — code schedules, AI executes

```python
@pipe.stage(reads=["change"], writes=["probe"],
            workers=4,           # parallel nodes
            timeout=120.0,       # seconds
            agent="researcher",  # optional role assignment
            depends_on=["fetch_ddl"])  # explicit cross-level dependency
async def extract_probes(change: Change, state: State, ai) -> list[Probe]:
    ...
```

- `fanout="auto"` (default): called once per input node
- `fanout="manual"`: called once with all nodes via `reads`-named kwargs
- Stage output is collected and stored in the `writes` level

### Checkpoint — human review as first-class state

```python
@pipe.checkpoint(on_reject="review", retry_limit=3)
def confirm(state: State) -> CheckpointResult:
    # inspect state, ask a human, return confirm / reject / route
    return CheckpointResult(action="confirm")
    # return CheckpointResult(action="route", target="fast_path")
```

`reject` rolls back to `on_reject` stage with full graph-aware propagation — all downstream stages re-run.

### Loop — iterative refinement

```python
@pipe.loop(rollback_to="generate_draft", exit_on=["approved"], max_rounds=5)
def quality_gate(state: State) -> str:
    score = evaluate(state.get_nodes("draft")[0])
    return "approved" if score > 0.9 else "retry"
```

### Router — conditional branching

```python
@pipe.router(targets=["fast_path", "full_path"])
def decide(state: State) -> str:
    return "fast_path" if state.data.skip_eligible else "full_path"
```

Non-selected targets are automatically skipped.

### Goal check — AI self-correction

```python
@pipe.goal_check(interval=3, rollback_to="extract_probes", max_checks=5)
async def check_alignment(state: State, ai) -> GoalCheckResult:
    return await ai.run_structured_async(
        "Are we still on track with the original goal?",
        GoalCheckResult,
        context={"goal": state.data.original_goal,
                 "completed": [r.name for r in state.history]},
    )
```

Fires every `interval` completed stages. Supports `"continue"`, `"adjust"` (patch `state.data`), or `"rollback"`.

---

## Typed pipeline state

Avoid the untyped-dict escape hatch:

```python
class ExploreCtx(BaseModel):
    mode: str = "explore_only"
    mental_model: MentalModel | None = None
    report_url: str | None = None

pipe = Pipeline("explore", hierarchy=[...], state_schema=ExploreCtx)

@pipe.stage(...)
def build_model(change, state: State, ai):
    state.data.mental_model = ai.run_structured(...)  # typed, IDE-complete
    state.update_data(mode="techplan")                # thread-safe batch write
```

State is serialized with sessions and survives crash/resume.

---

## Multi-agent

```python
researcher = Agent("researcher", ai=ClaudeAdapter(),
                   system_prompt="You are a data analyst. Be precise.")
reviewer   = Agent("reviewer",   ai=ClaudeAdapter(model="claude-haiku-4-5"),
                   system_prompt="You are a code reviewer. Be concise.")

pipe.add_agent(researcher)
pipe.add_agent(reviewer)

@pipe.stage(reads=["change"], writes=["finding"], agent="researcher")
async def analyze(change, state, ai):
    result = await ai.run_structured_async(...)
    state.post_message("researcher", f"Found {len(result)} issues", to_agent="reviewer")
    return result

@pipe.stage(reads=["finding"], agent="reviewer", fanout="manual")
async def review(finding, state, ai):
    msgs = state.get_messages(to_agent="reviewer")
    ...
```

Each stage receives the agent's adapter with its `system_prompt` pre-injected.

---

## Parallel stage groups (stage graph)

Stages between two barriers run as a DAG — dependencies inferred from `reads/writes`:

```python
@pipe.stage(reads=["file"], writes=["bug_issue"])   # ← runs in parallel
def find_bugs(file, state, ai): ...

@pipe.stage(reads=["file"], writes=["style_issue"]) # ← runs in parallel
def find_style(file, state, ai): ...

@pipe.stage(reads=["bug_issue", "style_issue"],     # ← waits for both
            writes=["comment"], fanout="manual")
def synthesize(bug_issue, style_issue, state, ai): ...
```

Add explicit cross-level dependencies with `depends_on=["stage_name"]`.

---

## Persistence & crash recovery

```python
from baton import SQLiteBackend

pipe = Pipeline("explore",
                hierarchy=[...],
                storage=SQLiteBackend("./runs.db"),   # zero extra deps
                heartbeat_interval=30)                # save every 30s

# Crash-safe: snapshots persisted, rollback survives restart
result = pipe.run(session_id="abc123")   # resume after crash
```

Or file-based (default):

```python
pipe = Pipeline("explore", hierarchy=[...], state_dir="./sessions")
```

---

## Pipeline versioning

Rename a stage without breaking in-flight sessions:

```python
@pipe.stage(reads=["prd"], writes=["requirement"],
            aliases=["parse_doc"])   # old name → new name mapping
def parse_prd(prd, state, ai): ...
```

Check compatibility before resuming:

```python
ok, reason = pipe.can_resume("abc123")
if not ok:
    print(f"Cannot resume: {reason}")
# "Stage 'old_name' was completed but no longer exists. Add aliases=['old_name']..."
```

---

## Observability

### Event hooks

```python
@pipe.on("stage_complete")
def log_perf(name, state, duration_ms, **kw):
    metrics.histogram("stage.duration", duration_ms, tags={"stage": name})

@pipe.on("stage_fail")
def alert(name, state, error, **kw):
    sentry.capture_exception(error)
```

Events: `stage_start`, `stage_complete`, `stage_fail`, `checkpoint`, `loop`, `goal_check`, `stage_progress`.

### OpenTelemetry

```python
from baton import BatonTracer

tracer = BatonTracer("my-service")
pipe = Pipeline("explore", hierarchy=[...], tracer=tracer)
tracer.attach_to_pipeline(pipe)   # stage spans + AI call spans (gen_ai.* semantic conventions)
```

Compatible with LangSmith, Jaeger, Honeycomb, OTLP.

### Progress heartbeat

```python
@pipe.stage(reads=["change"], writes=["probe"], progress_interval=10.0)
async def analyze(change, state, ai):
    # `stage_progress` event fires every 10s while this stage runs
    result = await ai.run_structured_async(...)
    return result
```

---

## Testing

```python
from baton import MockAIAdapter

def mock_handler(prompt: str, ctx: dict | None) -> str:
    if "requirements" in prompt:
        return json.dumps({"requirements": [...]})
    return "{}"

result = asyncio.run(pipe.run_async(
    ai=MockAIAdapter(handler=mock_handler),
    prd=[PRD(url="...", text="...")]
))
```

No API key needed. `MockAIAdapter` is a first-class citizen.

---

## DAGSpec — intra-stage parallelism

For complex tool-call dependencies inside a single stage:

```python
from baton import DAGSpec, DAGNode

@pipe.stage(reads=["probe"], writes=["probe"], fanout="manual")
async def execute_probes(probe, state, ai):
    dag = DAGSpec()
    dag.add(DAGNode("fetch_ddl", fn=lambda: fetch_ddl(table)))
    dag.add(DAGNode("analyze",   fn=lambda: analyze_probe(probe),
                                 depends_on=["fetch_ddl"]))
    result = await dag.run_async(workers=4)
    return [update_probe(p, result) for p in probe]
```

---

## vs LangGraph / ReAct

| | **Baton** | **LangGraph** | **ReAct** |
|---|---|---|---|
| Who decides next step? | Code (deterministic) | Code (graph edges) | AI (unpredictable) |
| Human confirmation | First-class `@checkpoint` | External callback | Not supported |
| Structured output | Built-in `run_structured` | Manual | Manual |
| Hierarchy + fan-out | Configurable N levels | Manual state design | No |
| Parallel execution | Stage graph (DAG) + `workers=N` | Manual (Send API) | No |
| Goal self-correction | `@goal_check` (built-in) | No | No |
| Typed pipeline state | `state_schema=MyModel` | TypedDict | No |
| Persistence | File / SQLite backends | External checkpointers | No |
| Versioning | `aliases=`, `can_resume()` | No | No |
| Testing | `MockAIAdapter` (first-class) | Mock HTTP layer | No |

---

## License

MIT
