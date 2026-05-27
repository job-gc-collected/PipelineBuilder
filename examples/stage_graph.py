"""Stage graph — parallel analysis with fan-in and DAGSpec.

Demonstrates:
- Two stages running in parallel (bug review + style review)
- Fan-in: synthesize waits for BOTH to complete
- DAGSpec for tool-call dependencies inside a stage
- Timing shows parallel execution (total ≈ max, not sum)

Run:
    python examples/stage_graph.py
"""
from __future__ import annotations

import asyncio
import json
import time
from pydantic import BaseModel
from pipeline_builder import DAGNode, DAGSpec, MockAIAdapter, Pipeline, State


# ─── Schemas ─────────────────────────────────────────────────────────────────

class File(BaseModel):
    path: str
    content: str


class BugIssue(BaseModel):
    line: int
    desc: str


class StyleIssue(BaseModel):
    rule: str
    desc: str


class Comment(BaseModel):
    file: str
    body: str


# ─── Pipeline ────────────────────────────────────────────────────────────────

pipe = Pipeline(
    "parallel_review",
    hierarchy=["file", "bug_issue", "style_issue", "comment"],
)


@pipe.stage(reads=["file"], writes=["bug_issue"])   # ← runs in parallel with find_style
async def find_bugs(file: File, state: State, ai) -> list[BugIssue]:
    await asyncio.sleep(0.1)  # simulates AI latency
    class Result(BaseModel):
        issues: list[BugIssue]
    r = await ai.run_structured_async("Find security bugs", Result,
                                       context={"path": file.path})
    return r.issues


@pipe.stage(reads=["file"], writes=["style_issue"])  # ← runs in parallel with find_bugs
async def find_style(file: File, state: State, ai) -> list[StyleIssue]:
    await asyncio.sleep(0.1)  # simulates AI latency
    class Result(BaseModel):
        issues: list[StyleIssue]
    r = await ai.run_structured_async("Find style violations", Result,
                                       context={"path": file.path})
    return r.issues


@pipe.stage(reads=["bug_issue", "style_issue"],      # ← fan-in: waits for BOTH
            writes=["comment"], fanout="manual")
async def synthesize(bug_issue: list[BugIssue], style_issue: list[StyleIssue],
                     state: State, ai) -> list[Comment]:
    """Runs only after both find_bugs AND find_style have completed."""
    class Result(BaseModel):
        comments: list[Comment]

    r = await ai.run_structured_async(
        "Synthesise bugs and style issues into review comments",
        Result,
        context={"bugs": [b.model_dump() for b in bug_issue],
                 "style": [s.model_dump() for s in style_issue]},
    )
    return r.comments


# ─── DAGSpec demo (intra-stage parallel tool calls) ──────────────────────────

async def demo_dagspec():
    """Show DAGSpec: fetch DDL before running SQL analysis."""
    print("\n── DAGSpec: tool-call dependency graph ──")
    order = []

    dag = DAGSpec()
    dag.add(DAGNode("fetch_ddl",  fn=lambda: (order.append("fetch_ddl"),  "ddl_result")))
    dag.add(DAGNode("fetch_etl",  fn=lambda: (order.append("fetch_etl"),  "etl_result")))
    dag.add(DAGNode("analyze_sql", fn=lambda: (order.append("analyze_sql"), "sql_result"),
                    depends_on=["fetch_ddl"]))   # waits for DDL
    dag.add(DAGNode("analyze_etl", fn=lambda: (order.append("analyze_etl"), "etl_analysis"),
                    depends_on=["fetch_etl"]))   # waits for ETL

    result = await dag.run_async(workers=4)
    print(f"  Execution order: {order}")
    print(f"  Results: {list(result.results.keys())}")
    assert order.index("fetch_ddl") < order.index("analyze_sql"), "DDL must precede SQL"
    print("  ✓ Dependencies respected")


# ─── Mock ─────────────────────────────────────────────────────────────────────

def make_mock():
    def handler(prompt: str, ctx: dict | None) -> str:
        path = (ctx or {}).get("path", "file.py")
        if "security" in prompt.lower() or "bug" in prompt.lower():
            return json.dumps({"issues": [{"line": 42, "desc": f"SQL injection in {path}"}]})
        if "style" in prompt.lower():
            return json.dumps({"issues": [{"rule": "PEP8", "desc": "Line too long"}]})
        if "synthesis" in prompt.lower() or "synthesise" in prompt.lower():
            return json.dumps({"comments": [{"file": path, "body": "1 security issue + 1 style issue"}]})
        return '{"issues": [], "comments": []}'
    return MockAIAdapter(handler=handler)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    files = [
        File(path="auth.py",   content="def login(u,p): execute(f'SELECT * FROM users WHERE user={u}')"),
        File(path="config.py", content="SECRET_KEY = 'hardcoded-bad-practice-here-this-is-way-too-long'"),
        File(path="utils.py",  content="def helper():  pass  # trailing spaces      "),
    ]

    t0 = time.monotonic()
    result = await pipe.run_async(ai=make_mock(), file=files)
    elapsed = time.monotonic() - t0

    bugs    = result.get_nodes("bug_issue")
    styles  = result.get_nodes("style_issue")
    comments = result.get_nodes("comment")

    print(f"\n✅ Parallel review complete in {elapsed:.2f}s")
    print(f"   Files reviewed:  {len(files)}")
    print(f"   Bug issues:      {len(bugs)}")
    print(f"   Style issues:    {len(styles)}")
    print(f"   Comments posted: {len(comments)}")
    print(f"\n   find_bugs + find_style ran in parallel.")
    print(f"   synthesize waited for both (fan-in).")

    await demo_dagspec()


if __name__ == "__main__":
    asyncio.run(main())
