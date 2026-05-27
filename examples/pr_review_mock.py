"""PR Review pipeline — mock AI, no API key needed.

Demonstrates:
- 3-level hierarchy: pr → file → issue
- Async stages with structured AI output
- workers=4 for parallel file review
- Checkpoint before posting comments
- MockAIAdapter for deterministic testing

Run from the project root:
    python examples/pr_review_mock.py
"""
from __future__ import annotations

import asyncio
import json
import sys

from pydantic import BaseModel, Field
from pipeline_builder import CheckpointResult, MockAIAdapter, Pipeline, State


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PR(BaseModel):
    title: str
    diff: str


class File(BaseModel):
    pr_title: str = ""
    path: str
    diff: str


class Issue(BaseModel):
    file_path: str
    severity: str = Field(description="critical / warning / suggestion")
    line: int | None = None
    message: str


# ─── Pipeline ─────────────────────────────────────────────────────────────────

pipe = Pipeline("pr_review", hierarchy=["pr", "file", "issue"],
                schemas={"pr": PR, "file": File, "issue": Issue})


@pipe.stage(reads=["pr"], writes=["file"])
async def split_files(pr: PR, state: State, ai: MockAIAdapter) -> list[File]:
    class FileSplit(BaseModel):
        files: list[File]

    result = await ai.run_structured_async("split diff into files", FileSplit,
                                            context={"diff": pr.diff})
    for f in result.files:
        f.pr_title = pr.title
    return result.files


@pipe.stage(reads=["file"], writes=["issue"], workers=4)   # 4 files in parallel
async def review_file(file: File, state: State, ai: MockAIAdapter) -> list[Issue]:
    class FileReview(BaseModel):
        issues: list[Issue]

    result = await ai.run_structured_async("review file for issues", FileReview,
                                            context={"file": file.path, "diff": file.diff})
    for i in result.issues:
        i.file_path = file.path
    return result.issues


@pipe.checkpoint(on_reject="review_file", retry_limit=2)
def confirm_post(state: State) -> CheckpointResult:
    issues = state.get_nodes("issue")
    print(f"\n⏸  {len(issues)} issue(s) found:")
    for i in issues:
        print(f"  [{i.severity}] {i.file_path}:{i.line or '?'} — {i.message}")

    if not sys.stdin.isatty():
        print("  (non-interactive: auto-confirming)")
        return CheckpointResult(action="confirm")

    ans = input("\nPost review comments? [y/n] ").strip().lower()
    return CheckpointResult(action="confirm" if ans == "y" else "reject")


# ─── Mock AI ──────────────────────────────────────────────────────────────────

def _make_mock() -> MockAIAdapter:
    def handler(prompt: str, ctx: dict | None) -> str:
        if "split" in prompt:
            return json.dumps({"files": [
                {"path": "auth.py",   "diff": "+ def login(): ..."},
                {"path": "config.py", "diff": "+ SECRET_KEY = 'hardcoded'"},
            ]})
        if "review" in prompt:
            path = (ctx or {}).get("file", "unknown.py")
            if "auth" in path:
                return json.dumps({"issues": [
                    {"file_path": path, "severity": "critical", "line": 3,
                     "message": "SQL injection: f-string in query"},
                    {"file_path": path, "severity": "warning", "line": 6,
                     "message": "Weak token: random.random() is not cryptographically secure"},
                ]})
            return json.dumps({"issues": [
                {"file_path": path, "severity": "critical", "line": 1,
                 "message": "Hardcoded secret key in source code"},
            ]})
        return "{}"
    return MockAIAdapter(handler=handler)


# ─── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    prs = [PR(title="Add login endpoint", diff="+ def login(u,p): ...\n+ SECRET='abc'")]

    result = await pipe.run_async(ai=_make_mock(), pr=prs)

    print(f"\n✅ Done  session={result.session_id}")
    print(f"   {len(result.get_nodes('file'))} files reviewed")
    print(f"   {len(result.get_nodes('issue'))} issues found")
    print(f"   stages: {[r.name + '/' + r.status for r in result.history]}")


if __name__ == "__main__":
    asyncio.run(main())
