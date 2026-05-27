"""Quickstart — 2-level pipeline with parallel review and human checkpoint.

Run with mock AI (no API key needed):
    python examples/quickstart.py

Run with real Claude:
    ANTHROPIC_API_KEY=... python examples/quickstart.py --real
"""
from __future__ import annotations

import asyncio
import json
import sys
from pydantic import BaseModel
from pipeline_builder import CheckpointResult, MockAIAdapter, Pipeline, State


class PR(BaseModel):
    title: str
    diff: str


class Issue(BaseModel):
    file: str
    severity: str   # "critical" | "warning" | "suggestion"
    message: str


# ─── Pipeline definition ──────────────────────────────────────────────────────

pipe = Pipeline("pr_review", hierarchy=["pr", "file", "issue"])


@pipe.stage(reads=["pr"], writes=["file"], fanout="manual")
async def split_files(pr: list[PR], state: State, ai) -> list:
    class FileSplit(BaseModel):
        files: list[dict]

    results = []
    for p in pr:
        r = await ai.run_structured_async("Split diff into files", FileSplit,
                                           context={"diff": p.diff[:500]})
        results.extend(r.files)
    return results


@pipe.stage(reads=["file"], writes=["issue"], workers=4)  # 4 files reviewed in parallel
async def review_file(file: dict, state: State, ai) -> list[Issue]:
    class FileReview(BaseModel):
        issues: list[Issue]

    r = await ai.run_structured_async("Review this file for issues", FileReview,
                                       context={"file": file})
    return r.issues


@pipe.checkpoint(on_reject="review_file", retry_limit=2)
def confirm_post(state: State) -> CheckpointResult:
    issues = state.get_nodes("issue")
    critical = [i for i in issues if i.severity == "critical"]
    print(f"\n⏸  {len(issues)} issues found ({len(critical)} critical)")
    for i in issues[:5]:
        print(f"  [{i.severity}] {i.file}: {i.message}")
    if len(issues) > 5:
        print(f"  ... and {len(issues) - 5} more")
    if not sys.stdin.isatty():
        print("  (non-interactive: auto-confirming)")
        return CheckpointResult(action="confirm")
    ans = input("\nPost review comments? [y/n] ").strip().lower()
    return CheckpointResult(action="confirm" if ans == "y" else "reject")


# ─── Mock AI for demo ─────────────────────────────────────────────────────────

def make_mock():
    def handler(prompt: str, ctx: dict | None) -> str:
        if "split" in prompt.lower():
            return json.dumps({"files": [
                {"path": "auth.py",   "diff": "+ def login(): ..."},
                {"path": "config.py", "diff": "+ SECRET = 'hardcoded'"},
            ]})
        if "review" in prompt.lower():
            path = (ctx or {}).get("file", {}).get("path", "unknown.py")
            if "config" in path:
                return json.dumps({"issues": [
                    {"file": path, "severity": "critical",
                     "message": "Hardcoded secret detected"},
                ]})
            return json.dumps({"issues": [
                {"file": path, "severity": "warning",
                 "message": "Missing input validation"},
            ]})
        return "{}"
    return MockAIAdapter(handler=handler)


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main(use_real_ai: bool = False):
    if use_real_ai:
        from pipeline_builder import ClaudeAdapter
        ai = ClaudeAdapter()
    else:
        ai = make_mock()

    prs = [PR(title="feat: add auth endpoint", diff="+ def login(): pass\n+ SECRET='abc'")]

    result = await pipe.run_async(ai=ai, pr=prs)
    issues = result.get_nodes("issue")
    print(f"\n✅ Done — {len(issues)} issue(s) recorded")


if __name__ == "__main__":
    asyncio.run(main(use_real_ai="--real" in sys.argv))
