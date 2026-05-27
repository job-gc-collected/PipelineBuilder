"""PR Review pipeline — real Claude AI.

Demonstrates:
- 3-level hierarchy: pr → file → issue
- Async AI stages with run_structured_async()
- workers=4 for parallel file review
- Checkpoint with human confirmation

Run from the project root:
    ANTHROPIC_API_KEY=sk-... python examples/pr_review.py
"""
from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel, Field
from pipeline_builder import CheckpointResult, ClaudeAdapter, Pipeline, State


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
async def split_files(pr: PR, state: State, ai: ClaudeAdapter) -> list[File]:
    """AI splits the PR diff into per-file chunks."""
    class FileSplit(BaseModel):
        files: list[File]

    result = await ai.run_structured_async(
        "Split this PR diff into individual files. "
        "Return each file's path and its diff section.",
        response_model=FileSplit,
        context={"pr_title": pr.title, "diff": pr.diff},
    )
    for f in result.files:
        f.pr_title = pr.title
    return result.files


@pipe.stage(reads=["file"], writes=["issue"], workers=4)   # 4 files reviewed in parallel
async def review_file(file: File, state: State, ai: ClaudeAdapter) -> list[Issue]:
    """AI reviews a single file and returns issues found."""
    class FileReview(BaseModel):
        issues: list[Issue]

    result = await ai.run_structured_async(
        "Review this code diff for bugs, security issues, and style problems. "
        "Return only real issues, not nitpicks.  Empty list if nothing found.",
        response_model=FileReview,
        context={"file": file.path, "diff": file.diff},
    )
    for issue in result.issues:
        issue.file_path = file.path
    return result.issues


@pipe.checkpoint(on_reject="review_file", retry_limit=2)
def confirm_post(state: State) -> CheckpointResult:
    issues = state.get_nodes("issue")
    print(f"\n⏸  Review complete — {len(issues)} issue(s) found:")
    for i in issues:
        marker = "🔴" if i.severity == "critical" else "🟡" if i.severity == "warning" else "💡"
        line_info = f":{i.line}" if i.line else ""
        print(f"  {marker} [{i.file_path}{line_info}] {i.message}")

    if not sys.stdin.isatty():
        print("  (non-interactive: auto-confirming)")
        return CheckpointResult(action="confirm")

    ans = input("\nPost review comments? [y/n] ").strip().lower()
    return CheckpointResult(action="confirm" if ans == "y" else "reject")


# ─── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    prs = [
        PR(
            title="Add user login endpoint",
            diff=(
                "--- a/auth.py\n+++ b/auth.py\n"
                "+def login(u, p):\n"
                "+    q = f\"SELECT * FROM users WHERE u='{u}' AND p='{p}'\"\n"
                "+    token = str(random.random())\n"
                "+    return token\n"
                "--- a/config.py\n+++ b/config.py\n"
                "+SECRET_KEY = 'hardcoded-secret-1234'\n"
            ),
        )
    ]

    result = await pipe.run_async(ai=ClaudeAdapter(), pr=prs)

    print(f"\n✅ Done  session={result.session_id}")
    print(f"   {len(result.get_nodes('file'))} files reviewed")
    print(f"   {len(result.get_nodes('issue'))} issues found")
    print(f"   stages: {[r.name + '/' + r.status for r in result.history]}")


if __name__ == "__main__":
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN to run this example.")
        sys.exit(1)
    asyncio.run(main())
