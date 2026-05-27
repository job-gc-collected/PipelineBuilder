"""Multi-agent pipeline with typed state and goal self-correction.

Two agents (researcher + reviewer) collaborate on document analysis.
A goal_check fires every 3 stages to verify alignment with the original task.

Run:
    python examples/multi_agent.py
"""
from __future__ import annotations

import asyncio
import json
from pydantic import BaseModel, Field
from pipeline_builder import Agent, GoalCheckResult, MockAIAdapter, Pipeline, State


# ─── Schemas ─────────────────────────────────────────────────────────────────

class Document(BaseModel):
    title: str
    content: str


class Finding(BaseModel):
    section: str
    insight: str
    confidence: float = 0.8


class Review(BaseModel):
    finding_id: str
    approved: bool
    note: str


# ─── Typed pipeline context ───────────────────────────────────────────────────

class AnalysisCtx(BaseModel):
    """State visible to all stages."""
    task: str = "Analyse the document"
    findings_reviewed: int = 0
    quality_threshold: float = 0.7


# ─── Pipeline ────────────────────────────────────────────────────────────────

pipe = Pipeline(
    "document_analysis",
    hierarchy=["document", "finding", "review"],
    state_schema=AnalysisCtx,
)

# Two agents with different roles and models
researcher = Agent("researcher", ai=MockAIAdapter(), system_prompt=(
    "You are a meticulous research analyst. "
    "Extract specific, evidence-backed insights."
))
reviewer = Agent("reviewer", ai=MockAIAdapter(), system_prompt=(
    "You are a critical reviewer. "
    "Approve only high-confidence, well-supported findings."
))

pipe.add_agent(researcher)
pipe.add_agent(reviewer)


@pipe.stage(reads=["document"], writes=["finding"],
            agent="researcher", workers=2)
async def extract_findings(doc: Document, state: State, ai) -> list[Finding]:
    class Result(BaseModel):
        findings: list[Finding]

    r = await ai.run_structured_async(
        f"Extract key insights from this document. Task: {state.data.task}",
        Result, context={"title": doc.title, "content": doc.content[:500]},
    )
    state.post_message("researcher", f"Found {len(r.findings)} insights in '{doc.title}'",
                       to_agent="reviewer")
    return r.findings


@pipe.stage(reads=["finding"], writes=["review"],
            agent="reviewer", fanout="auto")
async def review_finding(finding: Finding, state: State, ai) -> list[Review]:
    # Read messages from researcher
    msgs = state.get_messages(to_agent="reviewer", from_agent="researcher")

    class Result(BaseModel):
        approved: bool
        note: str

    r = await ai.run_structured_async(
        "Approve or reject this finding. Be critical.",
        Result, context={"finding": finding.model_dump(),
                         "researcher_notes": [m.content for m in msgs]},
    )
    state.update_data(findings_reviewed=state.data.findings_reviewed + 1)
    return [Review(finding_id=finding.section, approved=r.approved, note=r.note)]


@pipe.goal_check(interval=2, max_checks=3)
async def check_quality(state: State, ai) -> GoalCheckResult:
    """Every 2 stages: verify we're extracting high-quality findings."""
    reviews = state.get_nodes("review")
    if not reviews:
        return GoalCheckResult(verdict="continue")

    approved = sum(1 for r in reviews if r.approved)
    rate = approved / len(reviews)

    if rate < state.data.quality_threshold:
        return GoalCheckResult(
            verdict="adjust",
            note=f"Approval rate {rate:.0%} below threshold — lowering confidence bar",
            data_updates={"quality_threshold": state.data.quality_threshold - 0.1},
        )
    return GoalCheckResult(verdict="continue",
                           note=f"Quality OK: {rate:.0%} approved")


# ─── Mock handlers ────────────────────────────────────────────────────────────

def make_mock_researcher():
    def handler(prompt: str, ctx: dict | None) -> str:
        title = (ctx or {}).get("title", "doc")
        return json.dumps({"findings": [
            {"section": f"{title}_intro", "insight": "Key theme identified", "confidence": 0.9},
            {"section": f"{title}_body",  "insight": "Supporting evidence found", "confidence": 0.75},
        ]})
    return MockAIAdapter(handler=handler)


def make_mock_reviewer():
    def handler(prompt: str, ctx: dict | None) -> str:
        conf = (ctx or {}).get("finding", {}).get("confidence", 0.5)
        return json.dumps({"approved": conf >= 0.8, "note": "Confidence check passed" if conf >= 0.8 else "Needs more evidence"})
    return MockAIAdapter(handler=handler)


async def main():
    # Override agents with mock adapters for demo
    pipe._agents["researcher"].ai = make_mock_researcher()
    pipe._agents["reviewer"].ai = make_mock_reviewer()

    docs = [
        Document(title="Q4 Report", content="Revenue increased 23% driven by..."),
        Document(title="Risk Assessment", content="Three key risks identified..."),
    ]

    result = await pipe.run_async(document=docs)

    findings = result.get_nodes("finding")
    reviews  = result.get_nodes("review")
    approved = sum(1 for r in reviews if r.approved)

    print(f"\n✅ Analysis complete")
    print(f"   Documents:   {len(docs)}")
    print(f"   Findings:    {len(findings)}")
    print(f"   Reviews:     {len(reviews)}  ({approved} approved)")
    print(f"   Messages:    {len(result.messages)} inter-agent messages")
    print(f"   Final ctx:   quality_threshold={result.data.quality_threshold:.1f}, "
          f"findings_reviewed={result.data.findings_reviewed}")


if __name__ == "__main__":
    asyncio.run(main())
