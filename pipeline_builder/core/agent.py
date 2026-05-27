"""
Agent — 有身份的 AI 执行者。

设计：Agent 不驱动流程（流程由 Pipeline 代码控制），只持有 AI 配置。
stage 声明 agent= 参数后，框架注入 AgentAIAdapter，stage 代码透明使用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..ai.base import AIAdapter

T = TypeVar("T", bound=BaseModel)


@dataclass
class Agent:
    """An AI participant with a fixed identity and system prompt.

    Usage::

        researcher = Agent(
            name="researcher",
            ai=ClaudeAdapter(),
            system_prompt="You are a data analyst focused on accuracy.",
        )
        pipe.add_agent(researcher)

        @pipe.stage(reads=["topic"], writes=["finding"], agent="researcher")
        def analyze(topic: Topic, state: State, ai: AIAdapter) -> list[Finding]:
            # ai is automatically the researcher's AgentAIAdapter
            ...
    """
    name: str
    ai: "AIAdapter"
    system_prompt: str = ""


@dataclass
class AgentMessage:
    """A message posted by one agent, optionally directed at another.

    ``to_agent=None`` means broadcast — all agents can read it.

    Usage::

        state.post_message("researcher", "Found 3 issues", to_agent="reviewer")
        msgs = state.get_messages(to_agent="reviewer")
    """
    from_agent: str
    content: str
    to_agent: str | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "content": self.content,
            "to_agent": self.to_agent,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMessage":
        return cls(
            from_agent=d["from_agent"],
            content=d["content"],
            to_agent=d.get("to_agent"),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", 0.0),
        )


class AgentAIAdapter:
    """Wraps any AIAdapter and injects the agent's system_prompt.

    The system_prompt acts as a default; callers can override it by passing
    ``system=`` explicitly to run() / run_structured() / their async variants.
    """

    def __init__(self, delegate: "AIAdapter", system_prompt: str) -> None:
        self._delegate = delegate
        self._system_prompt = system_prompt

    def _effective_system(self, override: str | None) -> str | None:
        return override if override is not None else (self._system_prompt or None)

    # ------------------------------------------------------------------ #
    # Sync                                                                 #
    # ------------------------------------------------------------------ #

    def run(self, prompt: str, context: dict | None = None, system: str | None = None) -> str:
        return self._delegate.run(prompt, context, system=self._effective_system(system))

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        return self._delegate.run_structured(
            prompt, response_model, context, system=self._effective_system(system)
        )

    # ------------------------------------------------------------------ #
    # Async                                                                #
    # ------------------------------------------------------------------ #

    async def run_async(
        self, prompt: str, context: dict | None = None, system: str | None = None
    ) -> str:
        return await self._delegate.run_async(
            prompt, context, system=self._effective_system(system)
        )

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        return await self._delegate.run_structured_async(
            prompt, response_model, context, system=self._effective_system(system)
        )
