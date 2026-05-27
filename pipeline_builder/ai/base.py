from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AIAdapter(ABC):
    """Provider-agnostic AI interface for baton stages.

    Sync methods are abstract; async methods have default implementations that
    run the sync version in a thread pool.  Override the async methods in
    subclasses for true non-blocking execution (e.g. ClaudeAdapter uses
    AsyncAnthropic).
    """

    @abstractmethod
    def run(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        """Run a prompt and return a text response.

        system: optional system-level instructions (provider-specific behaviour:
                Claude uses the top-level system param; OpenAI prepends a system
                message; mock adapters ignore it).
        """
        ...

    @abstractmethod
    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        """Run a prompt and return a validated Pydantic model.

        system: see run() docstring.
        """
        ...

    async def run_streaming(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens as they arrive.

        Default implementation: runs sync ``run()`` in a thread and yields
        the entire text as a single chunk.  Subclasses should override this
        with true streaming (e.g. using the provider's streaming API) for
        real latency benefits.

        Usage in an async stage::

            async for token in ai.run_streaming("write a summary..."):
                buffer += token
        """
        text = await asyncio.to_thread(self.run, prompt, context, system)
        yield text

    async def run_structured_streaming(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
        on_chunk: Callable[[int, str], None] | None = None,
    ) -> T:
        """Like run_structured() but streams the AI response for progress visibility.

        ``on_chunk(chunk_count, partial_json)`` is called as each JSON delta
        arrives.  The final complete result is returned only when the full
        response has been received and validated.

        Default implementation: delegates to run_structured_async() with no
        streaming.  ClaudeAdapter overrides this to use the streaming API so
        ``on_chunk`` is called with real incremental JSON deltas.

        Usage::

            result = await ai.run_structured_streaming(
                "Analyse this change",
                AnalysisResult,
                on_chunk=lambda n, partial: print(f"[{n} chunks] {partial[:40]}..."),
            )
        """
        result = await self.run_structured_async(prompt, response_model, context, system)
        if on_chunk:
            on_chunk(1, result.model_dump_json())
        return result

    async def run_async(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        """Async variant. Default: runs sync run() in a thread pool."""
        return await asyncio.to_thread(self.run, prompt, context, system)

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        """Async variant. Default: runs sync run_structured() in a thread pool."""
        return await asyncio.to_thread(self.run_structured, prompt, response_model, context, system)

    def _build_messages(self, prompt: str, context: dict | None) -> list[dict]:
        content = prompt
        if context:
            content = f"{prompt}\n\nContext:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        return [{"role": "user", "content": content}]
