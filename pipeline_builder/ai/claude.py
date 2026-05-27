from __future__ import annotations

import os
from typing import AsyncGenerator, TypeVar

from pydantic import BaseModel

from .base import AIAdapter

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-4-7"

# Prompt cache TTL on Anthropic's side is 5 minutes.
# Marked "ephemeral" means: cache this content block for the session.
_CACHE_BLOCK_TYPE = {"type": "ephemeral"}


class ClaudeAdapter(AIAdapter):
    """Anthropic Claude adapter with prompt caching, streaming, and connection pooling.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ANTHROPIC_API_KEY or
        ANTHROPIC_AUTH_TOKEN environment variables.
    model:
        Model ID (default claude-opus-4-7).
    cache_system:
        When True (default), system prompts passed to run/run_structured are
        wrapped with ``cache_control: ephemeral`` so repeated calls within 5
        minutes reuse the cached KV state.  Cuts input tokens by 70-90% for
        multi-stage pipelines that pass the same knowledge-base system prompt.
    max_connections:
        httpx connection pool size for the underlying HTTP client.  Increase
        when using workers=N > 10 to avoid connection starvation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cache_system: bool = True,
        max_connections: int = 10,
    ) -> None:
        try:
            import anthropic
            import httpx
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        extra_headers: dict[str, str] = {}
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                extra_headers[k.strip()] = v.strip()

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        client_kw: dict = dict(
            api_key=resolved_key,
            default_headers=extra_headers or None,
        )
        self._client = anthropic.Anthropic(
            **client_kw, http_client=httpx.Client(limits=limits)
        )
        self._async_client = anthropic.AsyncAnthropic(
            **client_kw, http_client=httpx.AsyncClient(limits=limits)
        )
        self.model = model
        self._cache_system = cache_system

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _system_param(self, system: str | None) -> list | None:
        """Convert a system string to an Anthropic content-block list.

        When caching is enabled the block receives ``cache_control`` so
        Anthropic caches the encoded KV state for up to 5 minutes.
        """
        if not system:
            return None
        block: dict = {"type": "text", "text": system}
        if self._cache_system:
            block["cache_control"] = _CACHE_BLOCK_TYPE
        return [block]

    # ------------------------------------------------------------------ #
    # Sync                                                                 #
    # ------------------------------------------------------------------ #

    def run(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        messages = self._build_messages(prompt, context)
        kw: dict = dict(model=self.model, max_tokens=4096, messages=messages)
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks
        response = self._client.messages.create(**kw)
        return response.content[0].text

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        messages = self._build_messages(prompt, context)
        tools = [
            {
                "name": "respond",
                "description": "Return a structured response matching the schema exactly.",
                "input_schema": response_model.model_json_schema(),
            }
        ]
        kw: dict = dict(
            model=self.model,
            max_tokens=4096,
            tools=tools,
            tool_choice={"type": "tool", "name": "respond"},
            messages=messages,
        )
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks
        response = self._client.messages.create(**kw)
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return response_model.model_validate(tool_block.input)

    # ------------------------------------------------------------------ #
    # Async                                                                #
    # ------------------------------------------------------------------ #

    async def run_async(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        messages = self._build_messages(prompt, context)
        kw: dict = dict(model=self.model, max_tokens=4096, messages=messages)
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks
        response = await self._async_client.messages.create(**kw)
        return response.content[0].text

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        messages = self._build_messages(prompt, context)
        tools = [
            {
                "name": "respond",
                "description": "Return a structured response matching the schema exactly.",
                "input_schema": response_model.model_json_schema(),
            }
        ]
        kw: dict = dict(
            model=self.model,
            max_tokens=4096,
            tools=tools,
            tool_choice={"type": "tool", "name": "respond"},
            messages=messages,
        )
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks
        response = await self._async_client.messages.create(**kw)
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return response_model.model_validate(tool_block.input)

    async def run_structured_streaming(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
        on_chunk: "Callable[[int, str], None] | None" = None,
    ) -> T:
        """Run a structured prompt using the streaming API.

        As each ``input_json_delta`` event arrives, ``on_chunk(n, partial_json)``
        is called so callers can surface progress (e.g. emit a stage_progress
        event or update a UI).  The final validated model is returned only once
        the complete JSON has been received.

        This has the same latency as non-streaming run_structured() for short
        responses, but enables heartbeat events for long-running AI calls.
        """
        from typing import Callable as _C
        messages = self._build_messages(prompt, context)
        tools = [
            {
                "name": "respond",
                "description": "Return a structured response matching the schema exactly.",
                "input_schema": response_model.model_json_schema(),
            }
        ]
        kw: dict = dict(
            model=self.model,
            max_tokens=4096,
            tools=tools,
            tool_choice={"type": "tool", "name": "respond"},
            messages=messages,
        )
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks

        accumulated = ""
        chunk_count = 0

        async with self._async_client.messages.stream(**kw) as stream:
            async for event in stream:
                # input_json_delta events carry partial JSON for tool_use blocks
                if (
                    getattr(event, "type", None) == "content_block_delta"
                    and getattr(getattr(event, "delta", None), "type", None) == "input_json_delta"
                ):
                    partial = event.delta.partial_json
                    accumulated += partial
                    chunk_count += 1
                    if on_chunk:
                        on_chunk(chunk_count, accumulated)

        # Parse the complete accumulated JSON
        return response_model.model_validate_json(accumulated)

    async def run_streaming(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as they arrive.

        Usage in an async stage::

            async for token in ai.run_streaming("write a report...", system="..."):
                print(token, end="", flush=True)
        """
        messages = self._build_messages(prompt, context)
        kw: dict = dict(model=self.model, max_tokens=4096, messages=messages)
        sys_blocks = self._system_param(system)
        if sys_blocks:
            kw["system"] = sys_blocks

        async with self._async_client.messages.stream(**kw) as stream:
            async for text in stream.text_stream:
                yield text
