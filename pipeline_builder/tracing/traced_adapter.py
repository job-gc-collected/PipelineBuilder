"""TracedAIAdapter — wraps any AIAdapter and records OTel spans for each call."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..ai.base import AIAdapter
    from .tracer import BatonTracer

T = TypeVar("T", bound=BaseModel)


def _provider_name(adapter: "AIAdapter") -> str:
    """Infer provider string from adapter class name."""
    name = type(adapter).__name__.lower()
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    if "openai" in name:
        return "openai"
    return "unknown"


def _model_name(adapter: "AIAdapter") -> str:
    return getattr(adapter, "model", "unknown")


class TracedAIAdapter:
    """Wraps an AIAdapter and records gen_ai spans via BatonTracer.

    Injected by Pipeline._resolve_ai() when a tracer is configured.
    Stage code receives this transparently — it has the same interface as
    any AIAdapter (and AgentAIAdapter).
    """

    def __init__(self, delegate: "AIAdapter", tracer: "BatonTracer") -> None:
        self._delegate = delegate
        self._tracer = tracer
        self._provider = _provider_name(delegate)
        self._model = _model_name(delegate)

    # ------------------------------------------------------------------ #
    # Sync                                                                 #
    # ------------------------------------------------------------------ #

    def run(self, prompt: str, context: dict | None = None, system: str | None = None) -> str:
        with self._tracer.ai_call_span(
            self._provider, self._model, call_type="chat", prompt=prompt
        ):
            return self._delegate.run(prompt, context, system=system)

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        with self._tracer.ai_call_span(
            self._provider, self._model, call_type="structured", prompt=prompt
        ) as span:
            result = self._delegate.run_structured(
                prompt, response_model, context, system=system
            )
            # Record token usage if the underlying adapter returned it
            usage = getattr(result, "_usage", None)
            if usage:
                self._tracer.record_usage(
                    span,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                )
            return result

    # ------------------------------------------------------------------ #
    # Async                                                                #
    # ------------------------------------------------------------------ #

    async def run_async(
        self, prompt: str, context: dict | None = None, system: str | None = None
    ) -> str:
        with self._tracer.ai_call_span(
            self._provider, self._model, call_type="chat", prompt=prompt
        ):
            return await self._delegate.run_async(prompt, context, system=system)

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        with self._tracer.ai_call_span(
            self._provider, self._model, call_type="structured", prompt=prompt
        ) as span:
            result = await self._delegate.run_structured_async(
                prompt, response_model, context, system=system
            )
            return result
