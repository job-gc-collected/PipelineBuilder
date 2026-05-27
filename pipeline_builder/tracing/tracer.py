"""BatonTracer — thin OpenTelemetry wrapper with no-op fallback.

When opentelemetry-api is installed, creates real spans exported via the
configured OTel SDK (OTLP, Jaeger, LangSmith, etc.).

When not installed (or tracer=None passed to Pipeline), every method is a
no-op and the pipeline runs normally with no tracing overhead.

Span hierarchy::

    pipeline.run  [root]
      ├── stage.<name>
      │   └── gen_ai.chat   (one per AI call inside the stage)
      └── checkpoint.<name>

Attributes follow the OpenTelemetry gen_ai semantic conventions so traces
are readable in LangSmith, Jaeger, Zipkin, Honeycomb, etc.
"""
from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    pass

# gen_ai semantic convention schema URL (OpenTelemetry v1.27+)
_SEMCONV_URL = "https://opentelemetry.io/schemas/1.27.0"


# ─────────────────────────── No-op span / context ────────────────────────────

class _NoOpSpan:
    """Returned when OTel is not installed or tracing is disabled."""
    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exc: Exception) -> None: ...
    def set_status(self, *args, **kwargs) -> None: ...
    def end(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_): ...


_NOOP = _NoOpSpan()


# ─────────────────────────── BatonTracer ────────────────────────────────────

@contextmanager
def _noop_ctx():
    """Fallback context manager used when no tracer is configured."""
    yield None


class BatonTracer:
    """Wraps the OTel tracer.  All methods are safe to call even without OTel.

    Usage::

        tracer = BatonTracer("my-service")
        pipe = Pipeline("my_pipe", hierarchy=[...], tracer=tracer)

    For testing, pass an explicit ``tracer_provider`` to avoid the global
    OTel singleton::

        provider = TracerProvider()
        tracer = BatonTracer("test", tracer_provider=provider)
    """

    def __init__(self, service_name: str = "baton", tracer_provider: Any = None) -> None:
        self._service_name = service_name
        self._enabled = False
        self._tracer = None

        try:
            from opentelemetry import trace
            provider = tracer_provider or trace.get_tracer_provider()
            self._tracer = provider.get_tracer("baton", schema_url=_SEMCONV_URL)
            self._enabled = True
            self._trace_module = trace
        except ImportError:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ #
    # Context managers for pipeline / stage / AI call                     #
    # ------------------------------------------------------------------ #

    @contextmanager
    def pipeline_span(self, pipeline_name: str, session_id: str) -> Generator:
        """Root span for an entire pipeline run."""
        if not self._enabled:
            yield _NOOP
            return
        with self._tracer.start_as_current_span(
            f"pipeline.{pipeline_name}",
            kind=self._trace_module.SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("baton.pipeline.name", pipeline_name)
            span.set_attribute("baton.session_id", session_id)
            yield span

    @contextmanager
    def stage_span(
        self, stage_name: str, session_id: str, agent_name: str | None = None
    ) -> Generator:
        """Span for a single stage execution."""
        if not self._enabled:
            yield _NOOP
            return
        with self._tracer.start_as_current_span(
            f"stage.{stage_name}",
        ) as span:
            span.set_attribute("baton.stage.name", stage_name)
            span.set_attribute("baton.session_id", session_id)
            if agent_name:
                span.set_attribute("baton.agent.name", agent_name)
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(self._trace_module.StatusCode.ERROR, str(exc))
                raise

    @contextmanager
    def checkpoint_span(self, name: str, session_id: str) -> Generator:
        if not self._enabled:
            yield _NOOP
            return
        with self._tracer.start_as_current_span(f"checkpoint.{name}") as span:
            span.set_attribute("baton.checkpoint.name", name)
            span.set_attribute("baton.session_id", session_id)
            yield span

    @contextmanager
    def ai_call_span(
        self,
        provider: str,
        model: str,
        call_type: str = "chat",
        prompt: str = "",
    ) -> Generator:
        """Span for a single AI API call (gen_ai semantic conventions).

        provider: "anthropic" | "openai" | ...
        call_type: "chat" | "structured"
        """
        if not self._enabled:
            yield _NOOP
            return
        op_name = f"gen_ai.{call_type}"
        with self._tracer.start_as_current_span(op_name) as span:
            span.set_attribute("gen_ai.system", provider)
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.operation.name", call_type)
            if prompt:
                span.set_attribute("gen_ai.prompt", prompt[:500])  # truncated
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(self._trace_module.StatusCode.ERROR, str(exc))
                raise

    # ------------------------------------------------------------------ #
    # High-level: attach to a Pipeline via event hooks                    #
    # ------------------------------------------------------------------ #

    def attach_to_pipeline(self, pipe: Any) -> None:
        """Register stage-level OTel spans via event hooks.

        Call once after creating the pipeline::

            tracer = BatonTracer("my-service")
            pipe = Pipeline(..., tracer=tracer)
            tracer.attach_to_pipeline(pipe)

        This sets up stage spans as OTel children of the current active span
        (i.e., children of the pipeline root span).  AI call spans recorded by
        TracedAIAdapter will become grandchildren automatically via OTel context.
        """
        if not self._enabled:
            return

        import threading
        _active: dict[str, Any] = {}
        _lock = threading.Lock()

        @pipe.on("stage_start")
        def _on_start(name: str, state: Any, **kw: Any) -> None:
            span = self._tracer.start_span(f"stage.{name}")
            with _lock:
                _active[name] = span

        @pipe.on("stage_complete")
        def _on_complete(name: str, state: Any, duration_ms: float | None, **kw: Any) -> None:
            with _lock:
                span = _active.pop(name, None)
            if span is not None:
                if duration_ms is not None:
                    span.set_attribute("baton.stage.duration_ms", duration_ms)
                span.set_status(self._trace_module.StatusCode.OK)
                span.end()

        @pipe.on("stage_fail")
        def _on_fail(name: str, state: Any, error: Exception, **kw: Any) -> None:
            with _lock:
                span = _active.pop(name, None)
            if span is not None:
                span.record_exception(error)
                span.set_status(self._trace_module.StatusCode.ERROR, str(error))
                span.end()

    # ------------------------------------------------------------------ #
    # Convenience: record token usage after an AI call                    #
    # ------------------------------------------------------------------ #

    def record_usage(
        self,
        span: Any,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if not self._enabled or isinstance(span, _NoOpSpan):
            return
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
