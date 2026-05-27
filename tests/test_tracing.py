"""Tests for BatonTracer and TracedAIAdapter.

OTel is an optional dependency.  Tests here use an in-process SpanExporter
so no OTel infrastructure is needed.  All tests skip gracefully when
opentelemetry-sdk is not installed.
"""
import pytest
from pydantic import BaseModel

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from pipeline_builder import BatonTracer, MockAIAdapter, Pipeline, State
from pipeline_builder.tracing.traced_adapter import TracedAIAdapter

pytestmark = pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry-sdk not installed")


class Goal(BaseModel):
    name: str


class Task(BaseModel):
    goal: str
    action: str


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_tracer() -> tuple["BatonTracer", "InMemorySpanExporter"]:
    """Create a fresh tracer+exporter pair without touching the global OTel singleton."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Pass provider directly — avoids the "can only set once" global restriction
    tracer = BatonTracer("test-service", tracer_provider=provider)
    return tracer, exporter


# ------------------------------------------------------------------ #
# BatonTracer — no-op when OTel not installed                         #
# ------------------------------------------------------------------ #

def test_baton_tracer_enabled_when_otel_installed():
    tracer = BatonTracer()
    assert tracer.enabled is True


def test_baton_tracer_noop_span_is_safe():
    """Even without OTel, span context managers must not crash."""
    from pipeline_builder.tracing.tracer import _NoOpSpan
    noop = _NoOpSpan()
    noop.set_attribute("key", "val")
    noop.record_exception(Exception("x"))
    noop.end()
    # Also test via the BatonTracer context managers (OTel installed → real spans)
    tracer, _ = _make_tracer()
    with tracer.pipeline_span("t", "s"):
        pass  # must not crash


# ------------------------------------------------------------------ #
# pipeline_span creates a root span                                   #
# ------------------------------------------------------------------ #

def test_pipeline_span_creates_root_span():
    tracer, exporter = _make_tracer()
    with tracer.pipeline_span("my_pipe", "sess123") as span:
        pass
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "pipeline.my_pipe" in names
    root = next(s for s in spans if s.name == "pipeline.my_pipe")
    attrs = dict(root.attributes)
    assert attrs.get("baton.pipeline.name") == "my_pipe"
    assert attrs.get("baton.session_id") == "sess123"


# ------------------------------------------------------------------ #
# stage_span records stage attributes                                 #
# ------------------------------------------------------------------ #

def test_stage_span_records_name_and_session():
    tracer, exporter = _make_tracer()
    with tracer.stage_span("analyze", "sess1", agent_name="researcher"):
        pass
    spans = exporter.get_finished_spans()
    stage = next(s for s in spans if s.name == "stage.analyze")
    assert stage.attributes["baton.stage.name"] == "analyze"
    assert stage.attributes["baton.agent.name"] == "researcher"


def test_stage_span_records_error_on_exception():
    tracer, exporter = _make_tracer()
    with pytest.raises(ValueError):
        with tracer.stage_span("boom", "sess1"):
            raise ValueError("intentional")
    spans = exporter.get_finished_spans()
    stage = next(s for s in spans if s.name == "stage.boom")
    # OTel error status
    from opentelemetry.trace import StatusCode
    assert stage.status.status_code == StatusCode.ERROR


# ------------------------------------------------------------------ #
# ai_call_span records gen_ai attributes                              #
# ------------------------------------------------------------------ #

def test_ai_call_span_gen_ai_attributes():
    tracer, exporter = _make_tracer()
    with tracer.ai_call_span("anthropic", "claude-opus-4-7", "structured", "hello") as span:
        tracer.record_usage(span, input_tokens=100, output_tokens=50)
    spans = exporter.get_finished_spans()
    ai_span = next(s for s in spans if "gen_ai" in s.name)
    assert ai_span.attributes["gen_ai.system"] == "anthropic"
    assert ai_span.attributes["gen_ai.request.model"] == "claude-opus-4-7"
    assert ai_span.attributes["gen_ai.usage.input_tokens"] == 100
    assert ai_span.attributes["gen_ai.usage.output_tokens"] == 50


# ------------------------------------------------------------------ #
# TracedAIAdapter wraps AI calls in spans                             #
# ------------------------------------------------------------------ #

def test_traced_adapter_creates_span_per_call():
    tracer, exporter = _make_tracer()
    base_ai = MockAIAdapter(handler=lambda p, c: '{"action": "x"}')
    wrapped = TracedAIAdapter(delegate=base_ai, tracer=tracer)

    class R(BaseModel):
        action: str

    wrapped.run_structured("do something", R)
    spans = exporter.get_finished_spans()
    assert any("gen_ai" in s.name for s in spans)


def test_traced_adapter_run_creates_span():
    tracer, exporter = _make_tracer()
    base_ai = MockAIAdapter(handler=lambda p, c: "hello")
    wrapped = TracedAIAdapter(delegate=base_ai, tracer=tracer)
    wrapped.run("say hi")
    spans = exporter.get_finished_spans()
    assert any("gen_ai" in s.name for s in spans)


# ------------------------------------------------------------------ #
# attach_to_pipeline                                                  #
# ------------------------------------------------------------------ #

def test_attach_to_pipeline_creates_stage_spans():
    tracer, exporter = _make_tracer()
    pipe = Pipeline("test", hierarchy=["goal", "task"], tracer=tracer)
    tracer.attach_to_pipeline(pipe)

    @pipe.stage(reads=["goal"], writes=["task"])
    def analyze(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    spans = exporter.get_finished_spans()
    assert any(s.name == "stage.analyze" for s in spans)


def test_attach_to_pipeline_records_duration_ms():
    tracer, exporter = _make_tracer()
    pipe = Pipeline("test", hierarchy=["goal", "task"], tracer=tracer)
    tracer.attach_to_pipeline(pipe)

    @pipe.stage(reads=["goal"], writes=["task"])
    def make(goal: Goal, state: State) -> list[Task]:
        return [Task(goal=goal.name, action="x")]

    pipe.run(goal=[Goal(name="g")])
    spans = exporter.get_finished_spans()
    stage_span = next(s for s in spans if s.name == "stage.make")
    assert "baton.stage.duration_ms" in stage_span.attributes


def test_attach_to_pipeline_records_failed_stage():
    tracer, exporter = _make_tracer()
    pipe = Pipeline("test", hierarchy=["goal", "task"], tracer=tracer)
    tracer.attach_to_pipeline(pipe)

    @pipe.stage(reads=["goal"], writes=["task"])
    def boom(goal: Goal, state: State) -> list[Task]:
        raise ValueError("intentional")

    with pytest.raises(ValueError):
        pipe.run(goal=[Goal(name="g")])

    spans = exporter.get_finished_spans()
    from opentelemetry.trace import StatusCode
    stage_span = next(s for s in spans if s.name == "stage.boom")
    assert stage_span.status.status_code == StatusCode.ERROR


# ------------------------------------------------------------------ #
# Pipeline tracer= param wires TracedAIAdapter automatically          #
# ------------------------------------------------------------------ #

def test_pipeline_tracer_wraps_ai_in_traced_adapter():
    tracer, exporter = _make_tracer()
    pipe = Pipeline("test", hierarchy=["goal", "task"], tracer=tracer)
    received_ai_type = []

    @pipe.stage(reads=["goal"], writes=["task"])
    def go(goal: Goal, state: State, ai) -> list[Task]:
        received_ai_type.append(type(ai).__name__)
        return [Task(goal=goal.name, action="x")]

    pipe.run(ai=MockAIAdapter(), goal=[Goal(name="g")])
    assert received_ai_type == ["TracedAIAdapter"]
