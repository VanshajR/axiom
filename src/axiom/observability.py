"""Tracing and live numbers.

Tracing follows LLM Insight Engine: each model call is an OpenTelemetry span
with the GenAI semantic-convention attributes, exported over OTLP (to Langfuse)
when OTEL_EXPORTER_OTLP_ENDPOINT is set and a no-op otherwise, so providers and
the agent never import a vendor SDK.

Stats are the numbers Axiom will eventually quote about itself: latency
percentiles, which model answered, and how often fallback saved a request.
"""
import os
import time
from collections import Counter, deque
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer = trace.get_tracer("axiom")


def init_tracing(service_name: str = "axiom") -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    headers = dict(
        h.split("=", 1) for h in os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").split(",") if "=" in h
    )
    provider = TracerProvider(resource=Resource({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces", headers=headers)))
    trace.set_tracer_provider(provider)


@contextmanager
def model_call_span(provider: str, model: str):
    """One span per attempt at a model; the router records outcome attributes on it."""
    with tracer.start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        yield span


def record_usage(span, usage: dict, finish_reason: str | None) -> None:
    if usage.get("prompt_tokens") is not None:
        span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
    if usage.get("completion_tokens") is not None:
        span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])


class Stats:
    """In-memory, since restarts are rare and these are rolling numbers anyway."""

    def __init__(self, window: int = 500):
        self.started = time.time()
        self.latencies_ms: deque[float] = deque(maxlen=window)
        self.first_token_ms: deque[float] = deque(maxlen=window)
        self.answered_by: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.tool_calls: Counter[str] = Counter()
        self.requests = 0
        self.rescued_by_fallback = 0
        self.failed = 0

    def summary(self) -> dict:
        def pct(values, q):
            if not values:
                return None
            ordered = sorted(values)
            return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

        return {
            "uptime_s": round(time.time() - self.started),
            "requests": self.requests,
            "failed": self.failed,
            "rescued_by_fallback": self.rescued_by_fallback,
            "latency_ms": {"p50": pct(self.latencies_ms, 0.5), "p95": pct(self.latencies_ms, 0.95)},
            "first_token_ms": {"p50": pct(self.first_token_ms, 0.5), "p95": pct(self.first_token_ms, 0.95)},
            "answered_by": dict(self.answered_by.most_common()),
            "provider_failures": dict(self.failures.most_common()),
            "tool_calls": dict(self.tool_calls.most_common()),
            "mean_tools_per_request": round(sum(self.tool_calls.values()) / self.requests, 2) if self.requests else None,
        }


stats = Stats()
