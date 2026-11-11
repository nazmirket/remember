"""
Local tracing instrumentation for Hindsight API.

This module provides tracing for:
- LLM API calls with full prompts/completions following GenAI semantic conventions
- Token usage and model information
- Error tracking and finish reasons

Tracing is conditional and disabled by default. When enabled, spans stay in-process.
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider

    from .config import HindsightConfig, StaticConfigProxy

logger = logging.getLogger(__name__)


def _serialize_for_span(obj: Any) -> str:
    """Serialize an object for span recording, handling Pydantic models."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "model_dump_json"):
        # Pydantic v2 model
        return obj.model_dump_json()
    if hasattr(obj, "json"):
        # Pydantic v1 model
        return obj.json()
    if hasattr(obj, "model_dump"):
        # Pydantic v2 model - convert to dict then json
        return json.dumps(obj.model_dump())
    if hasattr(obj, "dict"):
        # Pydantic v1 model - convert to dict then json
        return json.dumps(obj.dict())
    # Fallback to json.dumps for dicts and other types
    return json.dumps(obj)


# No-op tracer for when tracing is disabled
class NoOpTracer:
    """No-op tracer that provides the same interface as OpenTelemetry Tracer but does nothing."""

    def start_as_current_span(self, name: str, **kwargs):
        """Return a no-op context manager that yields a NoOpSpan."""
        from contextlib import contextmanager

        @contextmanager
        def noop_span_context():
            yield NoOpSpan()

        return noop_span_context()

    def start_span(self, name: str, **kwargs):
        """Return a no-op span."""
        return NoOpSpan()


class NoOpSpan:
    """No-op span that provides the same interface as OpenTelemetry Span but does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op."""
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """No-op."""
        pass

    def end(self, end_time: int | None = None) -> None:
        """No-op."""
        pass


# Global tracer instance
_tracer: trace.Tracer | NoOpTracer = NoOpTracer()
_tracing_enabled: bool = False
# The SDK provider created by initialize_tracing(). Kept so shutdown_tracing()
# can shut it down directly instead of going through the global
# provider, which may be a proxy that never exposes shutdown().
_provider: "TracerProvider | None" = None


# GenAI semantic convention attribute names (based on v1.37 spec)
class GenAIAttributes:
    """GenAI semantic convention attribute names."""

    # Operation and provider
    OPERATION_NAME = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"

    # Model information
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_MODEL = "gen_ai.response.model"

    # Token usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Messages and prompts
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"

    # Response metadata
    FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Error tracking
    ERROR_TYPE = "error.type"


# Provider name mapping (Hindsight internal -> GenAI semantic convention)
PROVIDER_NAME_MAPPING = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "vertexai": "google",
    "groq": "groq",
    "ollama": "ollama",
    "ollama-cloud": "ollama",
    "lmstudio": "lmstudio",
    "openai-codex": "openai",
    "chatgpt": "openai",
    "claude-code": "anthropic",
    "cursor": "cursor",
    "github-copilot": "github",
    "mock": "mock",
}


def bank_id_span_processor() -> "SpanProcessor":
    """A span processor that stamps ``hindsight.bank_id`` on every span started inside a
    bank-scoped operation.

    Per-bank routing (e.g. an OTel collector sending each bank to its own backend project)
    keys on this attribute, and the model-call spans that carry token counts start deep
    inside the operation, so tagging only the operation span left the cost unattributed.
    Built here so the SDK still loads only when tracing is on.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    # Imported here: llm_trace imports this module.
    from .engine.llm_trace import current_trace_context
    from .schema_context import get_current_bank_id

    class BankIdSpanProcessor(SpanProcessor):
        def on_start(self, span: "Span", parent_context: "Context | None" = None) -> None:
            if span.attributes and "hindsight.bank_id" in span.attributes:
                return
            ctx = current_trace_context()
            bank_id = (ctx.bank_id if ctx else None) or get_current_bank_id()
            if bank_id:
                span.set_attribute("hindsight.bank_id", bank_id)

    return BankIdSpanProcessor()


def initialize_tracing(service_name: str, deployment_environment: str = "development") -> None:
    """Initialize local in-process tracing without an external exporter."""
    global _tracer, _tracing_enabled, _provider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    from hindsight_api import __version__

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": __version__,
            "deployment.environment.name": deployment_environment,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(bank_id_span_processor())
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    _tracing_enabled = True
    _provider = provider
    logger.info("Local tracing initialized: service=%s", service_name)


def initialize_tracing_from_config(
    config: "HindsightConfig | StaticConfigProxy", *, default_service_name: str | None = None
) -> bool:
    """Enable local tracing when configured; startup never depends on it."""
    from .config import DEFAULT_OTEL_SERVICE_NAME

    if not config.otel_traces_enabled:
        return False
    try:
        initialize_tracing(
            service_name=config.otel_service_name or default_service_name or DEFAULT_OTEL_SERVICE_NAME,
            deployment_environment=config.otel_deployment_environment,
        )
        create_span_recorder()
        return True
    except Exception as exc:
        logger.warning("Could not initialize local tracing: %s", exc)
        return False


def shutdown_tracing() -> None:
    """
    Shut down the local tracing pipeline. No-op when tracing was never initialized.
    """
    global _tracer, _tracing_enabled, _provider, _span_recorder

    provider = _provider
    if provider is None:
        return

    try:
        provider.shutdown()
    except Exception as e:
        logger.warning(f"Failed to shut down tracing cleanly: {e}")
    finally:
        _provider = None
        _tracing_enabled = False
        _tracer = NoOpTracer()
        if _span_recorder is not None:
            unregister_span_recorder(_span_recorder)
            _span_recorder = None


def get_tracer() -> trace.Tracer | NoOpTracer:
    """
    Get the global tracer instance.

    Returns a no-op tracer if tracing is disabled, so callers don't need to check for None.
    This improves code readability by allowing direct use without null checks.
    """
    return _tracer


@contextmanager
def create_operation_span(operation: str, bank_id: str | None = None):
    """
    Create a parent span for a Hindsight operation (retain, reflect, consolidation, etc.).

    This creates the span hierarchy:
    - hindsight.{operation} (parent)
      - chat {model} (child LLM calls)

    Args:
        operation: Operation name (retain, reflect, consolidation, mental_model_refresh)
        bank_id: Optional bank ID for context

    Returns:
        Span context manager
    """
    if not _tracing_enabled or _tracer is None:
        yield None
        return

    span_name = f"hindsight.{operation}"
    # OpenTelemetry returns a context manager here, not the span itself. Set
    # attributes on the span yielded by that manager so exporters receive them.
    with _tracer.start_as_current_span(span_name) as span:
        span.set_attribute("hindsight.operation", operation)
        if bank_id:
            span.set_attribute("hindsight.bank_id", bank_id)
        yield span


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _tracing_enabled


# Key under which the W3C trace context travels inside a task payload. Named
# like the other worker-only payload passengers (_tenant_id, _api_key_id).
TASK_TRACE_CONTEXT_KEY = "_traceparent"


def inject_task_trace_context(payload: dict[str, Any]) -> None:
    """
    Stamp the current trace context onto a task payload, in place.

    Queued work runs in the worker process, which has no way to know which
    request enqueued it: without this, the API's span for (say) an async retain
    and the worker's ``hindsight.retain`` span are two unrelated traces, and the
    API half contains nothing but the enqueue. Carrying the W3C traceparent in
    the payload lets the worker continue the caller's trace instead.

    No-op when tracing is disabled, so a payload never grows a null key.
    """
    if not _tracing_enabled:
        return
    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        if carrier:
            payload[TASK_TRACE_CONTEXT_KEY] = carrier
    except Exception as e:  # tracing must never break enqueueing
        logger.debug(f"Failed to inject trace context into task payload: {e}", exc_info=True)


def extract_task_trace_context(payload: dict[str, Any]) -> "Context | None":
    """
    Rebuild the enqueueing request's trace context from a task payload.

    Returns an OpenTelemetry ``Context`` to attach around task execution, or
    None when the payload carries no trace context (an internally scheduled
    task, or a payload queued while tracing was off).
    """
    if not _tracing_enabled:
        return None
    carrier = payload.get(TASK_TRACE_CONTEXT_KEY)
    if not isinstance(carrier, dict):
        return None
    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except Exception as e:
        logger.debug(f"Failed to extract trace context from task payload: {e}", exc_info=True)
        return None


# Maximum content length before truncation (to stay within span size limits)
MAX_CONTENT_LENGTH = 100_000  # characters


def _truncate_content(content: str) -> str:
    """Truncate content if too large for span."""
    if len(content) > MAX_CONTENT_LENGTH:
        return content[:MAX_CONTENT_LENGTH] + f"\n\n[TRUNCATED: {len(content) - MAX_CONTENT_LENGTH} chars omitted]"
    return content


class LLMSpanRecorder:
    """
    Records OpenTelemetry spans for LLM calls following GenAI semantic conventions.
    """

    def __init__(self, tracer: trace.Tracer):
        self.tracer = tracer

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        messages: list[dict[str, str]],
        # Text for LLM calls; structured for rerank scores and decision answers.
        response_content: str | dict[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        finish_reason: Optional[str] = None,
        error: Optional[Exception] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        cached_tokens: int = 0,
        thoughts_tokens: Optional[int] = None,
        served_by: Optional[str] = None,
        **_extra: Any,
    ) -> None:
        """
        Record a completed LLM call as a span with GenAI semantic conventions.

        This creates a span AFTER the call completes, using timestamps to
        set the correct start/end times. This approach works better with
        the existing sync metrics recording pattern.

        Args:
            provider: Hindsight provider name
            model: Model name
            scope: Scope identifier (memory, reflect, consolidation, etc.)
            messages: Input messages (chat history)
            response_content: Response text from LLM, or a structured result (rerank scores, decision answers)
            input_tokens: Input token count
            output_tokens: Output token count
            duration: Call duration in seconds
            finish_reason: Reason the model stopped (stop, length, tool_calls, etc.)
            error: Exception if call failed
            tool_calls: List of tool calls made (for function calling)
            cached_tokens: Cached/cache-read prompt tokens, when reported by the provider.
            thoughts_tokens: Reasoning tokens, when reported by the provider. Not part of
                ``output_tokens``, which is visible-only.
            served_by: The upstream a gateway routed the call to. Taken from ``error``
                when the failing call's exception carries it.
            _extra: Tolerated forward-compatible kwargs from other recorders.
        """
        served_by = served_by or getattr(error, "served_by", None)
        try:
            # Map provider name to GenAI semantic convention
            genai_provider = PROVIDER_NAME_MAPPING.get(provider.lower(), provider.lower())

            # Read off the scope, like the call's kind in the request history: embedding
            # and rerank calls are recorded here too, and must not pass as chat. Imported
            # here because llm_trace imports this module.
            from .engine.llm_trace import genai_operation_name

            operation_name = genai_operation_name(scope)

            # Create span name: "hindsight.{scope}" for consistency with parent spans
            # Model info is available in span attributes (gen_ai.request.model)
            if scope:
                span_name = f"hindsight.{scope}"
            else:
                # Fallback to chat {model} if no scope provided
                span_name = f"{operation_name} {model}"

            # Calculate timestamps
            end_time_ns = time.time_ns()  # noqa: TID251
            start_time_ns = end_time_ns - int(duration * 1_000_000_000)

            # Create span with explicit timestamps
            with self.tracer.start_as_current_span(
                span_name,
                start_time=start_time_ns,
                end_on_exit=False,  # We'll set end time manually
            ) as span:
                # Set required attributes
                span.set_attribute(GenAIAttributes.OPERATION_NAME, operation_name)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, genai_provider)
                span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
                span.set_attribute(GenAIAttributes.RESPONSE_MODEL, model)
                span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
                span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)
                if cached_tokens:
                    span.set_attribute("gen_ai.usage.cached_tokens", cached_tokens)
                # Providers pass this alongside cached_tokens; without an explicit
                # parameter it fell into **_extra and never reached the span, so the
                # reasoning half of the billed output was invisible to OTel consumers.
                if thoughts_tokens:
                    span.set_attribute("gen_ai.usage.reasoning_tokens", thoughts_tokens)

                # Add custom attributes for Hindsight context
                span.set_attribute("hindsight.scope", scope)
                span.set_attribute("hindsight.provider.internal", provider)
                # The upstream a gateway (OpenRouter) routed this call to: what lets a
                # failure rate be attributed to one endpoint rather than the model (#5612).
                if served_by:
                    span.set_attribute("hindsight.provider.served_by", served_by)

                # Add tool call information if present
                if tool_calls:
                    span.set_attribute("gen_ai.tool_calls.count", len(tool_calls))
                    # Add tool names as comma-separated list
                    tool_names = [tc.get("name", "") for tc in tool_calls]
                    span.set_attribute("gen_ai.tool_calls.names", ",".join(tool_names))

                # Format messages for GenAI conventions (as JSON)
                input_messages_json = self._format_messages(messages)
                output_messages_json = self._format_output(response_content, finish_reason)

                # Extract system instructions if present
                system_instructions = self._extract_system_instructions(messages)

                # Add event with prompts/completions following v1.37 conventions
                event_attrs = {}
                if input_messages_json:
                    event_attrs[GenAIAttributes.INPUT_MESSAGES] = input_messages_json
                if output_messages_json:
                    event_attrs[GenAIAttributes.OUTPUT_MESSAGES] = output_messages_json
                if system_instructions:
                    event_attrs[GenAIAttributes.SYSTEM_INSTRUCTIONS] = system_instructions
                if finish_reason:
                    event_attrs[GenAIAttributes.FINISH_REASONS] = json.dumps([finish_reason])

                span.add_event(
                    "gen_ai.client.inference.operation.details",
                    attributes=event_attrs,
                )

                # Add individual tool call events with details
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tool_event_attrs = {
                            "tool.name": tc.get("name", ""),
                            "tool.id": tc.get("id", ""),
                            "tool.arguments": json.dumps(tc.get("arguments", {})),
                        }
                        span.add_event(f"gen_ai.tool_call.{i}", attributes=tool_event_attrs)

                # Handle errors
                if error:
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    span.set_attribute(GenAIAttributes.ERROR_TYPE, type(error).__name__)
                    span.record_exception(error)
                else:
                    span.set_status(Status(StatusCode.OK))

                # Set end time
                span.end(end_time=end_time_ns)

        except Exception as e:
            # Don't let tracing errors break LLM calls
            logger.error(f"Failed to record LLM span: {e}", exc_info=True)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Format messages into GenAI semantic convention format (JSON array).

        Returns JSON string representation of message array.
        """
        try:
            formatted = []
            for msg in messages:
                content = msg.get("content", "")
                # Truncate if needed
                if isinstance(content, str):
                    content = _truncate_content(content)

                formatted.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": content,
                    }
                )

            return json.dumps(formatted)
        except Exception as e:
            logger.warning(f"Failed to format input messages: {e}")
            return "[]"

    def _format_output(
        self,
        content: str | dict[str, Any] | None,
        finish_reason: Optional[str],
    ) -> str:
        """Format output message into GenAI semantic convention format."""
        try:
            if content is None:
                return "[]"

            # Truncate if needed. Structured output (rerank scores, decision answers) is
            # serialized first so it is cut like text instead of exported whole.
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            content = _truncate_content(content)

            return json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": content,
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to format output message: {e}")
            return "[]"

    def _extract_system_instructions(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Extract system instructions from messages if present."""
        try:
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return _truncate_content(content)
                    return str(content)
        except Exception as e:
            logger.warning(f"Failed to extract system instructions: {e}")
        return None


class NoOpLLMSpanRecorder:
    """No-op span recorder for when tracing is disabled."""

    def record_llm_call(self, **kwargs) -> None:
        """No-op."""
        pass


class CompositeSpanRecorder:
    """Fans out ``record_llm_call`` to every registered recorder.

    This lets multiple local consumers observe the same LLM calls — e.g. the
    in-process span recorder and the per-bank DB tracer — through the single
    ``record_llm_call`` chokepoint each provider already calls. A failure in one
    recorder never affects the others or the LLM call itself.
    """

    def __init__(self) -> None:
        self._recorders: list[Any] = []

    def register(self, recorder: Any) -> None:
        if recorder not in self._recorders:
            self._recorders.append(recorder)

    def unregister(self, recorder: Any) -> None:
        if recorder in self._recorders:
            self._recorders.remove(recorder)

    def record_llm_call(self, **kwargs: Any) -> None:
        for recorder in self._recorders:
            try:
                recorder.record_llm_call(**kwargs)
            except Exception as e:  # never let one recorder break others
                logger.debug(f"Span recorder {type(recorder).__name__} failed: {e}", exc_info=True)


# Global composite recorder — always present; fans out to whatever is registered.
_composite_recorder = CompositeSpanRecorder()
# Backward-compat reference to the OTel recorder (if created).
_span_recorder: Optional[LLMSpanRecorder] = None


def get_span_recorder() -> CompositeSpanRecorder:
    """Get the global composite span recorder (fans out to all registered recorders)."""
    return _composite_recorder


def register_span_recorder(recorder: Any) -> None:
    """Register an additional GenAI recorder (e.g. the per-bank DB tracer)."""
    _composite_recorder.register(recorder)


def unregister_span_recorder(recorder: Any) -> None:
    """Remove a previously registered recorder."""
    _composite_recorder.unregister(recorder)


def create_span_recorder() -> LLMSpanRecorder:
    """Create and register the OpenTelemetry span recorder."""
    global _span_recorder
    tracer = get_tracer()
    if tracer is None:
        raise RuntimeError("Tracing not initialized. Call initialize_tracing() first.")
    # The None case is raised above; a NoOpTracer presents the same API and is what a
    # tracing-disabled deployment gets.
    _span_recorder = LLMSpanRecorder(cast("Tracer", tracer))
    register_span_recorder(_span_recorder)
    return _span_recorder
