"""The fork's built-in inference choices and local diagnostics."""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.cross_encoder import _create_cross_encoder_backend
from hindsight_api.engine.embeddings import _create_embeddings
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.knowledge.embedders import create_knowledge_embedder


@pytest.mark.parametrize("provider", ["gemini", "vertexai", "groq", "bedrock", "litellm", "cursor", "github-copilot"])
def test_removed_builtin_llm_providers_are_rejected(provider):
    with pytest.raises(ValueError, match="Invalid LLM provider"):
        LLMProvider(provider=provider, api_key="test", base_url="http://127.0.0.1:1234/v1", model="test")


def test_openai_compatible_adapter_rejects_removed_provider():
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    with pytest.raises(ValueError, match="only supports"):
        OpenAICompatibleLLM(provider="groq", api_key="test", base_url="", model="test")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "lmstudio", "mock"])
def test_retained_llm_provider_is_constructible(provider):
    config = LLMProvider(provider=provider, api_key="test", base_url="http://127.0.0.1:1234/v1", model="test")
    assert config.provider == provider


@pytest.mark.parametrize("provider", ["cohere", "google", "litellm", "hindsight"])
def test_removed_embedding_providers_are_rejected(provider):
    config = SimpleNamespace(embeddings_provider=provider, embeddings_query_prefix="", embeddings_passage_prefix="")
    with pytest.raises(ValueError, match="Unsupported embeddings provider"):
        _create_embeddings(config)


@pytest.mark.parametrize("provider", ["cohere", "google", "siliconflow", "litellm", "hindsight"])
def test_removed_reranker_providers_are_rejected(provider):
    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        _create_cross_encoder_backend(SimpleNamespace(provider=provider))


@pytest.mark.parametrize("provider", ["gemini", "vertex", "voyage", "cohere"])
def test_removed_multimodal_embedding_providers_are_rejected(provider):
    config = SimpleNamespace(
        knowledge_embeddings_provider=provider, knowledge_embeddings_dimensions=384, knowledge_embeddings_batch_size=1
    )
    with pytest.raises(ValueError, match="Unknown HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER"):
        create_knowledge_embedder(config, None)


def test_tracing_does_not_add_remote_exporter(monkeypatch):
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    from hindsight_api import tracing

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.invalid")
    tracing.initialize_tracing(service_name="local-test")
    try:
        processors = tracing._provider._active_span_processor._span_processors
        assert processors
        assert all(not isinstance(processor, (BatchSpanProcessor, SimpleSpanProcessor)) for processor in processors)
    finally:
        tracing.shutdown_tracing()
