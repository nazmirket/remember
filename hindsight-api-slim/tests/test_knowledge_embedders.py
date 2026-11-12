"""Knowledge-bank embedders: each provider's request and response, against a stub of its API, and
the server settings that decide what a knowledge bank can ingest (modalities, opt-outs, text-only).
"""

from dataclasses import replace

import pytest
from aiohttp import web

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.knowledge.embedders import (
    EmbedInput,
    TextEmbedder,
    UnsupportedModalityError,
    create_knowledge_embedder,
)
from tests.aiohttp_stub import stub_server
from tests.test_knowledge_banks import KB, _png, _text_pdf, _upload, fake_embedder, kb  # noqa: F401  (fixtures)

PNG = _png((255, 0, 0))
MP4 = b"\x00\x00\x00\x18ftypmp42"  # sent as bytes, never decoded by a remote provider


def _recorder(reply):
    """A stub API that records each request's path and body, answering ``reply(body)``."""
    seen: list[tuple[str, dict]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        seen.append((request.path, body))
        return web.json_response(reply(body))

    return handler, seen


# --- Voyage ----------------------------------------------------------------------------------------


# --- Cohere ----------------------------------------------------------------------------------------


# --- choosing a provider ---------------------------------------------------------------------------


def _config(**changes) -> HindsightConfig:
    return replace(HindsightConfig.from_env(), **changes)


async def test_the_text_provider_reuses_the_memory_banks_embedder_by_default(memory):
    embedder = create_knowledge_embedder(
        _config(knowledge_embeddings_provider="text", knowledge_text_embeddings=None), memory.embeddings
    )
    assert isinstance(embedder, TextEmbedder) and embedder.modalities == {"text"}
    assert embedder.dimension == memory.embeddings.dimension
    assert len(await embedder.embed_query("red")) == memory.embeddings.dimension
    with pytest.raises(UnsupportedModalityError):
        await embedder.embed_documents([EmbedInput(modality="image", data=PNG, mime="image/png")])


async def test_the_text_provider_can_use_any_memory_bank_provider_of_its_own(memory, monkeypatch):
    """HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_* reads every memory-bank embeddings setting again."""
    calls: list[str] = []

    async def tei(request: web.Request) -> web.StreamResponse:
        calls.append(request.path)
        if request.path == "/info":
            return web.json_response({"model_id": "stub-text"})
        inputs = (await request.json())["inputs"]
        return web.json_response([[1.0, 0.0, 0.0] for _ in (inputs if isinstance(inputs, list) else [inputs])])

    async with stub_server(tei) as base_url:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_PROVIDER", "tei")
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_TEI_URL", base_url)
        config = HindsightConfig.from_env()
        assert config.knowledge_text_embeddings is not None
        assert config.knowledge_text_embeddings.embeddings_provider == "tei"
        embedder = create_knowledge_embedder(replace(config, knowledge_embeddings_provider="text"), memory.embeddings)
        assert isinstance(embedder, TextEmbedder) and embedder.dimension == 0  # started on first use
        vectors = await embedder.embed_documents([EmbedInput(modality="text", text="Keys expire.", title="keys")])
    assert vectors == [[1.0, 0.0, 0.0]] and embedder.dimension == 3 and "/embed" in calls

    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_NOT_A_SETTING", "x")
    with pytest.raises(ValueError, match="Unknown knowledge_text embeddings variable"):
        HindsightConfig.from_env()


# --- what a server ingests -------------------------------------------------------------------------


@pytest.fixture
def disabled_media(monkeypatch):
    def disable(value: str) -> None:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES", value)
        clear_config_cache()

    yield disable
    monkeypatch.delenv("HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES", raising=False)
    clear_config_cache()


def test_an_unknown_disabled_media_type_is_a_startup_error(disabled_media):
    with pytest.raises(ValueError, match="unknown media types pictures"):
        disabled_media("pictures")
        HindsightConfig.from_env()
