"""Built-in services share the engine context."""

import os
from unittest.mock import MagicMock, patch

from remember_api.config import clear_config_cache
from remember_api.engine.memory_engine import MemoryEngine


def _build_engine(**kwargs) -> MemoryEngine:
    """Construct an engine without network, GPU or DB: __init__ never starts the pool."""
    mock_embeddings = MagicMock()
    mock_embeddings.dimension = 384

    with patch.dict(
        os.environ,
        {
            "REMEMBER_API_LLM_PROVIDER": "none",
            "REMEMBER_API_LLM_MODEL": "none",
            "REMEMBER_API_LLM_API_KEY": "test-key",
        },
        clear=False,
    ):
        clear_config_cache()
        engine = MemoryEngine(db_url="postgresql://localhost/remember_test", embeddings=mock_embeddings, **kwargs)

    # Drop the config cache this repopulated from the patched env, so "none"/chunks does
    # not bleed into other tests on this xdist worker.
    clear_config_cache()
    return engine


def test_built_in_services_need_no_external_context() -> None:
    engine = _build_engine()
    assert engine._authenticator is not None
    assert engine._memory_defense is not None
    assert not hasattr(engine, "_ext_ctx")
