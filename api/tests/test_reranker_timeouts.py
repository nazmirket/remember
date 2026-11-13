"""
Per-provider HTTP timeout wiring for remote rerankers.

Covers issue #1807: each HTTP-based reranker provider must accept a configurable
timeout via env var so workloads with large batches don't hit the previously
hardcoded 60s ceiling.
"""

from dataclasses import fields
from unittest.mock import patch

import pytest

from remember_api.config import RememberConfig
from remember_api.engine.cross_encoder import create_cross_encoder_from_env


def _make_config(**overrides) -> RememberConfig:
    """Build a RememberConfig with field-type-appropriate zero values plus overrides."""
    defaults: dict = {}
    for f in fields(RememberConfig):
        if f.type == "str":
            defaults[f.name] = ""
        elif f.type == "str | None":
            defaults[f.name] = None
        elif f.type == "int":
            defaults[f.name] = 0
        elif f.type == "int | None":
            defaults[f.name] = None
        elif f.type == "float":
            defaults[f.name] = 0.0
        elif f.type == "float | None":
            defaults[f.name] = None
        elif f.type == "bool":
            defaults[f.name] = False
        elif str(f.type).startswith("list["):
            defaults[f.name] = []
        else:
            defaults[f.name] = None
    defaults.update(overrides)
    return RememberConfig(**defaults)


@pytest.mark.parametrize(
    "provider, extra, timeout_field, attr_path",
    [
        (
            "typesafe",
            {"reranker_typesafe_api_key": "k"},
            "reranker_typesafe_timeout",
            ("timeout",),
        ),
    ],
)
def test_factory_threads_per_provider_timeout(provider, extra, timeout_field, attr_path):
    """Env-configured timeout reaches the provider instance (or its inner HTTP client)."""
    custom_timeout = 300.0
    config = _make_config(
        reranker_provider=provider,
        **{timeout_field: custom_timeout},
        **extra,
    )
    with patch("remember_api.config.get_config", return_value=config):
        encoder = create_cross_encoder_from_env()

    obj = encoder
    for part in attr_path:
        obj = getattr(obj, part)
    assert obj == custom_timeout


def test_default_timeouts_preserve_60s_behavior():
    """Unset env keeps the previously hardcoded 60.0s default for HTTP providers."""
    config = RememberConfig.from_env()
    assert config.reranker_typesafe_timeout == 60.0
