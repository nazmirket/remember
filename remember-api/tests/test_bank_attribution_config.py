"""
Config wiring for per-bank attribution and the configurable OpenRouter rerank URL.

- REMEMBER_API_LLM_SEND_BANK_AS_USER (default off, opt-in bool)
- REMEMBER_API_RERANKER_SEND_BANK_AS_HEADER (default off, opt-in bool)
- REMEMBER_API_RERANKER_OPENROUTER_BASE_URL (default = previously hardcoded URL)

Deterministic, no network.
"""

import os
from dataclasses import fields

from remember_api.config import RememberConfig
from remember_api.engine.bank_attribution import reranker_bank_attribution_headers
from remember_api.schema_context import _current_bank_id


def _restore_env(saved: dict[str, str | None]) -> None:
    from remember_api.config import clear_config_cache

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    clear_config_cache()


def _make_full_config(**overrides):
    """Build a complete RememberConfig from type-based defaults plus overrides.

    Mirrors the helper in test_reranker_timeouts.py so we can exercise the
    factory without touching real env/config.
    """
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


class TestSendBankAsUserConfig:
    def test_default_is_false(self):
        from remember_api.config import clear_config_cache

        saved = {"REMEMBER_API_LLM_SEND_BANK_AS_USER": os.environ.get("REMEMBER_API_LLM_SEND_BANK_AS_USER")}
        os.environ.pop("REMEMBER_API_LLM_SEND_BANK_AS_USER", None)
        clear_config_cache()
        try:
            assert RememberConfig.from_env().llm_send_bank_as_user is False
        finally:
            _restore_env(saved)

    def test_true_enables(self):
        from remember_api.config import clear_config_cache

        saved = {"REMEMBER_API_LLM_SEND_BANK_AS_USER": os.environ.get("REMEMBER_API_LLM_SEND_BANK_AS_USER")}
        os.environ["REMEMBER_API_LLM_SEND_BANK_AS_USER"] = "true"
        clear_config_cache()
        try:
            assert RememberConfig.from_env().llm_send_bank_as_user is True
        finally:
            _restore_env(saved)

    def test_reranker_bank_header_default_false_and_true(self):
        from remember_api.config import clear_config_cache

        key = "REMEMBER_API_RERANKER_SEND_BANK_AS_HEADER"
        saved = {key: os.environ.get(key)}
        os.environ.pop(key, None)
        clear_config_cache()
        try:
            assert RememberConfig.from_env().reranker_send_bank_as_header is False
            token = _current_bank_id.set("bank-disabled")
            try:
                assert reranker_bank_attribution_headers() == {}
            finally:
                _current_bank_id.reset(token)
            os.environ[key] = "true"
            clear_config_cache()
            assert RememberConfig.from_env().reranker_send_bank_as_header is True
            assert reranker_bank_attribution_headers() == {}
            token = _current_bank_id.set("bank-configured")
            try:
                assert reranker_bank_attribution_headers() == {"X-Remember-Bank-Id": "bank-configured"}
            finally:
                _current_bank_id.reset(token)
        finally:
            _restore_env(saved)
