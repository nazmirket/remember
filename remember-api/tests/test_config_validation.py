"""
Tests for configuration validation.

Verifies that config validation catches invalid parameter combinations.
"""

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from remember_api.config import clear_config_cache

    # Save original environment values
    env_vars_to_save = [
        "REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS",
        "REMEMBER_API_CONSOLIDATION_MAX_COMPLETION_TOKENS",
        "REMEMBER_API_RETAIN_CHUNK_SIZE",
        "REMEMBER_API_RETAIN_STRUCTURED_CHUNK_SIZE",
        "REMEMBER_API_LLM_PROVIDER",
        "REMEMBER_API_LLM_MODEL",
        "REMEMBER_API_LLM_REASONING_EFFORT",
        "REMEMBER_API_LLM_BEDROCK_SERVICE_TIER",
        "REMEMBER_API_LLM_GEMINI_SERVICE_TIER",
        "REMEMBER_API_SEMANTIC_MIN_SIMILARITY",
        "REMEMBER_API_GRAPH_SEED_MIN_SIMILARITY",
        "REMEMBER_API_TEMPORAL_SEMANTIC_MIN_SIMILARITY",
        "REMEMBER_API_SEMANTIC_LINK_MIN_SIMILARITY",
        "REMEMBER_API_CONSOLIDATION_DEDUP_THRESHOLD",
        "REMEMBER_API_DATABASE_URL",
        "REMEMBER_API_MIGRATION_DATABASE_URL",
    ]

    # Save original values
    original_values = {}
    for key in env_vars_to_save:
        original_values[key] = os.environ.get(key)

    clear_config_cache()

    yield

    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_retain_max_completion_tokens_must_be_greater_than_chunk_size():
    """Test that RETAIN_MAX_COMPLETION_TOKENS > RETAIN_CHUNK_SIZE validation works."""
    from remember_api.config import RememberConfig

    # Set invalid config: max_completion_tokens <= chunk_size
    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "1000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "2000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        RememberConfig.from_env()

    error_message = str(exc_info.value)

    # Verify error message contains helpful information
    assert "REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "1000" in error_message
    assert "REMEMBER_API_RETAIN_CHUNK_SIZE" in error_message
    assert "2000" in error_message
    assert "must be greater than" in error_message
    assert "You have two options to fix this:" in error_message
    assert "Increase REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "Use a model that supports" in error_message


def test_retain_max_completion_tokens_equal_to_chunk_size_fails():
    """Test that RETAIN_MAX_COMPLETION_TOKENS == RETAIN_CHUNK_SIZE also fails."""
    from remember_api.config import RememberConfig

    # Set invalid config: max_completion_tokens == chunk_size
    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "3000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        RememberConfig.from_env()

    error_message = str(exc_info.value)
    assert "must be greater than" in error_message


def test_valid_retain_config_succeeds():
    """Test that valid config with max_completion_tokens > chunk_size works."""
    from remember_api.config import RememberConfig

    # Set valid config: max_completion_tokens > chunk_size
    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    # Should not raise
    config = RememberConfig.from_env()
    assert config.retain_max_completion_tokens == 64000
    assert config.retain_chunk_size == 3000
    assert config.retain_structured_chunk_size is None


def test_retain_structured_chunk_size_reads_from_env():
    """Structured JSONL/conversation units can have an explicit character cap."""
    from remember_api.config import RememberConfig

    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_RETAIN_STRUCTURED_CHUNK_SIZE"] = "9000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    config = RememberConfig.from_env()
    assert config.retain_structured_chunk_size == 9000


def test_fail_on_extraction_errors_defaults_to_false(monkeypatch):
    """Silent-success behavior is preserved by default (issue #2700)."""
    from remember_api.config import ENV_FAIL_ON_EXTRACTION_ERRORS, RememberConfig

    monkeypatch.delenv(ENV_FAIL_ON_EXTRACTION_ERRORS, raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.fail_on_extraction_errors is False


def test_fail_on_extraction_errors_reads_true_from_env(monkeypatch):
    """The opt-in escape hatch parses truthy values from the environment."""
    from remember_api.config import ENV_FAIL_ON_EXTRACTION_ERRORS, RememberConfig

    monkeypatch.setenv(ENV_FAIL_ON_EXTRACTION_ERRORS, "true")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.fail_on_extraction_errors is True


def test_entity_intrabatch_merge_similarity_defaults_to_half(monkeypatch):
    """In-batch entity dedup merge cutoff defaults to 0.5 (issue #3107)."""
    from remember_api.config import ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, RememberConfig

    monkeypatch.delenv(ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    assert RememberConfig.from_env().entity_intrabatch_merge_similarity == 0.5


def test_entity_intrabatch_merge_similarity_reads_from_env(monkeypatch):
    from remember_api.config import ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, RememberConfig

    monkeypatch.setenv(ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, "0.8")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    assert RememberConfig.from_env().entity_intrabatch_merge_similarity == 0.8


def test_entity_intrabatch_merge_similarity_rejects_out_of_range(monkeypatch):
    """Must be in (0, 1] — a merge cutoff, same guard as the recall threshold."""
    from remember_api.config import ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")
    for bad in ("0", "1.5", "-0.1"):
        monkeypatch.setenv(ENV_ENTITY_INTRABATCH_MERGE_SIMILARITY, bad)
        with pytest.raises(ValueError, match="entity_intrabatch_merge_similarity"):
            RememberConfig.from_env()


def test_llm_ollama_num_ctx_defaults_to_none(monkeypatch):
    """Unset Ollama num_ctx override lets Ollama use its model/server default."""
    from remember_api.config import ENV_LLM_OLLAMA_NUM_CTX, RememberConfig

    monkeypatch.delenv(ENV_LLM_OLLAMA_NUM_CTX, raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.llm_ollama_num_ctx is None


def test_llm_ollama_num_ctx_keeps_direct_construction_default():
    """Direct RememberConfig construction should not require the new field."""
    from dataclasses import fields

    from remember_api.config import RememberConfig

    config_field = next(item for item in fields(RememberConfig) if item.name == "llm_ollama_num_ctx")

    assert config_field.default is None
    assert config_field.kw_only


def test_llm_ollama_num_ctx_reads_positive_int(monkeypatch):
    """The native Ollama context override is parsed as a positive integer."""
    from remember_api.config import ENV_LLM_OLLAMA_NUM_CTX, RememberConfig

    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "65536")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.llm_ollama_num_ctx == 65536


def test_llm_ollama_num_ctx_rejects_non_positive_values(monkeypatch):
    """Zero would be accepted by neither Ollama nor downstream range logic."""
    from remember_api.config import ENV_LLM_OLLAMA_NUM_CTX, RememberConfig

    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "0")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_LLM_OLLAMA_NUM_CTX):
        RememberConfig.from_env()


def test_retain_structured_chunk_size_can_be_less_than_chunk_size():
    """Structured-chunk cap can be smaller than the retain chunk target."""
    from remember_api.config import RememberConfig

    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_RETAIN_STRUCTURED_CHUNK_SIZE"] = "2000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    config = RememberConfig.from_env()
    assert config.retain_chunk_size == 3000
    assert config.retain_structured_chunk_size == 2000


def test_retain_strategy_structured_chunk_size_validation():
    """Retain strategies allow structured-chunk caps below chunk size."""
    from remember_api.config import RememberConfig
    from remember_api.config_resolver import apply_strategy

    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    config = RememberConfig.from_env()
    config.retain_strategies = {
        "jsonl": {
            "retain_structured_chunk_size": 2000,
        }
    }

    resolved = apply_strategy(config, "jsonl")
    assert resolved.retain_structured_chunk_size == 2000


def test_embedding_similarity_threshold_defaults_are_backward_compatible(monkeypatch):
    """Unset threshold settings preserve the five existing operating points."""
    from remember_api.config import (
        ENV_CONSOLIDATION_DEDUP_THRESHOLD,
        ENV_GRAPH_SEED_MIN_SIMILARITY,
        ENV_SEMANTIC_LINK_MIN_SIMILARITY,
        ENV_SEMANTIC_MIN_SIMILARITY,
        ENV_TEMPORAL_SEMANTIC_MIN_SIMILARITY,
        RememberConfig,
    )

    for env_name in (
        ENV_SEMANTIC_MIN_SIMILARITY,
        ENV_GRAPH_SEED_MIN_SIMILARITY,
        ENV_TEMPORAL_SEMANTIC_MIN_SIMILARITY,
        ENV_SEMANTIC_LINK_MIN_SIMILARITY,
        ENV_CONSOLIDATION_DEDUP_THRESHOLD,
    ):
        monkeypatch.delenv(env_name, raising=False)

    config = RememberConfig.from_env()

    assert config.semantic_min_similarity == 0.3
    assert config.graph_seed_min_similarity == 0.3
    assert config.temporal_semantic_min_similarity == 0.1
    assert config.semantic_link_min_similarity == 0.7
    assert config.consolidation_dedup_threshold == 0.97


@pytest.mark.parametrize(
    ("env_name", "field_name", "value"),
    [
        ("REMEMBER_API_SEMANTIC_MIN_SIMILARITY", "semantic_min_similarity", 0.58),
        ("REMEMBER_API_GRAPH_SEED_MIN_SIMILARITY", "graph_seed_min_similarity", 0.41),
        ("REMEMBER_API_TEMPORAL_SEMANTIC_MIN_SIMILARITY", "temporal_semantic_min_similarity", 0.22),
        ("REMEMBER_API_SEMANTIC_LINK_MIN_SIMILARITY", "semantic_link_min_similarity", 0.81),
    ],
)
def test_embedding_similarity_thresholds_read_from_env(env_name: str, field_name: str, value: float):
    """Each configurable similarity gate has an independent environment setting."""
    from remember_api.config import RememberConfig

    os.environ[env_name] = str(value)

    config = RememberConfig.from_env()

    assert getattr(config, field_name) == value


@pytest.mark.parametrize(
    ("env_name", "field_name"),
    [
        ("REMEMBER_API_SEMANTIC_MIN_SIMILARITY", "semantic_min_similarity"),
        ("REMEMBER_API_GRAPH_SEED_MIN_SIMILARITY", "graph_seed_min_similarity"),
        ("REMEMBER_API_TEMPORAL_SEMANTIC_MIN_SIMILARITY", "temporal_semantic_min_similarity"),
        ("REMEMBER_API_SEMANTIC_LINK_MIN_SIMILARITY", "semantic_link_min_similarity"),
    ],
)
@pytest.mark.parametrize("invalid_value", ["-0.01", "1.01"])
def test_embedding_similarity_thresholds_must_be_between_zero_and_one(
    env_name: str, field_name: str, invalid_value: str
):
    """All embedding-dependent gates fail fast outside the supported range."""
    from remember_api.config import RememberConfig

    os.environ[env_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        RememberConfig.from_env()


def test_consolidation_max_completion_tokens_defaults_to_unset():
    """By default consolidation sends no explicit output budget (backwards compatible)."""
    from remember_api.config import RememberConfig

    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("REMEMBER_API_CONSOLIDATION_MAX_COMPLETION_TOKENS", None)

    config = RememberConfig.from_env()
    assert config.consolidation_max_completion_tokens is None


def test_consolidation_max_completion_tokens_env_override():
    """REMEMBER_API_CONSOLIDATION_MAX_COMPLETION_TOKENS controls consolidation LLM output budget."""
    from remember_api.config import RememberConfig

    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"
    os.environ["REMEMBER_API_CONSOLIDATION_MAX_COMPLETION_TOKENS"] = "8192"

    config = RememberConfig.from_env()
    assert config.consolidation_max_completion_tokens == 8192


def test_log_config_masks_database_urls(caplog):
    """Config startup logs must not expose database credentials."""
    from remember_api.config import RememberConfig

    os.environ["REMEMBER_API_DATABASE_URL"] = "postgresql://remember_user:plain-password@db:5432/remember_db"
    os.environ["REMEMBER_API_MIGRATION_DATABASE_URL"] = (
        "postgresql://migration_user:migration-password@db-admin:5432/remember_db"
    )
    os.environ["REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["REMEMBER_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["REMEMBER_API_LLM_PROVIDER"] = "mock"

    caplog.set_level(logging.INFO, logger="remember_api.config")

    config = RememberConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "remember_user" not in log_output
    assert "plain-password" not in log_output
    assert "migration_user" not in log_output
    assert "migration-password" not in log_output
    assert "postgresql://***:***@db:5432/remember_db" in log_output
    assert "postgresql://***:***@db-admin:5432/remember_db" in log_output


def test_read_database_url_defaults_to_none_when_unset(monkeypatch):
    """Without REMEMBER_API_READ_DATABASE_URL, the field is None — engine
    will alias the read backend to the primary, preserving today's
    single-pool behaviour byte-for-bit. This is the most important guarantee
    of the change: zero-config means zero behaviour change.
    """
    from remember_api.config import RememberConfig

    monkeypatch.delenv("REMEMBER_API_READ_DATABASE_URL", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.read_database_url is None


def test_read_database_url_is_loaded_when_set(monkeypatch):
    """When REMEMBER_API_READ_DATABASE_URL is set, the value flows into
    config so MemoryEngine.initialize() will open a second backend against
    that URL for recall queries.
    """
    from remember_api.config import RememberConfig

    read_url = "postgresql://reader:secret@replica.example:5432/remember"
    monkeypatch.setenv("REMEMBER_API_READ_DATABASE_URL", read_url)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.read_database_url == read_url


def test_read_database_url_empty_string_is_treated_as_unset(monkeypatch):
    """Helm sometimes renders an unset env var as the empty string. Treat it
    the same as unset so deployments that conditionally set the var don't
    accidentally try to open a pool against `''`.
    """
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_READ_DATABASE_URL", "")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.read_database_url is None


def test_log_config_masks_read_database_url(monkeypatch, caplog):
    """Read-replica URL credentials must be masked in startup logs, same as
    the primary URL.
    """
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_DATABASE_URL", "postgresql://remember:pw@primary:5432/db")
    monkeypatch.setenv("REMEMBER_API_READ_DATABASE_URL", "postgresql://reader:replica-secret@replica:5432/db")
    monkeypatch.setenv("REMEMBER_API_RETAIN_MAX_COMPLETION_TOKENS", "64000")
    monkeypatch.setenv("REMEMBER_API_RETAIN_CHUNK_SIZE", "3000")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    caplog.set_level(logging.INFO, logger="remember_api.config")

    config = RememberConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "reader" not in log_output
    assert "replica-secret" not in log_output
    assert "Read database" in log_output
    assert "postgresql://***:***@replica:5432/db" in log_output


# Note: The BadRequestError wrapping is implemented in fact_extraction.py
# but requires a complex integration test setup. The functionality is
# straightforward: when a BadRequestError containing keywords like
# "max_tokens", "max_completion_tokens", or "maximum context" is caught,
# it's wrapped in a ValueError with helpful guidance.
#
# The config validation tests above ensure users get early feedback
# about invalid configurations before runtime errors occur.


# ---------------------------------------------------------------------------
# Multilingual BM25 configuration
# ---------------------------------------------------------------------------


def test_native_language_defaults_to_english(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.delenv("REMEMBER_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.text_search_extension_native_language == "english"


def test_native_language_loaded_from_env(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", "french")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.text_search_extension_native_language == "french"


def test_native_language_lowercased(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", "Spanish")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.text_search_extension_native_language == "spanish"


@pytest.mark.parametrize(
    "bad_value",
    ["en glish", "english;DROP TABLE", "english'", "1english", "english-extra", ""],
)
def test_native_language_rejects_invalid_identifiers(monkeypatch, bad_value):
    """text_search_extension_native_language is embedded into raw SQL — non-identifiers must be rejected."""
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", bad_value)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match="Invalid text_search_extension_native_language"):
        RememberConfig.from_env()


@pytest.mark.parametrize("extension", ["pgroonga", "vchord", "pg_textsearch", "pg_search"])
def test_local_text_search_rejects_other_extensions(monkeypatch, extension):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_TEXT_SEARCH_EXTENSION", extension)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")
    with pytest.raises(ValueError, match="Invalid text_search_extension"):
        RememberConfig.from_env()


def test_text_search_extension_rejects_unknown(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_TEXT_SEARCH_EXTENSION", "bogus")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match="Invalid text_search_extension"):
        RememberConfig.from_env()


def test_pg_search_bm25_columns_apply_tokenizer():
    from remember_api._pg_search import pg_search_bm25_columns

    assert pg_search_bm25_columns("id", ("text", "context"), "") == "id, text, context"
    assert pg_search_bm25_columns("id", ("text", "context"), "jieba") == "id, (text::pdb.jieba), (context::pdb.jieba)"
    assert pg_search_bm25_columns("id", ("text",), "ngram(2, 3)") == "id, (text::pdb.ngram(2,3))"
    assert pg_search_bm25_columns("id", ("text",), "edge_ngram(2, 5)") == "id, (text::pdb.edge_ngram(2,5))"


def test_llm_output_language_defaults_to_none(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.delenv("REMEMBER_API_LLM_OUTPUT_LANGUAGE", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.llm_output_language is None


def test_llm_output_language_loaded_from_env(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_OUTPUT_LANGUAGE", "Japanese")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.llm_output_language == "Japanese"


def test_llm_output_language_empty_string_is_unset(monkeypatch):
    """Empty env var (e.g. from Helm) should be treated as unset, not literal ''."""
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_OUTPUT_LANGUAGE", "")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.llm_output_language is None


def test_llm_reasoning_effort_unset_stays_none(monkeypatch):
    """Unset must stay None all the way down: no provider sends a level nobody set.

    Baking "low" in here made every REMEMBER_API_*_REASONING_EFFORT variable
    indistinguishable from the default one layer down, so a configured value could be
    silently discarded (issue #3449) while an unconfigured one was still transmitted.
    """
    from remember_api.config import RememberConfig
    from remember_api.engine.providers.mock_llm import MockLLM

    monkeypatch.delenv("REMEMBER_API_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.llm_reasoning_effort is None

    llm = MockLLM(provider="mock", api_key="", base_url="", model="mock", reasoning_effort=None)
    assert llm.reasoning_effort is None


def test_llm_reasoning_effort_loaded_from_env(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.llm_reasoning_effort == "xhigh"


# ---------------------------------------------------------------------------
# Recall candidate gating (BM25 score floor + per-source cap) — issue #1707
# ---------------------------------------------------------------------------


def test_bm25_min_score_defaults_to_zero(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.delenv("REMEMBER_API_BM25_MIN_SCORE", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.bm25_min_score == 0.0


def test_bm25_min_score_loaded_from_env(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_BM25_MIN_SCORE", "1.5")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.bm25_min_score == 1.5


def test_recall_max_candidates_per_source_defaults_to_disabled(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.delenv("REMEMBER_API_RECALL_MAX_CANDIDATES_PER_SOURCE", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.recall_max_candidates_per_source == 0


def test_recall_max_candidates_per_source_loaded_from_env(monkeypatch):
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_RECALL_MAX_CANDIDATES_PER_SOURCE", "150")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()
    assert config.recall_max_candidates_per_source == 150


# ---------------------------------------------------------------------------
# Bedrock service tier (REMEMBER_API_LLM_BEDROCK_SERVICE_TIER)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gemini service tier (REMEMBER_API_LLM_GEMINI_SERVICE_TIER)
# ---------------------------------------------------------------------------


def test_operation_retention_defaults(monkeypatch):
    from remember_api.config import (
        ENV_OPERATION_CLEANUP_BATCH_SIZE,
        ENV_OPERATION_RETENTION_DAYS,
        RememberConfig,
    )

    monkeypatch.delenv(ENV_OPERATION_RETENTION_DAYS, raising=False)
    monkeypatch.delenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.operation_retention_days == 0
    assert config.operation_cleanup_batch_size == 1000
    assert "operation_retention_days" in RememberConfig.get_static_fields()
    assert "operation_cleanup_batch_size" in RememberConfig.get_static_fields()


def test_operation_retention_env_overrides(monkeypatch):
    from remember_api.config import (
        ENV_OPERATION_CLEANUP_BATCH_SIZE,
        ENV_OPERATION_RETENTION_DAYS,
        RememberConfig,
    )

    monkeypatch.setenv(ENV_OPERATION_RETENTION_DAYS, "0")
    monkeypatch.setenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, "37")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    config = RememberConfig.from_env()

    assert config.operation_retention_days == 0
    assert config.operation_cleanup_batch_size == 37


@pytest.mark.parametrize("raw", ["-1", "not-an-int"])
def test_operation_retention_rejects_invalid_values(monkeypatch, raw):
    from remember_api.config import ENV_OPERATION_RETENTION_DAYS, RememberConfig

    monkeypatch.setenv(ENV_OPERATION_RETENTION_DAYS, raw)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_OPERATION_RETENTION_DAYS):
        RememberConfig.from_env()


@pytest.mark.parametrize("raw", ["0", "-1", "not-an-int"])
def test_operation_cleanup_batch_size_requires_positive_integer(monkeypatch, raw):
    from remember_api.config import ENV_OPERATION_CLEANUP_BATCH_SIZE, RememberConfig

    monkeypatch.setenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, raw)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_OPERATION_CLEANUP_BATCH_SIZE):
        RememberConfig.from_env()


# --- Worker per-type slot reservations: RESERVED_SLOTS (floor) + deprecated _MAX_SLOTS alias ---


def test_worker_reserved_slots_parse(monkeypatch):
    """RESERVED_SLOTS sets the reservation floor for an operation type."""
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REMEMBER_API_WORKER_RETAIN_RESERVED_SLOTS", "2")

    config = RememberConfig.from_env()

    assert config.worker_slot_reservations["retain"] == 2


def test_worker_legacy_max_slots_is_deprecated_alias_for_reserved(monkeypatch, caplog):
    """The legacy _MAX_SLOTS env var maps to the reservation floor and warns."""
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REMEMBER_API_WORKER_CONSOLIDATION_MAX_SLOTS", "3")

    with caplog.at_level(logging.WARNING):
        config = RememberConfig.from_env()

    assert config.worker_slot_reservations["consolidation"] == 3
    assert any("REMEMBER_API_WORKER_CONSOLIDATION_MAX_SLOTS" in r.message for r in caplog.records)


def test_worker_reserved_and_legacy_both_set_is_rejected(monkeypatch):
    """Setting both RESERVED_SLOTS and the deprecated _MAX_SLOTS alias is ambiguous."""
    from remember_api.config import RememberConfig

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REMEMBER_API_WORKER_RETAIN_RESERVED_SLOTS", "2")
    monkeypatch.setenv("REMEMBER_API_WORKER_RETAIN_MAX_SLOTS", "2")

    with pytest.raises(ValueError, match="RESERVED_SLOTS"):
        RememberConfig.from_env()
