"""Tests for multi-LLM env parsing and engine-chain resolution helpers.

Covers indexed-member discovery, strategy JSON parsing/validation, and the
per-slot resolution + wrapping logic (``_build_llm``)
without constructing a full MemoryEngine.
"""

import pytest

from hindsight_api.config import (
    HindsightConfig,
    LLMMemberConfig,
    LLMStrategyConfig,
    _parse_llm_members,
    _parse_llm_strategy,
)
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import _build_llm, _LLMCallDefaults
from hindsight_api.engine.multi_llm import MultiLLMProvider

# No per-request overrides — exercises the chain-resolution logic without
# touching timeout/retry defaults (those have their own tests).
_NO_CALL_DEFAULTS = _LLMCallDefaults(timeout=None, max_retries=None, initial_backoff=None, max_backoff=None)


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Strip all HINDSIGHT_API_*LLM* env so each test sets only what it needs."""
    import os

    for key in list(os.environ):
        if key.startswith("HINDSIGHT_API_") and "LLM" in key:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
    return monkeypatch


# ── member parsing ─────────────────────────────────────────────────────────────


def test_parse_members_empty_when_unset(clean_llm_env):
    assert _parse_llm_members("") == []
    assert _parse_llm_members("RETAIN_") == []


def test_parse_members_indexed(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "gsk-1")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MODEL", "llama-x")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_PROVIDER", "anthropic")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_API_KEY", "ak-2")

    members = _parse_llm_members("")
    assert [(m.provider, m.model) for m in members] == [
        ("groq", "llama-x"),
        ("anthropic", "claude-haiku-4-5"),  # model defaulted from provider
    ]
    assert members[0].api_key == "gsk-1"


def test_parse_members_stops_at_first_gap(clean_llm_env):
    # Index 2 present but 1 missing → scan stops at 1, member 2 ignored.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_2_API_KEY", "gsk-2")
    assert _parse_llm_members("") == []


def test_parse_members_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_PROVIDER", "groq")
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_API_KEY", "gsk-1")
    retain = _parse_llm_members("RETAIN_")
    assert [m.provider for m in retain] == ["groq"]
    assert _parse_llm_members("") == []  # global still empty


def test_parse_members_missing_api_key_raises(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "openai")  # requires key
    with pytest.raises(ValueError, match="API_KEY is required"):
        _parse_llm_members("")


def test_parse_members_no_key_provider_ok(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")  # no key needed
    members = _parse_llm_members("")
    assert members[0].provider == "ollama"
    assert members[0].api_key is None


def test_parse_members_codex_home(clean_llm_env):
    # An openai-codex member can carry its own credentials directory so a chain
    # can fail over between two independently authorized ChatGPT profiles.
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "openai-codex")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_CODEX_HOME", "/creds/codex-b")
    members = _parse_llm_members("")
    assert members[0].provider == "openai-codex"
    assert members[0].codex_home == "/creds/codex-b"


def test_parse_members_codex_home_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_PROVIDER", "openai-codex")
    clean_llm_env.setenv("HINDSIGHT_API_RETAIN_LLM_1_CODEX_HOME", "/creds/retain-b")
    retain = _parse_llm_members("RETAIN_")
    assert retain[0].codex_home == "/creds/retain-b"
    # Scoped to the prefix; the global chain is unaffected.
    assert _parse_llm_members("") == []


def test_parse_members_without_codex_home_defaults_none(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")
    assert _parse_llm_members("")[0].codex_home is None


# ── strategy parsing ───────────────────────────────────────────────────────────


def test_parse_strategy_none_when_unset():
    assert _parse_llm_strategy(None) is None
    assert _parse_llm_strategy("") is None
    assert _parse_llm_strategy("   ") is None


def test_parse_strategy_failover():
    s = _parse_llm_strategy('{"mode": "failover"}')
    assert s == LLMStrategyConfig(mode="failover", weights=None)


def test_parse_strategy_weighted_round_robin():
    s = _parse_llm_strategy('{"mode": "round-robin", "weights": [3, 1]}')
    assert s.mode == "round-robin"
    assert s.weights == [3, 1]


def test_parse_strategy_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_strategy("{not json")


def test_parse_strategy_invalid_mode():
    with pytest.raises(ValueError, match="Invalid LLM strategy mode"):
        _parse_llm_strategy('{"mode": "magic"}')


def test_parse_strategy_weights_require_round_robin():
    with pytest.raises(ValueError, match="only valid with mode"):
        _parse_llm_strategy('{"mode": "failover", "weights": [1, 1]}')


def test_parse_strategy_weights_must_be_positive_ints():
    with pytest.raises(ValueError, match="positive integers"):
        _parse_llm_strategy('{"mode": "round-robin", "weights": [1, 0]}')
    with pytest.raises(ValueError, match="positive integers"):
        _parse_llm_strategy('{"mode": "round-robin", "weights": []}')


# ── from_env integration ────────────────────────────────────────────────────────


def test_from_env_no_chain_is_empty(clean_llm_env):
    config = HindsightConfig.from_env()
    assert config.llm_members == []
    assert config.llm_strategy is None


def test_from_env_global_chain(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "ollama")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_STRATEGY", '{"mode": "failover"}')
    config = HindsightConfig.from_env()
    assert [m.provider for m in config.llm_members] == ["ollama"]
    assert config.llm_strategy.mode == "failover"


# ── chain resolution + wrapping (_build_llm) ────────────────────────────────────


def _empty_config(**overrides) -> HindsightConfig:
    """A minimal HindsightConfig from env, then patched with the given fields."""
    import dataclasses

    base = HindsightConfig.from_env()
    return dataclasses.replace(base, **overrides)


def _member(provider="ollama"):
    return LLMMemberConfig(
        provider=provider,
        api_key=None,
        model="m",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )


def _base_llm() -> LLMProvider:
    return LLMProvider(provider="mock", api_key="", base_url="", model="m0")


def test_build_llm_no_chain_returns_plain_provider(clean_llm_env):
    base = _base_llm()
    result = _build_llm(base, _empty_config(), "", _NO_CALL_DEFAULTS)
    assert result is base
    assert not isinstance(result, MultiLLMProvider)


def test_build_llm_global_chain_wraps(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    base = _base_llm()
    result = _build_llm(base, config, "", _NO_CALL_DEFAULTS)
    assert isinstance(result, MultiLLMProvider)
    assert result.members[0] is base  # primary stays index 0
    assert result.members[1].provider == "ollama"


def test_build_llm_per_op_inherits_global(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        retain_llm_members=[],
        retain_llm_strategy=None,
    )
    result = _build_llm(_base_llm(), config, "retain_", _NO_CALL_DEFAULTS)
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["ollama"]  # inherited


def test_build_llm_fallback_slot_inherits_global_when_empty(clean_llm_env):
    # Refresh -> empty reflect slot must keep going to the global chain (#5724).
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        reflect_llm_members=[],
        reflect_llm_strategy=None,
        mental_model_refresh_llm_members=[],
        mental_model_refresh_llm_strategy=None,
    )
    result = _build_llm(_base_llm(), config, "mental_model_refresh_", _NO_CALL_DEFAULTS, fallback_prefix="reflect_")
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["ollama"]
    assert result.strategy.mode == "failover"


def test_build_llm_fallback_slot_wins_over_global(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        reflect_llm_members=[_member("lmstudio")],
        reflect_llm_strategy=LLMStrategyConfig(mode="round-robin"),
        mental_model_refresh_llm_members=[],
        mental_model_refresh_llm_strategy=None,
    )
    result = _build_llm(_base_llm(), config, "mental_model_refresh_", _NO_CALL_DEFAULTS, fallback_prefix="reflect_")
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["lmstudio"]
    assert result.strategy.mode == "round-robin"


def test_build_llm_per_op_overrides_global(clean_llm_env):
    config = _empty_config(
        llm_members=[_member("ollama")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
        retain_llm_members=[_member("lmstudio")],
        retain_llm_strategy=LLMStrategyConfig(mode="round-robin"),
    )
    result = _build_llm(_base_llm(), config, "retain_", _NO_CALL_DEFAULTS)
    assert isinstance(result, MultiLLMProvider)
    assert [m.provider for m in result.members[1:]] == ["lmstudio"]
    assert result._strategy.mode == "round-robin"


def test_build_llm_members_without_strategy_stays_plain(clean_llm_env):
    # Members configured but no strategy → no wrapping (strategy is required).
    config = _empty_config(llm_members=[_member("ollama")], llm_strategy=None)
    base = _base_llm()
    assert _build_llm(base, config, "", _NO_CALL_DEFAULTS) is base


# ── vertexai member build path (_member_to_llm) ─────────────────────────────────


# ── codex member build path (_member_to_llm) ────────────────────────────────────


def _write_codex_auth(auth_dir, access_token):
    """Write a minimal chatgpt-mode ``auth.json`` under ``auth_dir``."""
    import json

    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "rt",
                    "account_id": "acct",
                },
            }
        )
    )
    return auth_dir


def test_member_to_llm_passes_codex_home(clean_llm_env, tmp_path, monkeypatch):
    """A codex member's own ``CODEX_HOME`` reaches the built provider.

    The process-wide ``CODEX_HOME`` points at a different profile, so this also
    proves the member's value wins over the ambient environment.
    """
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _member_to_llm

    ambient = _write_codex_auth(tmp_path / "ambient", "at-ambient")
    member_home = _write_codex_auth(tmp_path / "member", "at-member")
    monkeypatch.setenv("CODEX_HOME", str(ambient))
    clear_config_cache()

    member = LLMMemberConfig(
        provider="openai-codex",
        api_key=None,
        model="gpt-5-codex",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
        codex_home=str(member_home),
    )
    provider = _member_to_llm(member, _empty_config(llm_codex_home=None), _NO_CALL_DEFAULTS)

    assert provider._provider_impl.access_token == "at-member"


def test_member_to_llm_codex_home_falls_back_to_global(clean_llm_env, tmp_path, monkeypatch):
    """A member with no ``CODEX_HOME`` inherits the primary's configured one."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _member_to_llm

    _write_codex_auth(tmp_path / "ambient", "at-ambient")
    global_home = _write_codex_auth(tmp_path / "global", "at-global")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "ambient"))
    clear_config_cache()

    member = LLMMemberConfig(
        provider="openai-codex",
        api_key=None,
        model="gpt-5-codex",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )
    provider = _member_to_llm(member, _empty_config(llm_codex_home=str(global_home)), _NO_CALL_DEFAULTS)

    assert provider._provider_impl.access_token == "at-global"


def test_build_llm_codex_chain_spans_two_profiles(clean_llm_env, tmp_path, monkeypatch):
    """The #3793 use case end to end: a failover chain of two Codex profiles.

    Both members are ``openai-codex``; each resolves its own ``auth.json``, so
    falling over from a rate-limited profile actually reaches a second account.
    """
    from hindsight_api.config import clear_config_cache

    primary_home = _write_codex_auth(tmp_path / "primary", "at-primary")
    fallback_home = _write_codex_auth(tmp_path / "fallback", "at-fallback")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    clear_config_cache()

    fallback = LLMMemberConfig(
        provider="openai-codex",
        api_key=None,
        model="gpt-5-codex",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
        codex_home=str(fallback_home),
    )
    config = _empty_config(
        llm_codex_home=str(primary_home),
        llm_members=[fallback],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    base = LLMProvider(
        provider="openai-codex",
        api_key="",
        base_url="",
        model="gpt-5-codex",
        codex_home=config.llm_codex_home,
    )

    chain = _build_llm(base, config, "", _NO_CALL_DEFAULTS)

    assert isinstance(chain, MultiLLMProvider)
    assert [m._provider_impl.access_token for m in chain.members] == ["at-primary", "at-fallback"]


# ── litellmrouter member build path (_member_to_llm) ─────────────────────────────


# ── per-member request policy (timeout / max_retries) ─────────────────────────


def test_parse_members_without_request_policy_defaults_none(clean_llm_env):
    """Unset means inherit, so both fields must be None rather than a value."""
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "k")
    member = _parse_llm_members("")[0]
    assert member.timeout is None
    assert member.max_retries is None


def test_parse_members_request_policy(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "k")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_TIMEOUT", "45.5")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MAX_RETRIES", "2")
    member = _parse_llm_members("")[0]
    assert member.timeout == 45.5
    assert member.max_retries == 2


def test_parse_members_request_policy_zero_is_not_unset(clean_llm_env):
    """0 retries is a meaningful setting -- fail fast -- and must survive.

    Conflating it with "unset" would silently restore the operation default,
    which is the opposite of what the operator asked for.
    """
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "k")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MAX_RETRIES", "0")
    assert _parse_llm_members("")[0].max_retries == 0


def test_parse_members_request_policy_per_op_prefix(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_API_KEY", "k")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_MAX_RETRIES", "3")
    assert _parse_llm_members("REFLECT_")[0].max_retries == 3


def test_parse_members_request_policy_invalid_raises(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "k")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MAX_RETRIES", "soon")
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        _parse_llm_members("")


def _policy_member(**overrides):
    base = dict(
        provider="gemini",
        api_key="k",
        model="m",
        base_url=None,
        reasoning_effort=None,
        extra_body=None,
        default_headers=None,
        bedrock_service_tier=None,
        gemini_service_tier=None,
    )
    base.update(overrides)
    return LLMMemberConfig(**base)


def test_member_inherits_operation_policy_when_unset():
    """The no-override path must be byte-identical to the previous behaviour."""
    from hindsight_api.engine.memory_engine import _member_call_defaults

    defaults = _LLMCallDefaults(timeout=120.0, max_retries=5, initial_backoff=0.5, max_backoff=8.0)
    assert _member_call_defaults(_policy_member(), defaults) == defaults.as_kwargs()


def test_member_policy_overrides_operation_policy():
    from hindsight_api.engine.memory_engine import _member_call_defaults

    defaults = _LLMCallDefaults(timeout=120.0, max_retries=5, initial_backoff=0.5, max_backoff=8.0)
    got = _member_call_defaults(_policy_member(timeout=30.0, max_retries=0), defaults)
    assert got["timeout"] == 30.0
    assert got["max_retries"] == 0
    # untouched fields still come from the operation
    assert got["initial_backoff"] == 0.5
    assert got["max_backoff"] == 8.0


def test_member_policy_overrides_are_independent():
    """Setting one must not disturb the other."""
    from hindsight_api.engine.memory_engine import _member_call_defaults

    defaults = _LLMCallDefaults(timeout=120.0, max_retries=5, initial_backoff=0.5, max_backoff=8.0)
    only_retries = _member_call_defaults(_policy_member(max_retries=0), defaults)
    assert only_retries["max_retries"] == 0 and only_retries["timeout"] == 120.0
    only_timeout = _member_call_defaults(_policy_member(timeout=30.0), defaults)
    assert only_timeout["timeout"] == 30.0 and only_timeout["max_retries"] == 5


def test_chain_can_fail_fast_on_primary_while_fallback_still_retries(clean_llm_env):
    """The configuration this exists for.

    A failover chain is already a retry: when a non-terminal member fails the
    next one is tried, so retrying it first only delays the handoff. The
    terminal member has nowhere to fail over to, so it keeps a retry budget.
    """
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_PROVIDER", "gemini")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_API_KEY", "k")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_1_MAX_RETRIES", "2")

    from hindsight_api.engine.memory_engine import _member_call_defaults

    primary_policy = _LLMCallDefaults(timeout=120.0, max_retries=0, initial_backoff=0.5, max_backoff=8.0)
    fallback = _parse_llm_members("")[0]
    assert primary_policy.max_retries == 0
    assert _member_call_defaults(fallback, primary_policy)["max_retries"] == 2


# ── a member whose sign-in is gone (#6095) ──────────────────────────────────────


@pytest.fixture
def logged_out(clean_llm_env, tmp_path):
    """Every sign-in based provider looks in an empty home: nobody is logged in."""
    from hindsight_api.config import clear_config_cache

    clean_llm_env.setenv("HOME", str(tmp_path))
    for var in ("CODEX_HOME", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        clean_llm_env.delenv(var, raising=False)
    clear_config_cache()
    yield
    clear_config_cache()


# Every provider that reads a stored sign-in while it is built. A new one belongs here
# too, or a lost login on it crash-loops the server again.
@pytest.mark.parametrize("provider", ["chatgpt", "openai-codex"])
async def test_logged_out_provider_builds_and_fails_each_call(logged_out, provider):
    from hindsight_api.engine.llm_interface import ProviderUnavailableError
    from hindsight_api.engine.providers.none_llm import UnavailableLLM

    llm = LLMProvider(provider=provider, api_key="", base_url="", model="m")  # used to raise here

    assert isinstance(llm._provider_impl, UnavailableLLM)
    assert llm.provider == provider  # not mistaken for provider "none"
    with pytest.raises(ProviderUnavailableError):
        await llm.verify_connection()
    with pytest.raises(ProviderUnavailableError):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)


async def test_logged_out_primary_fails_over_to_next_member(logged_out):
    base = LLMProvider(provider="chatgpt", api_key="", base_url="", model="gpt-5")
    config = _empty_config(llm_members=[_member("mock")], llm_strategy=LLMStrategyConfig(mode="failover"))
    chain = _build_llm(base, config, "", _NO_CALL_DEFAULTS)

    await chain.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert len(chain.members[1]._provider_impl._mock_calls) == 1


async def test_logged_out_middle_member_keeps_the_rest_of_the_chain(logged_out):
    config = _empty_config(
        llm_members=[_member("chatgpt"), _member("mock")],
        llm_strategy=LLMStrategyConfig(mode="failover"),
    )
    primary = _base_llm()
    primary._provider_impl._mock_exception = RuntimeError("primary down")
    chain = _build_llm(primary, config, "", _NO_CALL_DEFAULTS)

    await chain.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    # Primary failed, the logged-out middle member failed, the last member answered.
    assert len(chain.members[2]._provider_impl._mock_calls) == 1


def test_unknown_provider_still_fails_startup(clean_llm_env):
    with pytest.raises(ValueError):
        LLMProvider(provider="no-such-provider", api_key="k", base_url="", model="m")
