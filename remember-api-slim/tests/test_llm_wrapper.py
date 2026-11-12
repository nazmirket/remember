import pytest

from remember_api.engine.llm_wrapper import create_llm_provider, sanitize_llm_output


def test_legacy_positional_settings_cannot_silently_change_timeout():
    from remember_api.engine.llm_wrapper import LLMProvider
    from remember_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    with pytest.raises(TypeError):
        LLMProvider("ollama", "", "", "llama3.2", "low", None)
    with pytest.raises(TypeError):
        create_llm_provider("ollama", "", "", "llama3.2", "low", None)
    with pytest.raises(TypeError):
        OpenAICompatibleLLM("ollama", "", "", "llama3.2", "low", 7.5, None)


def test_llm_provider_threads_ollama_num_ctx_to_provider_impl():
    """LLMProvider carries the native Ollama context override to the implementation."""
    from remember_api.engine.llm_wrapper import LLMProvider

    provider = LLMProvider(
        provider="ollama",
        api_key="",
        base_url="",
        model="llama3.2",
        ollama_num_ctx=65536,
    )

    assert provider.ollama_num_ctx == 65536
    assert provider._provider_impl.ollama_num_ctx == 65536


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "65536", True])
def test_llm_provider_rejects_invalid_ollama_num_ctx(bad_value):
    """Direct callers should fail before sending invalid Ollama request options."""
    from remember_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="ollama_num_ctx"):
        LLMProvider(
            provider="ollama",
            api_key="",
            base_url="",
            model="llama3.2",
            ollama_num_ctx=bad_value,
        )


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "65536", True])
def test_openai_compatible_llm_rejects_invalid_ollama_num_ctx(bad_value):
    """The provider implementation also validates direct construction."""
    from remember_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    with pytest.raises(ValueError, match="ollama_num_ctx"):
        OpenAICompatibleLLM(
            provider="ollama",
            api_key="",
            base_url="",
            model="llama3.2",
            ollama_num_ctx=bad_value,
        )


def test_llm_provider_from_env_reads_ollama_num_ctx(monkeypatch):
    """Direct env construction uses the same optional positive-int parser."""
    from remember_api.config import ENV_LLM_OLLAMA_NUM_CTX, clear_config_cache
    from remember_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "ollama")
    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "32768")
    clear_config_cache()

    provider = LLMProvider.from_env()

    assert provider.ollama_num_ctx == 32768
    assert provider._provider_impl.ollama_num_ctx == 32768
    clear_config_cache()


@pytest.mark.asyncio
async def test_native_ollama_omits_num_ctx_unless_configured():
    """Native Ollama calls should not override the model context window by default."""
    from pydantic import BaseModel

    from remember_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
    from tests.ollama_stub import chat_body, ollama_stub

    class Answer(BaseModel):
        ok: bool

    async with ollama_stub(chat_body('{"ok": true}')) as stub:
        default_provider = OpenAICompatibleLLM(
            provider="ollama",
            api_key="",
            base_url=stub.openai_base_url,
            model="llama3.2",
        )
        await default_provider._call_ollama_native(
            messages=[{"role": "user", "content": "ping"}],
            response_format=Answer,
            max_completion_tokens=None,
            temperature=None,
            max_retries=0,
            initial_backoff=1,
            max_backoff=1,
            skip_validation=False,
        )

        configured_provider = OpenAICompatibleLLM(
            provider="ollama",
            api_key="",
            base_url=stub.openai_base_url,
            model="llama3.2",
            ollama_num_ctx=65536,
        )
        await configured_provider._call_ollama_native(
            messages=[{"role": "user", "content": "ping"}],
            response_format=Answer,
            max_completion_tokens=None,
            temperature=None,
            max_retries=0,
            initial_backoff=1,
            max_backoff=1,
            skip_validation=False,
        )
        await default_provider.cleanup()
        await configured_provider.cleanup()

    calls = stub.requests
    assert "num_ctx" not in calls[0].json["options"]
    assert calls[0].json["options"]["num_batch"] == 512
    assert calls[1].json["options"]["num_ctx"] == 65536


@pytest.mark.parametrize(
    "input_text, expected",
    [
        # Null bytes stripped
        ("hello\x00world", "helloworld"),
        ("FIRST\u0000PAGE", "FIRSTPAGE"),
        # Multiple null bytes
        ("\x00\x00text\x00", "text"),
        # Other control characters stripped (non-whitespace)
        ("text\x01\x02\x03end", "textend"),
        ("text\x08end", "textend"),  # backspace
        ("text\x0cend", "textend"),  # form feed
        ("text\x0bend", "textend"),  # vertical tab
        ("text\x1fend", "textend"),  # unit separator
        ("text\x7fend", "textend"),  # DEL
        # Whitespace preserved
        ("hello\tworld", "hello\tworld"),
        ("hello\nworld", "hello\nworld"),
        ("hello\r\nworld", "hello\r\nworld"),
        # Unicode surrogates stripped
        ("text\ud800end", "textend"),
        ("text\udfffend", "textend"),
        # Clean text unchanged
        ("normal text", "normal text"),
        ("unicode: café naïve", "unicode: café naïve"),
        # Edge cases
        ("", ""),
        (None, None),
    ],
)
def test_sanitize_llm_output(input_text, expected):
    assert sanitize_llm_output(input_text) == expected


# --- parse_llm_json: structural repair of malformed LLM JSON (#2547/#2544) ---


def test_parse_llm_json_valid_passthrough():
    from remember_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_llm_json_strips_fences():
    from remember_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize(
    "malformed,expected",
    [
        ('{"a": 1,}', {"a": 1}),  # trailing comma
        ('{"a": "unterminated', {"a": "unterminated"}),  # unterminated string
        ("{'a': 'single quotes'}", {"a": "single quotes"}),  # single quotes
        (r'{"path": "C:\Users"}', {"path": "C:\\Users"}),  # invalid \escape (#2504)
        ('```json\n{"a": 1,}\n```', {"a": 1}),  # fenced + trailing comma
    ],
)
def test_parse_llm_json_repairs_structural_malformation(malformed, expected):
    from remember_api.engine.llm_wrapper import parse_llm_json

    assert parse_llm_json(malformed) == expected


def test_parse_llm_json_control_char_scrub_still_works():
    from remember_api.engine.llm_wrapper import parse_llm_json

    # Raw control char embedded in a string value — scrubbed to a space, no repair.
    assert parse_llm_json('{"a": "line\x01break"}') == {"a": "line break"}


def test_parse_llm_json_preserves_raw_line_breaks_in_strings():
    """A raw newline in a string is content, not noise (#3361).

    A model writing a markdown table into a JSON string often forgets to escape
    its line breaks. Blanking them out delivers the table already welded onto
    one line, and nothing downstream can tell that apart from prose.
    """
    from remember_api.engine.llm_wrapper import parse_llm_json

    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert parse_llm_json('{"text": "%s"}' % table.replace("\n", "\n")) == {"text": table}
    assert parse_llm_json('{"a": "tab\there"}') == {"a": "tab\there"}
    assert parse_llm_json('{"a": "back\\\\slash and \\" quote"}') == {"a": 'back\\slash and " quote'}


@pytest.mark.parametrize("garbage", ["", "   ", "not json at all !!!", "```json\n```"])
def test_parse_llm_json_unrecoverable_raises(garbage):
    """Repair that yields an empty result must not masquerade as success — the
    contract is to raise so retry ladders / the #1833 fail-loud path can act."""
    import json

    from remember_api.engine.llm_wrapper import parse_llm_json

    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(garbage)


def test_llm_provider_constructor_ignores_global_config(monkeypatch):
    """The constructor uses only its arguments — it never reads global config.

    Resolving the server-level default for an omitted field is the caller's job
    (MemoryEngine's per-op builds, _member_to_llm, and from_env). Keeping the
    constructor config-free makes a provider's effective settings a pure function
    of its arguments, which is what lets each member of a multi-LLM chain be
    configured independently.
    """
    import json

    from remember_api.config import (
        ENV_LLM_DEFAULT_HEADERS,
        clear_config_cache,
    )
    from remember_api.engine.llm_wrapper import LLMProvider

    # Global config sets a non-default header map...
    monkeypatch.setenv(ENV_LLM_DEFAULT_HEADERS, json.dumps({"x-from": "global"}))
    clear_config_cache()

    # ...but a directly-constructed provider that omits it does NOT inherit it.
    provider = LLMProvider(provider="mock", api_key="", base_url="", model="m")
    assert provider.default_headers is None

    clear_config_cache()


def test_reask_messages_appends_the_reply_as_text_and_the_correction():
    """The re-ask keeps the request as a prefix and replays any reply shape as a string (#4965)."""
    from pydantic import BaseModel

    from remember_api.engine.llm_wrapper import reask_messages

    class _Reply(BaseModel):
        facts: list[str]

    request = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for reply, text in [
        ({"a": "é"}, '{"a": "é"}'),
        ([{"a": 1}, {"a": 2}], '[{"a": 1}, {"a": 2}]'),
        (_Reply(facts=["x"]), '{"facts":["x"]}'),
        ("raw text", "raw text"),
        (None, ""),
    ]:
        out = reask_messages(request, reply, "fix it")
        assert out[:2] == request
        assert out[2:] == [{"role": "assistant", "content": text}, {"role": "user", "content": "fix it"}]
    assert len(request) == 2  # the caller's list is not mutated
