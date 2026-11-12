"""Test for validating that profile environment variables are loaded correctly when starting daemon.

This is a regression test for issue #305 where profile .env files were not loaded
before daemon startup, causing environment variables to be ignored.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Create a temporary home directory.

    Both POSIX and Windows env vars are set because `Path.home()` consults
    USERPROFILE (then HOMEDRIVE+HOMEPATH) on Windows, not HOME — without
    USERPROFILE override the tests would operate on the real user profile.
    """
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setenv("HOME", str(temp_home))
    monkeypatch.setenv("USERPROFILE", str(temp_home))
    return temp_home


def test_profile_config_is_loaded_for_daemon(temp_home):
    """Test that profile .env config is loaded when preparing daemon startup.

    Before the fix, the daemon would ignore all values from the profile's .env file,
    using only os.environ or hardcoded defaults. This caused profile-specific
    settings to be silently ignored.
    """
    # Create a profile with custom configuration
    profile_dir = temp_home / ".remember" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "test-timeout"
    profile_env_path = profile_dir / f"{profile_name}.env"

    # Write profile config with custom values
    profile_env_path.write_text(
        "REMEMBER_API_LLM_PROVIDER=openai\n"
        "REMEMBER_API_LLM_API_KEY=sk-test-fake-key\n"
        "REMEMBER_API_LLM_MODEL=gpt-4o-mini\n"
        "REMEMBER_API_LOG_LEVEL=debug\n"
    )

    # Create metadata to register the profile with a port
    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9876,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Verify that ProfileManager can load the profile config
    from remember_embed.profile_manager import ProfileManager

    pm = ProfileManager()

    # Verify profile exists
    assert pm.profile_exists(profile_name)

    # Get profile paths
    paths = pm.resolve_profile_paths(profile_name)
    assert paths.config.exists()
    assert paths.config == profile_env_path
    assert paths.port == 9876

    # Load profile config (this simulates what the fix should do)
    profile_config = pm.load_profile_config(profile_name)

    # Verify config was loaded correctly
    assert profile_config["REMEMBER_API_LLM_PROVIDER"] == "openai"
    assert profile_config["REMEMBER_API_LLM_API_KEY"] == "sk-test-fake-key"
    assert profile_config["REMEMBER_API_LLM_MODEL"] == "gpt-4o-mini"
    assert profile_config["REMEMBER_API_LOG_LEVEL"] == "debug"

    # Keys are returned under their environment-variable names only. The
    # lowercase aliases this used to also inject ("idle_timeout", "llm_base_url",
    # ...) are gone: they were a second, hand-maintained list a setting had to
    # appear in to be seen, which is how REMEMBER_API_LLM_BASE_URL went missing
    # (issue #4094).
    assert not [key for key in profile_config if key.islower()]


def test_load_config_file_uses_correct_profile(temp_home, monkeypatch):
    """Test that load_config_file() loads the correct profile and not default.

    This is the core fix for issue #305 - when a profile is specified, we should
    ONLY load that profile's .env, never the default profile's config.
    """
    import os

    # Clear any existing profile override state
    from remember_embed.cli import set_cli_profile_override

    set_cli_profile_override(None)

    # Must clear REMEMBER_EMBED_PROFILE env var to ensure test isolation
    monkeypatch.delenv("REMEMBER_EMBED_PROFILE", raising=False)

    from remember_embed.cli import load_config_file
    from remember_embed.profile_manager import ProfileManager

    # Create default profile with one provider
    default_config_dir = temp_home / ".remember"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    (default_config_dir / "embed").write_text("REMEMBER_API_LLM_PROVIDER=openai\nREMEMBER_API_LLM_MODEL=gpt-4o-mini\n")

    # Create a named profile with a DIFFERENT provider
    profile_dir = temp_home / ".remember" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "myapp"
    profile_env_path = profile_dir / f"{profile_name}.env"
    profile_env_path.write_text("REMEMBER_API_LLM_PROVIDER=groq\nREMEMBER_API_LLM_MODEL=llama-3.1-70b\n")

    # Create metadata
    import json

    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9876,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Clear env to ensure we're testing file loading
    monkeypatch.delenv("REMEMBER_API_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL", raising=False)

    # Test 1: Load default profile (no profile specified)
    set_cli_profile_override(None)
    load_config_file()

    assert os.environ.get("REMEMBER_API_LLM_PROVIDER") == "openai"
    assert os.environ.get("REMEMBER_API_LLM_MODEL") == "gpt-4o-mini"

    # Clear env
    monkeypatch.delenv("REMEMBER_API_LLM_PROVIDER")
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL")

    # Test 2: Load named profile - should load ONLY that profile, not default
    set_cli_profile_override(profile_name)
    load_config_file()

    # Should have loaded from the named profile
    assert os.environ.get("REMEMBER_API_LLM_PROVIDER") == "groq", "Should load from named profile, not default"
    assert os.environ.get("REMEMBER_API_LLM_MODEL") == "llama-3.1-70b", "Should load from named profile, not default"


def test_load_config_file_reads_utf8_profile(temp_home, monkeypatch):
    """Profile templates should load independently of the process locale."""
    import builtins
    import os

    from remember_embed.cli import load_config_file, set_cli_profile_override

    set_cli_profile_override(None)
    monkeypatch.delenv("REMEMBER_EMBED_PROFILE", raising=False)
    monkeypatch.delenv("REMEMBER_API_LLM_PROVIDER", raising=False)

    config_path = temp_home / ".remember" / "embed"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# memory — retained\nREMEMBER_API_LLM_PROVIDER=openai\n",
        encoding="utf-8",
    )

    original_open = builtins.open

    def locale_open(file, mode="r", *args, **kwargs):
        if file == config_path and "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "ascii"
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", locale_open)

    load_config_file()

    assert os.environ["REMEMBER_API_LLM_PROVIDER"] == "openai"


def test_load_config_file_respects_environment(temp_home, monkeypatch):
    from remember_embed.cli import load_config_file, set_cli_profile_override

    set_cli_profile_override(None)
    monkeypatch.delenv("REMEMBER_EMBED_PROFILE", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "environment")
    config_path = temp_home / ".remember" / "embed"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("REMEMBER_API_LLM_PROVIDER=profile\n", encoding="utf-8")

    load_config_file()

    assert os.environ["REMEMBER_API_LLM_PROVIDER"] == "environment"


def test_get_config_respects_profile(temp_home, monkeypatch):
    """Test that get_config() returns profile-specific values."""
    import os

    from remember_embed.cli import get_config, set_cli_profile_override
    from remember_embed.profile_manager import ProfileManager

    # Create default profile
    default_config_dir = temp_home / ".remember"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    (default_config_dir / "embed").write_text(
        "REMEMBER_API_LLM_PROVIDER=openai\n"
        "REMEMBER_API_LLM_MODEL=gpt-4o-mini\n"
        "REMEMBER_API_LLM_API_KEY=sk-default-key\n"
    )

    # Create named profile
    profile_dir = temp_home / ".remember" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_name = "production"
    profile_env_path = profile_dir / f"{profile_name}.env"
    profile_env_path.write_text(
        "REMEMBER_API_LLM_PROVIDER=anthropic\n"
        "REMEMBER_API_LLM_MODEL=claude-sonnet-4-20250514\n"
        "REMEMBER_API_LLM_API_KEY=sk-ant-production\n"
    )

    # Create metadata
    import json

    metadata_path = profile_dir / "metadata.json"
    metadata = {
        "version": 1,
        "profiles": {
            profile_name: {
                "port": 9900,
                "created_at": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Clear env
    monkeypatch.delenv("REMEMBER_API_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("REMEMBER_API_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Test with named profile
    set_cli_profile_override(profile_name)
    config = get_config()

    # Should have loaded from production profile, NOT default
    assert config["REMEMBER_API_LLM_PROVIDER"] == "anthropic", "Should use profile's provider"
    assert config["REMEMBER_API_LLM_MODEL"] == "claude-sonnet-4-20250514", "Should use profile's model"
    assert config["REMEMBER_API_LLM_API_KEY"] == "sk-ant-production", "Should use profile's API key"


def test_profile_env_propagates_arbitrary_remember_keys_to_daemon(temp_home, monkeypatch):
    """Regression test: REMEMBER_* keys in profile .env must reach the daemon subprocess env.

    Previously `_start_daemon` only copied a whitelist of keys (llm_*, log_level,
    idle_timeout) from the merged profile config into the daemon env. Anything else
    in the profile's .env — e.g. REMEMBER_API_EMBEDDINGS_PROVIDER or
    REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU on non-macOS — was silently dropped.
    """
    import json
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    # Write a profile .env containing non-whitelisted REMEMBER_API_* keys.
    profile_dir = temp_home / ".remember" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_name = "forwarding"
    (profile_dir / f"{profile_name}.env").write_text(
        "REMEMBER_API_LLM_PROVIDER=openai\n"
        "REMEMBER_API_LLM_API_KEY=sk-x\n"
        "REMEMBER_API_LLM_MODEL=gpt-4o-mini\n"
        "REMEMBER_API_EMBEDDINGS_PROVIDER=tei\n"
        "REMEMBER_API_EMBEDDINGS_TEI_URL=http://localhost:8080\n"
        "REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU=1\n"
    )
    (profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    profile_name: {
                        "port": 9877,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "last_used": "2024-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )

    # Clear shell env so we're only testing profile propagation.
    for key in (
        "REMEMBER_API_EMBEDDINGS_PROVIDER",
        "REMEMBER_API_EMBEDDINGS_TEI_URL",
        "REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU",
    ):
        monkeypatch.delenv(key, raising=False)

    manager = DaemonEmbedManager()
    captured: dict[str, dict[str, str]] = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    # Simulate Linux so the macOS default doesn't mask the test.
    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("remember_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(config={}, profile=profile_name)

    env = captured["env"]
    assert env.get("REMEMBER_API_EMBEDDINGS_PROVIDER") == "tei"
    assert env.get("REMEMBER_API_EMBEDDINGS_TEI_URL") == "http://localhost:8080"
    assert env.get("REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU") == "1"


def test_daemon_child_env_var_set_in_daemon_env(temp_home, monkeypatch):
    """remember-embed must set _REMEMBER_DAEMON_CHILD=1 so the daemon child
    skips the redundant re-exec in daemonize()."""
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict[str, dict[str, str]] = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    env = captured["env"]
    assert env.get("_REMEMBER_DAEMON_CHILD") == "1"


def test_daemon_preserves_uv_python_for_local_uv(temp_home, monkeypatch):
    """The local API launch inherits the interpreter selected by the integration."""
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    monkeypatch.setenv("UV_PYTHON", "3.13")
    manager = DaemonEmbedManager()
    captured = {}
    popen_called = [False]

    def fake_find_api_command(env=None):
        captured["find_env"] = env
        return ["uv", "run", "--project", "/fork/remember-api", "remember-api"]

    def fake_popen(cmd, env, **kwargs):
        captured["popen_env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", side_effect=fake_find_api_command),
        patch.object(manager, "is_running", side_effect=fake_is_running),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    assert captured["find_env"]["UV_PYTHON"] == "3.13"
    assert captured["popen_env"]["UV_PYTHON"] == "3.13"


def test_windows_popen_uses_windowless_process_flags(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    On Windows the daemon must be spawned with
    `creationflags=CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP`, preserving a
    windowless console for descendants such as uvx's uv child (#4562).
    Standard streams must still be redirected away from the parent.
    """
    import subprocess
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    # These subprocess constants only exist on Windows CPython; patch them in
    # so the test passes on Linux/macOS CI too. Values from Win32 API docs.
    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x00000008, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("remember_embed.daemon_embed_manager.platform.system", return_value="Windows"),
    ):
        assert manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    expected_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    assert kwargs.get("creationflags") == expected_flags
    # Windows ignores CREATE_NO_WINDOW if DETACHED_PROCESS is also set.
    assert not kwargs["creationflags"] & subprocess.DETACHED_PROCESS
    assert "start_new_session" not in kwargs
    assert kwargs.get("stdin") is subprocess.DEVNULL
    assert kwargs.get("stderr") is subprocess.STDOUT
    assert kwargs.get("close_fds") is True
    # stdout must still go to the daemon log instead of the parent's terminal.
    assert kwargs.get("stdout") is not None


def test_posix_popen_uses_start_new_session(temp_home, monkeypatch):
    """Lock down POSIX behavior so a future refactor doesn't flip it."""
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("remember_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True
    assert "creationflags" not in kwargs


def test_posix_popen_redirects_stdout_stderr_to_log(temp_home, monkeypatch):
    """Regression test for #1380: on POSIX, the daemon child must NOT inherit
    the parent's stdout/stderr — output would otherwise leak into a TUI
    parent communicating over stdio (Hermes, JSON-RPC gateways) and corrupt
    its rendering. Both streams must be redirected to a real file handle.
    """
    from unittest.mock import MagicMock, patch

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    captured: dict = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["kwargs"] = kwargs
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
        patch("remember_embed.daemon_embed_manager.platform.system", return_value="Linux"),
    ):
        manager._start_daemon(
            config={"llm_provider": "openai", "llm_api_key": "sk-x", "llm_model": "gpt-4o-mini"},
            profile="",
        )

    kwargs = captured["kwargs"]
    stdout = kwargs.get("stdout")
    stderr = kwargs.get("stderr")
    # Must be real file objects, not None (inherit) or DEVNULL.
    assert stdout is not None and hasattr(stdout, "write"), (
        f"daemon stdout must be redirected to a file handle, got {stdout!r}"
    )
    assert stderr is not None and hasattr(stderr, "write"), (
        f"daemon stderr must be redirected to a file handle, got {stderr!r}"
    )


def test_get_config_does_not_default_llm_model(temp_home, monkeypatch):
    """Regression test for issue #1360.

    When `REMEMBER_API_LLM_MODEL` is unset, `get_config()` must return None
    rather than a hardcoded `gpt-4o-mini`. Previously the CLI tried to import
    `remember_api.config.PROVIDER_DEFAULT_MODELS` to compute a provider-keyed
    default, but that import fails in standalone venvs (`uvx remember-embed`,
    bundled installs) where `remember-api` isn't installed, and the fallback
    silently routed every non-OpenAI provider to `gpt-4o-mini` — which they
    reject, causing retain to silently store zero memories.

    Leaving the value unset lets the daemon (which has remember-api) resolve
    the correct provider default itself.
    """
    from remember_embed.cli import get_config, set_cli_profile_override

    set_cli_profile_override(None)
    monkeypatch.delenv("REMEMBER_EMBED_PROFILE", raising=False)
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL", raising=False)
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "gemini")

    config = get_config()

    assert config["REMEMBER_API_LLM_PROVIDER"] == "gemini"
    assert "REMEMBER_API_LLM_MODEL" not in config, (
        "the model must stay unset when the env var is unset so the daemon "
        "resolves the provider default; got a hardcoded fallback instead"
    )


def test_configure_from_env_omits_model_when_unset(temp_home, monkeypatch):
    """Regression test for issue #1360.

    `_do_configure_from_env` must not write a hardcoded `REMEMBER_API_LLM_MODEL`
    line into the profile .env when the user didn't set one — that line would
    then be re-injected on every daemon start and suppress the daemon's
    provider-keyed default lookup.
    """
    from remember_embed import cli

    # CONFIG_DIR/CONFIG_FILE are computed from Path.home() at module import time,
    # so the temp_home fixture's HOME override doesn't reach them. Redirect the
    # module-level constants for the duration of this test.
    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REMEMBER_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL", raising=False)

    rc = cli._do_configure_from_env()
    assert rc == 0

    contents = (config_dir / "embed").read_text()
    assert "REMEMBER_API_LLM_PROVIDER=anthropic" in contents
    # The config is seeded from .env.example, which carries a commented
    # `# REMEMBER_API_LLM_MODEL=gpt-4o-mini` reference line. What must not
    # appear is an *active* (uncommented) model line — that's what would get
    # re-injected on daemon start and suppress the provider-keyed default.
    active_lines = [ln.strip() for ln in contents.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert not any(ln.startswith("REMEMBER_API_LLM_MODEL=") for ln in active_lines), (
        "active model line must be omitted when the user didn't set one, so the daemon picks the provider default"
    )


def test_configure_from_env_accepts_providers_outside_interactive_menu(temp_home, monkeypatch):
    """Regression test for issue #1360.

    `_do_configure_from_env` previously rejected any provider not in the small
    interactive-menu set (`PROVIDER_API_KEYS`) with "Unknown
    provider". remember-api supports ~18 providers (anthropic, claude-code,
    bedrock, openrouter, ...), so the gate blocked valid CI configurations.
    Validation belongs in the daemon, not in the CLI's UX-only menu list.
    """
    from remember_embed import cli

    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("REMEMBER_API_LLM_API_KEY", "sk-ant-test")
    monkeypatch.delenv("REMEMBER_API_LLM_MODEL", raising=False)

    rc = cli._do_configure_from_env()
    assert rc == 0, "anthropic must be accepted — the daemon validates providers, not the CLI"

    contents = (config_dir / "embed").read_text()
    assert "REMEMBER_API_LLM_PROVIDER=anthropic" in contents


def test_configure_from_env_accepts_codex_without_api_key(temp_home, monkeypatch):
    """OpenAI Codex authenticates through Codex CLI rather than an LLM API key."""
    from remember_embed import cli

    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "openai-codex")
    monkeypatch.setenv("REMEMBER_API_LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.delenv("REMEMBER_API_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")

    assert cli._has_non_interactive_env() is True
    assert cli._do_configure_from_env() == 0

    contents = (config_dir / "embed").read_text()
    assert "REMEMBER_API_LLM_PROVIDER=openai-codex" in contents
    assert "REMEMBER_API_LLM_MODEL=gpt-5.6-luna" in contents
    active_lines = [line for line in contents.splitlines() if line and not line.startswith("#")]
    assert not any(line.startswith("REMEMBER_API_LLM_API_KEY=") for line in active_lines)


def test_configure_from_env_uses_anthropic_key(temp_home, monkeypatch):
    from remember_embed import cli

    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("REMEMBER_API_LLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")

    assert cli._has_non_interactive_env() is True
    assert cli._do_configure_from_env() == 0
    contents = (config_dir / "embed").read_text()
    assert "REMEMBER_API_LLM_API_KEY=anthropic-test-key" in contents
    assert "unrelated-openai-key" not in contents


def test_configure_from_env_rejects_removed_provider(temp_home, monkeypatch):
    from remember_embed import cli

    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("REMEMBER_API_LLM_API_KEY", "test-key")

    assert cli._do_configure_from_env() == 1
    assert not (config_dir / "embed").exists()


def _windows_scripts_dir(tmp_path: Path, *, with_pythonw: bool) -> Path:
    """Build a fake Windows venv Scripts dir with remember-api.exe.

    The dir holds remember-api.exe and python.exe; when ``with_pythonw`` is
    True the GUI-subsystem interpreter (pythonw.exe) is created next to
    python.exe so `_windows_gui_api_command()` can find it.
    """
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    (scripts_dir / "remember-api.exe").write_bytes(b"MZ")
    (scripts_dir / "python.exe").write_bytes(b"MZ")
    if with_pythonw:
        (scripts_dir / "pythonw.exe").write_bytes(b"MZ")
    return scripts_dir


def _base_pythonw(tmp_path: Path) -> Path:
    """The base interpreter a fake venv's pyvenv.cfg points ``home`` at."""
    return tmp_path / "base" / "pythonw.exe"


def _windows_launcher_venv(tmp_path: Path, *, uv: bool, with_site_packages: bool = True) -> Path:
    """Fake a Windows venv whose Scripts/pythonw.exe is a launcher, not an interpreter.

    Returns the Scripts dir; the base interpreter is at `_base_pythonw(tmp_path)`.
    ``uv=True`` writes the ``uv =`` marker that identifies a uv trampoline venv;
    without it the venv looks like a stdlib one, whose own GUI venvwlauncher must
    be left alone. ``with_site_packages=False`` omits the venv's import root, so
    the base interpreter could not import remember_api.
    """
    scripts_dir = _windows_scripts_dir(tmp_path, with_pythonw=True)
    if with_site_packages:
        (tmp_path / "Lib" / "site-packages").mkdir(parents=True)
    base_pythonw = _base_pythonw(tmp_path)
    base_pythonw.parent.mkdir()
    base_pythonw.write_bytes(b"MZ")
    cfg = f"home = {base_pythonw.parent}\n"
    if uv:
        cfg += "uv = 0.12.3\n"
    (tmp_path / "pyvenv.cfg").write_text(cfg)
    return scripts_dir


def _windows_manager(monkeypatch, scripts_dir: Path):
    """A DaemonEmbedManager that believes it runs on Windows inside scripts_dir."""
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    manager = DaemonEmbedManager()
    monkeypatch.setattr(manager, "_dev_api_command", lambda: None)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "remember_embed.daemon_embed_manager.sysconfig.get_path",
        lambda name: str(scripts_dir),
    )
    monkeypatch.setattr(
        "remember_embed.daemon_embed_manager.sys.executable",
        str(scripts_dir / "python.exe"),
    )
    return manager


# Non-local providers keep _find_api_command past the local-ML check (#2676).
_EXTERNAL_PROVIDERS = {
    "REMEMBER_API_EMBEDDINGS_PROVIDER": "openai",
    "REMEMBER_API_RERANKER_PROVIDER": "openai",
}


def _capture_daemon_env(manager, profile: str, config: dict) -> dict[str, str]:
    """Run `_start_daemon` with Popen stubbed and return the child's environment."""
    from unittest.mock import MagicMock, patch

    captured: dict[str, dict[str, str]] = {}
    popen_called = [False]

    def fake_popen(cmd, env, **kwargs):
        captured["env"] = env
        popen_called[0] = True
        proc = MagicMock()
        proc.pid = 12345
        return proc

    def fake_is_running(profile=""):
        return popen_called[0]

    with (
        patch("remember_embed.daemon_embed_manager.subprocess.Popen", side_effect=fake_popen),
        patch("remember_embed.daemon_embed_manager.time.sleep"),
        patch.object(manager, "_clear_port", return_value=True),
        patch.object(manager, "_find_api_command", return_value=["remember-api"]),
        patch.object(manager, "is_running", side_effect=fake_is_running),
    ):
        manager._start_daemon(config=config, profile=profile)

    return captured["env"]


def _write_profile(temp_home: Path, name: str, body: str, port: int = 9878) -> None:
    profile_dir = temp_home / ".remember" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / f"{name}.env").write_text(body, encoding="utf-8")
    (profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    name: {
                        "port": port,
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "last_used": "2024-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )


def test_profile_base_url_reaches_daemon_env(temp_home, monkeypatch):
    """Regression test for issue #4094.

    A profile that sets REMEMBER_API_LLM_BASE_URL alongside KEY/MODEL was
    reported as having everything applied *except* the base URL, so LLM calls
    went to api.openai.com and failed with OpenAI's own 401. This walks the
    real path — `get_config()` -> `_start_daemon` -> the child's environment —
    rather than testing either half in isolation.
    """
    from remember_embed.cli import get_config, set_cli_profile_override
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    base_url = "https://maas-api.example.com/openapi/compatible-mode/v1"
    _write_profile(
        temp_home,
        "hermes",
        "REMEMBER_API_LLM_API_KEY=sk-7c7e0\n"
        "REMEMBER_API_LLM_MODEL=qwen3.8-27b\n"
        f"REMEMBER_API_LLM_BASE_URL={base_url}\n",
    )

    for key in ("REMEMBER_API_LLM_API_KEY", "REMEMBER_API_LLM_MODEL", "REMEMBER_API_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    set_cli_profile_override("hermes")
    try:
        config = get_config()
        assert config["REMEMBER_API_LLM_BASE_URL"] == base_url
        env = _capture_daemon_env(DaemonEmbedManager(), "hermes", config)
    finally:
        set_cli_profile_override(None)

    assert env.get("REMEMBER_API_LLM_BASE_URL") == base_url
    assert env.get("REMEMBER_API_LLM_MODEL") == "qwen3.8-27b"
    assert env.get("REMEMBER_API_LLM_API_KEY") == "sk-7c7e0"


def test_configure_from_env_persists_every_api_var(temp_home, monkeypatch):
    """Regression test for issue #4094.

    `_do_configure_from_env` wrote a hand-listed subset (provider, model, key),
    so a base URL — or any other REMEMBER_API_* setting — the user exported
    never reached the profile it generated. REMEMBER_EMBED_* is deliberately
    not persisted: those configure this wrapper for one invocation.
    """
    from remember_embed import cli

    config_dir = temp_home / ".remember"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "CONFIG_FILE", config_dir / "embed")

    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REMEMBER_API_LLM_API_KEY", "sk-x")
    monkeypatch.setenv("REMEMBER_API_LLM_BASE_URL", "https://maas-api.example.com/v1")
    monkeypatch.setenv("REMEMBER_API_EMBEDDINGS_PROVIDER", "tei")
    monkeypatch.setenv("REMEMBER_EMBED_API_URL", "http://elsewhere:9999")

    assert cli._do_configure_from_env() == 0

    active = [
        line.strip()
        for line in (config_dir / "embed").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "REMEMBER_API_LLM_BASE_URL=https://maas-api.example.com/v1" in active
    assert "REMEMBER_API_EMBEDDINGS_PROVIDER=tei" in active
    assert not any(line.startswith("REMEMBER_EMBED_API_URL=") for line in active)


def test_daemon_uses_local_pg0_even_with_remote_database_config(temp_home, monkeypatch):
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    external = "postgresql://remote.example/other"
    _write_profile(temp_home, "coding-agent", f"REMEMBER_API_DATABASE_URL={external}\n")
    monkeypatch.setenv("REMEMBER_EMBED_API_DATABASE_URL", external)
    monkeypatch.setenv("REMEMBER_API_DATABASE_URL", external)
    env = _capture_daemon_env(DaemonEmbedManager(), "coding-agent", {})
    assert env["REMEMBER_API_DATABASE_URL"] == "pg0://remember-embed-coding-agent"


def test_daemon_forces_loopback_and_embedded_worker(temp_home, monkeypatch):
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    _write_profile(temp_home, "coding-agent", "REMEMBER_API_HOST=0.0.0.0\nREMEMBER_API_WORKERS=8\n")
    monkeypatch.setenv("REMEMBER_API_WORKER_ENABLED", "false")
    env = _capture_daemon_env(DaemonEmbedManager(), "coding-agent", {})
    assert env["REMEMBER_API_HOST"] == "127.0.0.1"
    assert env["REMEMBER_API_WORKERS"] == "1"
    assert env["REMEMBER_API_WORKER_ENABLED"] == "true"
