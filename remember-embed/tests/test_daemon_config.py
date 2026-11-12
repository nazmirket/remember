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
    profile_name = "coding-agent"
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
        manager._start_daemon(config={})

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
        manager._start_daemon(config=config)

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


def test_daemon_uses_local_pg0_even_with_remote_database_config(temp_home, monkeypatch):
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    external = "postgresql://remote.example/other"
    _write_profile(temp_home, "coding-agent", f"REMEMBER_API_DATABASE_URL={external}\n")
    monkeypatch.setenv("REMEMBER_EMBED_API_DATABASE_URL", external)
    monkeypatch.setenv("REMEMBER_API_DATABASE_URL", external)
    monkeypatch.setenv("REMEMBER_API_READ_DATABASE_URL", external)
    monkeypatch.setenv("REMEMBER_API_MIGRATION_DATABASE_URL", external)
    env = _capture_daemon_env(DaemonEmbedManager(), "coding-agent", {})
    assert env["REMEMBER_API_DATABASE_URL"] == "pg0://remember-embed-coding-agent"
    assert "REMEMBER_API_READ_DATABASE_URL" not in env
    assert "REMEMBER_API_MIGRATION_DATABASE_URL" not in env


def test_daemon_forces_loopback_and_embedded_worker(temp_home, monkeypatch):
    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    _write_profile(temp_home, "coding-agent", "REMEMBER_API_HOST=0.0.0.0\nREMEMBER_API_WORKERS=8\n")
    monkeypatch.setenv("REMEMBER_API_WORKER_ENABLED", "false")
    env = _capture_daemon_env(DaemonEmbedManager(), "coding-agent", {})
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["REMEMBER_API_HOST"] == "127.0.0.1"
    assert env["REMEMBER_API_WORKERS"] == "1"
    assert env["REMEMBER_API_WORKER_ENABLED"] == "true"
