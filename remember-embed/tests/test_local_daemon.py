"""Local daemon configuration and lifecycle commands."""

import sys

import pytest

from remember_embed import cli
from remember_embed.local_state import daemon_paths, record_daemon_port


@pytest.fixture
def local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REMEMBER_HOME", str(tmp_path))
    monkeypatch.delenv("REMEMBER_API_PORT", raising=False)
    return tmp_path


def test_port_is_retained_for_later_commands(local_home, monkeypatch):
    assert daemon_paths().port == 8888
    monkeypatch.setenv("REMEMBER_API_PORT", "9077")
    record_daemon_port(daemon_paths().port)
    monkeypatch.delenv("REMEMBER_API_PORT")
    assert daemon_paths().port == 9077


def test_process_port_takes_priority_over_saved_port(local_home, monkeypatch):
    record_daemon_port(9077)
    monkeypatch.setenv("REMEMBER_API_PORT", "9123")
    assert daemon_paths().port == 9123


@pytest.mark.parametrize("port", ["0", "65536", "abc"])
def test_invalid_port_is_rejected(local_home, monkeypatch, port):
    monkeypatch.setenv("REMEMBER_API_PORT", port)
    with pytest.raises(ValueError, match="Invalid local daemon port"):
        daemon_paths()


def test_environment_overrides_local_config_without_writing_secrets(local_home, monkeypatch):
    path = daemon_paths().config
    path.parent.mkdir(parents=True)
    path.write_text("REMEMBER_API_LLM_PROVIDER=ollama\nREMEMBER_API_LLM_MODEL=saved\n")
    monkeypatch.setenv("REMEMBER_API_LLM_MODEL", "current")
    monkeypatch.setenv("REMEMBER_API_LLM_API_KEY", "private")
    config = cli.get_config()
    assert config["REMEMBER_API_LLM_MODEL"] == "current"
    assert config["REMEMBER_API_LLM_API_KEY"] == "private"
    assert "private" not in path.read_text()


def test_daemon_start_receives_provider_configuration(local_home, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["remember-embed", "daemon", "start"])
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "claude-code")
    monkeypatch.setattr(cli.daemon_client, "ensure_daemon_running", lambda config: calls.append(config) or True)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert calls[0]["REMEMBER_API_LLM_PROVIDER"] == "claude-code"


def test_profile_commands_are_not_exposed(local_home, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["remember-embed", "profile", "list"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_start_refuses_a_second_daemon_on_another_port(local_home, monkeypatch):
    from unittest.mock import Mock

    from remember_embed.daemon_embed_manager import DaemonEmbedManager

    record_daemon_port(9077)
    monkeypatch.setenv("REMEMBER_API_PORT", "9123")
    manager = DaemonEmbedManager()
    manager._is_port_in_use = Mock(return_value=True)
    manager._owned_pid_on_port = Mock(return_value=1234)
    manager._start_daemon_locked = Mock()
    assert manager._start_daemon({}) is False
    manager._start_daemon_locked.assert_not_called()
    assert (local_home / "profiles" / "coding-agent.port").read_text() == "9077"


def test_anthropic_uses_its_own_key(local_home, monkeypatch):
    monkeypatch.setenv("REMEMBER_API_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("REMEMBER_API_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    assert cli.get_config()["REMEMBER_API_LLM_API_KEY"] == "anthropic-key"
