"""Tests for daemon command routing and configuration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from remember_embed import cli, daemon_client, profile_manager
from remember_embed.daemon_embed_manager import DaemonEmbedManager
from remember_embed.profile_manager import ProfileManager


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileManager:
    """Keep profile selection and config loading independent of the developer's environment."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("REMEMBER_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(profile_manager, "_cli_profile_override", None)

    manager = ProfileManager()
    for name, port in (("", 18888), ("work", 19100), ("other", 19200)):
        paths = manager.resolve_profile_paths(name)
        paths.config.write_text(
            f"REMEMBER_API_PORT={port}\nREMEMBER_API_LLM_MODEL={name or 'default'}-model\n",
            encoding="utf-8",
        )
        paths.log.write_text(f"{name or 'default'} daemon log\n", encoding="utf-8")
        paths.ui_log.write_text(f"{name or 'default'} UI log\n", encoding="utf-8")
    return manager


def invoke_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["remember-embed", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0


@pytest.mark.parametrize("command", ["daemon"])
@pytest.mark.parametrize(
    "env_profile,flag,expected",
    [
        pytest.param(None, [], "work", id="active-profile"),
        pytest.param(None, ["-p", "other"], "other", id="cli-over-active"),
        pytest.param("work", ["--profile", "other"], "work", id="env-over-cli"),
        pytest.param(None, ["--profile", "default"], "default", id="explicit-default"),
    ],
)
def test_logs_read_the_selected_profile(
    profiles: ProfileManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    env_profile: str | None,
    flag: list[str],
    expected: str,
) -> None:
    profiles.set_active_profile("work")
    if env_profile is not None:
        monkeypatch.setenv("REMEMBER_EMBED_PROFILE", env_profile)

    invoke_cli(monkeypatch, [*flag, command, "logs"])

    assert capsys.readouterr().out == f"{expected} {'UI' if command == 'ui' else 'daemon'} log\n"


def test_daemon_status_displays_the_instance_it_probes(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_active_profile("work")
    monkeypatch.setenv("COLUMNS", "240")
    is_running = Mock(return_value=True)
    monkeypatch.setattr(daemon_client, "is_daemon_running", is_running)

    invoke_cli(monkeypatch, ["daemon", "status"])

    is_running.assert_called_once_with("work")
    output = capsys.readouterr().out
    assert "http://127.0.0.1:19100" in output
    assert "remember-embed-work" in output
    assert "http://127.0.0.1:18888" not in output


def test_profile_show_honours_explicit_default(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_active_profile("work")

    invoke_cli(monkeypatch, ["--profile", "default", "profile", "show", "-o", "json"])

    result = json.loads(capsys.readouterr().out)
    assert result["name"] == "default"
    assert result["source"] == "cli_flag"
    assert result["config"] == str(profiles.resolve_profile_paths("").config)


def test_configure_uses_the_creation_target_even_with_an_env_profile(
    profiles: ProfileManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REMEMBER_EMBED_PROFILE", "work")

    invoke_cli(
        monkeypatch,
        ["configure", "--profile", "new", "--port", "19300", "--env", "REMEMBER_API_LLM_MODEL=new-model"],
    )

    assert profiles.load_profile_config("new")["REMEMBER_API_LLM_MODEL"] == "new-model"
    assert profiles.load_profile_config("work")["REMEMBER_API_LLM_MODEL"] == "work-model"


def test_unknown_memory_cli_command_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["remember-embed", "bank", "list"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "unknown command" in capsys.readouterr().err
