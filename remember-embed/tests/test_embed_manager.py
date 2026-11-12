"""Tests for local daemon management."""

import io
import os
import signal
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from remember_embed._http_probe import ProbeResponse
from remember_embed.daemon_embed_manager import (
    DaemonEmbedManager,
    _detach_popen_kwargs,
    _runs_a_foreign_interpreter,
    _strip_parent_interpreter_env,
    _terminate_startup_process,
)


# Process groups and SIGKILL do not exist on Windows.
_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def _startup_manager(is_running: MagicMock) -> DaemonEmbedManager:
    manager = DaemonEmbedManager()
    manager._clear_port = MagicMock(return_value=True)
    manager.is_running = is_running
    manager._component_version = MagicMock(return_value="0.0.0")
    manager._find_api_command = MagicMock(return_value=["remember-api"])
    return manager


def _running_process() -> MagicMock:
    process = MagicMock(pid=4321)
    process.poll.return_value = None
    process.wait.return_value = 0
    return process


@_posix_only
def test_start_timeout_terminates_spawned_daemon_group(tmp_path, monkeypatch):
    manager = _startup_manager(MagicMock(return_value=False))
    process = _running_process()
    paths = SimpleNamespace(config=tmp_path / "config.env", log=tmp_path / "daemon.log", port=9177)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.DAEMON_STARTUP_TIMEOUT", 0)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: "Linux")
    killpg = MagicMock()
    monkeypatch.setattr("remember_embed.daemon_embed_manager.os.killpg", killpg)

    with patch("remember_embed.daemon_embed_manager.subprocess.Popen", return_value=process) as popen:
        assert manager._start_daemon_locked({}, paths) is False

    assert popen.call_args.args[0] == ["remember-api", "--port", "9177"]
    # The whole group, so a uvx launcher's remember-api grandchild goes too.
    killpg.assert_called_once_with(4321, signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=10)


def test_startup_keeps_polling_after_transient_stability_failure(tmp_path, monkeypatch):
    manager = _startup_manager(MagicMock(side_effect=[False, True, False, True, True]))
    process = _running_process()
    paths = SimpleNamespace(config=tmp_path / "config.env", log=tmp_path / "daemon.log", port=9177)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.time.time", lambda: 0)

    with patch("remember_embed.daemon_embed_manager.subprocess.Popen", return_value=process):
        assert manager._start_daemon_locked({}, paths) is True

    process.wait.assert_not_called()
    assert manager.is_running.call_count == 5


def test_startup_fails_fast_when_daemon_exits(tmp_path, monkeypatch):
    manager = _startup_manager(MagicMock(return_value=False))
    process = _running_process()
    process.poll.return_value = 1
    paths = SimpleNamespace(config=tmp_path / "config.env", log=tmp_path / "daemon.log", port=9177)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.time.sleep", MagicMock(side_effect=AssertionError))

    with patch("remember_embed.daemon_embed_manager.subprocess.Popen", return_value=process):
        assert manager._start_daemon_locked({}, paths) is False

    manager.is_running.assert_called_once()  # the pre-spawn check only


def _spawned_env(manager, tmp_path, monkeypatch) -> dict:
    """Run _start_daemon_locked up to the spawn and return the env Popen received."""
    process = _running_process()
    process.poll.return_value = 1  # exit at once so the startup loop ends
    paths = SimpleNamespace(config=tmp_path / "config.env", log=tmp_path / "daemon.log", port=9177)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.time.sleep", lambda _seconds: None)
    with patch("remember_embed.daemon_embed_manager.subprocess.Popen", return_value=process) as popen:
        manager._start_daemon_locked({}, paths)
    return popen.call_args.kwargs["env"]


def test_uvx_fallback_does_not_inherit_the_parent_interpreter_env(tmp_path, monkeypatch):
    """A host that finds its own site-packages via PYTHONPATH (Hermes' package-manager install
    exports its Python 3.14 generation) must not hand that directory to uvx's own interpreter:
    remember-api would import the host's pydantic and fail on the missing compiled extension."""
    monkeypatch.setenv("PYTHONPATH", "/host/venv/lib/python3.14/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("PATH", "/host/venv/bin:/usr/bin")
    manager = _startup_manager(MagicMock(return_value=False))
    manager._find_api_command = MagicMock(return_value=["uvx", "remember-api@0.0.0"])

    env = _spawned_env(manager, tmp_path, monkeypatch)

    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env and "VIRTUAL_ENV" not in env
    assert env["PATH"] == "/host/venv/bin:/usr/bin"  # uvx itself is found through PATH
    assert env["REMEMBER_API_DAEMON_LOG"] == str(tmp_path / "daemon.log")


def test_the_dev_uv_run_command_does_not_inherit_it_either(tmp_path, monkeypatch):
    """`uv run --project <api>` resolves that project's own environment, so it is as foreign
    as uvx: a monorepo checkout under a host that exports PYTHONPATH would hit the same crash."""
    monkeypatch.setenv("PYTHONPATH", "/host/venv/lib/python3.14/site-packages")
    manager = _startup_manager(MagicMock(return_value=False))
    manager._find_api_command = MagicMock(
        return_value=["uv", "run", "--project", "/repo/remember-api", "--extra", "all", "remember-api"]
    )

    assert "PYTHONPATH" not in _spawned_env(manager, tmp_path, monkeypatch)


def test_a_resolved_launcher_path_is_still_recognised(tmp_path, monkeypatch):
    """The guard matches the basename: the sibling candidates in _find_api_command are returned
    resolved, so resolving the launcher too must not silently stop the scrub."""
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    manager = _startup_manager(MagicMock(return_value=False))
    manager._find_api_command = MagicMock(return_value=["/opt/tools/uvx", "remember-api@0.0.0"])

    assert "PYTHONPATH" not in _spawned_env(manager, tmp_path, monkeypatch)


def test_only_uv_launchers_count_as_foreign():
    assert _runs_a_foreign_interpreter(["uvx", "remember-api@1.0.0"])
    assert _runs_a_foreign_interpreter(["uv", "run", "--project", "/x", "remember-api"])
    assert _runs_a_foreign_interpreter(["/opt/bin/uvx.exe", "remember-api@1.0.0"])
    assert not _runs_a_foreign_interpreter(["/venv/bin/remember-api"])
    assert not _runs_a_foreign_interpreter(["/target/bin/remember-api"])


def test_installed_binary_keeps_the_parent_interpreter_env(tmp_path, monkeypatch):
    """A `pip install --target` layout (issue #1240) needs PYTHONPATH to import remember_api;
    the scrub applies only to the uvx fallback."""
    monkeypatch.setenv("PYTHONPATH", "/target")
    manager = _startup_manager(MagicMock(return_value=False))
    manager._find_api_command = MagicMock(return_value=["/target/bin/remember-api"])

    env = _spawned_env(manager, tmp_path, monkeypatch)

    assert env["PYTHONPATH"] == "/target"


def test_strip_parent_interpreter_env_leaves_everything_else():
    env = {"PYTHONPATH": "x", "PYTHONSAFEPATH": "1", "REMEMBER_API_LLM_MODEL": "m", "HOME": "/h"}
    assert _strip_parent_interpreter_env(env) == {"REMEMBER_API_LLM_MODEL": "m", "HOME": "/h"}


@_posix_only
def test_start_timeout_kills_daemon_group_that_ignores_terminate(monkeypatch):
    process = _running_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("remember-api", 10), 0]
    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: "Linux")
    killpg = MagicMock()
    monkeypatch.setattr("remember_embed.daemon_embed_manager.os.killpg", killpg)

    _terminate_startup_process(process)

    assert killpg.call_args_list == [((4321, signal.SIGTERM),), ((4321, signal.SIGKILL),)]
    assert process.wait.call_count == 2


def test_start_timeout_kills_daemon_on_windows(monkeypatch):
    process = _running_process()
    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: "Windows")

    _terminate_startup_process(process)

    process.kill.assert_called_once_with()


def test_get_database_url_default():
    """Test database URL generation with default pg0."""
    manager = DaemonEmbedManager()

    assert manager.get_database_url() == "pg0://remember-embed-coding-agent"


def test_database_url_ignores_remote_settings(monkeypatch):
    manager = DaemonEmbedManager()
    monkeypatch.setenv("REMEMBER_EMBED_API_DATABASE_URL", "postgresql://remote/inherited")
    monkeypatch.setenv("REMEMBER_API_DATABASE_URL", "postgresql://remote/inherited")
    assert manager.get_database_url() == "pg0://remember-embed-coding-agent"


def test_manager_singleton():
    """Managers are independent and resolve the same local database."""
    manager1 = DaemonEmbedManager()
    manager2 = DaemonEmbedManager()

    # They should be independent instances but same type
    assert type(manager1) == type(manager2)

    # They should produce the same results
    assert manager1.get_database_url() == manager2.get_database_url()


def _embed_module_in_venv(venv: Path, site_packages: str) -> Path:
    """Create a venv (pyvenv.cfg at its root) holding remember_embed; return the module path."""
    pkg_dir = venv / site_packages / "remember_embed"
    pkg_dir.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /base\n")
    module = pkg_dir / "daemon_embed_manager.py"
    module.write_text("")
    return module


def test_listening_pids_windows_hides_netstat_console(monkeypatch):
    """Windows netstat probes must not flash a console window."""
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        result = MagicMock()
        result.returncode = 0
        result.stdout = "  TCP    127.0.0.1:9177    0.0.0.0:0    LISTENING    4321\n"
        return result

    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: "Windows")
    monkeypatch.setattr("remember_embed.daemon_embed_manager.subprocess.CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr("remember_embed.daemon_embed_manager.subprocess.run", fake_run)

    assert DaemonEmbedManager._listening_pids(9177) == [4321]
    assert calls[0][1]["creationflags"] == 0x08000000


def test_is_port_in_use_checks_both_loopback_families(monkeypatch):
    """An ::1-only listener occupies the port even though IPv4 refuses."""
    import socket

    attempted = []

    class FakeSocket:
        def __init__(self, family, type_):
            self.family = family

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def settimeout(self, value):
            pass

        def connect_ex(self, address):
            attempted.append(address[0])
            return 0 if self.family == socket.AF_INET6 else 1

    monkeypatch.setattr(socket, "socket", FakeSocket)

    assert DaemonEmbedManager._is_port_in_use(19177) is True
    assert attempted == ["127.0.0.1", "::1"]


class _RecordingProbe:
    """probe_get stand-in that answers 200 and records each probe's timeouts."""

    def __init__(self):
        self.reads: list[float] = []
        self.connects: list[float | None] = []

    def __call__(self, url, *, read_timeout, connect_timeout=None):
        self.reads.append(read_timeout)
        self.connects.append(connect_timeout)
        return ProbeResponse(status_code=200, text="")


def test_reclaim_probe_waits_long_enough_for_a_busy_daemon(monkeypatch):
    """Regression for #3099: a busy event loop must not read as a dead daemon.

    _port_health_ok is the probe whose false negative gets the listener killed,
    so it is the one that has to allow for a stalled loop.
    """
    from remember_embed import daemon_embed_manager

    assert daemon_embed_manager.HEALTH_PROBE_TIMEOUT >= 10.0

    probe = _RecordingProbe()
    monkeypatch.setattr("remember_embed.daemon_embed_manager.probe_get", probe)
    monkeypatch.setattr(daemon_embed_manager, "HEALTH_PROBE_TIMEOUT", 25.0)

    DaemonEmbedManager._port_health_ok(9177)
    assert probe.reads == [25.0]


def test_liveness_probes_stay_short(tmp_path, monkeypatch):
    """The daemon liveness probe must not inherit the longer reclaim budget."""
    from remember_embed import daemon_embed_manager

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    manager = DaemonEmbedManager()

    probe = _RecordingProbe()
    monkeypatch.setattr("remember_embed.daemon_embed_manager.probe_get", probe)

    assert manager.is_running() is True
    assert probe.reads == [2.0]
    assert all(c == daemon_embed_manager.PROBE_CONNECT_TIMEOUT for c in probe.connects)

    assert daemon_embed_manager.PROBE_CONNECT_TIMEOUT < 5.0


# ── #4344: the probe's decode is pinned, not left to the locale ──────────────
# `netstat`/`powershell`/`wmic` emit localized text in the console code page. With
# `text=True` alone the decode uses `locale.getpreferredencoding(False)`, which is
# `utf-8` in a UTF-8-mode process, and the mismatch raises inside `subprocess`'s own
# reader thread: the thread dies, `run()` returns with `stdout=None` and returncode 0,
# and `_windows_listening_pids()` reports no listeners on a host that has plenty.

# A localized zh-CN `netstat -ano -p TCP` answer: cp936 header, ASCII data lines.
_CP936_NETSTAT = (
    "\r\n活动连接\r\n\r\n"
    "  协议  本地地址          外部地址        状态           PID\r\n"
    "  TCP    127.0.0.1:8642         0.0.0.0:0              LISTENING       4242\r\n"
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4\r\n"
).encode("cp936")


def _emit(payload: bytes) -> list[str]:
    """A command that writes `payload` to stdout as raw bytes."""
    import sys as _sys

    return [
        _sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(%r); sys.stdout.buffer.flush()" % payload,
    ]


def test_run_probe_reads_output_that_is_not_utf8():
    """The probe returns the output instead of losing it in a dead reader thread."""
    output = DaemonEmbedManager._run_probe(_emit(_CP936_NETSTAT))

    assert output is not None
    # The localized header is replaced rather than raising; the data lines are intact.
    assert "LISTENING" in output
    assert "127.0.0.1:8642" in output
    assert "4242" in output


def test_windows_listening_pids_parses_a_localized_netstat(monkeypatch):
    """The existing parser needs no change: only the header carries non-ASCII."""
    decoded = _CP936_NETSTAT.decode("utf-8", errors="replace")
    monkeypatch.setattr(
        "remember_embed.daemon_embed_manager.DaemonEmbedManager._run_probe",
        staticmethod(lambda cmd, timeout=None: decoded),
    )

    assert DaemonEmbedManager._windows_listening_pids(8642) == [4242]
    assert DaemonEmbedManager._windows_listening_pids(445) == [4]
    assert DaemonEmbedManager._windows_listening_pids(9999) == []


def test_run_probe_still_reports_a_failed_command_as_none():
    """A non-zero exit is still None, so the encoding change did not widen success."""
    import sys as _sys

    assert DaemonEmbedManager._run_probe([_sys.executable, "-c", "raise SystemExit(3)"]) is None


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_detach_popen_kwargs_pins_stdin(system: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon child must never inherit the caller's fd 0.

    A caller can hold an fd 0 that is a socket opened with FD_CLOEXEC (e.g. a
    TUI/gateway parent that wires its IPC channel onto fds 0-2). An inherited
    fd 0 is closed by the kernel at exec, so the child would start with
    ``sys.stdin = None`` and crash in ``_redirect_stdio_to_log()``.
    """
    monkeypatch.setattr("remember_embed.daemon_embed_manager.platform.system", lambda: system)
    kwargs = _detach_popen_kwargs(io.BytesIO())
    assert kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console inheritance")
def test_windows_daemon_descendants_inherit_console(tmp_path: Path) -> None:
    """A console launcher must not leave its descendants to allocate a new console."""
    probe = dedent("""\
        import ctypes
        from ctypes import wintypes

        pids = (wintypes.DWORD * 32)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(pids, len(pids))
        assert 0 < count <= len(pids)
        print(*pids[:count])
    """)
    launcher = dedent("""\
        import subprocess
        import sys

        assert sys.stdin.read() == ""
        info = subprocess.STARTUPINFO()
        info.dwFlags = subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = subprocess.SW_HIDE
        subprocess.run([sys.executable, "-c", sys.argv[1]], startupinfo=info, check=True, timeout=15)
    """)
    # Keep a regressing implementation from flashing windows during the test.
    # SW_HIDE hides a newly allocated window; it does not change console membership.
    info = subprocess.STARTUPINFO()
    info.dwFlags = subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    log = tmp_path / "daemon.log"
    with log.open("wb") as log_handle:
        with subprocess.Popen(
            [sys.executable, "-c", launcher, probe], startupinfo=info, **_detach_popen_kwargs(log_handle)
        ) as process:
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise

    output = log.read_text()
    assert returncode == 0, output
    console_pids = {int(pid) for pid in output.split()}
    assert process.pid in console_pids
    assert os.getpid() not in console_pids


def test_api_command_uses_checkout():
    manager = DaemonEmbedManager()
    command = manager._find_api_command()
    assert command[:3] == ["uv", "run", "--project"]
    assert Path(command[3]).resolve().name == "remember-api"
    assert command[-1] == "remember-api"


def test_api_command_rejects_missing_checkout(monkeypatch):
    manager = DaemonEmbedManager()
    monkeypatch.setattr(manager, "_dev_api_command", lambda: None)
    with pytest.raises(RuntimeError, match="checkout is required"):
        manager._find_api_command()
