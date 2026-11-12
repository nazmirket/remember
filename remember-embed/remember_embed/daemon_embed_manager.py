"""Local daemon lifecycle and database selection."""

import logging
import math
import os
import platform
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ._http_probe import ProbeResponse, probe_get
from .profile_manager import ProfileLockTimeout, ProfileManager, lock_file, unlock_file

logger = logging.getLogger(__name__)
console = Console(stderr=True)


def _parse_float_env(name: str, default: float) -> float:
    """Parse a float environment variable, falling back on invalid values."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _safe_non_negative_float(value: float, fallback: float) -> float:
    """Return a finite non-negative float, or fallback for invalid values."""
    return value if math.isfinite(value) and value >= 0 else fallback


def _safe_positive_float(value: float, fallback: float) -> float:
    """Return a finite positive float, or fallback for invalid values."""
    return value if math.isfinite(value) and value > 0 else fallback


def _parse_non_negative_int(value: str | None, default: int, name: str) -> int:
    """Parse a non-negative integer, falling back for invalid or negative values."""
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Invalid %s=%r; using %d", name, value, default)
        return default
    return parsed


# Constants
# Allow CI/Windows to extend the startup budget — pg0-embedded's Windows wheel
# unpacks and runs initdb on first boot, which takes noticeably longer on cold
# runners than POSIX.
DAEMON_STARTUP_TIMEOUT = int(os.getenv("REMEMBER_EMBED_DAEMON_STARTUP_TIMEOUT", "180"))
DAEMON_STARTUP_TERMINATE_TIMEOUT = 10
ENV_DAEMON_LOG_MAX_BYTES = "REMEMBER_EMBED_DAEMON_LOG_MAX_BYTES"
ENV_DAEMON_LOG_BACKUP_COUNT = "REMEMBER_EMBED_DAEMON_LOG_BACKUP_COUNT"
DEFAULT_DAEMON_LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_DAEMON_LOG_BACKUP_COUNT = 3
# When another process is concurrently starting the daemon, the TCP port can be
# bound before /health returns 200. Give that warming daemon a short grace window
# before treating the listener as stale/foreign and attempting to reclaim it.
PORT_HEALTH_GRACE_TIMEOUT = _safe_non_negative_float(
    _parse_float_env("REMEMBER_EMBED_PORT_HEALTH_GRACE_TIMEOUT", 30.0),
    30.0,
)
PORT_HEALTH_CHECK_INTERVAL = _safe_positive_float(
    _parse_float_env("REMEMBER_EMBED_PORT_HEALTH_CHECK_INTERVAL", 0.5),
    0.5,
)
# Two probe budgets, split by what a wrong answer costs.
#
# The daemon serves /health on the same asyncio event loop that runs LLM calls
# and embeddings, so a slow provider call can stall the response for tens of
# seconds. Where a false negative makes us *kill* the listener — the reclaim
# decision in _port_health_ok — 2s was not enough to tell "busy" from "dead"
# (issue #3099), so that probe waits as long as the worker-side liveness
# threshold does.
HEALTH_PROBE_TIMEOUT = _safe_positive_float(
    _parse_float_env("REMEMBER_EMBED_HEALTH_PROBE_TIMEOUT", 10.0),
    10.0,
)
# Everywhere else the question is only "is it up?", asked on hot paths (every
# _ensure_started, profile delete, CLI status) and often on ports where nothing
# is listening at all. A false negative there is cheap — the caller re-runs
# ensure_running, which consults the long probe above before touching anything —
# so these stay short. They were briefly raised to HEALTH_PROBE_TIMEOUT, which
# pushed the control center's delete handler (one daemon probe + one UI probe
# per loopback family) past the 5s default client timeout on Windows.
LIVENESS_PROBE_TIMEOUT = 2.0
# Connecting is not the slow part for a busy daemon — being answered is. Cap
# connect separately so an unreachable address cannot spend the whole read
# budget, which is what turns two loopback families into double the wait.
PROBE_CONNECT_TIMEOUT = 1.0

# Both loopback families are probed: Next.js binds ::1 only when started with
# `--hostname localhost`, so an IPv4-only health check reported a perfectly
# healthy control plane as down (issue #3527).
LOOPBACK_HOSTS = ("127.0.0.1", "::1")

# Before signalling a PID we require its command line to identify it as one of
# ours. Selecting the victim purely by "who holds the port" killed unrelated
# services that happened to be on the same port (issue #3520).
DAEMON_PROCESS_MARKERS = ("remember-api", "remember_api")

# Deadline for the short read-only probes (netstat, ps, wmic, PowerShell) used to
# identify processes. Five seconds is ample for the POSIX ones, which read procfs
# or run `ps`.
_PROBE_TIMEOUT_S = 5.0

# Windows needs more. Its command-line lookup goes through WMI, and the wmic
# front-end is absent from current Windows images, so the fallback is a fresh
# PowerShell — whose startup alone can pass five seconds on a loaded machine
# before Get-CimInstance runs. Under the old shared deadline that surfaced as a
# daemon we own being classified as foreign, and as a flaky test-embed-windows.
_WINDOWS_PROBE_TIMEOUT_S = 30.0


def _probe(url: str, read: float) -> ProbeResponse | None:
    """GET url with a short connect and a caller-chosen read budget; None if nothing answered."""
    return probe_get(url, read_timeout=read, connect_timeout=min(read, PROBE_CONNECT_TIMEOUT))


# Interpreter-selection variables of the *parent* process. They are meaningful
# only for the interpreter that set them; forwarding them to a child that runs a
# different interpreter makes that child import the parent's packages.
_PARENT_INTERPRETER_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "VIRTUAL_ENV")


def _runs_a_foreign_interpreter(cmd: list[str]) -> bool:
    """Whether *cmd* launches remember-api on an interpreter that is not ours.

    True for uv-launched commands, including the
    monorepo ``uv run --project <api> ... remember-api`` dev command — because uv resolves
    an environment of its own, usually on a different Python than the caller. False for the
    installed-binary paths, which are a single executable from an environment we already share.

    Matched on the basename, not the literal string: the sibling candidate paths in
    ``_find_api_command`` are returned resolved (``[str(candidate)]``), so a future change that
    resolves the launcher the same way must not silently stop scrubbing the env.
    """
    return Path(cmd[0]).stem in {"uv", "uvx"}


def _strip_parent_interpreter_env(env: dict[str, str]) -> dict[str, str]:
    """Drop the parent interpreter's PYTHONPATH/PYTHONHOME/VIRTUAL_ENV from *env*.

    Only for the uv-launched commands, which run in an environment uv
    builds for them, usually on a different Python than the caller. Hosts that
    locate their own site-packages through PYTHONPATH (Hermes' package-manager
    install exports its Python 3.14 generation that way) otherwise hand that
    directory to the child interpreter, which then imports the host's pydantic
    and dies on the missing compiled extension::

        ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'

    The installed-binary paths keep the env untouched: a ``pip install --target``
    layout (issue #1240) relies on PYTHONPATH to find remember_api at all, and a
    binary from the caller's own venv runs the same interpreter, so nothing there
    is foreign.
    """
    return {key: value for key, value in env.items() if key not in _PARENT_INTERPRETER_ENV}


def _detach_popen_kwargs(log_handle: IO[bytes]) -> dict:
    """Cross-platform kwargs to spawn a subprocess detached from the caller.

    On POSIX, `start_new_session=True` calls setsid(2) so the child
    survives the parent's terminal; `stdin` is pinned to /dev/null so the
    child never inherits a caller fd 0 that may be CLOEXEC (closed at exec,
    leaving ``sys.stdin = None``). On Windows there is no setsid: we use
    `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` so console launchers such
    as uvx and their descendants share a windowless console. Previously,
    `DETACHED_PROCESS` left the launcher without a console, so its children
    could allocate a visible one (#4562). Do not combine the two flags:
    Windows ignores `CREATE_NO_WINDOW` when `DETACHED_PROCESS` is also set.

    `log_handle` receives the child's stdout/stderr on both platforms so
    output never leaks into the parent's terminal (which would corrupt a
    TUI parent communicating over stdio).
    """
    if platform.system() == "Windows":
        # Windows-only constants; use getattr so type checkers (e.g. ty) running
        # on Linux don't flag the attribute access. They're guaranteed present
        # at runtime because of the platform.system() guard above.
        create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {
            "creationflags": create_no_window | create_new_process_group,
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
    return {
        "start_new_session": True,
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": log_handle,
    }


def _signal_startup_process(process: subprocess.Popen, *, force: bool) -> None:
    """Signal the daemon child's whole process group on POSIX.

    The child may be a launcher rather than remember-api itself;
    signalling only its pid can leave the real API running as an orphan.
    ``_detach_popen_kwargs`` starts it with ``start_new_session``, so its pid is
    also its process group id. On Windows there is no group to signal, and
    ``terminate()`` is already an alias for ``kill()``.
    """
    if platform.system() == "Windows":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)


def _terminate_startup_process(process: subprocess.Popen) -> None:
    """Stop the daemon child owned by a failed startup attempt, escalating to kill."""
    if process.poll() is not None:
        return
    try:
        _signal_startup_process(process, force=False)
        try:
            process.wait(timeout=DAEMON_STARTUP_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            _signal_startup_process(process, force=True)
            process.wait(timeout=DAEMON_STARTUP_TERMINATE_TIMEOUT)
    except ProcessLookupError:
        return


class DaemonEmbedManager:
    """Production embed manager using daemon-based architecture with profile isolation."""

    def __init__(self):
        """Initialize the daemon embed manager."""
        self._profile_manager = ProfileManager()

    @staticmethod
    def _rotate_daemon_log(log_path: Path, max_bytes: int, backup_count: int) -> None:
        """Rotate a full daemon log before a new daemon opens it.

        Startup is serialized by the profile lock, and this runs only after
        any stale daemon has been stopped. That keeps child processes from
        writing to a renamed inode during rotation. Rotation only occurs at
        startup; an already-running daemon is left untouched.
        """
        if max_bytes == 0 or not log_path.exists() or log_path.stat().st_size < max_bytes:
            return

        if backup_count == 0:
            log_path.write_bytes(b"")
            return

        oldest = log_path.with_name(f"{log_path.name}.{backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(backup_count - 1, 0, -1):
            source = log_path.with_name(f"{log_path.name}.{index}")
            if source.exists():
                source.replace(log_path.with_name(f"{log_path.name}.{index + 1}"))
        log_path.replace(log_path.with_name(f"{log_path.name}.1"))

    def _sanitize_profile_name(self, profile: str | None) -> str:
        """Sanitize profile name for use in database names and file paths."""
        if profile is None:
            return "default"
        return re.sub(r"[^a-zA-Z0-9_-]", "-", profile)

    def get_database_url(self, profile: str) -> str:
        """Return the local pg0 database owned by this profile."""
        return f"pg0://remember-embed-{self._sanitize_profile_name(profile)}"

    def resolve_database_url(self, profile: str, config: Mapping[str, str] | None = None) -> str:
        """Resolve the daemon database locally, independent of inherited settings."""
        return self.get_database_url(profile)

    def get_url(self, profile: str) -> str:
        """
        Get the URL for the daemon serving this profile.

        Args:
            profile: Profile name

        Returns:
            URL string (e.g., "http://127.0.0.1:54321")

        Raises:
            RuntimeError: If daemon is not running
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        return f"http://127.0.0.1:{paths.port}"

    def is_running(self, profile: str) -> bool:
        """Check if daemon is running and responsive.

        Uses the short liveness budget, not HEALTH_PROBE_TIMEOUT: this runs on
        hot paths and a false negative only costs a re-run of ensure_running,
        which consults the long probe before deciding anything destructive.
        """
        daemon_url = self.get_url(profile)
        response = _probe(f"{daemon_url}/health", LIVENESS_PROBE_TIMEOUT)
        return response is not None and response.status_code == 200

    def _dev_api_command(self) -> list[str] | None:
        """Return the dev-mode launch command when running inside the monorepo."""
        dev_api_path = Path(__file__).parent.parent.parent / "remember-api"
        if dev_api_path.exists() and (dev_api_path / "pyproject.toml").exists():
            return ["uv", "run", "--project", str(dev_api_path), "--extra", "all", "remember-api"]
        return None

    def _find_api_command(self, env: Mapping[str, str] | None = None) -> list[str]:
        """Launch only the API from this Remember checkout."""
        command = self._dev_api_command()
        if command is None:
            raise RuntimeError("The Remember API checkout is required beside remember-embed")
        return command

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is in use using a socket connection (cross-platform).

        Probes both loopback families: a server bound to ::1 only (issue #3527)
        is invisible to an IPv4-only connect, which would make the caller treat
        an occupied port as free.
        """
        import socket

        for host in LOOPBACK_HOSTS:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    if sock.connect_ex((host, port)) == 0:
                        return True
            except OSError:
                # The family isn't available on this host (e.g. IPv6 disabled).
                continue
        return False

    @staticmethod
    def _run_probe(cmd: list[str], timeout: float = _PROBE_TIMEOUT_S) -> str | None:
        """Run a short read-only probe command, returning stdout or None.

        The decode is pinned rather than left to the locale. These probes run
        Windows-native CLIs (`netstat`, `powershell`, `wmic`) that emit localized
        text in the console code page, while `text=True` alone decodes with
        `locale.getpreferredencoding(False)` - which is `utf-8` in a UTF-8-mode
        process. The mismatch raises inside `subprocess`'s own reader thread, where
        nothing here can catch it: the thread dies, `run()` returns normally with
        `stdout=None` and returncode 0, and the caller reads "no listeners" from a
        host that has plenty.

        `errors="replace"` is the half that makes it deterministic; `encoding` alone
        still raises. Only the localized header carries non-ASCII, so every line the
        parsers read survives intact and neither needed changing.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return result.stdout if result.returncode == 0 else None

    @staticmethod
    def _windows_listening_pids(port: int) -> list[int]:
        """PIDs listening on `port` according to netstat."""
        output = DaemonEmbedManager._run_probe(["netstat", "-ano", "-p", "TCP"])
        if output is None:
            return []
        pids: list[int] = []
        for line in output.splitlines():
            fields = line.split()
            # proto, local address, foreign address, state, pid
            if len(fields) < 5 or fields[-2] != "LISTENING":
                continue
            # Match on the local address' port only: the daemon binds 127.0.0.1
            # but the UI can bind 0.0.0.0 or [::1] (issue #3527), and an
            # IPv4-literal match would miss those listeners entirely.
            local_address = fields[1]
            if local_address.rsplit(":", 1)[-1] != str(port):
                continue
            try:
                pids.append(int(fields[-1]))
            except ValueError:
                continue
        return pids

    @staticmethod
    def _posix_listening_pids(port: int) -> list[int]:
        """PIDs listening on `port`, via lsof, falling back to ss.

        `lsof` is the default on macOS but is absent from minimal containers and
        several Linux distributions, where it used to leave the caller with no
        PID at all (issue #3517). `ss` (iproute2) covers those hosts.
        """
        output = DaemonEmbedManager._run_probe(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"])
        pids: list[int] = []
        if output:
            for token in output.split():
                try:
                    pids.append(int(token))
                except ValueError:
                    continue
        if pids:
            return pids

        # Lines look like:
        #   LISTEN 0 4096 127.0.0.1:9177 0.0.0.0:* users:(("remember-api",pid=15774,fd=19))
        output = DaemonEmbedManager._run_probe(["ss", "-tlnp", f"sport = :{port}"])
        if output:
            for line in output.splitlines():
                if "users:" not in line:
                    continue
                pids.extend(int(match) for match in re.findall(r"pid=(\d+)", line))
        return pids

    @staticmethod
    def _listening_pids(port: int) -> list[int]:
        """All PIDs listening on a local port, de-duplicated, in discovery order."""
        if platform.system() == "Windows":
            found = DaemonEmbedManager._windows_listening_pids(port)
        else:
            found = DaemonEmbedManager._posix_listening_pids(port)
        return list(dict.fromkeys(found))

    @staticmethod
    def _process_command_line(pid: int) -> str | None:
        """The command line of `pid`, or None when it cannot be determined."""
        if platform.system() == "Windows":
            # Ask the kernel first. The subprocess probes below are slow: wmic is
            # gone from current Windows images, so the answer rests on a cold
            # PowerShell, which can take longer than any fixed deadline on a loaded
            # machine. A timeout there is not harmless — the caller reads None as
            # "not one of our processes" (see _process_matches), so the manager
            # refused to stop a daemon it owns (#6017).
            native = DaemonEmbedManager._windows_native_command_line(pid)
            if native:
                return native
            # Fallback for processes the native read cannot open. Both probes get
            # a longer deadline than the default for the cold start above.
            output = DaemonEmbedManager._run_probe(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/format:list"],
                timeout=_WINDOWS_PROBE_TIMEOUT_S,
            )
            if output is None:
                output = DaemonEmbedManager._run_probe(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
                    ],
                    timeout=_WINDOWS_PROBE_TIMEOUT_S,
                )
            if output is None:
                return None
            text = output.replace("CommandLine=", " ").strip()
            return text or None

        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            raw = proc_cmdline.read_bytes()
        except OSError:
            raw = None
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()

        # macOS (and Linux hosts without procfs).
        output = DaemonEmbedManager._run_probe(["ps", "-p", str(pid), "-o", "args="])
        return output.strip() if output and output.strip() else None

    @staticmethod
    def _windows_native_command_line(pid: int) -> str | None:
        """`pid`'s command line read in-process from the kernel, or None.

        Uses NtQueryInformationProcess(ProcessCommandLineInformation), available
        since Windows 8.1, which needs only limited query access — enough for any
        process of the same user, which is every daemon we started. No subprocess,
        so it answers in microseconds however loaded the machine is.
        """
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        process_command_line_information = 60

        class _UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),  # in bytes, without the terminator
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p),
            ]

        win_dll = getattr(ctypes, "WinDLL", None)  # only defined on Windows
        if win_dll is None:
            return None
        try:
            kernel32 = win_dll("kernel32", use_last_error=True)
            ntdll = win_dll("ntdll")
        except OSError:
            return None
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        query = ntdll.NtQueryInformationProcess
        query.restype = ctypes.c_long  # NTSTATUS; negative means failure
        query.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]

        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return None
        try:
            # The first call only reports the size the answer needs.
            size = wintypes.ULONG(0)
            query(handle, process_command_line_information, None, 0, ctypes.byref(size))
            if size.value < ctypes.sizeof(_UnicodeString):
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if query(handle, process_command_line_information, buffer, size, ctypes.byref(size)) < 0:
                return None
            text = _UnicodeString.from_buffer(buffer)
            if not text.Buffer or not text.Length:
                return None
            return ctypes.wstring_at(text.Buffer, text.Length // 2).strip() or None
        finally:
            kernel32.CloseHandle(handle)

    @staticmethod
    def _process_matches(pid: int, markers: tuple[str, ...]) -> bool:
        """True when `pid`'s command line identifies it as one of our processes."""
        cmdline = DaemonEmbedManager._process_command_line(pid)
        if cmdline is None:
            return False
        lowered = cmdline.lower()
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _owned_pid_on_port(port: int, markers: tuple[str, ...], description: str) -> int | None:
        """PID listening on `port` that we can positively identify as ours.

        Returns None — refusing to signal anything — when the listener cannot be
        identified. Failing to reclaim a port is recoverable; SIGTERMing an
        unrelated service that merely holds the port is not (issue #3520).
        """
        pids = DaemonEmbedManager._listening_pids(port)
        if not pids:
            logger.warning(f"Could not find PID for port {port}")
            return None
        for pid in pids:
            if DaemonEmbedManager._process_matches(pid, markers):
                return pid
        logger.warning(
            f"Port {port} is held by PID(s) {', '.join(str(p) for p in pids)}, which do not look like "
            f"a {description}; refusing to signal them"
        )
        return None

    @staticmethod
    def _kill_process(pid: int) -> bool:
        """Kill a process by PID and wait for it to exit. Returns True if process is gone."""
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    return True
        except OSError:
            return True  # Already gone
        return False

    @staticmethod
    def _port_health_ok(port: int) -> bool:
        """Return True when the listener on port responds like initialized Remember."""
        response = _probe(f"http://127.0.0.1:{port}/health", HEALTH_PROBE_TIMEOUT)
        if response is None or response.status_code != 200:
            return False
        try:
            health = response.json()
            return health.get("status") == "healthy" and health.get("database") == "connected"
        except (ValueError, AttributeError):
            # Not JSON, or JSON that is not an object: not Remember's payload.
            return False

    def _wait_for_port_health(self, port: int, timeout: float | None = None) -> bool:
        """Wait briefly for a just-bound daemon port to become healthy.

        `timeout` bounds when the last probe may *start*, not when it returns, so
        a listener that accepts and hangs can push the wall clock to roughly
        timeout + HEALTH_PROBE_TIMEOUT. That slack is deliberate: it is what lets
        a daemon stalled on a slow LLM call answer before we call it stale.
        """
        timeout = _safe_non_negative_float(
            PORT_HEALTH_GRACE_TIMEOUT if timeout is None else timeout,
            0.0,
        )
        interval = _safe_positive_float(PORT_HEALTH_CHECK_INTERVAL, 0.5)
        deadline = time.monotonic() + timeout
        while True:
            if not self._is_port_in_use(port):
                return False
            if self._port_health_ok(port):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(interval, remaining))

    def _clear_port(self, port: int) -> bool:
        """
        Ensure the port is free before starting a daemon.

        Behavior:
          * Port free → True (nothing to do).
          * Port occupied by a *healthy* remember daemon → True without killing.
            The caller's "start" is effectively a no-op: the daemon is already up.
            Killing it would race concurrent starts (one process kills the other's
            freshly-started daemon, both rush to rebind the port).
          * Port occupied but /health is unreachable, non-200, or does not return
            Remember's initialized health payload → treat as a stale daemon and
            attempt to reclaim it, but only after the listener's command line
            confirms it is a Remember daemon. This preserves the original intent
            of clearing stale daemons from version upgrades without SIGTERMing a
            foreign service that merely holds the port (issue #3520).
          * Kill failed, or unidentifiable/foreign process occupying the port → False.
        """
        if not self._is_port_in_use(port):
            return True

        # Port is occupied — check if it's a healthy Remember daemon. In
        # concurrent startup races the socket can bind before /health returns
        # 200, so wait briefly before deciding the listener is stale/foreign.
        if self._wait_for_port_health(port):
            logger.debug(f"Port {port} already serving a healthy Remember daemon; reusing it")
            return True

        # Unhealthy after grace window — attempt to reclaim by killing the listener.
        if not self._is_port_in_use(port):
            return True

        pid = self._owned_pid_on_port(port, DAEMON_PROCESS_MARKERS, "remember daemon")
        if pid is None:
            logger.warning(f"Port {port} is in use by another process")
            return False

        logger.info(f"Clearing unhealthy process on port {port} (PID {pid})")
        if self._kill_process(pid):
            logger.info(f"Stale process (PID {pid}) stopped")
            return True

        logger.warning(f"Process (PID {pid}) did not stop in time")
        return False

    def _start_daemon(self, config: dict, profile: str, extra_args: list[str] | None = None) -> bool:
        """Start the daemon in background.

        Serializes concurrent start attempts via an exclusive flock on the
        profile's lock file, so two processes calling `start()` at the same
        time cannot race into `_clear_port` and kill each other's daemons.
        Inside the lock we re-check `is_running()`; the second caller sees
        the first caller's daemon and short-circuits.

        The lock wait is bounded (see REMEMBER_EMBED_LOCK_TIMEOUT): a caller
        that cannot get in reports a startup failure instead of raising an
        opaque OS error (issue #3100).
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        paths.lock.parent.mkdir(parents=True, exist_ok=True)

        # Hold the per-profile start lock for the full startup sequence.
        # lock_file() retries until the lock is acquired or the wait budget
        # expires, on both Unix (flock) and Windows (msvcrt), so concurrent
        # callers serialize here.
        with open(paths.lock, "w") as lock_fd:
            try:
                lock_file(lock_fd)
            except ProfileLockTimeout as exc:
                logger.error(f"Cannot start daemon for profile '{profile}': {exc}")
                return False
            try:
                if self.is_running(profile):
                    logger.debug(f"Daemon for profile '{profile}' came up while waiting for start lock")
                    if profile:
                        self._register_profile(profile, paths.port, config)
                    return True
                return self._start_daemon_locked(config, profile, paths, extra_args=extra_args)
            finally:
                unlock_file(lock_fd)

    def _start_daemon_locked(
        self,
        config: dict,
        profile: str,
        paths,
        extra_args: list[str] | None = None,
    ) -> bool:
        """Perform the actual daemon startup. Caller must hold paths.lock."""
        profile_label = f"profile '{profile}'" if profile else "default profile"
        daemon_log = paths.log
        port = paths.port

        # Ensure port is free before starting (handles stale daemons from version upgrades)
        if not self._clear_port(port):
            logger.error(f"Cannot start daemon: port {port} is in use by another process")
            return False

        # _clear_port returns True without killing when a healthy remember daemon
        # already owns the port (started out-of-band, e.g. by another user or a
        # previous version upgrade). Re-check is_running so we don't spawn a
        # second daemon that would fail to bind.
        if self.is_running(profile):
            logger.debug(f"Daemon for profile '{profile}' already healthy; skipping spawn")
            if profile:
                self._register_profile(profile, port, config)
            return True

        # Load profile's .env file and merge with provided config
        # This fixes issue #305 where profile env vars were ignored
        profile_config = self._profile_manager.load_profile_config(profile)
        # Merge: profile config first, then override with explicitly provided config
        merged_config = {**profile_config, **config}
        config = merged_config

        # Build the daemon environment: every REMEMBER_* key in the merged
        # profile/explicit config, forwarded verbatim under its own name.
        #
        # There is deliberately no whitelist of known settings. There used to be
        # one — LLM/log/idle_timeout, keyed by lowercase aliases — and a setting
        # missing from it (REMEMBER_API_LLM_BASE_URL, issue #4094) was reported
        # as silently ignored while KEY/MODEL applied. remember-api owns the
        # set of settings that exist and adds to it every release; mirroring
        # that set here can only ever drift behind it.
        #
        # An explicit "" is forwarded, not skipped: that is how a caller clears
        # an inherited setting — e.g. blanking the API key for a local LLM
        # service that has no authentication (issue #3253). Only None means
        # "not specified".
        env = os.environ.copy()
        for key, value in config.items():
            if key.startswith("REMEMBER_") and value is not None:
                env[key] = str(value)

        env["REMEMBER_API_DATABASE_URL"] = self.resolve_database_url(profile, config)
        env["REMEMBER_API_HOST"] = "127.0.0.1"
        env["REMEMBER_API_WORKERS"] = "1"
        env["REMEMBER_API_WORKER_ENABLED"] = "true"

        database_url = env["REMEMBER_API_DATABASE_URL"]
        is_pg0 = database_url.startswith("pg0://")

        # Set defaults if not provided
        if "REMEMBER_API_LOG_LEVEL" not in env:
            env["REMEMBER_API_LOG_LEVEL"] = "info"

        # Tell the daemon child it was already launched in a detached session
        # (via our Popen below) so daemonize() skips the redundant re-exec.
        env["_REMEMBER_DAEMON_CHILD"] = "1"

        # Create log directory
        daemon_log.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = _parse_non_negative_int(
            env.get(ENV_DAEMON_LOG_MAX_BYTES),
            DEFAULT_DAEMON_LOG_MAX_BYTES,
            ENV_DAEMON_LOG_MAX_BYTES,
        )
        backup_count = _parse_non_negative_int(
            env.get(ENV_DAEMON_LOG_BACKUP_COUNT),
            DEFAULT_DAEMON_LOG_BACKUP_COUNT,
            ENV_DAEMON_LOG_BACKUP_COUNT,
        )
        try:
            self._rotate_daemon_log(daemon_log, max_bytes=max_bytes, backup_count=backup_count)
        except OSError as exc:
            logger.warning("Could not rotate daemon log %s: %s", daemon_log, exc)
        env["REMEMBER_API_DAEMON_LOG"] = str(daemon_log)

        # Build command
        # No --daemon: _REMEMBER_DAEMON_CHILD above already puts remember-api
        # in daemon mode, and this Popen already detaches it. Passing --daemon
        # too was redundant, and API versions older than the Popen-based
        # daemonize() (#1519) fork on it, which can deadlock native libraries
        # on macOS.
        try:
            cmd = self._find_api_command(env=env)
        except RuntimeError as exc:
            logger.error("Cannot start local daemon: %s", exc)
            console.print(f"[red]Cannot start local daemon: {exc}[/red]")
            return False
        cmd += [
            "--port",
            str(port),
        ]
        if extra_args:
            cmd.extend(extra_args)
        if _runs_a_foreign_interpreter(cmd):
            env = _strip_parent_interpreter_env(env)

        try:
            # Start daemon. Redirect stdout/stderr to the daemon log so that
            # any output from the subprocess (e.g. dependency startup,
            # Python library init messages) does not leak into the parent
            # process terminal. Python's own logging (via
            # REMEMBER_API_DAEMON_LOG) will append to the same path.
            # Popen dups the fd into the child during spawn, so the parent
            # can close its handle as soon as Popen returns.
            with open(daemon_log, "ab") as daemon_log_handle:
                daemon_process = subprocess.Popen(cmd, env=env, **_detach_popen_kwargs(daemon_log_handle))

            # Wait for daemon to be ready with rich UI
            start_time = time.time()
            last_check_time = start_time
            last_log_position = 0
            log_lines = [f"Starting daemon for {profile_label}...", ""]

            title = f"[bold cyan]Starting Daemon[/bold cyan] [dim]({profile} @ :{port})[/dim]"

            with Live(console=console, auto_refresh=False) as live:
                content = Text("\n".join(log_lines), style="dim")
                panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                live.update(panel)
                live.refresh()

                while time.time() - start_time < DAEMON_STARTUP_TIMEOUT:
                    # Polling keeps going through a missed health probe (below),
                    # so a child that actually died is detected here instead of
                    # waiting out the whole startup deadline.
                    exit_code = daemon_process.poll()
                    if exit_code is not None:
                        log_lines.append("")
                        log_lines.append(f"✗ Daemon exited during initialization (exit code {exit_code})")
                        log_lines.append(f"See full log: {daemon_log}")
                        content = Text("\n".join(log_lines), style="dim")
                        fail_title = f"[bold red]✗ Daemon Failed[/bold red] [dim]({profile} @ :{port})[/dim]"
                        panel = Panel(content, title=fail_title, border_style="red", padding=(1, 2))
                        live.update(panel)
                        live.refresh()
                        console.print()
                        return False

                    # Tail daemon logs
                    if daemon_log.exists():
                        try:
                            with open(daemon_log, "r", encoding="utf-8", errors="replace") as f:
                                f.seek(last_log_position)
                                new_lines = f.readlines()
                                last_log_position = f.tell()
                                for line in new_lines:
                                    line = line.rstrip()
                                    if line:
                                        log_lines.append(line)
                                log_lines = log_lines[-4:]
                        except Exception:
                            pass

                    if self.is_running(profile):
                        log_lines.append("")
                        log_lines.append("✓ Daemon responding, verifying stability...")
                        content = Text("\n".join(log_lines), style="dim")
                        panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                        live.update(panel)
                        live.refresh()

                        time.sleep(2)
                        if self.is_running(profile):
                            log_lines.append("✓ Daemon started successfully!")
                            log_lines.append("")
                            log_lines.append(f"Logs: {daemon_log}")

                            if is_pg0:
                                pg0_name = database_url.replace("pg0://", "")
                                pg0_path = Path.home() / ".pg0" / "instances" / pg0_name
                                log_lines.append(f"Database: {pg0_path}")

                            content = Text("\n".join(log_lines), style="dim")
                            success_title = (
                                f"[bold green]✓ Daemon Started[/bold green] [dim]({profile} @ :{port})[/dim]"
                            )
                            panel = Panel(content, title=success_title, border_style="green", padding=(1, 2))
                            live.update(panel)
                            live.refresh()
                            console.print()
                            # Register profile in metadata so CLI can discover it
                            if profile:
                                self._register_profile(profile, port, config)
                            return True
                        else:
                            log_lines.append("")
                            log_lines.append("Health probe failed after readiness; continuing to wait...")
                            content = Text("\n".join(log_lines), style="dim")
                            panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                            live.update(panel)
                            live.refresh()
                            # /health shares the daemon event loop with startup
                            # backlog processing. A single missed stability probe
                            # is not evidence that the process exited; keep polling
                            # within the existing startup deadline.
                            continue

                    # Periodic progress
                    if time.time() - last_check_time > 3:
                        elapsed = int(time.time() - start_time)
                        status_msg = f"⏳ Waiting for daemon... ({elapsed}s elapsed)"
                        if log_lines and log_lines[-1].startswith("⏳"):
                            log_lines[-1] = status_msg
                        else:
                            log_lines.append(status_msg)
                        last_check_time = time.time()

                    content = Text("\n".join(log_lines), style="dim")
                    panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                    live.update(panel)
                    live.refresh()
                    time.sleep(0.5)

            # Timeout
            log_lines.append("")
            log_lines.append("✗ Daemon failed to start (timeout)")
            log_lines.append("")
            log_lines.append(f"See full log: {daemon_log}")
            content = Text("\n".join(log_lines), style="dim")
            timeout_title = f"[bold red]✗ Daemon Failed (Timeout)[/bold red] [dim]({profile} @ :{port})[/dim]"
            panel = Panel(content, title=timeout_title, border_style="red", padding=(1, 2))
            console.print(panel)
            console.print()
            # Before this cleanup, the manager returned at the deadline but left
            # the detached child initializing. A supervisor retry could then
            # accumulate overlapping daemons that had not bound the port yet, so
            # port-based cleanup could not see them.
            _terminate_startup_process(daemon_process)
            return False

        except FileNotFoundError:
            error_msg = (
                f"Command not found: {cmd[0]}\nFull command: {' '.join(cmd)}\n\nInstall remember-api from this checkout"
            )
            error_panel = Panel(
                Text(error_msg, style="red"),
                title="[bold red]✗ Command Not Found[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print(error_panel)
            console.print()
            return False
        except Exception as e:
            error_msg = f"Failed to start daemon: {e}\n\nCommand: {' '.join(cmd)}\nLog file: {daemon_log}"
            error_panel = Panel(
                Text(error_msg, style="red"),
                title="[bold red]✗ Startup Error[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print(error_panel)
            console.print()
            return False

    def _register_profile(self, profile: str, port: int, config: dict) -> None:
        """Register a named profile in metadata so it's discoverable by the CLI.

        Saves non-secret REMEMBER_API_* settings. Credentials remain in the
        process environment unless explicitly configured in the profile.
        Silently ignores errors to avoid blocking daemon startup.
        """
        try:
            api_config = {
                k: v
                for k, v in config.items()
                if k.startswith("REMEMBER_API_") and not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", k, re.I)
            }
            if not api_config:
                return
            # Additive: seed keys the profile doesn't have yet, never overwrite
            # ones it does. create_profile rewrites the whole file, so the
            # existing keys have to be carried forward regardless (UI port, idle
            # timeout, ...), and `config` here is the merged profile + this
            # invocation's ambient REMEMBER_* environment — letting it win would
            # make a one-off `REMEMBER_API_LLM_MODEL=... remember-embed recall`
            # permanently rewrite the user's profile. The profile file is owned
            # by `configure` and the control center; a daemon start only fills
            # in what is missing.
            existing = self._profile_manager.load_profile_config(profile)
            merged = {**api_config, **existing}
            self._profile_manager.create_profile(profile, port, merged)
        except Exception as e:
            logger.debug(f"Failed to register profile '{profile}' in metadata: {e}")

    def ensure_running(self, config: dict, profile: str, extra_args: list[str] | None = None) -> bool:
        """
        Ensure daemon is running, starting it if needed.

        Args:
            config: Environment configuration dict (REMEMBER_* vars)
            profile: Profile name for isolation
            extra_args: Extra CLI arguments to pass to remember-api (e.g. ["--offline"])

        Returns:
            True if daemon is running (started or already running), False on failure
        """
        if self.is_running(profile):
            logger.debug(f"Daemon already running for profile '{profile}'")
            if profile:
                paths = self._profile_manager.resolve_profile_paths(profile)
                self._register_profile(profile, paths.port, config)
            return True
        return self._start_daemon(config, profile, extra_args=extra_args)

    def stop(self, profile: str) -> bool:
        """
        Stop the daemon for this profile.

        Args:
            profile: Profile name

        Returns:
            True if stopped successfully, False otherwise
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        port = paths.port

        # Every decision here is based on port occupancy, never on /health.
        # A daemon that is alive but busy fails the responsiveness probe
        # (issue #3169), so using it as the already-stopped guard made stop()
        # report success without sending any signal. Reclaiming the profile's
        # port from an unresponsive listener is the same policy _clear_port()
        # applies on the start path.
        if not self._is_port_in_use(port):
            logger.debug(f"Daemon not running for profile '{profile}'")
            return True

        pid = self._owned_pid_on_port(port, DAEMON_PROCESS_MARKERS, "remember daemon")
        if pid is None:
            logger.warning(f"Port {port} is bound but no Remember daemon could be identified on it")
            return False

        logger.debug(f"Found daemon PID {pid} on port {port}")
        if not self._kill_process(pid):
            logger.warning(f"Daemon process (PID {pid}) did not stop in time")
            return False

        # The process is gone; wait for the listener to disappear so a
        # follow-up start doesn't race the closing socket.
        for _ in range(30):
            if not self._is_port_in_use(port):
                return True
            time.sleep(0.1)

        return not self._is_port_in_use(port)
