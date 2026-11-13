"""Paths, configuration and startup locking for the local daemon."""

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from .remember_home import remember_home

logger = logging.getLogger(__name__)

# ==============================================================================
# Cross-platform file locking implementation
# ==============================================================================
# Why not use a library like portalocker or fasteners?
#
# 1. Minimal dependency: Our use case is narrow — exclusive locking for
#    metadata persistence and daemon-start serialization. Adding a new
#    dependency for that is unnecessary.
#
# 2. Portability: We only need to support the two major platforms (Unix and
#    Windows), both of which expose a non-blocking exclusive lock primitive
#    that we drive from one shared retry loop below.
#
# 3. Maintainability: The code is straightforward and has no external
#    dependencies to track or update. The locking logic is localized here,
#    making it easy to understand and modify if needed.
#
# Locks are acquired with a bounded wait, never an unbounded blocking call:
# the Windows `msvcrt.LK_LOCK` mode retries internally exactly 10 times and
# then raises, which made concurrent daemon starts for the same daemon fail
# non-deterministically (issue #3100). Both platforms now take the
# non-blocking mode in a retry loop with exponential backoff, and time out
# with an error naming the lock file and its recorded holder.
# ==============================================================================

ENV_LOCK_TIMEOUT = "REMEMBER_EMBED_LOCK_TIMEOUT"
# The daemon-start path holds the daemon lock for the whole startup sequence
# (up to REMEMBER_EMBED_DAEMON_STARTUP_TIMEOUT, 180s by default), so the wait
# budget has to comfortably exceed that or a legitimate concurrent start would
# time out waiting for the winner.
DEFAULT_LOCK_TIMEOUT = 300.0
_LOCK_RETRY_INITIAL = 0.01
_LOCK_RETRY_MAX = 0.25


class DaemonLockTimeout(TimeoutError):
    """Raised when an exclusive lock could not be acquired within the timeout."""


def _default_lock_timeout() -> float:
    """Lock wait budget in seconds, overridable via REMEMBER_EMBED_LOCK_TIMEOUT."""
    raw = os.getenv(ENV_LOCK_TIMEOUT)
    if raw is None:
        return DEFAULT_LOCK_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    # Reject nan/inf/non-positive: an unbounded wait is the bug being fixed.
    if not math.isfinite(value) or value <= 0:
        logger.warning("Invalid %s=%r; using %s", ENV_LOCK_TIMEOUT, raw, DEFAULT_LOCK_TIMEOUT)
        return DEFAULT_LOCK_TIMEOUT
    return value


if sys.platform != "win32":
    import fcntl

    def _try_lock(file_obj: IO[str]) -> bool:
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _release_lock(file_obj: IO[str]) -> None:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
else:
    import msvcrt

    # msvcrt.locking(fd, mode, N) locks N bytes starting at the *current* file
    # position, and LK_UNLCK must be called with the file pointer positioned
    # at the start of the same region. Callers typically lock immediately
    # after `open(..., "w")` (position 0), then write JSON (advancing the
    # position), then unlock — at which point the unlock request targets a
    # byte past the data and Windows returns EACCES. Seek to 0 on both sides
    # so lock and unlock always act on byte 0.
    def _try_lock(file_obj: IO[str]) -> bool:
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _release_lock(file_obj: IO[str]) -> None:
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)


def _owner_path(file_obj: IO[str]) -> Optional[Path]:
    """Sidecar file recording the PID currently holding `file_obj`'s lock.

    The lock file itself is opened with mode "w" by callers, so a waiter
    truncates it before it ever tries to lock — the holder's identity cannot
    live there. The sidecar is only ever written by the lock holder.
    """
    name = getattr(file_obj, "name", None)
    if not isinstance(name, str):
        return None
    return Path(name + ".owner")


def _record_lock_owner(file_obj: IO[str]) -> None:
    path = _owner_path(file_obj)
    if path is None:
        return
    try:
        path.write_text(str(os.getpid()))
    except OSError:
        logger.debug("Could not record lock owner for %s", path, exc_info=True)


def _clear_lock_owner(file_obj: IO[str]) -> None:
    path = _owner_path(file_obj)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not clear lock owner for %s", path, exc_info=True)


def _describe_lock_holder(file_obj: IO[str]) -> str:
    path = _owner_path(file_obj)
    if path is None:
        return ""
    try:
        holder = path.read_text().strip()
    except OSError:
        return ""
    return f" (held by PID {holder})" if holder else ""


def lock_file(file_obj: IO[str], timeout: Optional[float] = None) -> None:
    """Acquire an exclusive lock on `file_obj`, waiting up to `timeout` seconds.

    Raises:
        DaemonLockTimeout: if the lock is still held when the budget expires.
    """
    budget = _default_lock_timeout() if timeout is None else timeout
    deadline = time.monotonic() + max(budget, 0.0)
    delay = _LOCK_RETRY_INITIAL
    while True:
        if _try_lock(file_obj):
            _record_lock_owner(file_obj)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            name = getattr(file_obj, "name", "<unknown>")
            raise DaemonLockTimeout(
                f"Timed out after {budget:g}s waiting for the lock on {name}{_describe_lock_holder(file_obj)}. "
                f"Set {ENV_LOCK_TIMEOUT} to wait longer, or remove the lock file if the holder is gone."
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, _LOCK_RETRY_MAX)


def unlock_file(file_obj: IO[str]) -> None:
    """Release a lock acquired with lock_file()."""
    _clear_lock_owner(file_obj)
    _release_lock(file_obj)


def parse_env_file(path: Path) -> dict[str, str]:
    """Read a daemon .env into ``{KEY: value}``, the way dotenv does.

    Skips blanks and comments, accepts ``export KEY=value``, and drops one pair
    of matching surrounding quotes, so a hand-written ``KEY="value"`` yields
    ``value`` rather than forwarding the quotes into a credential (#5301).
    """
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key.strip()] = value
    return env


@dataclass(frozen=True)
class DaemonPaths:
    config: Path
    lock: Path
    log: Path
    port: int


def saved_daemon_port() -> int | None:
    """Return the last successfully started daemon's port, if recorded."""
    port_file = remember_home() / "profiles" / "coding-agent.port"
    if not port_file.exists():
        return None
    try:
        port = int(port_file.read_text().strip())
    except ValueError:
        return None
    return port if 1 <= port <= 65535 else None


def daemon_paths() -> DaemonPaths:
    """Resolve the one daemon's state and configured loopback port."""
    state = remember_home() / "profiles"
    config = state / "coding-agent.env"
    settings = parse_env_file(config)
    saved_port = saved_daemon_port()
    raw_port = os.environ.get("REMEMBER_API_PORT") or saved_port or settings.get("REMEMBER_API_PORT") or "8888"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"Invalid local daemon port: {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid local daemon port: {raw_port!r}")
    return DaemonPaths(config=config, lock=state / "coding-agent.lock", log=state / "coding-agent.log", port=port)


def record_daemon_port(port: int) -> None:
    """Remember the running daemon's port for later status and stop commands."""
    state = remember_home() / "profiles"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / f".coding-agent.port.{os.getpid()}"
    temporary.write_text(str(port), encoding="utf-8")
    temporary.replace(state / "coding-agent.port")
