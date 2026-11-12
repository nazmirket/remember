"""
Helpers for managing the local daemon.
"""

from .daemon_embed_manager import DaemonEmbedManager
from .profile_manager import resolve_active_profile

_manager = DaemonEmbedManager()


def get_daemon_url(profile: str | None = None) -> str:
    """Get daemon URL for a profile.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        URL for daemon.
    """
    if profile is None:
        profile = resolve_active_profile()
    return _manager.get_url(profile)


def ensure_daemon_running(config: dict, profile: str | None = None, extra_args: list[str] | None = None) -> bool:
    """
    Ensure daemon is running, starting it if needed.

    Args:
        config: Overrides for the daemon, keyed by environment variable name
                ("REMEMBER_API_LLM_API_KEY", ...). Every REMEMBER_* key is
                forwarded to the daemon process; the profile's own .env is
                merged underneath it.
        profile: Profile name (None = resolve from priority).
        extra_args: Extra CLI arguments to pass to remember-api (e.g. ["--offline"]).

    Returns:
        True if daemon is running.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.ensure_running(config, profile, extra_args=extra_args)


def stop_daemon(profile: str | None = None) -> bool:
    """Stop the running daemon and wait for it to fully stop.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon stopped successfully.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.stop(profile)


def is_daemon_running(profile: str | None = None) -> bool:
    """Check if daemon is running for a profile.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon is running and responsive.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.is_running(profile)
