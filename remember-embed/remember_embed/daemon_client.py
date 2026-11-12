"""Helpers for managing the local daemon."""

from .daemon_embed_manager import DaemonEmbedManager

_manager = DaemonEmbedManager()


def get_daemon_url() -> str:
    """Return the local daemon URL."""
    return _manager.get_url()


def ensure_daemon_running(config: dict) -> bool:
    """Start the local daemon if needed."""
    return _manager.ensure_running(config)


def stop_daemon() -> bool:
    """Stop the local daemon."""
    return _manager.stop()


def is_daemon_running() -> bool:
    """Check whether the local daemon is responding."""
    return _manager.is_running()
