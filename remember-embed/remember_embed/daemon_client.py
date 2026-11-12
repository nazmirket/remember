"""
CLI utilities for daemon and CLI management.

This module provides CLI-specific functions for managing the daemon
and the remember Rust CLI binary.
"""

import logging
import os
from pathlib import Path

from .daemon_embed_manager import DaemonEmbedManager
from .profile_manager import resolve_active_profile
from .remember_home import remember_home

logger = logging.getLogger(__name__)

# Singleton manager instance
_manager = DaemonEmbedManager()

# CLI paths - check multiple locations
CLI_INSTALL_DIRS = [
    Path.home() / ".local" / "bin",  # Standard location from get-cli installer
    remember_home() / "bin",  # Alternative location
]


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


def find_cli_binary() -> Path | None:
    """Find the remember CLI binary in known locations or PATH."""
    import shutil
    import sys

    binary_names = ("remember.exe", "remember") if sys.platform == "win32" else ("remember",)

    # Check standard install locations. On Windows we look for both the
    # `.exe` (produced by `cargo build`) and the bare `remember` in case a
    # user dropped a WSL/Git-Bash build there.
    for install_dir in CLI_INSTALL_DIRS:
        for name in binary_names:
            binary = install_dir / name
            if binary.exists() and (sys.platform == "win32" or os.access(binary, os.X_OK)):
                return binary

    # Check PATH. shutil.which handles PATHEXT on Windows, so passing
    # "remember" finds remember.exe automatically.
    path_binary = shutil.which("remember")
    if path_binary:
        return Path(path_binary)

    return None


def is_cli_installed() -> bool:
    """Check if the remember CLI is installed."""
    return find_cli_binary() is not None


def ensure_cli_installed() -> bool:
    """Use an installed Remember CLI; do not download a binary."""
    if is_cli_installed():
        return True
    import sys

    print(
        "Remember CLI not found. Install it from this checkout with: cargo install --path remember-cli", file=sys.stderr
    )
    return False


def run_cli(args: list[str], config: dict, profile: str | None = None) -> int:
    """
    Run the remember CLI with the given arguments.

    Ensures daemon is running (unless REMEMBER_API_URL is already set) and passes the API URL.

    Args:
        args: CLI arguments (e.g., ["memory", "retain", "bank", "content"])
        config: Configuration dict with llm settings
        profile: Profile name (None = resolve from priority)

    Returns:
        Exit code from CLI
    """
    import subprocess
    import sys

    if profile is None:
        profile = resolve_active_profile()

    # Ensure CLI is installed
    if not ensure_cli_installed():
        return 1

    cli_binary = find_cli_binary()
    if not cli_binary:
        print("Error: Remember CLI not found", file=sys.stderr)
        return 1

    # Build environment
    env = os.environ.copy()

    # Check if user wants to use external API
    api_url = env.get("REMEMBER_EMBED_API_URL")

    if not api_url:
        # No external API specified - ensure our daemon is running
        if not ensure_daemon_running(config, profile):
            print("Error: Failed to start daemon", file=sys.stderr)
            return 1
        api_url = get_daemon_url(profile)
    else:
        # Using external API - skip daemon startup
        logger.debug(f"Using external API at {api_url}")

    # Set the API URL for the CLI (using the standard REMEMBER_API_URL var that the CLI expects)
    env["REMEMBER_API_URL"] = api_url

    # Pass through API token if set (using the standard REMEMBER_API_KEY var that the CLI expects)
    api_token = env.get("REMEMBER_EMBED_API_TOKEN")
    if api_token:
        env["REMEMBER_API_KEY"] = api_token

    # Run CLI
    try:
        result = subprocess.run(
            [str(cli_binary)] + args,
            env=env,
        )
        return result.returncode
    except Exception as e:
        print(f"Error running CLI: {e}", file=sys.stderr)
        return 1
