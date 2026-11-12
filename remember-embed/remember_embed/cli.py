"""Manage the local Remember daemon."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from . import daemon_client
from .daemon_embed_manager import DaemonEmbedManager
from .local_state import daemon_paths, parse_env_file


def get_config() -> dict[str, str]:
    """Merge the local daemon's saved settings with this process's overrides."""
    config = parse_env_file(daemon_paths().config)
    config.update({key: value for key, value in os.environ.items() if key.startswith("REMEMBER_")})
    provider = config.get("REMEMBER_API_LLM_PROVIDER", "openai")
    key_env = {"openai": "OPENAI_API_KEY", "openai-responses": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
        provider
    )
    if key_env and "REMEMBER_API_LLM_API_KEY" not in config and os.environ.get(key_env):
        config["REMEMBER_API_LLM_API_KEY"] = os.environ[key_env]
    return config


def main() -> None:
    """Run one local daemon lifecycle command."""
    parser = argparse.ArgumentParser(prog="remember-embed", description="Manage the local Remember daemon")
    commands = parser.add_subparsers(dest="command", required=True)
    daemon = commands.add_parser("daemon", help="Manage the local daemon")
    actions = daemon.add_subparsers(dest="action", required=True)
    actions.add_parser("start", help="Start the daemon")
    actions.add_parser("stop", help="Stop the daemon")
    actions.add_parser("status", help="Show daemon status")
    logs = actions.add_parser("logs", help="Show daemon logs")
    logs.add_argument("--follow", "-f", action="store_true")
    logs.add_argument("--lines", "-n", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("REMEMBER_EMBED_LOG_LEVEL", "INFO").upper(), stream=sys.stderr)
    paths = daemon_paths()
    if args.action == "start":
        success = daemon_client.ensure_daemon_running(get_config())
        if not success:
            print("Failed to start the local daemon", file=sys.stderr)
        raise SystemExit(0 if success else 1)
    if args.action == "stop":
        success = daemon_client.stop_daemon()
        print("Daemon stopped" if success else "Failed to stop the local daemon")
        raise SystemExit(0 if success else 1)
    if args.action == "status":
        if not daemon_client.is_daemon_running():
            print("Daemon is not running")
            raise SystemExit(1)
        database = DaemonEmbedManager().get_database_url().removeprefix("pg0://")
        print(f"Daemon is running\n  URL: http://127.0.0.1:{paths.port}\n  Logs: {paths.log}")
        print(f"  Database: {Path.home() / '.pg0' / 'instances' / database}")
        return
    if not paths.log.exists():
        print(f"No daemon logs found: {paths.log}", file=sys.stderr)
        raise SystemExit(1)
    if args.follow:
        try:
            subprocess.run(["tail", "-f", str(paths.log)], check=False)
        except KeyboardInterrupt:
            pass
    else:
        with paths.log.open(encoding="utf-8", errors="replace") as stream:
            print("".join(stream.readlines()[-args.lines :]), end="")


if __name__ == "__main__":
    main()
