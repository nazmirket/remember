"""
Command-line interface for Remember API.

Run the server with:
    remember-api

Run as background daemon:
    remember-api --daemon

Stop with Ctrl+C.
"""

import argparse
import asyncio
import atexit
import dataclasses
import errno
import os
import signal
import socket
import sys
import time
import warnings
from typing import Any

import uvicorn

from . import __version__
from .banner import print_banner
from .config import (
    DEFAULT_ACCESS_LOG,
    DEFAULT_HOST,
    ENV_ACCESS_LOG,
    ENV_WORKERS,
    RememberConfig,
    _get_raw_config,
    load_dotenv_for_entrypoint,
)
from .daemon import (
    DEFAULT_DAEMON_PORT,
    ENV_DAEMON_CHILD,
    daemonize,
)

# Keep CLI argument parsing and diagnostics lightweight until startup needs the engine.
_LAZY_IMPORTS: "dict[str, tuple[str, str]]" = {
    "MemoryEngine": (".", "MemoryEngine"),
    "create_app": (".api", "create_app"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __package__), attribute)
    globals()[name] = value
    return value


# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global reference for cleanup
_memory: "MemoryEngine | None" = None


def _cleanup():
    """Synchronous cleanup function to stop resources on exit."""
    global _memory
    if _memory is not None and _memory._pg0 is not None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_memory._pg0.stop())
            loop.close()
            print("\npg0 stopped.")
        except Exception as e:
            print(f"\nError stopping pg0: {e}")


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM to ensure cleanup."""
    print(f"\nReceived signal {signum}, shutting down...")
    _cleanup()
    sys.exit(0)


def _port_bind_error(host: str, port: int) -> OSError | None:
    """Return the error uvicorn's bind would raise for host:port, or None if it would succeed.

    uvicorn only binds after the app's startup has run — embedded PostgreSQL, model loading,
    migrations — so an occupied port otherwise costs a full initialization before failing, which
    a supervisor with Restart=always turns into a restart storm (#4281).

    This mirrors uvicorn's own TCP bind (Config.bind_socket: address family from the host string,
    SO_REUSEADDR) so it can only fail where uvicorn would, and it never calls listen(): nothing can
    connect to the probe, and the socket is released before uvicorn binds for real. A listener
    appearing between the two is not caught here; uvicorn then fails exactly as it did before.
    """
    family = socket.AF_INET6 if host and ":" in host else socket.AF_INET
    with socket.socket(family=family) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            return exc
    return None


# How long an "address in use" is retried before giving up. Before the pre-flight probe, the
# ~10s of initialization ahead of uvicorn's bind silently absorbed a previous instance that was
# still releasing the port (a restart that doesn't wait for the old process to exit); failing on
# the first attempt would turn that into a spurious exit. Matches uvicorn's graceful-shutdown cap.
_PORT_IN_USE_GRACE_SECONDS = 5.0
_PORT_IN_USE_RETRY_INTERVAL = 0.5
# Windows reports WSAEADDRINUSE (10048) rather than errno.EADDRINUSE.
_ADDR_IN_USE_ERRNOS = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE)}


def _wait_for_port(host: str, port: int) -> OSError | None:
    """Probe the bind, retrying "address in use" for a short grace window; return the final error."""
    deadline = time.monotonic() + _PORT_IN_USE_GRACE_SECONDS
    error = _port_bind_error(host, port)
    while error is not None and error.errno in _ADDR_IN_USE_ERRNOS and time.monotonic() < deadline:
        time.sleep(_PORT_IN_USE_RETRY_INTERVAL)
        error = _port_bind_error(host, port)
    return error


@dataclasses.dataclass(frozen=True)
class ParsedCliArgs:
    args: argparse.Namespace
    explicit_port: bool


def _parse_cli_args(argv: list[str], config: RememberConfig) -> ParsedCliArgs:
    parser = argparse.ArgumentParser(
        prog="remember-api",
        description="Remember API Server",
    )

    # Server options
    parser.add_argument(
        "--host",
        default=config.host or DEFAULT_HOST,
        choices=["127.0.0.1", "::1", "localhost"],
        help="Loopback host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Port to bind to (default: {config.port}, env: REMEMBER_API_PORT)",
    )
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {config.log_level}, env: REMEMBER_API_LOG_LEVEL)",
    )

    # Access log options
    parser.add_argument(
        "--access-log",
        action="store_true",
        default=config.access_log,
        help=f"Enable access log (env: {ENV_ACCESS_LOG}, default: {DEFAULT_ACCESS_LOG})",
    )
    parser.add_argument(
        "--no-access-log",
        dest="access_log",
        action="store_false",
        help="Disable access log (overrides env and default)",
    )

    # Daemon mode options
    parser.add_argument(
        "--daemon",
        action="store_true",
        help=f"Run as background daemon (uses port {DEFAULT_DAEMON_PORT})",
    )

    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error(f"local daemon requires a loopback host, got {args.host!r}")

    explicit_port = hasattr(args, "port")
    if not explicit_port:
        args.port = config.port

    return ParsedCliArgs(args=args, explicit_port=explicit_port)


def main():
    """Main entry point for the CLI."""
    global _memory

    load_dotenv_for_entrypoint()

    # Arm profiling here, after .env is loaded and before anything starts serving, so a
    # report covers the run rather than beginning halfway through it. No-op unless
    # REMEMBER_API_PROFILE is set.
    from remember_api.profiling import install as _install_profiling

    _install_profiling()

    # One local process owns the daemon and its concurrency limits.
    os.environ[ENV_WORKERS] = "1"
    config = _get_raw_config()

    parsed_cli_args = _parse_cli_args(sys.argv[1:], config)
    args = parsed_cli_args.args

    # Daemon mode handling.
    # is_daemon_child is True when we are the re-exec'd child spawned by
    # daemonize() or by remember-embed's DaemonEmbedManager.  The child
    # does not have --daemon in its argv, but must still behave as a daemon
    # (resolve host/port, suppress banner, etc.).
    is_daemon_child = os.environ.get(ENV_DAEMON_CHILD) == "1"
    is_daemon = args.daemon or is_daemon_child

    if is_daemon and not parsed_cli_args.explicit_port:
        args.port = DEFAULT_DAEMON_PORT

    # Fail before any expensive initialization if the port cannot be bound. For --daemon this
    # runs in the foreground parent too, so the error reaches the terminal instead of the log.
    bind_error = _wait_for_port(args.host, args.port)
    if bind_error is not None:
        print(f"Error: cannot bind {args.host}:{args.port}: {bind_error}", file=sys.stderr)
        sys.exit(1)

    from .pg0 import parse_pg0_url

    if not parse_pg0_url(config.database_url).is_pg0 or config.read_database_url or config.migration_database_url:
        print("Error: the local daemon requires its embedded pg0 database", file=sys.stderr)
        sys.exit(1)

    if is_daemon:
        # Detach into background (parent re-execs and exits; child redirects
        # stdio to log file).  No lockfile needed — port binding prevents
        # duplicate daemons.
        daemonize()

    # Print banner (not in daemon mode)
    if not is_daemon:
        print()
        print_banner()

    # Configure Python logging based on log level
    # Update config with CLI override if provided
    if args.log_level != config.log_level:
        config = dataclasses.replace(config, host=args.host, port=args.port, log_level=args.log_level)
    config.configure_logging()
    if not is_daemon:
        config.log_config()

    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Bind the lazily-resolved names through the MODULE, not as bare globals: a module
    # `__getattr__` (PEP 562) is consulted for `module.X` access, but NOT for a plain global lookup
    # inside this module's own functions — that raises NameError. Reading them off the module object
    # both triggers the lazy import and picks up anything a test has patched onto the module, which
    # a local `from .x import y` would silently shadow.
    _this = sys.modules[__name__]
    MemoryEngine = _this.MemoryEngine
    create_app = _this.create_app
    _memory = MemoryEngine(run_migrations=config.run_migrations_on_startup)
    app = create_app(memory=_memory, initialize_memory=True)

    # Check for uvloop/winloop availability
    loop_impl = "asyncio"
    if sys.platform == "win32":
        try:
            import winloop

            winloop.install()  # Patches asyncio globally — uvicorn uses "asyncio" but gets winloop
            loop_impl = "asyncio"  # Tell uvicorn "asyncio" — it's now winloop underneath
            print("winloop installed as asyncio event loop policy (Windows uvloop port)")
        except ImportError:
            print("winloop not installed, using default asyncio event loop")
    else:
        try:
            import uvloop  # noqa: F401

            loop_impl = "uvloop"
            print("uvloop available, will use for event loop")
        except ImportError:
            print("uvloop not installed, using default asyncio event loop")

    # `dict[str, Any]`, explicitly: this is a kwargs BAG, assembled conditionally from parsed
    # CLI args below, and unpacked into a signature with ~30 differently-typed parameters.
    # Inferred, its value type is the union of everything put in it, and the `**` unpack is
    # then checked as if every key could be every type -- one diagnostic per uvicorn
    # parameter, none of them a real defect. The types that matter here are argparse's, and
    # they are checked where the args are declared.
    uvicorn_config: dict[str, Any] = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
        "proxy_headers": False,
        "ws": "wsproto",  # Use wsproto instead of websockets to avoid deprecation warnings
        "loop": loop_impl,  # Explicitly set event loop implementation
        "timeout_keep_alive": 30,  # Exceed aiohttp's 15s client timeout so the client always closes first
        "timeout_graceful_shutdown": 5,  # Cap graceful shutdown at 5s; also enables force-kill on second Ctrl+C
    }

    # Print startup info (not in daemon mode)
    if not is_daemon:
        from .banner import print_startup_info

        print_startup_info(
            host=args.host,
            port=args.port,
            database_url=config.database_url,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embeddings_provider=(
                f"{(config.query_embeddings or config).embeddings_provider} (queries), "
                f"{(config.passage_embeddings or config).embeddings_provider} (passages)"
                if config.query_embeddings or config.passage_embeddings
                else config.embeddings_provider
            ),
            # Recall ranks with the decision model instead when one is set up for it.
            reranker_provider=(
                "decision model" if config.decision_model_config("recall") else config.reranker_provider
            ),
            version=__version__,
            vector_extension=config.vector_extension,
            text_search_extension=config.text_search_extension,
        )

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
