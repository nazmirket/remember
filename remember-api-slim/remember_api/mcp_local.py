"""
Local MCP server entry point for use with Claude Code (HTTP transport).

This is a thin wrapper around the main remember-api server that pre-configures
sensible defaults for local use (embedded PostgreSQL via pg0, warning log level).

The full API runs on localhost:8888. Configure Claude Code's MCP settings:
    claude mcp add --transport http remember http://localhost:8888/mcp/

Or pinned to a specific bank (single-bank mode):
    claude mcp add --transport http remember http://localhost:8888/mcp/default/

Run with:
    remember-local-mcp

Environment variables:
    REMEMBER_API_LLM_API_KEY: API key for providers that require one.
    REMEMBER_API_LLM_PROVIDER: Optional. LLM provider (default: "openai").
    REMEMBER_API_LLM_MODEL: Optional. LLM model (default: "gpt-4o-mini").
    REMEMBER_API_DATABASE_URL: Optional. Override database URL (default: pg0://remember-mcp).
"""

import os


def main() -> None:
    """Start the Remember API server with local defaults."""
    # Seeds the environment that RememberConfig then parses — a write, not a read
    # that bypasses the config. It has to happen before the config is first built,
    # so it cannot be expressed as a config field.
    from remember_api.config import ENV_DATABASE_URL

    os.environ.setdefault(ENV_DATABASE_URL, "pg0://remember-mcp")

    from remember_api.main import main as api_main

    api_main()


if __name__ == "__main__":
    main()
