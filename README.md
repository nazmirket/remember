# Remember

Remember is a headless, persistent memory layer for Codex and Claude Code. Both agents share a local API daemon and a local pg0 database. The memory engine retains and retrieves observations, supports reasoning and Pages, and keeps the background worker that maintains them.

Remember is maintained at [nazmirket/remember](https://github.com/nazmirket/remember). It originated from [Hindsight](https://github.com/vectorize-io/hindsight), created by Vectorize, and is developed as its own project. The original copyright notice and MIT license are preserved in [LICENSE](LICENSE).

The service runs on demand on loopback. Memory storage uses local PostgreSQL with pgvector and native text search; inference can use OpenAI, Anthropic, their Codex or Claude Code adapters, or a local model. A hosted provider receives the content it processes. Server logs, metrics, and optional in-process traces stay local.

## Setup

See [the coding-agent setup guide](integrations/coding-agents/README.md) for prerequisites, installation, configuration and diagnostics. Install from this checkout so the daemon runs Remember's API and embedded manager.

## Configuration

Remember stores its configuration and daemon state in `~/.remember` by default. Set `REMEMBER_HOME` to choose another location or `REMEMBER_CONFIG` to select the coding-agent configuration file. The embedded database is separate, under `~/.pg0/instances/remember-embed-coding-agent`; changing `REMEMBER_HOME` does not move it. Command-line tools and environment settings use the `remember` and `REMEMBER_` prefixes.

## Development

- `api/`: API, memory engine, worker and database migrations.
- `embed/`: local daemon lifecycle.
- `integrations/coding-agents/`: shared integration, Codex and Claude Code hooks and MCP tools.
- `system-tests/`: blackbox memory tests using local database and deterministic model stubs.
- `clients/python/`: the system-test API client.
- `schemas/`: generated local API schema.

```sh
(cd integrations/coding-agents && npm ci && npm test && npm run build)
(cd api && uv sync --frozen --extra embedded-db)
(cd system-tests && uv run pytest tests/)
PYTHONPATH=embed .venv/bin/python -m pytest embed/tests
```

For direct API development, `./scripts/dev/start.sh` starts the API on loopback. Run `./scripts/hooks/lint.sh` after Python or TypeScript changes.
