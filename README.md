# Local Hindsight memory

A headless, persistent memory layer for Codex and Claude Code. Both agents share a local API daemon and a local pg0 database. The memory engine retains and retrieves observations, supports reasoning and Pages, and keeps the background worker that maintains them.

The service runs on demand on loopback. Memory storage is local; inference can use OpenAI, Anthropic, their Codex or Claude Code adapters, or a local model. A hosted provider receives the content it processes. Server logs, metrics, and optional in-process traces stay local.

## Setup

See [the coding-agent setup guide](hindsight-integrations/coding-agents/README.md) for prerequisites, installation, configuration and diagnostics. Install from this checkout so the daemon runs this fork's API and embedded manager.

## Development

- `hindsight-api-slim/`: API, memory engine, worker and database migrations.
- `hindsight-embed/`: local daemon and profile lifecycle.
- `hindsight-integrations/coding-agents/`: shared integration, Codex and Claude Code hooks and MCP tools.
- `hindsight-system-tests/`: blackbox memory tests using local database and deterministic model stubs.
- `hindsight-clients/python/` and `hindsight-cli/`: the system-test client and standalone command-line tool.
- `schemas/`: generated local API schema.

```sh
(cd hindsight-integrations/coding-agents && npm ci && npm test && npm run build)
(cd hindsight-api-slim && uv sync --frozen --extra embedded-db)
(cd hindsight-system-tests && uv run pytest tests/)
PYTHONPATH=hindsight-embed .venv/bin/python -m pytest hindsight-embed/tests
```

For direct API development, `./scripts/dev/start.sh` starts the API on loopback. Run `./scripts/hooks/lint.sh` after Python or TypeScript changes.
