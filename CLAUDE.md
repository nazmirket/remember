# Project guidance

Remember is a local, headless memory layer for Codex and Claude Code, maintained at `nazmirket/remember`. It originated from Hindsight; retain the upstream copyright notice and license. Both agents share an on-demand loopback API, a profile-specific pg0 database and the memory worker. Model inference can be local or hosted. Preserve session capture, retain/recall/reflect, Pages, consolidation, graph and temporal retrieval, git seeding, surveys, project isolation and linked-worktree sharing.

## Development

- `remember-api-slim/`: FastAPI service, memory engine, worker and PostgreSQL migrations.
- `remember-embed/`: daemon lifecycle, profiles and headless command forwarding.
- `remember-integrations/coding-agents/`: shared agent integration, Codex/Claude hooks and MCP tools.
- `remember-clients/python/`: Python API client, including the system-test client.
- `remember-cli/`: standalone CLI and its internal generated Rust client.
- `remember-system-tests/`: blackbox memory tests using pg0 and deterministic model stubs.
- `schemas/`: generated OpenAPI and bank-template schemas.

```sh
./scripts/dev/setup.sh
(cd remember-integrations/coding-agents && npm test && npx tsc --noEmit && npm run build)
PYTHONPATH=remember-embed .venv/bin/python -m pytest remember-embed/tests -q
(cd remember-system-tests && uv run pytest tests/)
./scripts/hooks/lint.sh
```

Run `./scripts/generate-openapi.sh` after changing API endpoints. Generated API data lives in `schemas/openapi.json`; the Rust CLI's internal client reads that file. The coding-agent companion skill is generated from the marked sections of its README: run `npm run skill:build` in `remember-integrations/coding-agents` after changing those sections.

## Conventions

Read `.claude/skills/code-review/SKILL.md` before code changes and perform its self-review before delivery. Python targets 3.11+, uses async I/O and Ruff with a 120-character line limit. Use typed structures for known structured data; preserve bank scoping in every database query. Avoid unrelated formatting or behavior changes.

PostgreSQL migrations live in `remember-api-slim/remember_api/alembic/versions/`. Preserve existing revision identifiers and the revision chain. New migrations must use `run_for_dialect(pg=...)` and provide `_pg_upgrade`; use an explicit no-op for historical PostgreSQL-free revisions. Test migrations against a fresh pg0 database as well as relevant upgrade behavior.

Use deterministic tests for mechanics. Changes to model interpretation require a real-model evaluation, not assertions that rely on a mock model to reproduce language behavior. Add blackbox coverage when user-visible memory capabilities interact across components. Isolate test homes, ports and databases; synthetic hooks do not prove actual host capture/recall.
