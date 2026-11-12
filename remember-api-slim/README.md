# Local memory API

This package contains the local API, background worker, memory engine, and PostgreSQL migrations used by the Codex and Claude Code integrations.

Set up and run Remember from its checkout as described in the [project README](../README.md). For direct API development, run `./scripts/dev/start.sh` from the repository root. The API listens on loopback and uses the configured local database. The coding-agent integration normally manages the local daemon automatically.

The API retains memories and provides retrieval, reasoning, entities, and Pages. Its schema is generated at [schemas/openapi.json](../schemas/openapi.json).
