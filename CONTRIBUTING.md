# Contributing

This repository is a local, headless memory layer for Codex and Claude Code. See
[README.md](README.md) for the project overview and the
[coding-agent setup guide](remember-integrations/coding-agents/README.md) for
installation and configuration.

## Set up a development checkout

Install Python, [uv](https://docs.astral.sh/uv/), Node.js, npm, and Rust, then run:

```sh
./scripts/dev/setup.sh
```

The script synchronizes the API environment with embedded PostgreSQL support,
installs the coding-agent dependencies, and builds the integration. Add the
appropriate local model extras or hosted-provider credentials for your setup.
The setup guide explains the available choices.

For direct API development, start the service on loopback:

```sh
./scripts/dev/start.sh
```

## Verify changes

Choose checks that cover the behavior changed. The main project checks are:

```sh
(cd remember-integrations/coding-agents && npm test && npx tsc --noEmit && npm run build)
PYTHONPATH=remember-embed .venv/bin/python -m pytest remember-embed/tests -q
(cd remember-system-tests && uv run pytest tests/)
./scripts/hooks/lint.sh
```

The system tests use local PostgreSQL and deterministic model stubs. They do not
replace testing real Codex or Claude Code sessions or live model providers.
Follow the project conventions in [CLAUDE.md](CLAUDE.md), including preserving
the PostgreSQL migration chain and bank isolation.

After changing API endpoints, regenerate the schema with
`./scripts/generate-openapi.sh`. When changing the marked companion-skill
sections in the coding-agent README, run `npm run skill:build` in that directory.

## Reporting problems

Use this repository's issue tracker for bugs and feature requests. Include the
steps to reproduce, expected and actual behavior, and relevant environment
versions. Report security issues privately as described in [SECURITY.md](SECURITY.md).
