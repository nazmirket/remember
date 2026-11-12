#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
(cd "$ROOT_DIR/remember-api" && uv sync --frozen --extra embedded-db)
(cd "$ROOT_DIR/remember-integrations/coding-agents" && npm ci && npm run build)
printf 'Local memory dependencies are ready. See remember-integrations/coding-agents/README.md for installation.\n'
