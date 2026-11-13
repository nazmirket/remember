#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
(cd "$ROOT_DIR/api" && uv sync --frozen --extra embedded-db)
(cd "$ROOT_DIR/integrations/coding-agents" && npm ci && npm run build)
printf 'Local memory dependencies are ready. See integrations/coding-agents/README.md for installation.\n'
