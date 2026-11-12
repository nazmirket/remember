#!/bin/sh
set -eu

: "${REMEMBER_E2E_INSTALL_COMMAND:?missing harness install command}"
: "${REMEMBER_CONFIG:?missing Remember config path}"

# The tarball is produced from the checkout under test. Installing it here, instead of bind-mounting
# its source, verifies the published-package surface: bins, bundled hooks, MCP server, and skill.
npm install --global /plugin/*.tgz

# Credentials arrive on a READ-ONLY mount and are copied into the location the CLI expects.
#
# Copying rather than bind-mounting onto the target is deliberate. Some harnesses authenticate from
# a single file (codex, cursor, grok) but others keep a whole directory with live SQLite session
# stores (copilot, cline) that the CLI opens read-write — a read-only mount there fails at startup.
# Copying also guarantees a test run can never modify the host's real subscription credentials.
if [ -n "${REMEMBER_E2E_CREDENTIAL_TARGET:-}" ] && [ -e /remember-credentials/source ]; then
  mkdir -p "$(dirname "$REMEMBER_E2E_CREDENTIAL_TARGET")"
  cp -a /remember-credentials/source "$REMEMBER_E2E_CREDENTIAL_TARGET"
fi

# Each adapter owns this small, static command. It is intentionally an environment value so this
# runner stays harness-neutral as more CLI adapters are added.
sh -ceu "$REMEMBER_E2E_INSTALL_COMMAND"

exec "$@"
