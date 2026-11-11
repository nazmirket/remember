import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    deepen: "src/deepen.ts",
    installer: "src/installer.ts",
    status: "src/status.ts",
    "claude-hook": "src/claude-hook.ts",
    "claude-stop-hook": "src/claude-stop-hook.ts",
    "claude-sessionstart-hook": "src/claude-sessionstart-hook.ts",
    "codex-hook": "src/codex-hook.ts",
    "codex-stop-hook": "src/codex-stop-hook.ts",
    "codex-sessionstart-hook": "src/codex-sessionstart-hook.ts",
    "daemon-start": "src/daemon-start.ts",
    "survey-supervisor": "src/survey-supervisor.ts",
    "mcp-server": "src/mcp-server.ts",
    "hindsight-seed": "src/hindsight-seed.ts",
  },
  format: ["esm"],
  target: "node18",
  clean: true,
  dts: { entry: "src/index.ts" },
  shims: false,
  // Each bin entry (claude-hook.js, cursor-hook.js, codex-hook.js, deepen.js, status.js, mcp-server.js,
  // hindsight-seed.js) must be a single self-contained file: hosts wire a hook by absolute path to
  // ONE dist file and never load the rest of the package, so shared code can't live in a separate
  // chunk-*.js.
  splitting: false,
  // mcp-server.js additionally needs its npm deps (the MCP SDK + zod) inlined, since the wrapper
  // only copies the single bundle file, not node_modules. The regexes catch subpath imports too
  // (e.g. "@modelcontextprotocol/sdk/server/mcp.js", "zod/v4/...").
  // hindsight-all (the local-daemon lifecycle manager) is inlined for the same reason: hooks are
  // wired by absolute path to ONE dist file and never load the package's node_modules. It has zero
  // dependencies of its own, so inlining costs almost nothing.
  // jsonc-parser likewise: installer.js is staged to ~/.hindsight/coding-agents as dist + skill +
  // package.json — never node_modules — so an external import there is unresolvable, and re-running
  // `install` from the staged copy (the upgrade path) dies with ERR_MODULE_NOT_FOUND.
  noExternal: [
    /^@modelcontextprotocol\/sdk/,
    /^zod/,
    /^@vectorize-io\/hindsight-all/,
    // smol-toml likewise: installer.js parses ~/.grok/config.toml with it.
    /^smol-toml/,
  ],
});
