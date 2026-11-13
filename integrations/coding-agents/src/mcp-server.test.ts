import { describe, expect, it, vi } from "vitest";
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import {
  buildMcpServer,
  isDirectExecution,
  resolveHarness,
  resolveProjectCwd,
  selectTools,
} from "./mcp-server";
import { resolveConfig } from "./core/config";
import type { RememberClient } from "./core/remember";

// Plain stub — no SDK, no network. selectTools only needs a client reference to hand to
// buildPageTools when enabled; it never calls any client method itself.
const stubClient = {} as RememberClient;

it("recognizes direct execution through a symlink without starting on import", () => {
  const dir = mkdtempSync(join(tmpdir(), "remember-mcp-entry-"));
  try {
    const entry = join(dir, "server.js");
    const alias = join(dir, "alias.js");
    const other = join(dir, "other.js");
    writeFileSync(entry, "");
    writeFileSync(other, "");
    symlinkSync(entry, alias);
    expect(isDirectExecution(alias, pathToFileURL(entry).href)).toBe(true);
    expect(isDirectExecution(other, pathToFileURL(entry).href)).toBe(false);
    expect(isDirectExecution(undefined, pathToFileURL(entry).href)).toBe(false);
    expect(isDirectExecution(join(dir, "missing"), pathToFileURL(entry).href)).toBe(false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("selectTools", () => {
  it("returns [] when cfg.disabled is true — a disabled Remember exposes NO tools", () => {
    const cfg = resolveConfig({ disabled: true });
    expect(selectTools(cfg, stubClient, "b")).toEqual([]);
  });

  it("returns the eight remember_* tool specs when enabled", () => {
    const cfg = resolveConfig({});
    const tools = selectTools(cfg, stubClient, "b");
    expect(tools.map((t) => t.name).sort()).toEqual(
      [
        "remember_sync_status",
        "remember_diagnose",
        "remember_search_pages",
        "remember_list_pages",
        "remember_read_page",
        "remember_reflect",
        "remember_capture_initiative",
        "remember_ingest_document",
      ].sort()
    );
  });

  it("propagates the configured harness to diagnostics", async () => {
    const cfg = resolveConfig({ harness: "codex" });
    const tools = selectTools(cfg, stubClient, "b");
    const diagnose = tools.find((tool) => tool.name === "remember_diagnose");

    expect(diagnose).toBeDefined();
    const result = await diagnose!.handler({});
    expect(JSON.parse(result.content[0].text)).toMatchObject({ harness: "codex" });
  });

  // #3590: selectTools built the tools WITHOUT the reflect settings, so remember_reflect fell
  // back to the client's hardcoded 120s deadline and reflectTimeoutMs was dead config.
  it("forwards the resolved reflect timeout and budget into remember_reflect", async () => {
    const reflect = vi.fn().mockResolvedValue("answer");
    const client = { reflect } as unknown as RememberClient;
    const cfg = resolveConfig({ reflectToolTimeoutMs: 660_000, reflectBudget: "mid" });
    const tools = selectTools(cfg, client, "b");

    await tools.find((tool) => tool.name === "remember_reflect")!.handler({ query: "why?" });
    expect(reflect).toHaveBeenCalledWith("why?", { budget: "mid", timeoutMs: 660_000 });
  });

  it("also attributes documents ingested through the MCP tool to that harness", async () => {
    // The same option feeds remember_ingest_document, which until now stamped nothing: the
    // documents list resolves a document's agent logo from `metadata.harness` / `harness:<id>`,
    // so MCP-ingested documents used to show up unattributed.
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { retain } as unknown as RememberClient;
    const tools = selectTools(resolveConfig({ harness: "codex" }), client, "b");
    const ingest = tools.find((tool) => tool.name === "remember_ingest_document");

    await ingest!.handler({ title: "Runbook", content: "steps" });
    const [, , , tags, , opts] = retain.mock.calls[0];
    expect(tags).toContain("harness:codex");
    expect(opts.metadata).toMatchObject({ harness: "codex" });
  });
});

/**
 * #3603: this used to fall back to "claude-code" when the registration named no harness. Every
 * host launches the same binary, so that fallback silently served Codex (and cursor-cli,
 * copilot-cli, grok-build — none of whose installers set the variable) as Claude Code: their
 * ingests came back tagged `harness:claude-code` and they resolved Claude Code's bank. Refusing to
 * start is recoverable by re-running the installer; mis-stamped documents are not.
 */
describe("resolveHarness", () => {
  it("returns the harness its registration declares", () => {
    expect(resolveHarness({ REMEMBER_MCP_HARNESS: "codex" })).toBe("codex");
    expect(resolveHarness({ REMEMBER_MCP_HARNESS: "claude-code" })).toBe("claude-code");
  });

  it.each([{}, { REMEMBER_MCP_HARNESS: "" }])(
    "refuses to start on %j rather than guessing",
    (env) => {
      expect(() => resolveHarness(env)).toThrow(/REMEMBER_MCP_HARNESS is not set/);
      // The message has to name the way out — the fix is a re-install, not editing a config by hand.
      expect(() => resolveHarness(env)).toThrow(/install <harness>/);
    }
  );
});

describe("resolveProjectCwd", () => {
  it("uses the survey project override", () => {
    expect(resolveProjectCwd({ REMEMBER_MCP_PROJECT_CWD: "/repo" }, "/home")).toBe("/repo");
  });

  it("uses the working directory when the override is missing or uninterpolated", () => {
    expect(resolveProjectCwd({}, "/home")).toBe("/home");
    expect(resolveProjectCwd({ REMEMBER_MCP_PROJECT_CWD: "${workspaceFolder}" }, "/home")).toBe(
      "/home"
    );
  });
});

/**
 * Registration parity across the places that launch this binary (#3603).
 *
 * The harness used to default to "claude-code", so a registration that named none was silently
 * served as Claude Code: four installers and both inline survey recipes wrote one, and Codex
 * ingests came back tagged `harness:claude-code` in Claude Code's bank. The default is gone — a
 * nameless registration now refuses to start — but that only converts a silent wrong answer into a
 * loud one at the site that forgot.
 *
 * A per-site test can't catch the next one: the site that forgets is by definition the one whose
 * test nobody wrote (survey.ts was found by hand, not by the installer sweep). So this asserts the
 * shape — a module that points something at mcp-server.js is registering it, and every one of them
 * must also name the harness.
 */
describe("every mcp-server.js registration names a harness", () => {
  const SRC = fileURLToPath(new URL(".", import.meta.url));

  /** Modules that reference the binary WITHOUT registering it, and why. */
  const EXEMPT: Record<string, string> = {
    "mcp-server.ts": "the server itself — it READS the variable, it does not register anything",
    "core/util.ts": "isOurMcpEntry only RECOGNISES an existing registration, it writes none",
  };

  /**
   * An ASSIGNMENT of the variable, in any of the three dialects a registration is written in:
   * a JS env object (`REMEMBER_MCP_HARNESS: "codex"`), a TOML/`-c` override
   * (`REMEMBER_MCP_HARNESS = "codex"`), or a CLI pair (`REMEMBER_MCP_HARNESS=codex`).
   * Deliberately not a bare-name search: prose about the variable — including the comments this
   * very fix added next to each registration — would satisfy that and let a site slip through.
   */
  const SETS_HARNESS = /(?:REMEMBER|REMEMBER)_MCP_HARNESS\s*[:=]/;

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no module that points at mcp-server.js without setting the harness", () => {
    const nameless = sourceFiles(SRC).filter((rel) => {
      if (rel in EXEMPT) return false;
      const src = readFileSync(join(SRC, rel), "utf8");
      return src.includes("mcp-server.js") && !SETS_HARNESS.test(src);
    });
    expect(nameless).toEqual([]);
  });
});

describe("buildMcpServer", () => {
  it("answers tools/list with an empty list when Remember is disabled", async () => {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildMcpServer([]);
    const client = new Client({ name: "test-client", version: "0.1.0" });

    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      expect(client.getServerCapabilities()).toMatchObject({ tools: { listChanged: false } });
      await expect(client.listTools()).resolves.toEqual({ tools: [] });
    } finally {
      await client.close();
      await server.close();
    }
  });

  /**
   * Dcode computes an "is this tool coherently read-only" verdict from the MCP annotations and,
   * in headless mode, REJECTS every call that fails it; Codex Auto-review reads the same metadata
   * to tell a safe page read from a write. So the annotations are a functional requirement,
   * not documentation. Assert them over the wire (what a client actually sees), not on the specs.
   */
  it("publishes explicit safety annotations for every enabled tool", async () => {
    const readOnly = {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    };
    // The writes are additive, never idempotent: clients should still gate them.
    const nonDestructiveWrite = {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    };
    const expected = {
      remember_sync_status: readOnly,
      remember_diagnose: readOnly,
      remember_search_pages: readOnly,
      remember_list_pages: readOnly,
      remember_read_page: readOnly,
      remember_reflect: readOnly,
      remember_capture_initiative: nonDestructiveWrite,
      remember_ingest_document: nonDestructiveWrite,
    };
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = buildMcpServer(selectTools(resolveConfig({}), stubClient, "b"));
    const client = new Client({ name: "test-client", version: "0.1.0" });

    await server.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      const listed = await client.listTools();
      expect(Object.fromEntries(listed.tools.map((tool) => [tool.name, tool.annotations]))).toEqual(
        expected
      );
    } finally {
      await client.close();
      await server.close();
    }
  });
});
