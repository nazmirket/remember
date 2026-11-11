import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { INSTALLERS, MARKER, nodeCommand, run, type InstallCtx } from "./installer";
import { SKILL_DIRS } from "./core/skill-dirs";

/** The uninstall sweep and the hook seed both shell out to the system sqlite3 — absent on some
 *  runners, so the DB-backed tests gate on a probe instead of assuming. */
const hasSqlite3 = (() => {
  try {
    execFileSync("sqlite3", ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
    return true;
  } catch {
    return false;
  }
})();

// Every test gets a FRESH temp dir as ctx.home (never the real $HOME) and a stubbed
// claudeMcp so the real `claude` CLI is never executed. run() is always called with
// explicit harness names so detect() (which probes PATH) never runs.

const homes: string[] = [];

// TraeCode resolves its user-level MCP file by Electron's OS conventions, which consult APPDATA
// (win32) and XDG_CONFIG_HOME (linux). Left set, the install writes OUTSIDE the temp ctx.home —
// the CI runner's XDG_CONFIG_HOME is exactly how the family sweep at "MCP registrations name the
// calling harness" lost the file and failed on linux while every traecode-specific test (which
// pinned the env locally) passed. Pin the whole file instead: no per-harness describe should have
// to remember this, the same lesson as SKILL_DIRS — the list everyone forgets lives once.
let savedEnv: Record<string, string | undefined>;
beforeAll(() => {
  savedEnv = { APPDATA: process.env.APPDATA, XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME };
  delete process.env.APPDATA;
  delete process.env.XDG_CONFIG_HOME;
});
afterAll(() => {
  for (const [k, v] of Object.entries(savedEnv)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
});

function makeCtx(): InstallCtx & {
  claudeMcp: ReturnType<typeof vi.fn>;
  qwenMcp: ReturnType<typeof vi.fn>;
  clinePlugin: ReturnType<typeof vi.fn>;
  dcodePlugin: ReturnType<typeof vi.fn>;
  nodeSqlite: ReturnType<typeof vi.fn>;
} {
  const home = mkdtempSync(join(tmpdir(), "hindsight-installer-test-"));
  homes.push(home);
  const pkgRoot = join("/opt", MARKER); // contains the marker, like the real package path
  return {
    home,
    pkgRoot,
    dist: join(pkgRoot, "dist"),
    claudeMcp: vi.fn(() => true),
    // qwen-code registers through the `qwen` CLI, exactly as claude-code does through `claude`.
    // Stub it for the same reason: the suite must never execute a real host CLI.
    qwenMcp: vi.fn(() => true),
    clinePlugin: vi.fn(() => true),
    dcodePlugin: vi.fn(() => true),
    // Stubbed like the CLI seams above, so the suite never depends on the Node running it.
    nodeSqlite: vi.fn(() => true),
    // Never let a developer's real ~/.hindsight/claude-code.json steer the tests.
  };
}

function readJson(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8"));
}

function writeJsonAt(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

// configureServer honors HINDSIGHT_CONFIG — a developer shell exporting it must not leak the
// suite's --server writes into their real config file ("" is falsy → the per-test home is used).
//
// DSH_HOME leaks the same way, and more sharply: `dshHome()` is `process.env.DSH_HOME ||
// join(c.home, ".dsh")`, so the roster tests that install the dsh entrypoint into a temp `ctx.home`
// write the DEVELOPER'S own home patch whenever their shell exports it — and the row they leave
// behind points at a temp dir this suite then deletes, i.e. a dsh that no longer boots. Unset for
// the whole FILE, not just the dsh block: the loops that leak live outside that block, and the
// reader in core/history.ts honors DSH_HOME too.
beforeEach(() => {
  vi.stubEnv("HINDSIGHT_CONFIG", "");
  vi.stubEnv("DSH_HOME", "");
  delete process.env.DSH_HOME;
});

afterEach(() => {
  vi.unstubAllEnvs();
  while (homes.length) rmSync(homes.pop()!, { recursive: true, force: true });
  vi.clearAllMocks();
});

// The env guards above are load-bearing rather than hygiene, and dropping one is silent on CI: the
// leak they prevent only shows up on a machine whose shell exports the variable. This is what
// notices.
describe("env isolation", () => {
  it("unsets DSH_HOME, so no test can write a real dsh home patch", () => {
    expect(process.env.DSH_HOME).toBeUndefined();
  });
});

describe("claude-code installer", () => {
  const settingsPath = (ctx: InstallCtx) => join(ctx.home, ".claude", "settings.json");

  it("install writes the 3 hook events with our dist commands and timeouts 30/30/60", () => {
    const ctx = makeCtx();
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    const settings = readJson(settingsPath(ctx));
    const hooks = settings.hooks;
    expect(Object.keys(hooks).sort()).toEqual(["SessionStart", "Stop", "UserPromptSubmit"]);
    const inner = (ev: string) => hooks[ev][0].hooks[0];
    expect(inner("SessionStart").command).toContain(join(ctx.dist, "claude-sessionstart-hook.js"));
    expect(inner("UserPromptSubmit").command).toContain(join(ctx.dist, "claude-hook.js"));
    expect(inner("Stop").command).toContain(join(ctx.dist, "claude-stop-hook.js"));
    expect(inner("SessionStart").timeout).toBe(30);
    expect(inner("UserPromptSubmit").timeout).toBe(30);
    expect(inner("Stop").timeout).toBe(60);
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      expect(inner(ev).type).toBe("command");
    }
  });

  it("preserves pre-existing foreign hook entries and appends ours", () => {
    const ctx = makeCtx();
    const foreign = { hooks: [{ type: "command", command: "echo other-tool", timeout: 5 }] };
    writeJsonAt(settingsPath(ctx), { hooks: { SessionStart: [foreign] } });
    run(["install", "claude-code"], ctx);
    const events = readJson(settingsPath(ctx)).hooks.SessionStart;
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual(foreign);
    expect(JSON.stringify(events[1])).toContain(MARKER);
  });

  it("re-install is idempotent — exactly ONE of our entries per event", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    run(["install", "claude-code"], ctx);
    const hooks = readJson(settingsPath(ctx)).hooks;
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      const ours = hooks[ev].filter((e: unknown) => JSON.stringify(e).includes(MARKER));
      expect(ours).toHaveLength(1);
      expect(hooks[ev]).toHaveLength(1);
    }
  });

  it("removes before adding so an existing registration is REPOINTED, not skipped", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    const calls = ctx.claudeMcp.mock.calls.map((c) => c[0]);
    const removeAt = calls.findIndex((a) => a[1] === "remove");
    const addAt = calls.findIndex((a) => a[1] === "add");
    // `claude mcp add` refuses a name that already exists, so without the remove a re-install
    // silently leaves a stale (possibly dead) server path registered.
    expect(removeAt).toBeGreaterThanOrEqual(0);
    expect(removeAt).toBeLessThan(addAt);
    expect(calls[removeAt]).toEqual(["mcp", "remove", "--scope", "user", "hindsight"]);
  });

  it("registers the MCP server via `claude mcp add` (user scope)", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    expect(ctx.claudeMcp).toHaveBeenCalledWith([
      "mcp",
      "add",
      "--scope",
      "user",
      "hindsight",
      // Every host launches the same mcp-server.js, so the registration has to name its harness —
      // otherwise the server falls back to claude-code and mis-attributes the other hosts' writes.
      "--env",
      "HINDSIGHT_MCP_HARNESS=claude-code",
      "--",
      "node",
      join(ctx.dist, "mcp-server.js"),
    ]);
  });

  it("still succeeds when claudeMcp reports the CLI is unusable (manual instructions)", () => {
    const ctx = makeCtx();
    ctx.claudeMcp.mockReturnValue(false);
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(logs.join("\n")).toContain("claude mcp add");
    // hooks were still written despite the MCP failure
    expect(existsSync(settingsPath(ctx))).toBe(true);
  });

  it("uninstall strips our entries, keeps foreign ones, and calls `claude mcp remove`", () => {
    const ctx = makeCtx();
    const foreign = { hooks: [{ type: "command", command: "echo other-tool", timeout: 5 }] };
    writeJsonAt(settingsPath(ctx), { hooks: { Stop: [foreign] } });
    run(["install", "claude-code"], ctx);
    run(["uninstall", "claude-code"], ctx);
    const settings = readJson(settingsPath(ctx));
    expect(settings.hooks.Stop).toEqual([foreign]);
    expect(settings.hooks.SessionStart).toBeUndefined();
    expect(settings.hooks.UserPromptSubmit).toBeUndefined();
    expect(JSON.stringify(settings)).not.toContain(MARKER);
    expect(ctx.claudeMcp).toHaveBeenCalledWith(["mcp", "remove", "--scope", "user", "hindsight"]);
  });

  it("uninstall removes the hooks object entirely when nothing else remains", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    run(["uninstall", "claude-code"], ctx);
    expect(readJson(settingsPath(ctx)).hooks).toBeUndefined();
  });
});

// WorkBuddy and CodeBuddy share one installer (same engine, different home folder and MCP file).
describe("codex installer", () => {
  const hooksPath = (ctx: InstallCtx) => join(ctx.home, ".codex", "hooks.json");
  const tomlPath = (ctx: InstallCtx) => join(ctx.home, ".codex", "config.toml");
  it("install writes the 3 hook events into hooks.json", () => {
    const ctx = makeCtx();
    expect(run(["install", "codex"], ctx)).toBe(0);
    const hooks = readJson(hooksPath(ctx)).hooks;
    expect(Object.keys(hooks).sort()).toEqual(["SessionStart", "Stop", "UserPromptSubmit"]);
    expect(hooks.SessionStart[0].hooks[0].command).toContain("codex-sessionstart-hook.js");
    expect(hooks.UserPromptSubmit[0].hooks[0].command).toContain("codex-hook.js");
    expect(hooks.Stop[0].hooks[0].command).toContain("codex-stop-hook.js");
  });

  it("creates config.toml with the features flag and mcp_servers section when missing", () => {
    const ctx = makeCtx();
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml).toContain("[features]\nhooks = true");
    expect(toml).toContain("[mcp_servers.hindsight]");
    expect(toml).toContain(join(ctx.dist, "mcp-server.js"));
    expect(toml).toContain('env = { HINDSIGHT_MCP_HARNESS = "codex" }');
  });

  it("does NOT duplicate an existing [features] section (only appends mcp) and backs up the toml", () => {
    const ctx = makeCtx();
    const original = "[features]\nsome_flag = true\n";
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    writeFileSync(tomlPath(ctx), original);
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml.match(/^\[features\]/gm)).toHaveLength(1);
    expect(toml).not.toContain("hooks = true"); // user is told to add it manually
    expect(toml).toContain("[mcp_servers.hindsight]");
    expect(readFileSync(`${tomlPath(ctx)}.hindsight-backup`, "utf8")).toBe(original);
  });

  it("appends nothing features-related when hooks is already present", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    writeFileSync(tomlPath(ctx), "[features]\nhooks = true\n");
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml.match(/hooks/g)).toHaveLength(1);
    expect(toml.match(/^\[features\]/gm)).toHaveLength(1);
    expect(toml).toContain("[mcp_servers.hindsight]");
  });

  // Regression: Codex sessions ingested documents tagged `harness:claude-code`. Its registration
  // carried no HINDSIGHT_MCP_HARNESS, and install skipped whenever a block already existed — so
  // the stale, harness-less block survived every re-install and the shared mcp-server.js kept
  // falling back to claude-code for the bank AND the retain stamp.
  it("re-install REPLACES a harness-less mcp block instead of leaving it stale", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    writeFileSync(
      tomlPath(ctx),
      '[features]\nhooks = true\n\n[mcp_servers.hindsight]\ncommand = "node"\n' +
        `args = ["${join(ctx.dist, "mcp-server.js")}"]\n\n[ui]\ntheme = "dark"\n`
    );
    run(["install", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml.match(/^\[mcp_servers\.hindsight\]/gm)).toHaveLength(1);
    expect(toml).toContain('env = { HINDSIGHT_MCP_HARNESS = "codex" }');
    expect(toml).toContain('[ui]\ntheme = "dark"'); // foreign sections survive the rewrite
  });

  it("uninstall removes the mcp_servers.hindsight block and leaves the rest of the toml", () => {
    const ctx = makeCtx();
    run(["install", "codex"], ctx);
    run(["uninstall", "codex"], ctx);
    const toml = readFileSync(tomlPath(ctx), "utf8");
    expect(toml).not.toContain("[mcp_servers.hindsight]");
    expect(toml).toContain("hooks = true"); // flag deliberately left in place
    const hooks = readJson(hooksPath(ctx)).hooks;
    expect(Object.keys(hooks)).toHaveLength(0);
  });
});

describe("the published manifest", () => {
  it("omits configuration for unsupported hosts", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8")
    ) as Record<string, unknown>;
    expect(manifest.pi).toBeUndefined();
    expect(manifest.dsh).toBeUndefined();
    expect(manifest.cline).toBeUndefined();
  });
});

describe("run() CLI behavior", () => {
  it("returns 1 for an unknown harness name and touches nothing", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run(["install", "not-a-harness"], ctx)).toBe(1);
    expect(logs.join("\n")).toContain('unknown harness "not-a-harness"');
    expect(existsSync(join(ctx.home, ".claude"))).toBe(false);
    expect(ctx.claudeMcp).not.toHaveBeenCalled();
  });

  it("returns 0 with usage when no command is given", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    expect(run([], ctx)).toBe(0);
    expect(logs.join("\n")).toContain("usage:");
  });

  it("returns 1 for an unknown command", () => {
    const ctx = makeCtx();
    expect(run(["frobnicate"], ctx)).toBe(1);
  });

  it("MARKER identifies our entries under BOTH the npm and repo-checkout layouts", () => {
    // Dedupe-on-reinstall and uninstall both key off this substring appearing in the package path.
    // It silently stopped matching a checkout when the directory dropped its `hindsight-` prefix.
    expect("/usr/lib/node_modules/@vectorize-io/hindsight-coding-agents/dist").toContain(MARKER);
    expect("/repo/hindsight-integrations/coding-agents/dist").toContain(MARKER);
  });

  it("re-install from a repo-checkout path leaves exactly one hook entry per event", () => {
    const ctx = makeCtx();
    const repoStyle = {
      ...ctx,
      pkgRoot: "/repo/hindsight-integrations/coding-agents",
      dist: "/repo/hindsight-integrations/coding-agents/dist",
    };
    run(["install", "claude-code"], repoStyle);
    run(["install", "claude-code"], repoStyle);
    const hooks = readJson(join(ctx.home, ".claude", "settings.json")).hooks;
    for (const ev of ["SessionStart", "UserPromptSubmit", "Stop"]) {
      expect(hooks[ev]).toHaveLength(1);
    }
  });
});

/**
 * Every host launches the SAME `dist/mcp-server.js`, so the command line cannot say who is calling
 * — only `HINDSIGHT_MCP_HARNESS` can. A registration that omits it silently inherits the server's
 * claude-code fallback, which is how a Codex `hindsight_ingest_document` was stored tagged
 * `harness:claude-code` on a machine running both (#3603).
 *
 * Swept over INSTALLERS rather than a hand-written list of config paths: the harness that gets this
 * wrong is by construction the one nobody wrote a test for, so the guard has to find the
 * registration itself — any file the install wrote that names mcp-server.js, wherever it landed.
 */
describe("MCP registrations name the calling harness", () => {
  /** Every file under `dir`, recursively. */
  function filesUnder(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
      e.isDirectory() ? filesUnder(join(dir, e.name)) : [join(dir, e.name)]
    );
  }

  // These hosts have no INSTALLER-written MCP registration. The in-process ones load our
  // plugin/extension in-process (src/kilo.ts, src/dsh.ts, src/pi.ts, src/prime-agent.ts,
  // dist/index.js for opencode, index.js -> dist/opencode2.js for opencode2), and that entry
  // hands its own harness name straight to RuntimeCore. TraeCode's registration is per-repo
  // `<repo>/.trae/mcp.json`, written by the SessionStart hook (covered in core/traecode-mcp.test.ts)
  // — the installer deliberately writes no user-level MCP file (see installer.ts's traecode docs).
  const NO_INSTALLER_MCP = new Set([
    "opencode",
    "opencode2",
    "kilo",
    "pi",
    "prime-agent",
    "dsh",
    "dcode",
    "traecode",
  ]);
  const MCP_HOSTS = INSTALLERS.map((i) => i.name).filter((n) => !NO_INSTALLER_MCP.has(n));

  it.each(MCP_HOSTS)("%s", (harness) => {
    const ctx = makeCtx();
    expect(run(["install", harness], ctx)).toBe(0);
    const registrations = [
      ...filesUnder(ctx.home)
        .map((f) => readFileSync(f, "utf8"))
        // claude-code and qwen-code register through their host CLI instead of a file we write,
        // so their registration is the argv we handed the mock.
        .concat(ctx.claudeMcp.mock.calls.map((c) => c[0].join(" ")))
        .concat(ctx.qwenMcp.mock.calls.map((c) => c[0].join(" ")))
        .filter((text) => text.includes("mcp-server.js")),
    ];
    expect(registrations.length).toBeGreaterThan(0);
    for (const text of registrations) expect(text).toContain(`HINDSIGHT_MCP_HARNESS`);
    for (const text of registrations) expect(text).toContain(harness);
  });
});

/**
 * `all` is an explicit target rather than the default for a bare command: wiring every detected
 * agent rewrites a lot of a machine's config and should not happen by accident.
 */
describe("all vs named harnesses", () => {
  it("`install all` wires every DETECTED agent", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    expect(run(["install", "all"], ctx)).toBe(0);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".codex", "hooks.json"))).toBe(true);
  });

  it("a bare `install` changes NOTHING and explains the choice", () => {
    const ctx = makeCtx();
    const logs: string[] = [];
    ctx.log = (m) => logs.push(m);
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    expect(run(["install"], ctx)).toBe(1);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
    expect(logs.join("\n")).toContain("all");
  });

  it("a named harness wires only that one", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    mkdirSync(join(ctx.home, ".codex"), { recursive: true });
    run(["install", "claude-code"], ctx);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(true);
    expect(existsSync(join(ctx.home, ".codex", "hooks.json"))).toBe(false);
  });

  it("`uninstall all` is accepted too, so the pair stays symmetric", () => {
    const ctx = makeCtx();
    mkdirSync(join(ctx.home, ".claude"), { recursive: true });
    run(["install", "all"], ctx);
    expect(run(["uninstall", "all"], ctx)).toBe(0);
    expect(readJson(join(ctx.home, ".claude", "settings.json")).hooks).toBeUndefined();
  });
});

/**
 * Running from an npx cache used to be refused: the wiring is absolute paths into this package, and
 * a cache eviction would break every hook silently. The runtime is now copied somewhere stable
 * first, so `npx` works and nobody needs a global install of a tool that only sets other tools up.
 */
describe("runtime staging", () => {
  /** A package layout convincing enough to be staged: staging keys off a built dist. */
  function fakePackage(root: string): { pkgRoot: string; dist: string } {
    const dist = join(root, "dist");
    mkdirSync(dist, { recursive: true });
    writeFileSync(join(dist, "installer.js"), "// built");
    writeFileSync(join(dist, "claude-hook.js"), "// built");
    writeFileSync(join(root, "package.json"), JSON.stringify({ name: "x", main: "dist/index.js" }));
    mkdirSync(join(root, "skill"), { recursive: true });
    writeFileSync(join(root, "skill", "SKILL.md"), "# skill");
    return { pkgRoot: root, dist };
  }

  it("installs from an npx cache, wiring the stable copy instead of the cache", () => {
    const ctx = makeCtx();
    const cache = mkdtempSync(join(tmpdir(), "npx-cache-"));
    homes.push(cache);
    Object.assign(ctx, fakePackage(join(cache, "_npx", "abc123", "node_modules", "coding-agents")));

    expect(run(["install", "claude-code"], ctx)).toBe(0);
    const staged = join(ctx.home, ".hindsight", "coding-agents");
    const command = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart[0]
      .hooks[0].command as string;
    expect(command).toContain(join(staged, "dist"));
    // The whole point: nothing in a host config may reference the evictable cache.
    expect(command).not.toContain("_npx");
    expect(existsSync(join(staged, "dist", "claude-hook.js"))).toBe(true);
  });

  // MARKER matching is what makes re-install replace and uninstall remove, and it looks for this
  // substring in the command path — so the staged location must keep it.
  it("stages somewhere the marker still matches", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    Object.assign(ctx, fakePackage(src));

    run(["install", "claude-code"], ctx);
    expect(join(ctx.home, ".hindsight", "coding-agents")).toContain(MARKER);
    run(["uninstall", "claude-code"], ctx);
    expect(readJson(join(ctx.home, ".claude", "settings.json")).hooks).toBeUndefined();
  });

  it("upgrading replaces the staged runtime, stale files and all", () => {
    const ctx = makeCtx();
    const v1 = mkdtempSync(join(tmpdir(), "v1-"));
    const v2 = mkdtempSync(join(tmpdir(), "v2-"));
    homes.push(v1, v2);
    fakePackage(v1);
    writeFileSync(join(v1, "dist", "old-only.js"), "// dropped in the next version");
    fakePackage(v2);
    writeFileSync(join(v2, "dist", "new-only.js"), "// added in the next version");

    Object.assign(ctx, { pkgRoot: v1, dist: join(v1, "dist") });
    run(["install", "claude-code"], ctx);
    Object.assign(ctx, { pkgRoot: v2, dist: join(v2, "dist") });
    run(["install", "claude-code"], ctx);

    const staged = join(ctx.home, ".hindsight", "coding-agents", "dist");
    expect(existsSync(join(staged, "new-only.js"))).toBe(true);
    // Merging instead of replacing would leave an entry point a host config could still name.
    expect(existsSync(join(staged, "old-only.js"))).toBe(false);
    const events = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart;
    expect(events).toHaveLength(1);
  });

  // Re-running the STAGED installer must not delete the dist it is executing from.
  it("is a no-op when run from the staged copy itself", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    fakePackage(src);
    Object.assign(ctx, { pkgRoot: src, dist: join(src, "dist") });
    run(["install", "claude-code"], ctx);

    const staged = join(ctx.home, ".hindsight", "coding-agents");
    Object.assign(ctx, { pkgRoot: staged, dist: join(staged, "dist") });
    expect(run(["install", "claude-code"], ctx)).toBe(0);
    expect(existsSync(join(staged, "dist", "installer.js"))).toBe(true);
  });

  // `update` is what core/auto-update.ts spawns unattended, so its blast radius IS the contract:
  // it refreshes the staged runtime and rewrites nothing a host reads.
  it("`update` re-stages the runtime without touching any host config", () => {
    const ctx = makeCtx();
    const v1 = mkdtempSync(join(tmpdir(), "v1-"));
    const v2 = mkdtempSync(join(tmpdir(), "v2-"));
    homes.push(v1, v2);
    fakePackage(v1);
    fakePackage(v2);
    writeFileSync(join(v2, "dist", "new-only.js"), "// added in the next version");

    Object.assign(ctx, { pkgRoot: v1, dist: join(v1, "dist") });
    run(["install", "claude-code"], ctx);
    const settingsPath = join(ctx.home, ".claude", "settings.json");
    const before = readFileSync(settingsPath, "utf8");

    Object.assign(ctx, { pkgRoot: v2, dist: join(v2, "dist") });
    expect(run(["update"], ctx)).toBe(0);

    const staged = join(ctx.home, ".hindsight", "coding-agents", "dist");
    expect(existsSync(join(staged, "new-only.js"))).toBe(true);
    // Byte-identical: the hooks already point at the staged path, so an update has no reason to
    // rewrite them — and rewriting is exactly what would let an unattended run wire agents the
    // user never installed.
    expect(readFileSync(settingsPath, "utf8")).toBe(before);
  });

  it("`update` needs no harness argument and wires nothing when nothing is installed", () => {
    const ctx = makeCtx();
    const src = mkdtempSync(join(tmpdir(), "pkg-"));
    homes.push(src);
    fakePackage(src);
    Object.assign(ctx, { pkgRoot: src, dist: join(src, "dist") });

    expect(run(["update"], ctx)).toBe(0);
    expect(existsSync(join(ctx.home, ".hindsight", "coding-agents", "dist"))).toBe(true);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
  });

  // The marker is what core/auto-update.ts reads to decide whether it may replace this runtime,
  // so staging has to write it — an absent marker fails closed and auto-update never runs.
  it("records where the runtime was staged from", () => {
    const ctx = makeCtx();
    const cache = mkdtempSync(join(tmpdir(), "npx-cache-"));
    homes.push(cache);
    const src = join(cache, "_npx", "abc123", "node_modules", "coding-agents");
    Object.assign(ctx, fakePackage(src));

    run(["install", "claude-code"], ctx);
    const origin = readJson(
      join(ctx.home, ".hindsight", "coding-agents", ".install-origin.json")
    ) as { source: string };
    expect(origin.source).toBe(src);
  });

  // A checkout whose dist was never built has nothing to copy; wiring the source path is better
  // than pointing every hook at a directory that does not exist.
  it("wires in place when there is nothing to stage", () => {
    const ctx = makeCtx();
    run(["install", "claude-code"], ctx);
    const command = readJson(join(ctx.home, ".claude", "settings.json")).hooks.SessionStart[0]
      .hooks[0].command as string;
    expect(command).toContain(ctx.dist);
    expect(existsSync(join(ctx.home, ".hindsight", "coding-agents", "dist"))).toBe(false);
  });
});

describe("local server setup", () => {
  const configPath = (ctx: InstallCtx) => join(ctx.home, ".hindsight", "coding-agent.json");

  it.each(["claude-code", "codex"])("configures %s with the fork daemon", (harness) => {
    const ctx = makeCtx();
    expect(run(["install", harness], ctx)).toBe(0);
    const cfg = readJson(configPath(ctx));
    expect(cfg.serverMode).toBe("daemon");
    expect(cfg.embedPackagePath).toContain("hindsight-embed");
    expect(cfg.apiUrl).toBeUndefined();
  });

  it("honors HINDSIGHT_CONFIG without writing the default file", () => {
    const ctx = makeCtx();
    const override = join(ctx.home, "elsewhere", "config.json");
    vi.stubEnv("HINDSIGHT_CONFIG", override);
    expect(run(["install", "claude-code", "--server", "daemon"], ctx)).toBe(0);
    expect(readJson(override).serverMode).toBe("daemon");
    expect(existsSync(configPath(ctx))).toBe(false);
  });

  it.each(["cloud", "self-hosted", "hybrid"])("rejects %s server mode", (mode) => {
    const ctx = makeCtx();
    expect(run(["install", "claude-code", "--server", mode], ctx)).toBe(1);
    expect(existsSync(configPath(ctx))).toBe(false);
    expect(existsSync(join(ctx.home, ".claude", "settings.json"))).toBe(false);
  });

  it.each(["--api-url", "--api-token"])("rejects unsupported %s flags", (flag) => {
    const ctx = makeCtx();
    expect(run(["install", "codex", flag, "value"], ctx)).toBe(1);
    expect(existsSync(configPath(ctx))).toBe(false);
  });

  it("does not overwrite an existing remote server configuration", () => {
    const ctx = makeCtx();
    const original = { serverMode: "self-hosted", apiUrl: "http://mine:8888" };
    writeJsonAt(configPath(ctx), original);
    expect(run(["install", "claude-code"], ctx)).toBe(1);
    expect(readJson(configPath(ctx))).toEqual(original);
  });
});

/**
 * Companion-skill parity across every host that installs one (#3524 shape).
 *
 * The installer writes the skill and core/skill-sync.ts re-copies it on drift at every session
 * start, so `npm update -g` upgrades the skill too. Those were two hand-maintained path lists, and
 * the self-update one covered four of the ten hosts — the six it missed kept whichever SKILL.md
 * they were installed with, forever, with nothing failing. A per-host test cannot catch that: the
 * host that is forgotten is by definition the one nobody wrote a test for.
 *
 * So drive the real thing over the WHOLE family: install each harness into a temp home, find where
 * the skill actually landed, and require that the self-update refreshes that same copy.
 */
describe("every installed companion skill is kept current by the session-start self-update", () => {
  const SKILL_NAME = "hindsight-coding-agent";

  /** makeCtx's pkgRoot is a synthetic /opt path, so the packaged skill can't be staged in it.
   *  Build a real temp package root holding a SKILL.md the installer can copy. */
  function ctxWithPackagedSkill(body: string): InstallCtx {
    const ctx = makeCtx();
    const pkgRoot = mkdtempSync(join(tmpdir(), "hs-pkg-skillsync-"));
    homes.push(pkgRoot);
    mkdirSync(join(pkgRoot, "skill"), { recursive: true });
    writeFileSync(join(pkgRoot, "skill", "SKILL.md"), body);
    return { ...ctx, pkgRoot, dist: join(pkgRoot, "dist") };
  }

  /** Every directory named `hindsight-coding-agent` under `root`, home-relative. */
  function findSkillCopies(root: string, prefix: string[] = []): string[][] {
    return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
      if (!entry.isDirectory()) return [];
      const rel = [...prefix, entry.name];
      return entry.name === SKILL_NAME ? [rel] : findSkillCopies(join(root, entry.name), rel);
    });
  }

  /** Hosts that install no skill at all, and why — so a host that silently STOPS installing one
   *  fails here instead of passing as "nothing to check". opencode (both major versions) and its
   *  Kilo fork have no skills mechanism; Devin and Dcode read no user-level skills directory
   *  either. */
  const NO_SKILL_MECHANISM: string[] = [];

  it("lands in the mapped directory and is refreshed there, for every host that has one", async () => {
    const { syncCompanionSkill } = await import("./core/skill-sync");
    const withoutSkill: string[] = [];
    for (const harness of INSTALLERS.map((i) => i.name)) {
      const ctx = ctxWithPackagedSkill("packaged v1");
      run(["install", harness], ctx);
      const copies = findSkillCopies(ctx.home);
      if (!copies.length) {
        withoutSkill.push(harness);
        continue;
      }
      // Where it landed must be the directory the shared map names, so the self-update looks there.
      expect(
        copies.map((parts) => parts.slice(0, -1)),
        harness
      ).toEqual([SKILL_DIRS[harness]]);

      // And the self-update must actually refresh THIS host's copy when the package moves on.
      const installed = join(ctx.home, ...copies[0], "SKILL.md");
      const srcDir = mkdtempSync(join(tmpdir(), "hs-pkg-newer-"));
      homes.push(srcDir);
      writeFileSync(join(srcDir, "SKILL.md"), "packaged v2");
      syncCompanionSkill(harness, { home: ctx.home, srcDir });
      expect(readFileSync(installed, "utf8"), harness).toBe("packaged v2");
    }
    expect(withoutSkill.sort()).toEqual(NO_SKILL_MECHANISM);
  });
});

describe("nodeCommand", () => {
  it("quotes the script path only when it contains a space", () => {
    expect(nodeCommand("C:\\Users\\me\\dist\\hook.js")).toBe("node C:\\Users\\me\\dist\\hook.js");
    expect(nodeCommand("/Users/my name/dist/hook.js")).toBe('node "/Users/my name/dist/hook.js"');
  });
});

it("only exposes the two supported agents", () => {
  expect(INSTALLERS.map((i) => i.name)).toEqual(["claude-code", "codex"]);
});
