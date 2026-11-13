#!/usr/bin/env node
import { rememberHome } from "./core/remember-home";
/**
 * remember-coding-agents install|uninstall [harness...]
 *
 * ONE setup command for every supported coding agent. With no harness arguments it detects which
 * agents exist on this machine (binary on PATH or config dir present) and wires each one's NATIVE
 * integration — hooks + MCP where the host wants them:
 *
 *   claude-code  hooks in ~/.claude/settings.json and MCP configuration
 *   codex        hooks in ~/.codex/hooks.json and MCP configuration in config.toml
 *
 * IDEMPOTENT: our entries are recognized by the package path in their command ("remember-coding-
 * agents"), replaced on re-install (so moving the package just needs `install` again) and removed
 * on `uninstall`. Everything else in the host's config files is preserved; files are created when
 * missing. Backups: the first time we touch an existing file we write `<file>.remember-backup`.
 */
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { isatty } from "node:tty";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { HOOK_HARNESSES, type HookHarnessName } from "./harness/hook-lifecycle";
import { importLocalHistory } from "./core/history";
import { detectLlm, isRememberCheckout, hasUv, type LlmChoice } from "./core/daemon";
import { resolveSkillDirs } from "./core/skill-dirs";
import { companionSkillPath } from "./core/skill-sync";
import { formatUsageReport, readUsage } from "./core/usage";
import { createInstallerUi, type SelectOption } from "./install-ui";

/**
 * Substring that identifies OUR entries in a host's config, so a re-install replaces them and
 * `uninstall` removes exactly what we added.
 *
 * It must appear in the package path under BOTH layouts: npm
 * (`node_modules/remember-coding-agents/...`) and a repo checkout
 * (`integrations/coding-agents/...`). The old value was the full package name, which the
 * repo path stopped containing when the directory dropped its `remember-` prefix — silently
 * breaking dedupe-on-reinstall and uninstall for anyone running from a checkout.
 */
export const MARKER = "coding-agents";

export interface InstallCtx {
  home: string;
  pkgRoot: string; // package root
  dist: string; // built entry points
  /** Runs `claude mcp ...`; injectable for tests. Returns false when the CLI isn't usable. */
  claudeMcp?: (args: string[]) => boolean;
  /** Runs `qwen mcp ...`; injectable for tests. Returns false when the CLI isn't usable. */
  qwenMcp?: (args: string[]) => boolean;
  /** Runs `cline plugin ...`; injectable for tests. Returns false when the CLI isn't usable. */
  clinePlugin?: (args: string[]) => boolean;
  /** Runs native Dcode plugin commands; injectable for tests. Returns false when unusable. */
  /** Reports whether `node:sqlite` works in the node that runs hooks; injectable for tests. */
  nodeSqlite?: () => boolean;
  /** Whether stdin can be prompted. Defaults to the real TTY check at the CLI entry; tests set it
   *  explicitly so the suite never blocks on a read. */
  interactive?: boolean;
  /** Daemon prerequisite probes; injectable for tests. */
  hasUv?: () => boolean;
  detectLlm?: () => LlmChoice | undefined;
  /** Reads an old per-agent plugin's endpoint; injectable for tests. */
  /** The raw CLI arguments after the command word — harness installers read flags off this
   *  (e.g. traecode's --enable-workspace-mcp) without each one re-parsing argv. */
  args?: readonly string[];
  log?: (m: string) => void;
  /** Styles an interactive readLineSync prompt (the CLI passes the InstallerUi rail style). */
  promptStyle?: (q: string) => string;
  /** Arrow-key picker (the CLI passes InstallerUi.select). Chosen index; null = cancelled;
   *  undefined = raw TTY unavailable, fall back to the numeric prompt. */
  selectPrompt?: (
    question: string,
    options: SelectOption[],
    defaultIndex: number
  ) => number | null | undefined;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * JSON.parse accepts non-object top-level values (`[]`, `null`, a number), and every caller here
 * treats the result as an object it can index and merge into — an array config would be written
 * back out as an array, quietly replacing the host's file with something it cannot load. A
 * top-level value that is not a plain object is as broken as an unparseable file, so both fall
 * back to `{}`.
 */
function readJson(path: string): Record<string, any> {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    return isPlainObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  backupOnce(path);
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
}

/** The first time we touch an existing file, keep a copy of what the user had. */
function backupOnce(path: string): void {
  if (existsSync(path) && !existsSync(`${path}.remember-backup`)) {
    copyFileSync(path, `${path}.remember-backup`);
  }
}

/** Hook-array merge for claude/codex-style files: drop our old entries, append the new one. */
function mergeHookEvent(existing: any[] | undefined, entry: unknown): any[] {
  const kept = (existing ?? []).filter((e) => !JSON.stringify(e).includes(MARKER));
  return [...kept, entry];
}

function stripOurs(existing: any[] | undefined): any[] {
  return (existing ?? []).filter((e) => !JSON.stringify(e).includes(MARKER));
}

/** Remove the event key entirely when nothing else remains (leave the host file tidy). */
function setOrDelete(obj: Record<string, any>, key: string, arr: any[]): void {
  if (arr.length) obj[key] = arr;
  else delete obj[key];
}

const cmdHook = (dist: string, file: string, timeout: number) => ({
  hooks: [{ type: "command", command: `node "${join(dist, file)}"`, timeout }],
});

/** Quote the script path when it contains spaces. */
export const nodeCommand = (script: string): string =>
  script.includes(" ") ? `node "${script}"` : `node ${script}`;

/** Install/uninstall consume the same lifecycle declaration as the runtime entrypoints. Keeping
 * event names and command files here would allow a host to run a different lifecycle than it installs. */
function mergeHarnessHooks(
  hooks: Record<string, any>,
  harness: HookHarnessName,
  dist: string
): void {
  for (const hook of Object.values(HOOK_HARNESSES[harness].install)) {
    hooks[hook.event] = mergeHookEvent(hooks[hook.event], cmdHook(dist, hook.entry, hook.timeout!));
  }
}

function stripHarnessHooks(hooks: Record<string, any>, harness: HookHarnessName): void {
  for (const hook of Object.values(HOOK_HARNESSES[harness].install)) {
    setOrDelete(hooks, hook.event, stripOurs(hooks[hook.event]));
  }
}

/** This host's skills directory, from the map core/skill-sync.ts also reads — see SKILL_DIRS for
 *  why the two sides must not keep separate copies of these paths. */
function skillsBaseFor(c: InstallCtx, harness: string): string {
  const parts = resolveSkillDirs(harness, c.home);
  if (!parts) throw new Error(`${harness} installs a skill but names no directory in SKILL_DIRS`);
  return join(c.home, ...parts);
}

/** Copy the packaged companion SKILL into a host's skills directory (idempotent overwrite).
 * The log line carries the harness prefix like every adapter message: several adapters install
 * the skill before their first own log, and an unprefixed line would render under the PREVIOUS
 * harness's group in the CLI output. */
function installSkill(c: InstallCtx, harness: string): void {
  const src = join(c.pkgRoot, "skill");
  if (!existsSync(join(src, "SKILL.md"))) return;
  const skillsBase = skillsBaseFor(c, harness);
  const dst = companionSkillPath(skillsBase);
  mkdirSync(skillsBase, { recursive: true });
  cpSync(src, dst, { recursive: true });
  c.log?.(`${harness}: skill installed at ${dst}`);
}

function uninstallSkill(c: InstallCtx, harness: string): void {
  rmSync(companionSkillPath(skillsBaseFor(c, harness)), { recursive: true, force: true });
}

// ── per-harness adapters ────────────────────────────────────────────────────────

export interface HarnessInstaller {
  name: string;
  detect(ctx: InstallCtx): boolean;
  /**
   * Blocking environment check, run before anything is written. Returns the reason when this
   * machine cannot support the harness, so a doomed setup fails at install time instead of
   * reporting success and then never retaining anything.
   */
  preflight?(ctx: InstallCtx): string | undefined;
  install(ctx: InstallCtx): void | boolean;
  uninstall(ctx: InstallCtx): void | boolean;
}

function onPath(bin: string): boolean {
  try {
    execFileSync("which", [bin], { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

const claudeCode: HarnessInstaller = {
  name: "claude-code",
  detect: (c) => onPath("claude") || existsSync(join(c.home, ".claude")),
  install(c) {
    const path = join(c.home, ".claude", "settings.json");
    const settings = readJson(path);
    settings.hooks = settings.hooks ?? {};
    mergeHarnessHooks(settings.hooks, "claude-code", c.dist);
    writeJson(path, settings);
    c.log?.(`claude-code: hooks merged into ${path}`);
    // Companion SKILL: every skills-capable host gets it (claude/antigravity/cursor native dirs;
    // codex via the ~/.agents/skills standard).
    installSkill(c, "claude-code");
    const mcp = c.claudeMcp ?? defaultClaudeMcp;
    // `claude mcp add` REFUSES when the name is taken ("MCP server remember already exists in
    // user config") — so on a machine that already had Remember, a re-install could never
    // repoint the server. After the package moved, the stale registration survived and Claude
    // Code reported "Failed to connect — Connection closed" with the remember_* tools dead.
    // Remove first (a no-op when absent) so the add always lands, matching how every other host
    // wiring is replaced rather than skipped.
    mcp(["mcp", "remove", "--scope", "user", "remember"]);
    if (
      mcp([
        "mcp",
        "add",
        "--scope",
        "user",
        "remember",
        "--env",
        "REMEMBER_MCP_HARNESS=claude-code",
        "--",
        "node",
        join(c.dist, "mcp-server.js"),
      ])
    ) {
      c.log?.("claude-code: MCP server registered (claude mcp add, user scope)");
    } else {
      c.log?.(
        `claude-code: could not run \`claude mcp add\` — register the tools manually:\n` +
          `  claude mcp add --scope user remember --env REMEMBER_MCP_HARNESS=claude-code -- node "${join(c.dist, "mcp-server.js")}"`
      );
    }
  },
  uninstall(c) {
    const path = join(c.home, ".claude", "settings.json");
    if (existsSync(path)) {
      const settings = readJson(path);
      if (settings.hooks) {
        stripHarnessHooks(settings.hooks, "claude-code");
        if (!Object.keys(settings.hooks).length) delete settings.hooks;
        writeJson(path, settings);
      }
    }
    const mcp = c.claudeMcp ?? defaultClaudeMcp;
    mcp(["mcp", "remove", "--scope", "user", "remember"]);
    uninstallSkill(c, "claude-code");
    c.log?.("claude-code: hooks + MCP registration + skill removed");
  },
};

function defaultClaudeMcp(args: string[]): boolean {
  try {
    execFileSync("claude", args, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Our `[mcp_servers.remember]` table (plus any sub-table of it): the header line through to the
 * next table header or EOF. Shared by install — which REPLACES the block — and uninstall.
 */
const CODEX_MCP_BLOCK_RE = /^\[mcp_servers\.remember(?:\.[^\]]+)?\][^\n]*\n(?:(?!\[)[^\n]*\n?)*/gm;

/** Inline `env`, so CODEX_MCP_BLOCK_RE never has to straddle a `[mcp_servers.remember.env]` table. */
const codexMcpBlock = (dist: string) =>
  `[mcp_servers.remember]\ncommand = "node"\nargs = [${JSON.stringify(join(dist, "mcp-server.js"))}]\n` +
  `env = { REMEMBER_MCP_HARNESS = "codex" }`;

const codex: HarnessInstaller = {
  name: "codex",
  detect: (c) => onPath("codex") || existsSync(join(c.home, ".codex")),
  install(c) {
    const hooksPath = join(c.home, ".codex", "hooks.json");
    const cfg = readJson(hooksPath);
    cfg.hooks = cfg.hooks ?? {};
    mergeHarnessHooks(cfg.hooks, "codex", c.dist);
    writeJson(hooksPath, cfg);
    c.log?.(`codex: hooks merged into ${hooksPath}`);

    // config.toml: append-only for anything that is not ours (TOML round-tripping is not worth the
    // risk). Our OWN mcp block is stripped and rewritten instead of skipped when present: appending
    // only when absent made this install-once-only, so the harness-less registration that
    // attributed every Codex write to claude-code could never be repaired by re-running install.
    const tomlPath = join(c.home, ".codex", "config.toml");
    const existing = existsSync(tomlPath) ? readFileSync(tomlPath, "utf8") : "";
    const toml = existing.replace(CODEX_MCP_BLOCK_RE, "");
    const additions: string[] = [];
    // Codex ≥ 0.145 deprecates `codex_hooks` for `[features].hooks`; accept either as "already
    // enabled", write the modern name for new installs.
    if (!/^\s*(codex_hooks|hooks)\s*=/m.test(toml)) {
      if (/^\[features\]/m.test(toml)) {
        c.log?.(
          "codex: add `hooks = true` under your existing [features] section in ~/.codex/config.toml"
        );
      } else {
        additions.push("[features]\nhooks = true");
      }
    }
    additions.push(codexMcpBlock(c.dist));
    const next = `${toml.replace(/\n*$/, "\n\n")}${additions.join("\n\n")}\n`;
    if (next !== existing) {
      if (existsSync(tomlPath) && !existsSync(`${tomlPath}.remember-backup`)) {
        copyFileSync(tomlPath, `${tomlPath}.remember-backup`);
      }
      mkdirSync(dirname(tomlPath), { recursive: true });
      writeFileSync(tomlPath, next);
      c.log?.(`codex: wrote ${additions.length} section(s) to ${tomlPath}`);
    }
    installSkill(c, "codex"); // agentskills-standard shared dir
  },
  uninstall(c) {
    const hooksPath = join(c.home, ".codex", "hooks.json");
    if (existsSync(hooksPath)) {
      const cfg = readJson(hooksPath);
      if (cfg.hooks) {
        stripHarnessHooks(cfg.hooks, "codex");
        writeJson(hooksPath, cfg);
      }
    }
    uninstallSkill(c, "codex");
    const tomlPath = join(c.home, ".codex", "config.toml");
    if (existsSync(tomlPath)) {
      const toml = readFileSync(tomlPath, "utf8");
      const cleaned = toml.replace(CODEX_MCP_BLOCK_RE, "");
      if (cleaned !== toml) writeFileSync(tomlPath, cleaned);
    }
    c.log?.(
      "codex: hooks + MCP section + skill removed ([features] hooks flag left as-is — other hooks may use it)"
    );
  },
};

// ── runtime staging ─────────────────────────────────────────────────────────────

/**
 * Where the runtime is copied to, and therefore what every hook command points at.
 *
 * Under ~/.remember, which this package already owns (the config file lives there), and named
 * `coding-agents` on purpose: MARKER matching is what lets a re-install replace our entries and
 * `uninstall` remove them, and it looks for exactly that substring in the command path.
 */
export function runtimeDir(home: string): string {
  return join(rememberHome(home), "coding-agents");
}

/** Names the directory `stageRuntime` copied from — read by core/auto-update.ts. */
export const ORIGIN_FILE = ".install-origin.json";

/**
 * Copy the runtime out of wherever this was executed from and into a stable location, then point
 * the wiring at the copy.
 *
 * `dist` is baked into each supported agent's hook commands and MCP registration.
 *
 * Copying is skipped when there is nothing to copy — running from a checkout whose dist has not
 * been built, and in tests — so the wiring falls back to the source path rather than a directory
 * that does not exist. It is also skipped when already running from the staged copy, which is what
 * makes re-running `install` cheap.
 */
function stageRuntime(c: InstallCtx): InstallCtx {
  const target = runtimeDir(c.home);
  // Compared through realpath: re-running the STAGED installer must not reach the copy below, which
  // deletes the very dist it is executing from. A symlinked or differently-spelled path to the same
  // directory would slip past a string compare.
  const same = (a: string, b: string): boolean => {
    try {
      return realpathSync(a) === realpathSync(b);
    } catch {
      return a === b;
    }
  };
  if (same(c.pkgRoot, target)) return c;
  if (!existsSync(join(c.dist, "installer.js"))) return c;
  try {
    // Replaced wholesale rather than merged: a stale entry point left behind by an older version
    // would still be reachable from a host config that references it by name.
    rmSync(join(target, "dist"), { recursive: true, force: true });
    mkdirSync(target, { recursive: true });
    cpSync(c.dist, join(target, "dist"), { recursive: true });
    const skill = join(c.pkgRoot, "skill");
    if (existsSync(skill)) cpSync(skill, join(target, "skill"), { recursive: true });
    const pkgJson = join(c.pkgRoot, "package.json");
    if (existsSync(pkgJson)) copyFileSync(pkgJson, join(target, "package.json"));
    // Record WHERE this copy came from, LAST — the marker means "a complete copy landed here".
    // core/auto-update.ts will only replace a runtime it can prove was downloaded by npx: a copy
    // staged from a global `npm i -g`, a project dependency or a local checkout belongs to whoever
    // manages that source, and silently overwriting it would either fight their package manager
    // (leaving `npm ls -g` reporting a version that is no longer what runs) or destroy a
    // developer's locally-built dist.
    writeFileSync(
      join(target, ORIGIN_FILE),
      JSON.stringify({ source: c.pkgRoot, stagedAt: new Date().toISOString() })
    );
    c.log?.(`runtime staged at ${target}`);
    return { ...c, pkgRoot: target, dist: join(target, "dist") };
  } catch (error) {
    // A failed copy must not wire hooks at a half-written directory.
    c.log?.(`could not stage the runtime at ${target}: ${String(error)}`);
    return c;
  }
}

// ── local daemon setup ──────────────────────────────────────────────────────────

/** Value of `--name value` or `--name=value`. */
export function flagValue(args: string[], name: string): string | undefined {
  const inline = args.find((a) => a.startsWith(`--${name}=`));
  if (inline) return inline.slice(name.length + 3);
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Args that are VALUES of a preceding flag, so they aren't mistaken for harness names. */
export function flagValueArgs(args: string[], names: string[]): Set<string> {
  const taken = new Set<string>();
  for (const name of names) {
    const i = args.indexOf(`--${name}`);
    if (i >= 0 && args[i + 1] && !args[i + 1].startsWith("--")) taken.add(args[i + 1]);
  }
  return taken;
}

const CONFIG_RELATIVE = ["coding-agent.json"];
/** The runtime resolves its config as REMEMBER_CONFIG || ~/.remember/coding-agent.json
 *  (core/config.ts CONFIG_PATH); every installer read/write must honor the same override. */
const rememberConfigPath = (c: InstallCtx): string =>
  process.env.REMEMBER_CONFIG || join(rememberHome(c.home), ...CONFIG_RELATIVE);

/**
 * Resolve and persist the connection mode into ~/.remember/coding-agent.json.
 *
 * Returns false to abort the install. Everything else here is advisory: a daemon whose
 * prerequisites are missing is still worth configuring, because `uv` or an API key can be
 * installed right after — unlike the harness preflights, which gate wiring that could never work.
 */
function configureServer(
  c: InstallCtx,
  args: string[],
  _installing: readonly string[],
  sourceRoot: string
): boolean {
  const requested = flagValue(args, "server");
  if (flagValue(args, "api-url") || flagValue(args, "api-token")) {
    c.log?.("Remember does not accept remote API connection flags.");
    return false;
  }
  if (requested && requested !== "daemon") {
    c.log?.("Remember supports only the local daemon (--server daemon).");
    return false;
  }
  const configPath = rememberConfigPath(c);
  const existing = readJson(configPath);
  if ((existing.serverMode && existing.serverMode !== "daemon") || existing.apiUrl) {
    c.log?.(`Existing server configuration requires review: ${configPath}`);
    return false;
  }
  const candidates = [
    process.env.REMEMBER_EMBED_PACKAGE_PATH,
    existing.embedPackagePath,
    join(sourceRoot, "..", "..", "embed"),
    join(process.cwd(), "..", "..", "embed"),
  ];
  const embedPackagePath = candidates.find((candidate) => isRememberCheckout(candidate));
  if (!embedPackagePath) {
    c.log?.("Set REMEMBER_EMBED_PACKAGE_PATH to Remember's embed directory.");
    return false;
  }
  const next: Record<string, unknown> = {
    ...existing,
    embedPackagePath,
  };
  delete next.serverMode;
  delete next.apiUrl;
  delete next.apiToken;
  writeJson(configPath, next);
  reportDaemonPrereqs(c);
  c.log?.(`server: local daemon (${configPath})`);
  return true;
}

/**
 * Daemon mode has two prerequisites the plugin can't supply. Report both up front rather than
 * letting the first session fail quietly with nothing but a diagnostic line.
 */
function reportDaemonPrereqs(c: InstallCtx): void {
  if (!(c.hasUv ?? hasUv)()) {
    c.log?.(
      `⚠️  \`uv\` is not on PATH. The local daemon is run with it, so memory stays inert\n` +
        `    until you install it: https://docs.astral.sh/uv/`
    );
  }
  const llm = (c.detectLlm ?? detectLlm)();
  if (llm) {
    c.log?.(`   local extraction will use ${llm.provider} (from ${llm.source})`);
  } else {
    c.log?.(
      `⚠️  No LLM available for local fact extraction. Set OPENAI_API_KEY or ANTHROPIC_API_KEY,\n` +
        `    or install the Claude Code or Codex CLI (which needs no key).`
    );
  }
}

export const INSTALLERS: HarnessInstaller[] = [claudeCode, codex];

// The public executable was renamed from Gemini CLI to Antigravity's `agy`. Keep the
// integration's stable internal name for bank identity and hook compatibility, while accepting
// the executable name users naturally type at the installer prompt.
const HARNESS_ALIASES: Record<string, string> = {};

// ── CLI ─────────────────────────────────────────────────────────────────────────

/**
 * Backfill this repo's past sessions for one harness, by handing them to the same deepen engine a
 * session start uses. Deepen dedups by document id, so re-running is safe.
 *
 * Scoped to the current directory: history is per-repo, and a machine can hold thousands of
 * sessions across unrelated projects that nobody wants extracted.
 */
function importConversations(harness: string, ctx: InstallCtx): void {
  const repo = process.cwd();
  const found = importLocalHistory(harness, repo);
  if (!found.supported) {
    ctx.log?.(`${harness}: --import-conversations skipped — ${found.reason}`);
    return;
  }
  if (found.unattributed) {
    // Never silently: a session we cannot attribute is one the user might expect to see imported.
    ctx.log?.(
      `${harness}: skipped ${found.unattributed} session(s) that do not record which directory ` +
        `they ran in — importing them could file another repo's conversation into this bank`
    );
  }
  if (!found.sessions.length) {
    ctx.log?.(`${harness}: no past sessions found on disk for ${repo}`);
    return;
  }
  const turns = found.sessions.reduce((n, s) => n + s.turns.length, 0);
  const file = join(mkdtempSync(join(tmpdir(), "remember-import-")), "conversations.json");
  writeFileSync(file, JSON.stringify(found.sessions));
  ctx.log?.(
    `${harness}: importing ${found.sessions.length} past sessions (${turns} turns) from ${repo} — ` +
      `this runs extraction and may take a while`
  );
  try {
    execFileSync("node", [join(ctx.dist, "deepen.js"), "--repo", repo, "--conversations", file], {
      stdio: "inherit",
    });
  } catch {
    // The wiring is already in place; a failed backfill must not make `install` look failed.
    ctx.log?.(`${harness}: conversation import did not finish — re-run it any time with:`);
    ctx.log?.(`  node "${join(ctx.dist, "deepen.js")}" --repo "${repo}" --conversations "${file}"`);
  }
}

export function run(argv: string[], ctxIn: InstallCtx): number {
  let ctx = ctxIn;
  const [command, ...rawArgs] = argv;
  ctx = { ...ctx, args: rawArgs };
  // `--import-conversations` backfills this repo's PAST sessions for the harness being installed —
  // the migration path off the older per-agent plugins, whose banks the server cannot merge into
  // this one. Opt-in: it re-extracts history and therefore costs tokens.
  const importHistory = rawArgs.includes("--import-conversations");
  // A flag's VALUE (`--server daemon`) is a bare word too — excluding it keeps "daemon" from being
  // read as a harness name and rejected.
  const valueArgs = flagValueArgs(rawArgs, ["server", "api-url", "api-token"]);
  const names = rawArgs.filter((a) => !a.startsWith("--") && !valueArgs.has(a));
  // Read-only: summarizes the local usage log (core/usage.ts) and touches nothing else.
  if (command === "stats") {
    ctx.log?.(formatUsageReport(readUsage()));
    return 0;
  }
  // Everything we write into a host's config is an ABSOLUTE path into this package. Run straight
  // from an npx cache those paths die on the first eviction and every hook stops SILENTLY, which is
  // why installing from a cache used to be refused outright. Copying the runtime somewhere stable
  // first removes the problem instead of pushing it onto the user: `npx` now works, and nobody has
  // to keep a global install of a tool whose only job is to set other tools up.
  if (command === "update") ctx = stageRuntime(ctx);
  // `update` is `install`'s staging half and nothing else: it replaces the runtime every wired
  // agent already points at (~/.remember/coding-agents/dist, a path that is stable across
  // versions) and writes to NO host config. That separation is what makes it safe to run
  // unattended from `core/auto-update.ts` — an `install` would need a harness list, and choosing
  // one on the user's behalf would rewire agents they never asked us to touch.
  //
  // The cost of not rewiring: a version that introduces a NEW hook entry point is staged but not
  // referenced, so that one feature waits for a manual `install`. Every existing entry point picks
  // the new code up on its next spawn.
  if (command === "update") {
    ctx.log?.(
      ctx.pkgRoot === ctxIn.pkgRoot
        ? "runtime already up to date — nothing staged"
        : "runtime updated — every wired agent picks it up on its next session"
    );
    return 0;
  }
  if (command !== "install" && command !== "uninstall") {
    ctx.log?.(
      `usage: remember-coding-agents <install|uninstall> <all|harness...>\n` +
        `       remember-coding-agents update\n` +
        `       remember-coding-agents stats\n` +
        `       [--server daemon]\n` +
        `       [--import-conversations] [--enable-workspace-mcp]\n` +
        `  all      every agent detected on this machine\n` +
        `  harness  ${INSTALLERS.map((i) => i.name).join(", ")}\n` +
        `  update   re-stage the runtime only, leaving every host config untouched\n` +
        `  agents/CI: without a TTY nothing ever prompts — pass --server (and --api-url/--api-token) to choose`
    );
    return command ? 1 : 0;
  }
  let targets: HarnessInstaller[];
  // `all` is spelled out rather than implied by a bare command: this rewrites the config of EVERY
  // detected agent — hooks, MCP registration and the companion skill — which is too much of a
  // machine to change by accident from `install` alone.
  if (names.includes("all")) {
    targets = INSTALLERS.filter((i) => i.detect(ctx));
    if (!targets.length) {
      ctx.log?.("no supported coding agents detected — name one explicitly to wire it anyway");
      return 1;
    }
    ctx.log?.(`detected: ${targets.map((t) => t.name).join(", ")}`);
  } else if (names.length) {
    targets = [];
    for (const n of names) {
      const hit = INSTALLERS.find((i) => i.name === (HARNESS_ALIASES[n] ?? n));
      if (!hit) {
        ctx.log?.(
          `unknown harness "${n}" — expected "all" or one of: ${INSTALLERS.map((i) => i.name).join(", ")}`
        );
        return 1;
      }
      targets.push(hit);
    }
  } else {
    ctx.log?.(
      `${command}: name a harness, or "all" for every agent detected on this machine.\n` +
        `  remember-coding-agents ${command} claude-code\n` +
        `  remember-coding-agents ${command} all\n` +
        `harnesses: ${INSTALLERS.map((i) => i.name).join(", ")}`
    );
    return 1;
  }
  // Preflight runs before server or host config is written. It only blocks the harness that failed:
  // on `install all` the other agents are still worth wiring. The non-zero exit keeps the failure
  // visible to whatever script invoked this.
  const blocked = new Set<string>();
  if (command === "install") {
    for (const t of targets) {
      const problem = t.preflight?.(ctx);
      if (!problem) continue;
      ctx.log?.(`\n❌ ${t.name}: ${problem}`);
      blocked.add(t.name);
    }
  }
  const runnable = targets.filter((t) => !blocked.has(t.name));
  // Resolve the server only when at least one harness can be installed, and before wiring any of
  // them so the first session already has a config to read.
  if (
    command === "install" &&
    runnable.length > 0 &&
    !configureServer(
      ctx,
      rawArgs,
      runnable.map((t) => t.name),
      ctxIn.pkgRoot
    )
  )
    return 1;

  if (command === "install" && runnable.length > 0) ctx = stageRuntime(ctx);
  const failed: string[] = [];
  for (const t of runnable) {
    if (t[command](ctx) === false) failed.push(t.name);
  }
  if (command === "install" && importHistory) {
    for (const t of runnable) importConversations(t.name, ctx);
  }
  if (blocked.size || failed.length) {
    const notInstalled = [...blocked, ...failed];
    ctx.log?.(`\n❌ not installed: ${notInstalled.join(", ")} — see the messages above.`);
    return 1;
  }
  // No completion message here: the CLI entry's InstallerUi outro reports success (and where the
  // settings live), so run() stays a silent-on-success engine for programmatic use.
  return 0;
}

/* c8 ignore start */
// npm exposes `bin` entries through a symlink. Node leaves argv[1] at that symlink path, so a
// direct URL comparison incorrectly makes the installed CLI a silent no-op. Resolve it first;
// source/dev invocations still work through the installer.js suffix fallback.
const mainPath = process.argv[1]
  ? (() => {
      try {
        return realpathSync(process.argv[1]);
      } catch {
        return process.argv[1];
      }
    })()
  : undefined;
const isMain = mainPath && import.meta.url === pathToFileURL(mainPath).href;
if (isMain || mainPath?.endsWith("installer.js")) {
  const dist = dirname(fileURLToPath(import.meta.url));
  const pkgRoot = dirname(dist);
  let version: string | undefined;
  try {
    version = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")).version;
  } catch {
    version = undefined; // cosmetic only — a header without a version beats a crashed installer
  }
  const ui = createInstallerUi({
    home: homedir(),
    command: process.argv[2],
    version,
    harnessNames: INSTALLERS.map((i) => i.name),
    // configureServer's messages render as their own "server" group, but don't count as an agent.
    auxNames: ["server"],
    configPath: process.env.REMEMBER_CONFIG || undefined,
  });
  ui.intro();
  const code = run(process.argv.slice(2), {
    home: homedir(),
    pkgRoot,
    dist,
    // Only a real terminal gets prompted; piped/CI installs take the documented default.
    // tty.isatty, NOT process.stdin.isTTY: the process.stdin getter initializes the stdin TTY
    // stream, which puts fd 0 in non-blocking mode and breaks readLineSync (see there).
    interactive: Boolean(isatty(0) && isatty(1)),
    log: ui.log,
    promptStyle: ui.prompt,
    selectPrompt: ui.select,
  });
  ui.outro(code);
  process.exit(code);
}
/* c8 ignore stop */
