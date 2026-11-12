/** On-demand lifecycle for the checkout-local memory daemon. */
import { execFileSync, spawn as realSpawn } from "node:child_process";
import { dirname, join } from "node:path";
import { connect } from "node:net";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import type { Config } from "./config";
import { USER_AGENT } from "./remember";
import { diag } from "./diag";
import { log } from "./log";

/**
 * How long each hook is willing to block waiting for a cold daemon. Both sit well inside the
 * corresponding hook timeouts declared in harness/hook-lifecycle.ts (SessionStart 30s, Stop 60s),
 * because overrunning those gets the hook killed by the host.
 */
export const DAEMON_WAIT_SESSION_START_MS = 12_000;
export const DAEMON_WAIT_RETAIN_MS = 40_000;

/** Probe a base URL's /health. Never throws — an unreachable server is just "not running". */
export async function isServerHealthy(baseUrl: string, timeoutMs = 2_000): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/health`, {
      headers: { "User-Agent": USER_AGENT },
      signal: AbortSignal.timeout(timeoutMs),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Refuse to reclaim an occupied port, even if the listener is unhealthy. */
export async function isPortOccupied(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = connect({ host: "127.0.0.1", port });
    socket.setTimeout(1000);
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => {
      socket.destroy();
      resolve(false);
    });
    socket.once("timeout", () => {
      socket.destroy();
      resolve(true);
    });
  });
}

/** Is `uv` on PATH? The local daemon runs through it. */
export function hasUv(): boolean {
  try {
    execFileSync("uv", ["--version"], {
      stdio: "pipe",
      timeout: 10_000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

interface ProviderProbe {
  provider: string;
  keyEnv: string;
}

/**
 * Providers the daemon can use for local fact extraction, most-specific first. Mirrors the old
 * plugin's list. `claude-code` needs no key — it drives the Claude Code CLI the user already has,
 * which is why it is the last resort rather than an error.
 */
const PROVIDER_PROBES: ProviderProbe[] = [
  { provider: "openai", keyEnv: "OPENAI_API_KEY" },
  { provider: "anthropic", keyEnv: "ANTHROPIC_API_KEY" },
];

export interface LlmChoice {
  provider: string;
  /** Absent for providers that need no credential. */
  apiKey?: string;
  /** Where it came from, for the installer's output and diagnostics. */
  source: string;
}

/**
 * Decide which LLM the daemon extracts facts with.
 *
 * An explicit `REMEMBER_API_LLM_PROVIDER` always wins — including for providers this code knows
 * nothing about, so a local Ollama or a gateway can be pointed at without a code change here.
 */
export function detectLlm(env: NodeJS.ProcessEnv = process.env): LlmChoice | undefined {
  const explicit = env.REMEMBER_API_LLM_PROVIDER?.trim();
  if (explicit) {
    return {
      provider: explicit,
      apiKey: env.REMEMBER_API_LLM_API_KEY?.trim() || undefined,
      source:
        env.REMEMBER_API_LLM_PROVIDER !== undefined
          ? "REMEMBER_API_LLM_PROVIDER"
          : "REMEMBER_API_LLM_PROVIDER",
    };
  }
  for (const probe of PROVIDER_PROBES) {
    const key = env[probe.keyEnv]?.trim();
    if (key) return { provider: probe.provider, apiKey: key, source: probe.keyEnv };
  }
  if (onPath("claude")) {
    return { provider: "claude-code", source: "the Claude Code CLI on PATH (no API key needed)" };
  }
  if (onPath("codex")) {
    return { provider: "openai-codex", source: "the Codex CLI on PATH (no API key needed)" };
  }
  return undefined;
}

function onPath(bin: string): boolean {
  try {
    execFileSync("which", [bin], {
      stdio: "pipe",
      timeout: 5_000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

/** Environment handed to the daemon: the LLM choice and any REMEMBER_API_* the
 *  user already exports (so provider-specific knobs pass through without being enumerated here). */
export function daemonEnv(
  cfg: Config,
  env: NodeJS.ProcessEnv = process.env
): Record<string, string | undefined> {
  const out: Record<string, string | undefined> = {};
  for (const [key, value] of Object.entries(env)) {
    if (key.startsWith("REMEMBER_API_") && key !== "REMEMBER_API_DATABASE_URL" && value) {
      out[key] = value;
    }
  }
  // Use the same optional local token for the daemon and its clients.
  if (cfg.apiToken) out.REMEMBER_API_TOKEN = cfg.apiToken;
  // The local memory service must not inherit an externally exposed bind address.
  out.REMEMBER_API_HOST = "127.0.0.1";
  // Database selection is owned by remember-embed and always uses local pg0.
  const llm = detectLlm(env);
  if (llm) {
    out.REMEMBER_API_LLM_PROVIDER = llm.provider;
    if (llm.apiKey) {
      out.REMEMBER_API_LLM_API_KEY = llm.apiKey;
    }
  }
  return out;
}

/** Require the Remember checkout so uv cannot silently launch a published upstream daemon. */
export function isRememberCheckout(path: string | undefined): boolean {
  return (
    !!path &&
    existsSync(join(path, "pyproject.toml")) &&
    existsSync(join(path, "..", "remember-api", "pyproject.toml"))
  );
}

/**
 * Launch the starter as a DETACHED process, the way seeding and the codebase survey are launched.
 *
 * A cold start pays for dependency setup and model load and routinely outlives any hook's timeout,
 * so it cannot be awaited inline: the harness would kill the hook mid-start and the daemon would
 * never come up. The child outlives this process and keeps going.
 *
 * One spawn per cold start is the goal, and `ensureDaemon` is what holds a burst of callers to it.
 * The starter also re-checks health, so a spawn that slips through anyway leaves the daemon the
 * winner started alone — but racing starters are not free: they queue on the daemon's file lock.
 */
export function startDaemonDetached(
  cfg: Config,
  harness: string,
  spawnFn: typeof realSpawn = realSpawn
): void {
  try {
    const starter = join(dirname(fileURLToPath(import.meta.url)), "daemon-start.js");
    const child = spawnFn("node", [starter, "--harness", harness], {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    // spawn() failures often surface ASYNCHRONOUSLY as an 'error' event; unhandled, that would
    // crash the hook.
    child.on("error", () => {});
    child.unref();
    diag(harness, "daemon_start_spawned", { apiUrl: cfg.apiUrl });
  } catch {
    /* best-effort: a failed spawn must not break the caller */
  }
}

/**
 * Cold starts currently in flight, keyed by API URL.
 *
 * Every ensure point re-checks health and starts its own daemon when the port is dead, and the
 * persistent-plugin hosts fire several of them within the same tick — one SessionStart per session
 * plus a Stop per finished turn. Each of them therefore observes the same dead port before the
 * first daemon answers, and each spawns a starter of its own. The starters then serialise on the
 * profile's file lock, where the losers sit for the full 300 s lock budget before failing
 * (#1155, #3100). One entry per URL collapses that burst onto a single spawn.
 *
 * Keyed by URL rather than held as a module singleton because a caller may be pointed at any
 * daemon port. The entry is dropped as soon as its attempt settles, so a later turn that still
 * finds the port dead starts a fresh attempt instead of inheriting a spent one.
 *
 * Module-global, and deliberately without a lock: Node runs one thread, and the check-then-set
 * below contains no `await`, so two callers cannot interleave between the `get` and the `set`.
 */
const coldStarts = new Map<string, Promise<void>>();

/**
 * Await `p` for at most `ms`. The timer is cleared on the way out: a follower that gives up early
 * must not leave an abandoned `setTimeout` holding the hook process open behind it.
 */
async function waitBounded(p: Promise<void>, ms: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      p,
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, Math.max(ms, 0));
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Make sure a daemon is serving `cfg.apiUrl`. A no-op in every other mode.
 *
 * Side effect only — callers proceed regardless of whether it came up, so that a request against a
 * down daemon fails through exactly the same path as a request against a down Cloud or self-hosted
 * server. Never throws: memory is best-effort and must not break the agent.
 *
 * `waitMs` bounds how long the caller is willing to block for a cold start — always well under the
 * calling hook's own timeout, so a slow start costs memory for one turn rather than a killed hook.
 *
 * A start already underway is joined rather than duplicated; see `coldStarts`.
 */
export async function ensureDaemon(
  cfg: Config,
  harness: string,
  opts: { waitMs?: number; spawnFn?: typeof realSpawn } = {}
): Promise<void> {
  if (cfg.serverMode !== "daemon") return;
  if (await isServerHealthy(cfg.apiUrl)) return;
  if (!preflightDaemon(cfg, harness)) return;

  const waitMs = opts.waitMs ?? 0;
  const running = coldStarts.get(cfg.apiUrl);
  if (running) {
    // Join the start already underway instead of spawning a competitor, so a burst costs one
    // starter rather than one per caller. Never for longer than this caller's own budget — a Stop
    // hook waiting 40 s must not push a SessionStart hook past its 30 s timeout — and whatever is
    // left of that budget once the shared attempt settles goes on waiting for the daemon the
    // attempt set in motion, so joining never leaves a caller with less patience than it would
    // have had on its own.
    const startedAt = Date.now();
    // Awaited for its side effect only: a follower does not inherit the attempt's failure.
    await waitBounded(
      running.catch(() => undefined),
      waitMs
    );
    const remaining = waitMs - (Date.now() - startedAt);
    if (remaining > 0) await waitForHealth(cfg.apiUrl, remaining);
    return;
  }

  const attempt = (async () => {
    if (await isPortOccupied(cfg.apiPort)) {
      log.warn(harness, `Memory daemon port ${cfg.apiPort} is occupied; not starting`);
      return;
    }
    startDaemonDetached(cfg, harness, opts.spawnFn);
    await waitForHealth(cfg.apiUrl, waitMs);
  })();
  coldStarts.set(cfg.apiUrl, attempt);
  try {
    await attempt;
  } finally {
    // Only the owner clears the entry, so a follower can never drop a newer attempt's guard.
    if (coldStarts.get(cfg.apiUrl) === attempt) coldStarts.delete(cfg.apiUrl);
  }
}

/** Poll /health until it answers or the budget runs out. `budgetMs <= 0` means "don't wait". */
export async function waitForHealth(
  baseUrl: string,
  budgetMs: number,
  pollMs = 1_000
): Promise<boolean> {
  const deadline = Date.now() + budgetMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollMs));
    if (await isServerHealthy(baseUrl)) return true;
  }
  return false;
}

/** The two things daemon mode cannot work without. Reported, never thrown. */
export function preflightDaemon(cfg: Config, harness: string): boolean {
  if (!hasUv()) {
    diag(harness, "daemon_uv_missing", { apiUrl: cfg.apiUrl });
    log.warn(harness, "daemon mode needs `uv` on PATH — see https://docs.astral.sh/uv/");
    return false;
  }
  if (!detectLlm()) {
    diag(harness, "daemon_no_llm", {});
    log.warn(
      harness,
      "daemon mode needs an LLM for fact extraction — set OPENAI_API_KEY (or ANTHROPIC_API_KEY / " +
        "REMEMBER_API_LLM_PROVIDER), or install the Claude Code or Codex CLI"
    );
    return false;
  }
  return true;
}
