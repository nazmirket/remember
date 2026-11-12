import { spawn } from "child_process";
import { getEmbedCommand } from "./command.js";
import { silentLogger } from "./logger.js";
import type { Logger } from "./logger.js";
import type { RememberServerOptions } from "./types.js";

const DEFAULT_PORT = 8888;
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PROFILE = "default";
const DEFAULT_READY_TIMEOUT_MS = 30_000;
const DEFAULT_READY_POLL_INTERVAL_MS = 1_000;

/** Manages the checkout's embedded daemon and waits for readiness. */
export class RememberServer {
  private readonly profile: string;
  private readonly port: number;
  private readonly host: string;
  private readonly baseUrl: string;
  private readonly embedPackagePath: string | undefined;
  private readonly userEnv: Record<string, string | undefined>;
  private readonly extraProfileCreateArgs: string[];
  private readonly extraDaemonStartArgs: string[];
  private readonly platformCpuWorkaround: boolean;
  private readonly readyTimeoutMs: number;
  private readonly readyPollIntervalMs: number;
  private readonly logger: Logger;

  constructor(opts: RememberServerOptions = {}) {
    this.profile = opts.profile ?? DEFAULT_PROFILE;
    this.port = opts.port ?? DEFAULT_PORT;
    this.host = opts.host ?? DEFAULT_HOST;
    this.baseUrl = `http://${this.host}:${this.port}`;
    this.embedPackagePath = opts.embedPackagePath;
    this.userEnv = opts.env ?? {};
    this.extraProfileCreateArgs = opts.extraProfileCreateArgs ?? [];
    this.extraDaemonStartArgs = opts.extraDaemonStartArgs ?? [];
    this.platformCpuWorkaround = opts.platformCpuWorkaround ?? process.platform === "darwin";
    this.readyTimeoutMs = opts.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS;
    this.readyPollIntervalMs = opts.readyPollIntervalMs ?? DEFAULT_READY_POLL_INTERVAL_MS;
    this.logger = opts.logger ?? silentLogger;
  }

  /** The base URL the daemon listens on (`http://host:port`). */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /** The profile name this server operates on. */
  getProfile(): string {
    return this.profile;
  }

  /**
   * Ensure the daemon is configured and running. Idempotent — the underlying
   * `profile create --merge` and `daemon start` commands tolerate re-runs.
   */
  async start(): Promise<void> {
    this.logger.info(`[remember] starting daemon for profile "${this.profile}"`);

    const env = this.buildEnv();
    await this.configureProfile(env);
    await this.startDaemon(env);
    await this.waitForReady();

    this.logger.info(`[remember] daemon ready at ${this.baseUrl}`);
  }

  /** Stop the daemon. Never throws — logs and resolves even on failure. */
  async stop(): Promise<void> {
    this.logger.info(`[remember] stopping daemon for profile "${this.profile}"`);

    const [cmd, ...baseArgs] = getEmbedCommand({
      embedPackagePath: this.embedPackagePath,
    });
    const args = [...baseArgs, "daemon", "--profile", this.profile, "stop"];

    // Resolve the CLI and its profile with the same caller overrides as start.
    // Inheriting process.env here loses a custom PATH or configuration root.
    const child = spawn(cmd, args, { stdio: "pipe", env: this.buildEnv(), windowsHide: true });
    this.pipeOutput(child, "daemon.stop");

    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        this.logger.warn(`[remember] daemon stop timed out after 5s`);
        resolve();
      }, 5_000);
      child.on("exit", () => {
        clearTimeout(timeout);
        this.logger.info(`[remember] daemon stopped`);
        resolve();
      });
      child.on("error", (err) => {
        clearTimeout(timeout);
        this.logger.warn(`[remember] error stopping daemon: ${err.message}`);
        resolve();
      });
    });
  }

  /** Probe `/health` once with a short timeout. */
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(2_000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  /**
   * Merge the process env, the caller-supplied `env`, and (on macOS) the
   * embeddings CPU workaround. Caller-supplied values always win over the
   * workaround; undefined values are dropped.
   */
  private buildEnv(): NodeJS.ProcessEnv {
    const merged: NodeJS.ProcessEnv = { ...process.env };

    if (this.platformCpuWorkaround && process.platform === "darwin") {
      merged["REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1";
      merged["REMEMBER_API_RERANKER_LOCAL_FORCE_CPU"] = "1";
    }

    for (const [key, value] of Object.entries(this.userEnv)) {
      if (value !== undefined) {
        merged[key] = value;
      }
    }

    return merged;
  }

  /**
   * Run `profile create <name> --merge --port <port> [--env K=V ...]`.
   * Every entry in the merged env that was passed via {@link userEnv} (or
   * auto-applied by the CPU workaround) is forwarded as `--env`.
   */
  private async configureProfile(env: NodeJS.ProcessEnv): Promise<void> {
    this.logger.info(`[remember] configuring profile "${this.profile}"`);

    const [cmd, ...baseArgs] = getEmbedCommand({
      embedPackagePath: this.embedPackagePath,
    });
    const createArgs = [
      ...baseArgs,
      "profile",
      "create",
      this.profile,
      "--merge",
      "--port",
      String(this.port),
    ];

    // Forward every env var that the caller intended for the daemon as --env.
    // We only forward keys the caller explicitly set (userEnv) plus the CPU
    // workaround values — not the entire process.env, to avoid leaking random
    // host state into profile config.
    const envForProfile = this.collectProfileEnv(env);
    for (const [key, value] of Object.entries(envForProfile)) {
      createArgs.push("--env", `${key}=${value}`);
    }

    createArgs.push(...this.extraProfileCreateArgs);

    await this.runCommand(cmd, createArgs, env, "profile.create");
  }

  /** Collect only the env vars that should be written into the profile file. */
  private collectProfileEnv(env: NodeJS.ProcessEnv): Record<string, string> {
    const out: Record<string, string> = {};

    // 1. User-supplied env — always forwarded.
    for (const [key, value] of Object.entries(this.userEnv)) {
      if (value !== undefined) {
        out[key] = value;
      }
    }

    // 2. CPU workaround — only if auto-applied and not already overridden.
    if (this.platformCpuWorkaround && process.platform === "darwin") {
      const cpuKeys = [
        "REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU",
        "REMEMBER_API_RERANKER_LOCAL_FORCE_CPU",
      ];
      for (const key of cpuKeys) {
        if (!(key in out) && env[key] !== undefined) {
          out[key] = env[key] as string;
        }
      }
    }

    return out;
  }

  private async startDaemon(env: NodeJS.ProcessEnv): Promise<void> {
    const [cmd, ...baseArgs] = getEmbedCommand({
      embedPackagePath: this.embedPackagePath,
    });
    const args = [
      ...baseArgs,
      "daemon",
      "--profile",
      this.profile,
      "start",
      ...this.extraDaemonStartArgs,
    ];

    await this.runCommand(cmd, args, env, "daemon.start");
  }

  /**
   * Spawn `cmd` with `args`, pipe its output through the logger, and resolve
   * once it exits with code 0. Rejects on non-zero exit or spawn error.
   */
  private async runCommand(
    cmd: string,
    args: string[],
    env: NodeJS.ProcessEnv,
    label: string
  ): Promise<void> {
    const child = spawn(cmd, args, { stdio: "pipe", env, windowsHide: true });
    let output = "";
    child.stdout?.on("data", (data: Buffer) => {
      const text = data.toString();
      output += text;
      for (const line of text.trimEnd().split("\n")) {
        if (line) this.logger.info(`[remember:${label}] ${line}`);
      }
    });
    child.stderr?.on("data", (data: Buffer) => {
      const text = data.toString();
      output += text;
      for (const line of text.trimEnd().split("\n")) {
        if (line) this.logger.warn(`[remember:${label}] ${line}`);
      }
    });

    await new Promise<void>((resolve, reject) => {
      child.on("exit", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`${label} failed with code ${code}: ${output.trim()}`));
        }
      });
      child.on("error", (err) => {
        reject(new Error(`${label} failed to spawn: ${err.message}`, { cause: err }));
      });
    });
  }

  /** Stream a spawned child's stdout/stderr through the logger without blocking. */
  private pipeOutput(child: ReturnType<typeof spawn>, label: string): void {
    child.stdout?.on("data", (data: Buffer) => {
      for (const line of data.toString().trimEnd().split("\n")) {
        if (line) this.logger.info(`[remember:${label}] ${line}`);
      }
    });
    child.stderr?.on("data", (data: Buffer) => {
      for (const line of data.toString().trimEnd().split("\n")) {
        if (line) this.logger.warn(`[remember:${label}] ${line}`);
      }
    });
  }

  /** Poll `/health` until it succeeds or `readyTimeoutMs` elapses. */
  private async waitForReady(): Promise<void> {
    const deadline = Date.now() + this.readyTimeoutMs;
    let attempt = 0;
    while (Date.now() < deadline) {
      attempt++;
      try {
        const res = await fetch(`${this.baseUrl}/health`, {
          signal: AbortSignal.timeout(this.readyPollIntervalMs),
        });
        if (res.ok) {
          this.logger.debug(`[remember] health check passed (attempt ${attempt})`);
          return;
        }
      } catch {
        // expected while the daemon is still booting
      }
      await new Promise((resolve) => setTimeout(resolve, this.readyPollIntervalMs));
    }
    throw new Error(
      `Remember daemon did not become ready within ${this.readyTimeoutMs}ms at ${this.baseUrl}`
    );
  }
}
