import type { Logger } from "./logger.js";

/**
 * Options for {@link RememberServer}.
 *
 * The server is intentionally thin and pass-through: anything configurable
 * on the daemon side (env vars or CLI flags) can be set here without needing
 * a new dedicated option. Use {@link env} for `REMEMBER_*` / `OPENAI_API_KEY` /
 * local or hosted provider settings.
 *
 * For talking to the daemon after `start()`, use `remember-client`
 * against `server.getBaseUrl()`. This package does not ship its own HTTP
 * client.
 */
export interface RememberServerOptions {
  /** TCP port the daemon listens on. Default: `8888`. */
  port?: number;
  /** Local path to a `embed` directory — required. */
  embedPackagePath?: string;
  /**
   * Environment variables passed to the daemon process. This is the preferred way
   * to surface any `REMEMBER_API_*` / `REMEMBER_EMBED_*` setting — adding a
   * new daemon env var never requires a wrapper update.
   *
   * Values of `undefined` are dropped (so you can spread conditionally).
   */
  env?: Record<string, string | undefined>;
  /**
   * On macOS, automatically set
   * `REMEMBER_API_EMBEDDINGS_LOCAL_FORCE_CPU=1` and
   * `REMEMBER_API_RERANKER_LOCAL_FORCE_CPU=1` to avoid Metal/MPS crashes in
   * daemon mode. Default: `true` on `darwin`, ignored elsewhere. Any value set
   * explicitly in {@link env} wins over the auto-applied value.
   */
  platformCpuWorkaround?: boolean;
  /** Max time (ms) to wait for `/health` to return 200. Default: `30_000`. */
  readyTimeoutMs?: number;
  /** Polling interval (ms) while waiting for `/health`. Default: `1_000`. */
  readyPollIntervalMs?: number;
  /** Optional pluggable logger. Default: silent. */
  logger?: Logger;
}
