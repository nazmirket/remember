#!/usr/bin/env node
/**
 * Start the local `remember-embed` daemon. Spawned DETACHED by core/daemon.ts — never run inline
 * by a hook, because a cold start (uvx download + model load) outlives every hook timeout.
 *
 * Blocking here is the point: this process exists purely to own that wait, and exits once the
 * daemon answers /health. It writes only to the plugin log, so its stdio can be discarded.
 */
import { loadConfig } from "./core/config";
import {
  daemonEnv,
  isRememberCheckout,
  isPortOccupied,
  isServerHealthy,
  preflightDaemon,
} from "./core/daemon";
import { diag } from "./core/diag";
import { log, setLogLevel } from "./core/log";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

async function main(): Promise<void> {
  const harness = arg("harness") ?? "coding-agent";
  const cfg = loadConfig({ harness });
  setLogLevel(cfg.logLevel);

  // Several sessions can race to start one daemon; whoever loses simply finds it healthy.
  if (await isServerHealthy(cfg.apiUrl)) return;
  if (await isPortOccupied(cfg.apiPort))
    throw new Error(`Memory daemon port ${cfg.apiPort} is occupied`);
  if (!preflightDaemon(cfg, harness)) return;
  if (!isRememberCheckout(cfg.embedPackagePath)) {
    throw new Error("Set REMEMBER_EMBED_PACKAGE_PATH to Remember's embed directory");
  }

  // An agent can start in any repository. Its .env must not override the local
  // bind address, database, or provider settings inherited by the daemon.
  process.env.PYTHON_DOTENV_DISABLED = "1";
  // Pass provider settings and credentials through the process environment.
  const daemonSettings = daemonEnv(cfg);
  const { RememberServer } = await import("./core/local-server/server");
  const server = new RememberServer({
    port: cfg.apiPort,
    env: daemonSettings,
    embedPackagePath: cfg.embedPackagePath,
    logger: {
      debug: (m) => log.debug(harness, m),
      info: (m) => log.info(harness, m),
      warn: (m) => log.warn(harness, m),
      error: (m) => log.error(harness, m),
    },
  });
  await server.start();
  diag(harness, "daemon_started", { apiUrl: cfg.apiUrl });
}

main().catch((error) => {
  diag(arg("harness") ?? "coding-agent", "daemon_start_failed", { error: String(error) });
});
