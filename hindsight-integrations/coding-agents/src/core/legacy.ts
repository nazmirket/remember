/** Detect an older Claude plugin that would duplicate memory capture. */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, relative, isAbsolute } from "node:path";

/** The old per-agent Claude Code plugin's name in Claude's plugin registry (`<name>@<marketplace>`). */
const LEGACY_CLAUDE_PLUGIN = "hindsight-memory";

/**
 * The registry key (`hindsight-memory@<marketplace>`) of the old Claude Code plugin when it is still
 * installed AND active for `cwd`, else undefined.
 *
 * Running both is not harmless: each recalls into every prompt and retains every transcript, so the
 * agent sees two memory blocks and conversations land twice — the old plugin's copy in its single
 * static `claude_code` bank. The installer does not uninstall it (Claude owns its plugin registry),
 * so the session start is where the user finds out.
 *
 * Read from Claude's own files: `plugins/installed_plugins.json` for what is installed (user scope,
 * or a project/local scope whose `projectPath` contains `cwd`) and `settings.json` `enabledPlugins`
 * for an explicit disable. Any read failure means "not detected" — this only ever adds a warning.
 */
export function detectLegacyClaudePlugin(
  cwd: string,
  claudeDir: string = join(homedir(), ".claude")
): string | undefined {
  const registry = readJson(join(claudeDir, "plugins", "installed_plugins.json"));
  const plugins = registry?.plugins;
  if (!plugins || typeof plugins !== "object") return undefined;
  const enabled = readJson(join(claudeDir, "settings.json"))?.enabledPlugins as
    Record<string, unknown> | undefined;
  for (const [key, raw] of Object.entries(plugins as Record<string, unknown>)) {
    if (key.split("@")[0] !== LEGACY_CLAUDE_PLUGIN) continue;
    if (enabled && enabled[key] === false) continue;
    // v2 registry: an array of installs; v1: a single install object.
    const installs = (Array.isArray(raw) ? raw : [raw]) as Array<Record<string, unknown> | null>;
    const active = installs.some((i) => {
      if (!i || typeof i !== "object") return false;
      if (typeof i.projectPath !== "string") return true; // user scope: every project
      const rel = relative(i.projectPath, cwd);
      return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
    });
    if (active) return key;
  }
  return undefined;
}

function readJson(path: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

/** The user-visible SessionStart warning for a still-active old plugin. */
export function legacyClaudePluginWarning(key: string): string {
  return (
    `⚠️ The old Hindsight Claude Code plugin (${key}) is still installed — it runs alongside this ` +
    `one, so memory is recalled and retained twice.\n` +
    `  ↳ remove it: claude plugin uninstall ${key}`
  );
}
