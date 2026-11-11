import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { detectLegacyClaudePlugin } from "./legacy";

const homes: string[] = [];

afterEach(() => {
  while (homes.length) rmSync(homes.pop()!, { recursive: true, force: true });
});

describe("detectLegacyClaudePlugin", () => {
  /** A fake `~/.claude` with the given plugin registry and user settings. */
  function claudeDir(plugins: unknown, settings?: unknown): string {
    const dir = mkdtempSync(join(tmpdir(), "hindsight-claude-"));
    homes.push(dir);
    mkdirSync(join(dir, "plugins"), { recursive: true });
    writeFileSync(join(dir, "plugins", "installed_plugins.json"), JSON.stringify(plugins));
    if (settings !== undefined) writeFileSync(join(dir, "settings.json"), JSON.stringify(settings));
    return dir;
  }
  const userInstall = { scope: "user", installPath: "/x", version: "0.7.5" };

  it("finds a user-scope install, whatever the marketplace is called", () => {
    const dir = claudeDir({ version: 2, plugins: { "hindsight-memory@hindsight": [userInstall] } });
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBe("hindsight-memory@hindsight");
  });

  it("reads the v1 registry shape (one install object, not an array)", () => {
    const dir = claudeDir({ plugins: { "hindsight-memory@vectorize": userInstall } });
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBe("hindsight-memory@vectorize");
  });

  it("ignores other plugins, including similarly named ones", () => {
    const dir = claudeDir({
      version: 2,
      plugins: { "hindsight-memory-extra@x": [userInstall], "telegram@official": [userInstall] },
    });
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBeUndefined();
  });

  it("skips a plugin the user explicitly disabled", () => {
    const dir = claudeDir(
      { version: 2, plugins: { "hindsight-memory@hindsight": [userInstall] } },
      { enabledPlugins: { "hindsight-memory@hindsight": false } }
    );
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBeUndefined();
  });

  it("counts a project-scope install only inside that project", () => {
    const dir = claudeDir({
      version: 2,
      plugins: { "hindsight-memory@hindsight": [{ scope: "project", projectPath: "/work/app" }] },
    });
    expect(detectLegacyClaudePlugin("/work/app", dir)).toBe("hindsight-memory@hindsight");
    expect(detectLegacyClaudePlugin("/work/app/src", dir)).toBe("hindsight-memory@hindsight");
    expect(detectLegacyClaudePlugin("/work/app-other", dir)).toBeUndefined();
  });

  it("is undefined when Claude has no plugin registry or it is unreadable", () => {
    const dir = mkdtempSync(join(tmpdir(), "hindsight-claude-"));
    homes.push(dir);
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBeUndefined();
    mkdirSync(join(dir, "plugins"));
    writeFileSync(join(dir, "plugins", "installed_plugins.json"), "{not json");
    expect(detectLegacyClaudePlugin("/any/repo", dir)).toBeUndefined();
  });
});
