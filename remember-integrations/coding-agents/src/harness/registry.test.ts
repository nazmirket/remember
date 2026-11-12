import { describe, expect, it } from "vitest";
import { INSTALLERS } from "../installer";
import { HOOK_HARNESSES } from "./hook-lifecycle";
import { getHarness, HARNESS_NAMES } from "./registry";

describe("supported agent registry", () => {
  it("keeps installer, hooks and history adapters aligned", () => {
    expect(HARNESS_NAMES).toEqual(["claude-code", "codex"]);
    expect(INSTALLERS.map((entry) => entry.name)).toEqual(HARNESS_NAMES);
    expect(Object.keys(HOOK_HARNESSES)).toEqual(HARNESS_NAMES);
  });
  it.each(HARNESS_NAMES)("resolves %s", async (name) => {
    const adapter = await getHarness(name);
    expect(adapter.name).toBe(name);
  });
  it("rejects unsupported agents", async () => {
    await expect(getHarness("opencode")).rejects.toThrow("unknown harness");
  });
});
