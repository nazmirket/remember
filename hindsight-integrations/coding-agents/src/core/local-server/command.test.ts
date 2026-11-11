import { describe, expect, it } from "vitest";
import { getEmbedCommand } from "./command";

describe("local embedded command", () => {
  it("uses the fork checkout", () => {
    expect(getEmbedCommand({ embedPackagePath: "/fork/hindsight-embed" })).toEqual([
      "uv",
      "run",
      "--directory",
      "/fork/hindsight-embed",
      "hindsight-embed",
    ]);
  });
  it("never falls back to a published upstream package", () => {
    expect(() => getEmbedCommand()).toThrow("checkout is required");
  });
});
