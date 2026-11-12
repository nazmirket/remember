import { describe, expect, it } from "vitest";
import { getEmbedCommand } from "./command";

describe("local embedded command", () => {
  it("uses the Remember checkout", () => {
    expect(getEmbedCommand({ embedPackagePath: "/fork/remember-embed" })).toEqual([
      "uv",
      "run",
      "--directory",
      "/fork/remember-embed",
      "remember-embed",
    ]);
  });
  it("never falls back to a published upstream package", () => {
    expect(() => getEmbedCommand()).toThrow("checkout is required");
  });
});
