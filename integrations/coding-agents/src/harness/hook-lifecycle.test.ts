import { describe, expect, it } from "vitest";
import { HOOK_HARNESSES, type HookHarnessName } from "./hook-lifecycle";

describe("supported hook lifecycles", () => {
  it.each(Object.keys(HOOK_HARNESSES) as HookHarnessName[])(
    "keeps %s events aligned",
    (harness) => {
      const spec = HOOK_HARNESSES[harness];
      expect(Object.keys(spec.install).sort()).toEqual(["prompt", "sessionStart", "stop"]);
      expect(spec.sessionStart.harness).toBe(harness);
      expect(spec.prompt.harness).toBe(harness);
      expect(spec.retain.harness).toBe(harness);
    }
  );
  it("accepts Codex's user_prompt field", () => {
    expect(HOOK_HARNESSES.codex.prompt.parse({ user_prompt: "hello" }).prompt).toBe("hello");
  });
  it("retains Claude's final assistant message", () => {
    expect(
      HOOK_HARNESSES["claude-code"].retain.parse({ last_assistant_message: "done" })
        .lastAssistantMessage
    ).toBe("done");
  });
});
