import { describe, expect, it } from "vitest";
import { stripInjectedMemory } from "./transcript-util";

describe("stripInjectedMemory", () => {
  it("strips injected blocks without removing surrounding conversation", () => {
    for (const prefix of ["remember"]) {
      for (const name of ["memory", "pages", "page_refresh"]) {
        expect(
          stripInjectedMemory(`before <${prefix}_${name}>injected</${prefix}_${name}> after`)
        ).toBe("before  after");
      }
    }
  });
});
