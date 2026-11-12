import { join } from "node:path";
import { expect, it } from "vitest";
import { rememberHome } from "./remember-home";

it("uses the default home or an explicit override", () => {
  expect(rememberHome("/home/test", {})).toBe(join("/home/test", ".remember"));
  expect(rememberHome("/home/test", { REMEMBER_HOME: "/custom" })).toBe("/custom");
});
