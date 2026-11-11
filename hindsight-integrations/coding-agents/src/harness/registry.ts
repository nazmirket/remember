/** Normalized history import adapters for the supported hook-based agents. */
import { readFileSync } from "node:fs";
import type { ChatSession, HarnessAdapter } from "../core/types";

export const HARNESS_NAMES = ["claude-code", "codex"];

export const jsonChatReader = (harness: string) => ({
  describe: `${harness} sessions via a normalized JSON export`,
  async read(opts: { conversations?: string }): Promise<ChatSession[]> {
    if (!opts.conversations) return [];
    return JSON.parse(readFileSync(opts.conversations, "utf8")) as ChatSession[];
  },
});

export async function getHarness(name: string): Promise<HarnessAdapter> {
  if (!HARNESS_NAMES.includes(name)) {
    throw new Error(`unknown harness '${name}'. available: ${HARNESS_NAMES.join(", ")}`);
  }
  return {
    name,
    chatReader: jsonChatReader(name),
  };
}
