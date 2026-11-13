import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TransportTurn } from "./chat";
import {
  formatUsageReport,
  memoryUsageCursorStore,
  readUsage,
  recordUsage,
  summarizeTurns,
} from "./usage";

const t = (role: string, content: string): TransportTurn => ({ role, content });

let root: string;
let usageFile: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-usage-"));
  usageFile = join(root, "usage.jsonl");
  vi.stubEnv("REMEMBER_USAGE_FILE", usageFile);
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

const lines = () =>
  readFileSync(usageFile, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));

describe("summarizeTurns", () => {
  it("counts only Remember tools, whatever prefix the host gives them", () => {
    const usage = summarizeTurns([
      t("system", "REF-ID: x"), // before the first user turn: belongs to no turn
      t("user", "how do we round?"),
      t("action", "mcp__remember__remember_search_pages rounding"),
      t("action", "Read src/round.ts"),
      t("action", "remember.remember_reflect why half-up"),
      t("assistant", "> 🧠 **From Remember memory (Conventions)** — half up"),
      t("user", "thanks"),
      t("action", "remember_read_page kp-1"),
      t("assistant", "done"),
    ]);
    expect(usage).toEqual([
      {
        turn: 1,
        calls: ["remember_search_pages", "remember_reflect"],
        credited: true,
      },
      { turn: 2, calls: ["remember_read_page"], credited: false },
    ]);
  });

  it("recognizes tool calls and memory credits", () => {
    expect(
      summarizeTurns([
        t("user", "q"),
        t("action", "mcp__remember__remember_reflect q"),
        t("assistant", "From Remember memory: answer"),
      ])
    ).toEqual([{ turn: 1, calls: ["remember_reflect"], credited: true }]);
  });

  it("reads credit from Gemini-shaped `model` turns, and never from the user's own words", () => {
    const usage = summarizeTurns([
      t("user", "did it come From Remember memory?"),
      t("user", "second"),
      t("model", "🧠 From Remember memory — yes"),
    ]);
    expect(usage.map((u) => u.credited)).toEqual([false, true]);
  });
});

describe("recordUsage", () => {
  const base = { harness: "claude-code", sessionId: "s1", bankId: "bank-1" };

  it("records each turn once across repeated Stops over the growing transcript", () => {
    const cursors = memoryUsageCursorStore();
    const first = [t("user", "a"), t("action", "mcp__remember__remember_reflect q")];
    recordUsage({ ...base, turns: first, cursors, lastTurnComplete: true });
    recordUsage({ ...base, turns: first, cursors, lastTurnComplete: true }); // Stop re-delivered
    recordUsage({
      ...base,
      turns: [...first, t("user", "b"), t("assistant", "ok")],
      cursors,
      lastTurnComplete: true,
    });

    expect(lines()).toMatchObject([
      {
        harness: "claude-code",
        session: "s1",
        bank: "bank-1",
        turn: 1,
        calls: ["remember_reflect"],
      },
      { turn: 2, calls: [], credited: false },
    ]);
  });

  it("re-reads the previous turn on the next Stop, so a late credit line is not lost", () => {
    const cursors = memoryUsageCursorStore();
    const search = t("action", "remember_search_pages q");
    // Stop fires before the host flushed the reply — and that reply is where the
    // credit blockquote lives, so turn 1 is written uncredited.
    recordUsage({
      ...base,
      turns: [t("user", "a"), search],
      cursors,
      lastTurnComplete: true,
      reviseLastTurn: true,
    });
    expect(lines().at(-1)).toMatchObject({ turn: 1, credited: false });

    // The next Stop reads a complete transcript and corrects it.
    recordUsage({
      ...base,
      turns: [
        t("user", "a"),
        search,
        t(
          "assistant",
          "> 🧠 **From Remember memory (Pricing decisions)** — the threshold is post-discount"
        ),
        t("user", "b"),
        t("assistant", "ok"),
      ],
      cursors,
      lastTurnComplete: true,
      reviseLastTurn: true,
    });

    const written = lines();
    // readUsage keeps the LAST line per turn, so the corrected one wins.
    expect(written.filter((l) => l.turn === 1).at(-1)).toMatchObject({ turn: 1, credited: true });
    expect(written.at(-1)).toMatchObject({ turn: 2 });
  });

  it("holds back an unanswered last turn instead of recording it empty", () => {
    const cursors = memoryUsageCursorStore();
    recordUsage({ ...base, turns: [t("user", "a")], cursors, lastTurnComplete: false });
    expect(() => readFileSync(usageFile)).toThrow(); // nothing finished yet

    recordUsage({
      ...base,
      turns: [t("user", "a"), t("action", "remember_reflect q"), t("assistant", "done")],
      cursors,
      lastTurnComplete: true,
    });
    expect(lines()).toMatchObject([{ turn: 1, calls: ["remember_reflect"] }]);
  });
});

describe("readUsage", () => {
  it("aggregates per harness, dedupes re-recorded turns and includes the rotated generation", () => {
    const line = (o: object) => JSON.stringify({ ts: "x", bank: "b", ...o }) + "\n";
    writeFileSync(
      `${usageFile}.1`,
      line({
        harness: "codex",
        session: "a",
        turn: 1,
        calls: ["remember_reflect"],
        credited: true,
      })
    );
    writeFileSync(
      usageFile,
      line({
        harness: "codex",
        session: "a",
        turn: 1,
        calls: ["remember_reflect"],
        credited: true,
      }) +
        line({ harness: "codex", session: "a", turn: 2, calls: [], credited: false }) +
        // A write-only turn: counts as a call, never against the credit rate.
        line({
          harness: "codex",
          session: "a",
          turn: 3,
          calls: ["remember_ingest_document"],
          credited: false,
        }) +
        line({
          harness: "codex",
          session: "b",
          turn: 1,
          calls: ["remember_reflect", "remember_search_pages"],
          credited: false,
        }) +
        "not json\n"
    );

    const usage = readUsage(usageFile);
    expect(usage).toEqual([
      {
        harness: "codex",
        turns: 4,
        turnsWithCalls: 3,
        calls: 4,
        turnsWithRetrieval: 2,
        creditedTurnsWithRetrieval: 1,
        byTool: {
          remember_reflect: 2,
          remember_search_pages: 1,
          remember_ingest_document: 1,
        },
      },
    ]);
    expect(formatUsageReport(usage, usageFile)).toContain(
      "codex: 4 turns · 3 with a Remember call (75%) · 1.00 calls/turn · " +
        "credited after retrieval 1/2 (50%)"
    );
  });

  it("says so when nothing is recorded", () => {
    expect(formatUsageReport(readUsage(usageFile), usageFile)).toBe(
      `no Remember usage recorded yet (${usageFile})`
    );
  });
});
