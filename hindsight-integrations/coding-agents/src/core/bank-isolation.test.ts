import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { deriveBankId } from "./bank";

it("separates same-named repositories while sharing a bank between agents and worktrees", () => {
  const root = mkdtempSync(join(tmpdir(), "hindsight-bank-isolation-"));
  try {
    const first = join(root, "first", "project");
    const second = join(root, "second", "project");
    const worktree = join(root, "worktree");
    mkdirSync(first, { recursive: true });
    mkdirSync(second, { recursive: true });
    for (const repo of [first, second]) {
      execFileSync("git", ["init", "-q", repo]);
      execFileSync("git", [
        "-C",
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "initial",
      ]);
    }
    execFileSync("git", ["-C", first, "worktree", "add", "-q", "-b", "test-worktree", worktree]);
    const claudeBank = deriveBankId({}, first, "claude-code");
    expect(deriveBankId({}, first, "codex")).toBe(claudeBank);
    expect(deriveBankId({}, worktree, "codex")).toBe(claudeBank);
    expect(deriveBankId({}, second, "codex")).not.toBe(claudeBank);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
