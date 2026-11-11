/** Import past Claude Code and Codex sessions for a single repository. */
import {
  closeSync,
  existsSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  statSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ChatSession } from "./types";
import { readClaudeTranscript } from "./transcript";
import { readCodexTranscript } from "./transcript-codex";
import type { TransportTurn } from "./chat";

export interface HistoryImport {
  supported: boolean;
  /** Why, when unsupported — surfaced to the user rather than failing silently. */
  reason?: string;
  sessions: ChatSession[];
  /** Sessions skipped because nothing in them proves which repo they belong to. */
  unattributed?: number;
}

/** Claude encodes a project directory as its absolute path with EVERY non-alphanumeric character
 *  replaced by `-` — separators, dots, underscores, spaces, `+`, `@`, all of them. Case is kept,
 *  and runs are not collapsed (`/a-1/-b` -> `-a-1--b`), so this is a 1:1 character substitution.
 *  Verified against Claude Code 2.1.241: `hs_under_test` -> `hs-under-test`, and
 *  `hs+odd@repo v2` -> `hs-odd-repo-v2`. */
export function claudeProjectDir(repoDir: string, home = homedir()): string {
  return join(home, ".claude", "projects", repoDir.replace(/[^a-zA-Z0-9]/g, "-"));
}

/** Whether a recorded session directory belongs to this repository. */
export function withinRepo(dir: string | undefined, repoDir: string): boolean {
  return (
    !!dir && (dir === repoDir || dir.startsWith(repoDir.endsWith("/") ? repoDir : repoDir + "/"))
  );
}

/** First `cwd` recorded in a Claude transcript, i.e. where that session was working. */
function claudeSessionCwd(file: string): string | undefined {
  try {
    for (const line of readFileSync(file, "utf8").split("\n", 400)) {
      if (!line.includes('"cwd"')) continue;
      const cwd = (JSON.parse(line) as { cwd?: string }).cwd;
      if (cwd) return cwd;
    }
  } catch {
    /* unreadable or truncated transcript */
  }
  return undefined;
}

function toSession(id: string, turns: TransportTurn[]): ChatSession | undefined {
  // `action` turns are tool-call breadcrumbs; the interchange format carries prose only.
  const prose = turns
    .filter((t) => t.role === "user" || t.role === "assistant")
    .map((t) => ({
      role: t.role,
      text: t.content,
      ...(t.timestamp ? { timestamp: t.timestamp } : {}),
    }));
  return prose.length ? { id, turns: prose } : undefined;
}

/**
 * First line of a file, read in chunks.
 *
 * Codex's `session_meta` header is a single line that carries the agent's full base instructions —
 * tens of KB. Reading a fixed prefix and splitting on newline truncated it mid-string, so every
 * rollout failed to parse and the import silently found nothing. Capped so a file with no newline
 * can't pull an unbounded amount into memory.
 */
function firstLine(path: string, cap = 1_000_000): string | undefined {
  const fd = openSync(path, "r");
  try {
    const chunk = Buffer.alloc(64 * 1024);
    let acc = "";
    while (acc.length < cap) {
      const n = readSync(fd, chunk, 0, chunk.length, null);
      if (n <= 0) break;
      acc += chunk.subarray(0, n).toString("utf8");
      const nl = acc.indexOf("\n");
      if (nl !== -1) return acc.slice(0, nl);
    }
    return acc.length && acc.length < cap ? acc : undefined;
  } finally {
    closeSync(fd);
  }
}

/**
 * The entries of a directory, or none when there is nothing to list.
 *
 * `existsSync` answered only half of that: it is TRUE for a regular file sitting where a directory
 * was expected — a stray `~/.claude/projects/-Users-x-dev-repo-sub` — and the `readdirSync` behind
 * it then threw ENOTDIR out of `importLocalHistory`, which promises never to throw. One junk entry
 * killed the whole `--import-conversations` run instead of costing that entry (#3771).
 */
function listDir(dir: string): string[] {
  try {
    return readdirSync(dir);
  } catch {
    return []; // missing, unreadable, or a stray file where a directory was expected
  }
}

function jsonlFiles(dir: string): string[] {
  return listDir(dir)
    .filter((f) => f.endsWith(".jsonl"))
    .map((f) => join(dir, f));
}

/**
 * Claude Code: one directory per LAUNCH directory, one .jsonl per session.
 *
 * Running Claude from a subdirectory creates its own project dir, so an exact match on the repo
 * root silently misses that history (on a real machine, 64 of 107 project dirs were nested under
 * another). Candidate dirs are prefiltered by encoded-name prefix — cheap, since the encoding is
 * order-preserving — and then confirmed against the `cwd` recorded INSIDE each session, because the
 * name alone is ambiguous: `/` and `.` both encode to `-`, so `repo-sub` may be the subdirectory
 * `repo/sub` or an unrelated sibling repo called `repo-sub`.
 */
function claudeHistory(repoDir: string, home: string): HistoryImport {
  const root = join(home, ".claude", "projects");
  const exact = claudeProjectDir(repoDir, home);
  const prefix = exact + "-";
  const dirs = listDir(root)
    .map((d) => join(root, d))
    .filter((d) => d === exact || d.startsWith(prefix));
  const sessions: ChatSession[] = [];
  let unattributed = 0;
  for (const file of dirs.flatMap(jsonlFiles)) {
    // ONLY the cwd recorded inside the session may attribute it to a repo. Falling back to the
    // directory name would be a guess: every non-alphanumeric encodes to `-`, so `repo-sub` is either the
    // subdirectory `repo/sub` or an unrelated sibling repo — and a wrong guess files someone
    // else's conversation into this repo's memory, which is worse than importing nothing.
    // (Measured: 400/400 sampled sessions record a cwd, so this skips ~nothing in practice.)
    const cwd = claudeSessionCwd(file);
    if (!cwd) {
      unattributed++;
      continue;
    }
    if (!withinRepo(cwd, repoDir)) continue;
    try {
      const id = file
        .split("/")
        .pop()!
        .replace(/\.jsonl$/, "");
      const s = toSession(id, readClaudeTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* a single unreadable transcript must not abort the import */
    }
  }
  return { supported: true, sessions, unattributed };
}

/**
 * Codex: rollouts are partitioned by DATE, not project, so the repo is read from the `session_meta`
 * header each file opens with — cheap enough to check without parsing the whole transcript.
 */
function codexHistory(repoDir: string, home: string): HistoryImport {
  const root = join(home, ".codex", "sessions");
  if (!existsSync(root)) return { supported: true, sessions: [] };
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry);
      if (statSync(p).isDirectory()) walk(p);
      else if (entry.endsWith(".jsonl")) files.push(p);
    }
  };
  try {
    walk(root);
  } catch {
    return { supported: true, sessions: [] };
  }

  const sessions: ChatSession[] = [];
  for (const file of files) {
    try {
      const head = firstLine(file);
      if (!head) continue;
      const meta = JSON.parse(head) as { payload?: { cwd?: string; id?: string } };
      if (!withinRepo(meta?.payload?.cwd, repoDir)) continue;
      const s = toSession(meta.payload?.id ?? file, readCodexTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* skip unreadable/short files */
    }
  }
  return { supported: true, sessions };
}

/**
 * Dcode: transcripts are named for their THREAD, and carry no cwd of their own — the working
 * directory lives only in the LangGraph checkpoint store (`.state/sessions.db`), whose schema is
 * exactly the internal, unversioned kind this module refuses to read.
 *
 * `dcode threads list --json` is the supported way to ask the same question. It is a declared
 * contract, not a schema leak: the payload is `{schema_version, command, data:[…]}` and each row
 * carries `thread_id` and `cwd`. Verified against deepagents-code 0.1.65.
 */
/** Read past sessions for one repository from either supported coding agent. */
export function importLocalHistory(
  harness: string,
  repoDir: string,
  home = homedir()
): HistoryImport {
  switch (harness) {
    case "claude-code":
      return claudeHistory(repoDir, home);
    case "codex":
      return codexHistory(repoDir, home);
    default:
      return { supported: false, reason: `Unsupported agent: ${harness}`, sessions: [] };
  }
}
