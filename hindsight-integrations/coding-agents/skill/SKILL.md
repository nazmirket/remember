---
name: hindsight-coding-agent
description: How this machine's Hindsight coding-agent memory works — the plugin behind the 🧠 banner. Use when the user says "store/remember this in hindsight", asks what the memory/pages are, wants to configure per-repo memory (disable, rename banks, git depth), or something memory-related looks broken.
---

<!-- GENERATED from README.md (its skill:begin regions) + skill-src/preamble.md.
     Edit those, then run: npm run skill:build -->

# Hindsight Coding-Agent Memory

This machine runs the `hindsight-coding-agents` plugin: long-term project memory for coding
sessions, backed by a Hindsight server. You (the agent) are already wired into it — this skill
explains what happens automatically, which tools you have, and how to configure or debug it.

## What happens automatically (no action needed)

- **Per-repo memory bank**: each repository resolves to a bank (shown in the session banner:
  `↳ memory bank “coding-agent::<repo>-<hash>”`). Worktrees share the main repo's bank.
- **Ingestion builds itself**: on first open, the bank is seeded from recent commit messages and a
  read-only codebase survey; every session start, a background engine tops it up (new commits, new
  conversations) and keeps 5 pages current. There is NO ingest command to run.
- **Session synthesis**: by default, the first prompt of a session triggers one deep memory
  synthesis (`reflect`) injected into context. `autoInject` switches the source: `pages` (page
  search hits), `recall` (recalled observations), or `none` (nothing injected; the agent
  searches the pages first and reflects only when they are too shallow).
- **Write-back**: the session transcript is retained into the bank automatically at session end
  . The user never needs to "save" a conversation.

## Storing things deliberately

When the user says "store this in hindsight" / "remember this":

- The **current conversation** is captured automatically at session end — say so; no tool needed.
- An **external document, notes, or durable findings** → `hindsight_ingest_document(title, content)`.
- A **new feature/initiative being started** → `hindsight_capture_initiative(title, summary)`,
  right after the plan is agreed and before code is written.
- A **plan that materially changed** (goal, scope, or rationale — including mid-implementation) →
  call `hindsight_capture_initiative` again with `relates_to_page_id` set to that initiative's page
  id, summarising the _current_ intent. Same page, updated plan — never a second page. Trivial
  course-corrections don't count.

## Retrieving

- `hindsight_search_pages(query)` — FIRST STOP for project questions (components,
  conventions, past decisions, initiatives). Server-side hybrid search, fast.
- `hindsight_read_page(page_id)` / `hindsight_list_pages` — read pages fully.
- `hindsight_reflect(query)` — deep reasoning over the whole memory for WHY questions and exact
  decided values; slower (seconds), use deliberately.
- Credit visibly whenever memory informs an answer: start that part with
  `🧠 From Hindsight memory (<page>): …` — and never credit memory that didn't contribute.

## Correcting wrong or stale memory

If you verify that something Hindsight served is wrong or outdated (the code, git, or an external
source contradicts it), FIX THE RECORD — don't just ignore it. Call
`hindsight_ingest_document` with:

- **title**: `Correction: <topic>` (e.g. `Correction: retry policy 4xx set`)
- **content**: (1) what memory claimed, (2) what is verifiably true now, (3) the evidence you
  checked (file/commit/output). Quote exact values verbatim.

Newer facts supersede older ones in retrieval, so one clear correction permanently outranks the
stale memory. Do this whenever you catch a wrong injected memory, a stale page claim, or
an outdated decision — silent disregard leaves the trap armed for the next session.

## Local operation

A session starts the daemon on demand. The service binds to `127.0.0.1` and stores its database locally in a profile-specific pg0 database. It uses the API source in this checkout and refuses to send memory requests to an unrelated service occupying the port. A daemon serves both agents and remains running between sessions.

Set `HINDSIGHT_API_LLM_PROVIDER` and the provider's required environment variables before launching the agent. For example, a local model provider can be configured through the API's provider settings. Hosted providers transmit content for processing; local storage does not make hosted inference local. If no provider is explicitly selected, the integration detects supported API keys or a Claude Code CLI on PATH.

The default bank identity combines the repository name with a stable hash of the canonical main checkout path. Codex, Claude Code and linked git worktrees share that bank; unrelated repositories with the same name do not. Explicit bank mappings can intentionally share or separate memories.

## Configuration

The installer writes `~/.hindsight/coding-agent.json`, or the path in `HINDSIGHT_CONFIG`. Edit settings there; restart the agent for changes to take effect. Scalar fields can also use `HINDSIGHT_` plus the field name in uppercase snake case, for example `HINDSIGHT_API_PORT`. Maps such as `mapPathToBank`, `harnesses`, `banks` and `retainMetadata` are file-only.

Useful settings include `disabled`, `retainSessions`, `autoInject`, `gitIngest`, `pages`, `bankId`, `bankIdTemplate`, `dynamicBankId`, `mapPathToBank` and per-bank `banks`. A `banks.<bankId>` entry can rename a bank with `bank`; explicit mappings override the default isolation. Do not use remote `apiUrl` or `serverMode` settings: this fork uses the local daemon.

The supported configuration fields and their definitions are in `src/core/config.ts` (`RawConfig`):

`serverMode`, `apiUrl`, `apiToken`, `apiPort`, `daemonIdleTimeout`, `daemonProfile`, `embedPackagePath`, `bankId`, `dynamicBankId`, `bankIdTemplate`, `mapPathToBank`, `resolveWorktrees`, `optInOnly`, `optInPaths`, `harness`, `disabled`, `retainSessions`, `maxParallelRetains`, `reflectTimeoutMs`, `reflectToolTimeoutMs`, `injectTimeoutMs`, `reflectBudget`, `autoInject`, `autoReflect`, `pageSearchLimit`, `recallOptions`, `toolGuideExtra`, `pageRefreshEveryTurns`, `pageTriggerType`, `pageTriggerCron`, `pages`, `customPages`, `autoSeed`, `seedLimit`, `codebaseSurvey`, `surveyModel`, `surveyBudgetUsd`, `logLevel`, `surveyRefreshCommits`, `gitIngest`, `retainTags`, `retainMetadata`, `retainContext`, `manageBankConfig`, `retainExtractionMode`, `defaultBankConfig`, `observationScopes`, `harnesses`, `banks`, `paths`

## Troubleshooting

Check `~/.hindsight/coding-agents/plugin.log`, ensure `uv` and the selected provider are available to the agent process, and verify that the configured checkout still exists. If another service uses the daemon port, stop or reconfigure it before retrying. Memory is best effort: an unavailable daemon can leave a session without recall or retention.
