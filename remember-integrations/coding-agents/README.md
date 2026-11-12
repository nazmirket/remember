# Remember coding-agent memory

**Claude Code** and **Codex CLI** share a local, persistent memory service. The integration captures session transcripts, recalls relevant memories, seeds a project from git, and provides memory tools through MCP.

## Install from this checkout

Requires Node.js, npm, `uv`, Python and a configured inference provider. On macOS, a current Rust toolchain may be needed to build Python dependencies. The daemon and its pg0 database run locally; a hosted inference provider may receive memory content.

```sh
cd remember-integrations/coding-agents
npm ci
npm run build
node dist/installer.js install claude-code
node dist/installer.js install codex
```

The installer locates the embedded package in the checkout and records its absolute path. If installing from a different working directory, set `REMEMBER_EMBED_PACKAGE_PATH` to this checkout's `remember-embed` directory. The installer writes the selected agent's hooks and MCP configuration; inspect its output and restart the agent. Use `uninstall claude-code` or `uninstall codex` to remove the integration.

### Claude Code

The `claude-code` target installs session, prompt and stop hooks and the memory MCP tools.

### Codex CLI

The `codex` target installs hooks and the memory MCP tools. Codex hooks must be enabled in its configuration.

<!-- skill:begin -->

## Local operation

A session starts the daemon on demand. The service binds to `127.0.0.1` and stores its database locally in a profile-specific pg0 database. It uses the API source in this checkout and refuses to send memory requests to an unrelated service occupying the port. A daemon serves both agents and remains running between sessions.

Set `REMEMBER_API_LLM_PROVIDER` and the provider's required environment variables before launching the agent. For example, a local model provider can be configured through the API's provider settings. Hosted providers transmit content for processing; local storage does not make hosted inference local. If no provider is explicitly selected, the integration detects supported API keys or a Claude Code CLI on PATH.

The default bank identity combines the repository name with a stable hash of the canonical main checkout path. Codex, Claude Code and linked git worktrees share that bank; unrelated repositories with the same name do not. Explicit bank mappings can intentionally share or separate memories.

## Configuration

The installer writes `~/.remember/coding-agent.json`. Set `REMEMBER_HOME` to choose a different data directory, or `REMEMBER_CONFIG` to select a configuration file. Edit settings there and restart the agent for changes to take effect. Scalar fields can also use `REMEMBER_` plus the field name in uppercase snake case, for example `REMEMBER_API_PORT`. Maps such as `mapPathToBank`, `harnesses`, `banks` and `retainMetadata` are file-only.

Useful settings include `disabled`, `retainSessions`, `autoInject`, `gitIngest`, `pages`, `bankId`, `bankIdTemplate`, `dynamicBankId`, `mapPathToBank` and per-bank `banks`. A `banks.<bankId>` entry can rename a bank with `bank`; explicit mappings override the default isolation. Do not use remote `apiUrl` or `serverMode` settings: Remember uses the local daemon.

The supported configuration fields and their definitions are in `src/core/config.ts` (`RawConfig`):

`serverMode`, `apiUrl`, `apiToken`, `apiPort`, `daemonIdleTimeout`, `daemonProfile`, `embedPackagePath`, `bankId`, `dynamicBankId`, `bankIdTemplate`, `mapPathToBank`, `resolveWorktrees`, `optInOnly`, `optInPaths`, `harness`, `disabled`, `retainSessions`, `maxParallelRetains`, `reflectTimeoutMs`, `reflectToolTimeoutMs`, `injectTimeoutMs`, `reflectBudget`, `autoInject`, `autoReflect`, `pageSearchLimit`, `recallOptions`, `toolGuideExtra`, `pageRefreshEveryTurns`, `pageTriggerType`, `pageTriggerCron`, `pages`, `customPages`, `autoSeed`, `seedLimit`, `codebaseSurvey`, `surveyModel`, `surveyBudgetUsd`, `logLevel`, `surveyRefreshCommits`, `gitIngest`, `retainTags`, `retainMetadata`, `retainContext`, `manageBankConfig`, `retainExtractionMode`, `defaultBankConfig`, `observationScopes`, `harnesses`, `banks`, `paths`

## Troubleshooting

Check `coding-agents/plugin.log` under the selected Remember home, ensure `uv` and the selected provider are available to the agent process, and verify that the configured checkout still exists. If another service uses the daemon port, stop or reconfigure it before retrying. Memory is best effort: an unavailable daemon can leave a session without recall or retention.
<!-- skill:end -->
