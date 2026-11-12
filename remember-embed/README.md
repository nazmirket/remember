# remember-embed

Remember embedded CLI - local memory operations with automatic daemon management.

This package provides a simple CLI for storing and recalling memories using Remember's memory engine. It automatically manages a background daemon for fast operations - no manual server setup required.

## How It Works

`remember-embed` uses a background daemon architecture for optimal performance:

1. **First command**: Automatically starts a local daemon (first run downloads dependencies and loads ML models - can take 1-3 minutes)
2. **Subsequent commands**: Near-instant responses (~1-2s) since daemon is already running
3. **Auto-shutdown**: Daemon automatically exits after 5 minutes of inactivity

The daemon runs on `localhost:8888` and uses an embedded PostgreSQL database (pg0) - everything stays local on your machine.

## Installation

Install and run the embedded manager from this checkout. See the [project setup guide](../README.md) for the supported local workflow.

## Quick Start

```bash
# Interactive setup (configures default profile)
remember-embed configure

# Or set your LLM API key manually
export OPENAI_API_KEY=sk-...

# Store a memory (bank_id = "default")
remember-embed memory retain default "User prefers dark mode"

# Recall memories
remember-embed memory recall default "What are user preferences?"
```

All commands use the "default" profile unless you specify a different one with `--profile` or `REMEMBER_EMBED_PROFILE`.

## Commands

### configure

Configure the default profile or create/update named profiles:

```bash
# Interactive setup for default profile
remember-embed configure

# Create/update named profile with single command
remember-embed configure --profile my-app \
  --env REMEMBER_API_LLM_PROVIDER=openai \
  --env REMEMBER_API_LLM_API_KEY=sk-xxx

# Create/update named profile interactively
remember-embed configure --profile staging
```

This will:
- Let you choose an LLM provider (OpenAI, Groq, Google, Ollama, GitHub Copilot)
- Configure credentials when the provider requires them
- Set the model
- Start the daemon with your configuration

### memory retain

Store a memory:

```bash
remember-embed memory retain default "User prefers dark mode"
remember-embed memory retain default "Meeting on Monday" --context work
remember-embed memory retain myproject "API uses JWT authentication"
```

### memory recall

Search memories:

```bash
remember-embed memory recall default "user preferences"
remember-embed memory recall default "upcoming events"
```

Use `-o json` for JSON output:
```bash
remember-embed memory recall default "user preferences" -o json
```

### memory reflect

Get contextual answers that synthesize multiple memories:

```bash
remember-embed memory reflect default "How should I set up the dev environment?"
```

### bank list

List all memory banks:

```bash
remember-embed bank list
```

### profile

Manage configuration profiles:

```bash
# List all profiles with status
remember-embed profile list

# Show current active profile
remember-embed profile show

# Set active profile (persists across commands)
remember-embed profile set-active my-app

# Clear active profile (revert to default)
remember-embed profile set-active --none

# Delete a profile
remember-embed profile delete my-app
```

### daemon

Manage the background daemon:

```bash
remember-embed daemon status    # Check if daemon is running
remember-embed daemon start     # Start the daemon
remember-embed daemon stop      # Stop the daemon
remember-embed daemon logs      # View last 50 lines of logs
remember-embed daemon logs -f   # Follow logs in real-time
remember-embed daemon logs -n 100  # View last 100 lines
```

## Configuration

### Interactive Setup

Run `remember-embed configure` for a guided setup that saves to `~/.remember/embed`.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REMEMBER_EMBED_PROFILE` | Profile name to use (overrides active profile) | None (uses default profile) |
| `REMEMBER_API_LLM_API_KEY` | LLM API key (or use `OPENAI_API_KEY`); required only when the selected provider uses an API key | Provider-dependent |
| `REMEMBER_API_LLM_PROVIDER` | LLM provider (OpenAI, Anthropic, Codex, Claude Code, or local) | `openai` |
| `REMEMBER_API_LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `REMEMBER_EMBED_API_URL` | Use external API server instead of starting local daemon | None (starts local daemon) |
| `REMEMBER_EMBED_API_TOKEN` | Authentication token for external API (sent as Bearer token) | None |
| `REMEMBER_EMBED_API_DATABASE_URL` | Database URL for daemon (`REMEMBER_API_DATABASE_URL` is used when this is unset) | `pg0://remember-embed-<profile>` |
| `REMEMBER_EMBED_DAEMON_LOG_MAX_BYTES` | Rotate the daemon log at startup at this size; `0` disables rotation | `10485760` (10 MiB) |
| `REMEMBER_EMBED_DAEMON_LOG_BACKUP_COUNT` | Retained backups; `0` truncates a full log at startup | `3` |

The size is checked only when a daemon starts, so a single uninterrupted run is never truncated and can
grow past `MAX_BYTES` — and at the next start that whole file is kept as the first backup. Retained size
is therefore around `MAX_BYTES × (BACKUP_COUNT + 1)` (40 MiB by default) only for daemons that restart
regularly; a daemon left running for weeks keeps whatever it wrote. Restart it, or lower
`REMEMBER_EMBED_DAEMON_IDLE_TIMEOUT`, to keep the bound meaningful.

**Using an External API Server:**

To connect to an existing Remember API server instead of starting the local daemon:

```bash
export REMEMBER_EMBED_API_URL=http://your-server:8000
export REMEMBER_EMBED_API_TOKEN=your-api-token  # Optional, if API requires auth
remember-embed memory recall default "query"
```

**Custom Database:**

To use an external PostgreSQL database instead of the embedded pg0 database (useful when running as root or in containerized environments):

```bash
export REMEMBER_EMBED_API_DATABASE_URL=postgresql://user:password@localhost:5432/dbname
remember-embed daemon start
```

**Note:** All banks share a single database. Bank isolation happens within the database via the `bank_id` parameter passed to CLI commands.

### Configuration Profiles

Profiles let you maintain multiple independent configurations (e.g., different API endpoints, LLM providers, or projects). Each profile runs its own daemon on a unique port (8889-9888).

**The Default Profile:**

When you run `remember-embed configure` without specifying a profile, it configures the "default" profile. This uses the configuration at `~/.remember/embed` and runs on port 8888.

**Creating Named Profiles:**

```bash
# Create a profile with single command
remember-embed configure --profile my-app \
  --env REMEMBER_API_LLM_PROVIDER=openai \
  --env REMEMBER_API_LLM_API_KEY=sk-xxx \
  --env REMEMBER_API_LLM_MODEL=gpt-4o-mini

# Create a profile interactively
remember-embed configure --profile staging
```

**Using Profiles:**

```bash
# Option 1: Environment variable (recommended for apps)
REMEMBER_EMBED_PROFILE=my-app remember-embed memory retain default "text"

# Option 2: CLI flag
remember-embed --profile my-app memory recall default "query"

# Option 3: Set as active (persists across commands)
remember-embed profile set-active my-app
remember-embed memory recall default "query"  # Uses my-app profile

# Clear active profile (revert to default)
remember-embed profile set-active --none
```

**Profile Management:**

```bash
# List all profiles with status
remember-embed profile list

# Show active profile
remember-embed profile show

# Delete a profile
remember-embed profile delete my-app
```

**Profile Resolution Priority:**
1. `REMEMBER_EMBED_PROFILE` environment variable (highest)
2. `--profile` CLI flag
3. Active profile from `~/.remember/active_profile` file
4. Default profile (lowest)

**Note:** If a profile is specified but doesn't exist, the command will fail with an error. Profiles must be explicitly created using `remember-embed configure --profile <name>`.

### Files

**Default Profile:**
| Path | Description |
|------|-------------|
| `~/.remember/embed` | Configuration file for default profile |
| `~/.remember/daemon.log` | Daemon logs for default profile |
| `~/.remember/daemon.lock` | Daemon lock file (PID) for default profile |

**Named Profiles:**
| Path | Description |
|------|-------------|
| `~/.remember/profiles/<name>.env` | Configuration file for profile |
| `~/.remember/profiles/<name>.log` | Daemon logs for profile |
| `~/.remember/profiles/<name>.lock` | Daemon lock file (PID) for profile |
| `~/.remember/profiles/metadata.json` | Profile metadata (ports, timestamps) |
| `~/.remember/active_profile` | Active profile name (when set with `profile set-active`) |

## Use with AI Coding Assistants

For Codex and Claude Code setup, use the [coding-agent integration guide](../remember-integrations/coding-agents/README.md).

## Troubleshooting

**Daemon won't start:**
```bash
# Check logs for errors
remember-embed daemon logs

# Stop any stuck daemon and restart
remember-embed daemon stop
remember-embed daemon start
```

**Slow first command:**
This is expected - the first command needs to download dependencies, start the daemon, and load ML models. First run can take 1-3 minutes depending on network speed. Subsequent commands will be fast (~1-2s).

**Change configuration:**
```bash
# Re-run configure (automatically restarts daemon)
remember-embed configure
```

## License

Apache 2.0
