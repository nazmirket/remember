---
name: code-review
description: Review local-memory changes against repository behavior, safety and test conventions.
user_invocable: true
---

# Code review

Review the actual changes and their connected callers, tests and configuration. Check:

- Both Claude Code and Codex retain, retrieve and inject the correct project bank; linked worktrees share it and separate projects do not.
- The on-demand daemon uses Remember's source, loopback and local pg0, does not persist inherited provider credentials, and handles a foreign or unavailable service safely.
- Worker processing, Pages, consolidation, graph and temporal retrieval, text document storage, inline attachments and migrations still work when affected.
- Database queries remain bank-scoped; preserve migration revision IDs and fresh/upgrade paths.
- No unsupported integrations, deployment paths, stale imports, generated artifacts or secrets are introduced.
- User-visible changes have focused deterministic tests; model-interpretation claims need real-model evaluation. Report when real-host or provider checks were not run.

Run `./scripts/hooks/lint.sh` and relevant tests. Report findings and verification honestly; self-review is not independent review.
