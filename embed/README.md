# Remember embedded daemon

This package manages the local Remember daemon and its pg0 database for the Codex and Claude Code integrations. Install from the project checkout using the [setup guide](../integrations/coding-agents/README.md).

The daemon runs on loopback and stays running until stopped. Memory is stored locally; hosted inference providers receive the content they process.

`remember-embed daemon start`, `stop`, `status`, and `logs` control the daemon. The coding-agent integration configures and starts it on demand. `REMEMBER_HOME` selects the local state directory.
