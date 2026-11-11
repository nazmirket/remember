# Hindsight integration tests

The MCP end-to-end tests require a running local Hindsight API server with its
MCP endpoints enabled. Start the server using the local fork, then run:

```bash
cd hindsight-integration-tests
HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests/test_mcp_e2e.py -v
```

`HINDSIGHT_API_URL` defaults to `http://localhost:8888`. These tests exercise
the general API MCP endpoint and bank-specific MCP routing.
