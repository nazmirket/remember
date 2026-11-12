# Remember integration tests

The MCP end-to-end tests require a running local Remember API server with its
MCP endpoints enabled. Start the local Remember server, then run:

```bash
cd remember-integration-tests
REMEMBER_API_URL=http://localhost:8888 uv run pytest tests/test_mcp_e2e.py -v
```

`REMEMBER_API_URL` defaults to `http://localhost:8888`. These tests exercise
the general API MCP endpoint and bank-specific MCP routing.
