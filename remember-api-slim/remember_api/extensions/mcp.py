"""MCP Extension for registering additional MCP tools.

This extension allows external packages (like remember-cloud) to register
additional MCP tools on the Remember MCP server.

Example:
    REMEMBER_API_MCP_EXTENSION=remember_cloud.extensions:CloudMCPExtension
"""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from remember_api.extensions.base import Extension

if TYPE_CHECKING:
    # Type-only: fastmcp loads when the MCP server starts, so a broken MCP
    # install can't stop a server that doesn't use it (#5453).
    from fastmcp import FastMCP

    from remember_api import MemoryEngine

logger = logging.getLogger(__name__)


class MCPExtension(Extension):
    """Base class for MCP extensions that register additional tools.

    Subclass this to add MCP tools in extension packages.

    Example:
        class CloudMCPExtension(MCPExtension):
            def register_tools(self, mcp: "FastMCP", memory: "MemoryEngine") -> None:
                # output_schema=None: a str result is sent as text only, not
                # duplicated into structuredContent (#6279).
                @mcp.tool(output_schema=None)
                async def my_custom_tool(query: str) -> str:
                    return "result"
    """

    @abstractmethod
    def register_tools(self, mcp: "FastMCP", memory: "MemoryEngine") -> None:
        """Register additional MCP tools.

        Args:
            mcp: FastMCP server instance to register tools on
            memory: MemoryEngine instance for accessing memory operations
        """
        pass
