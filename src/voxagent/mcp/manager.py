"""MCP Server Manager for voxagent.

This module provides MCPServerManager, which handles:
- Connecting to MCP servers
- Extracting tools from MCP servers
- Converting MCP tools to ToolDefinition format
- Managing server lifecycle (connect/disconnect)
"""

from __future__ import annotations

import logging
from typing import Any

from voxagent.mcp.tool import MCPToolDefinition, sanitize_tool_name
from voxagent.tools.definition import ToolDefinition

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages MCP server connections and tool extraction.

    This class handles the lifecycle of MCP servers and provides methods
    to extract and convert their tools to voxagent's ToolDefinition format.
    """

    def __init__(self) -> None:
        """Initialize the MCPServerManager."""
        self._servers: list[Any] = []
        self._connected_servers: list[Any] = []
        self._tools: list[ToolDefinition] = []

    async def add_servers(self, servers: list[Any]) -> None:
        """Add MCP servers to manage.

        Args:
            servers: List of MCP server instances (PydanticAI MCP classes).
        """
        self._servers.extend(servers)

    async def connect_all(self) -> list[ToolDefinition]:
        """Connect to all MCP servers and extract their tools.

        Returns:
            List of ToolDefinition objects from all connected servers.
        """
        self._tools = []
        self._connected_servers = []

        for server in self._servers:
            try:
                # Enter the async context manager
                await server.__aenter__()
                self._connected_servers.append(server)

                # Extract tools from this server
                tools = await self._extract_tools(server)
                self._tools.extend(tools)

                server_name = getattr(server, "name", None) or getattr(
                    server, "tool_prefix", "unknown"
                )
                logger.info(
                    f"Connected to MCP server '{server_name}' with {len(tools)} tools"
                )

            except Exception as e:
                server_name = getattr(server, "name", None) or getattr(
                    server, "tool_prefix", "unknown"
                )
                logger.warning(f"Failed to connect to MCP server '{server_name}': {e}")
                # Continue with other servers

        return self._tools

    async def disconnect_all(self) -> None:
        """Disconnect from all connected MCP servers.

        Note: PydanticAI MCP servers use anyio cancel scopes, which must be
        entered and exited from the same task. If called from a different task,
        the disconnect will be skipped to avoid RuntimeError.
        """
        for server in self._connected_servers:
            try:
                await server.__aexit__(None, None, None)
            except RuntimeError as e:
                # anyio cancel scope errors when called from different task
                if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                    server_name = getattr(server, "name", None) or getattr(
                        server, "tool_prefix", "unknown"
                    )
                    logger.debug(
                        f"MCP server '{server_name}' disconnect skipped (task mismatch): {e}"
                    )
                else:
                    raise
            except Exception as e:
                server_name = getattr(server, "name", None) or getattr(
                    server, "tool_prefix", "unknown"
                )
                logger.warning(f"Error disconnecting from MCP server '{server_name}': {e}")

        self._connected_servers = []
        self._tools = []

    async def _extract_tools(self, server: Any) -> list[ToolDefinition]:
        """Extract tools from an MCP server and convert to ToolDefinition.

        Args:
            server: The connected MCP server instance.

        Returns:
            List of ToolDefinition objects.
        """
        tools: list[ToolDefinition] = []

        try:
            # Get the server name/prefix for tool naming
            server_name = getattr(server, "name", None) or getattr(
                server, "tool_prefix", None
            )

            # List tools from the MCP server
            mcp_tools = await server.list_tools()

            for mcp_tool in mcp_tools:
                try:
                    tool_def = self._convert_mcp_tool(mcp_tool, server, server_name)
                    tools.append(tool_def)
                except Exception as e:
                    tool_name = getattr(mcp_tool, "name", "unknown")
                    logger.warning(f"Failed to convert MCP tool '{tool_name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to list tools from MCP server: {e}")

        return tools

    def _convert_mcp_tool(
        self, mcp_tool: Any, server: Any, server_name: str | None
    ) -> MCPToolDefinition:
        """Convert an MCP tool to MCPToolDefinition.

        Args:
            mcp_tool: The MCP tool object from list_tools().
            server: The MCP server instance.
            server_name: The server name/prefix.

        Returns:
            MCPToolDefinition instance.
        """
        # Get tool attributes
        original_name = mcp_tool.name
        description = mcp_tool.description or f"MCP tool: {original_name}"

        # Build the full tool name with prefix
        if server_name:
            full_name = f"{server_name}_{sanitize_tool_name(original_name)}"
        else:
            full_name = sanitize_tool_name(original_name)

        # Get input schema (parameters)
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            parameters = mcp_tool.inputSchema
        elif hasattr(mcp_tool, "input_schema") and mcp_tool.input_schema:
            parameters = mcp_tool.input_schema

        return MCPToolDefinition(
            name=full_name,
            description=description,
            parameters=parameters,
            mcp_server=server,
            original_tool_name=original_name,
            server_name=server_name or "unknown",
        )

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get all extracted tools."""
        return self._tools

    @property
    def connected_server_count(self) -> int:
        """Get the number of connected servers."""
        return len(self._connected_servers)

