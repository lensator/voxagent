"""Model Context Protocol (MCP) integration.

This subpackage provides:
- MCPToolDefinition: ToolDefinition subclass for MCP tools
- MCPServerManager: Manages MCP server lifecycle and tool extraction
- sanitize_tool_name: Utility to sanitize MCP tool names
"""

from voxagent.mcp.manager import MCPServerManager
from voxagent.mcp.tool import MCPToolDefinition, sanitize_tool_name

__all__ = [
    "MCPServerManager",
    "MCPToolDefinition",
    "sanitize_tool_name",
]
