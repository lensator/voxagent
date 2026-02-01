"""MCP Tool Definition for voxagent.

This module provides MCPToolDefinition, a ToolDefinition subclass that wraps
MCP server tools and forwards execution calls to the MCP server.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from voxagent.tools.definition import ToolContext, ToolDefinition

if TYPE_CHECKING:
    pass


class MCPToolDefinition(ToolDefinition):
    """A ToolDefinition that wraps an MCP server tool.

    This class forwards tool execution to the MCP server that provides the tool.
    It stores a reference to the MCP server and the original tool name.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        mcp_server: Any,
        original_tool_name: str,
        server_name: str,
    ) -> None:
        """Initialize MCPToolDefinition.

        Args:
            name: The sanitized tool name (alphanumeric and underscores).
            description: Tool description for the LLM.
            parameters: JSON Schema for the tool's parameters.
            mcp_server: The MCP server instance that provides this tool.
            original_tool_name: The original tool name from the MCP server.
            server_name: The name of the MCP server (for logging/debugging).
        """
        # Don't call parent __init__ with execute - we override run() instead
        # Validate name manually
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Tool name '{name}' is invalid. "
                "Must contain only alphanumeric characters and underscores, "
                "and cannot start with a digit."
            )

        self.name = name
        self.description = description
        self.parameters = parameters
        self.is_async = True  # MCP calls are always async

        # MCP-specific attributes
        self._mcp_server = mcp_server
        self._original_tool_name = original_tool_name
        self._server_name = server_name

        # Set execute to None - we override run() instead
        self.execute = self._execute_mcp_tool

    async def _execute_mcp_tool(self, **params: Any) -> Any:
        """Execute the MCP tool by calling the MCP server.

        This is called by the parent's run() method, but we override run()
        to handle MCP-specific execution.
        """
        # This shouldn't be called directly - run() handles execution
        raise NotImplementedError("Use run() instead")

    async def run(self, params: dict[str, Any], context: ToolContext) -> Any:
        """Execute the tool by forwarding to the MCP server.

        Args:
            params: Dictionary of parameter values.
            context: The ToolContext for this execution.

        Returns:
            The result from the MCP server tool execution.
        """
        try:
            # Call the MCP server's direct_call_tool method
            # This bypasses PydanticAI's context requirements
            result = await self._mcp_server.direct_call_tool(
                self._original_tool_name,
                params,
            )

            # MCP returns a list of content items, extract text
            if hasattr(result, "content") and result.content:
                # Concatenate all text content
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                return "\n".join(texts) if texts else str(result)

            return str(result)

        except Exception as e:
            # Return error as string - the executor will handle it
            return f"MCP tool error ({self._server_name}/{self._original_tool_name}): {e}"

    @property
    def mcp_server(self) -> Any:
        """Get the MCP server instance."""
        return self._mcp_server

    @property
    def original_tool_name(self) -> str:
        """Get the original MCP tool name."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Get the MCP server name."""
        return self._server_name


def sanitize_tool_name(name: str) -> str:
    """Sanitize a tool name to be valid for voxagent.

    MCP tool names may contain dashes or other characters that are not
    valid in Python identifiers. This function converts them to underscores.

    Args:
        name: The original tool name.

    Returns:
        A sanitized tool name with only alphanumeric characters and underscores.
    """
    # Replace dashes and other invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized

    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_tool"

    return sanitized

