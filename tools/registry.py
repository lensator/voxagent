"""Tool registry for voxagent."""

from __future__ import annotations

from voxagent.tools.definition import ToolDefinition


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool not found: {name}")


class ToolAlreadyRegisteredError(Exception):
    """Raised when trying to register a tool that already exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool already registered: {name}")


class ToolRegistry:
    """Registry for managing tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition, prefix: str | None = None) -> None:
        """Register a tool.

        Args:
            tool: The tool definition to register
            prefix: Optional namespace prefix (e.g., "mcp_server_name")

        Raises:
            ToolAlreadyRegisteredError: If tool with same name already registered
        """
        name = f"{prefix}_{tool.name}" if prefix else tool.name
        if name in self._tools:
            raise ToolAlreadyRegisteredError(name)
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Raises:
            ToolNotFoundError: If tool not found
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        del self._tools[name]

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name, returns None if not found."""
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> ToolDefinition:
        """Get a tool by name.

        Raises:
            ToolNotFoundError: If tool not found
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

