"""Tool system for agent tool definitions and execution.

This subpackage provides:
- Tool definitions with typed parameters
- Tool registry for dynamic registration
- Tool policies for allow/deny filtering
- Tool execution with abort signal support
"""

from voxagent.tools.context import AbortError, ToolContext
from voxagent.tools.decorator import tool
from voxagent.tools.definition import ToolDefinition
from voxagent.tools.executor import execute_tool
from voxagent.tools.policy import ToolPolicy, apply_tool_policies
from voxagent.tools.registry import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
)

__all__ = [
    "AbortError",
    "ToolAlreadyRegisteredError",
    "ToolContext",
    "ToolDefinition",
    "ToolNotFoundError",
    "ToolPolicy",
    "ToolRegistry",
    "apply_tool_policies",
    "execute_tool",
    "tool",
]
