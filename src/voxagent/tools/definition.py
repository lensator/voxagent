"""Tool definition classes.

This module provides ToolDefinition for the agent tool system.
ToolContext is now defined in context.py for centralized management.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from voxagent.tools.context import ToolContext


class ToolDefinition:
    """Definition of a tool that can be called by an agent.

    Attributes:
        name: The tool name (alphanumeric and underscores only).
        description: Human-readable description for the LLM.
        parameters: JSON Schema for the tool's parameters.
        execute: The callable that implements the tool.
        is_async: Whether the execute function is async.
    """

    def __init__(
        self,
        name: str,
        description: str,
        execute: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
        is_async: bool = False,
    ) -> None:
        """Initialize ToolDefinition.

        Args:
            name: Tool name (alphanumeric and underscores only).
            description: Description for the LLM.
            execute: The function that implements the tool.
            parameters: JSON Schema for parameters. Defaults to empty dict.
            is_async: Whether execute is an async function.

        Raises:
            ValueError: If name contains invalid characters.
        """
        # Validate name
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Tool name '{name}' is invalid. "
                "Must contain only alphanumeric characters and underscores, "
                "and cannot start with a digit."
            )

        self.name = name
        self.description = description
        self.execute = execute
        self.parameters = parameters if parameters is not None else {}
        self.is_async = is_async

    async def run(self, params: dict[str, Any], context: ToolContext) -> Any:
        """Execute the tool with given parameters.

        Args:
            params: Dictionary of parameter values.
            context: The ToolContext for this execution.

        Returns:
            The result from the tool execution.
        """
        # Check if execute function accepts 'context' or 'ctx' parameter
        # Prefer 'ctx' over 'context' since 'context' might be a user parameter
        sig = inspect.signature(self.execute)
        accepts_ctx = "ctx" in sig.parameters
        accepts_context = "context" in sig.parameters

        call_params = dict(params)
        if accepts_ctx:
            call_params["ctx"] = context
        elif accepts_context:
            call_params["context"] = context

        if self.is_async:
            return await self.execute(**call_params)
        else:
            # Run sync function directly
            return self.execute(**call_params)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI's function calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters if self.parameters else {"type": "object", "properties": {}},
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool format.

        Returns:
            Dictionary in Anthropic's tool format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters if self.parameters else {"type": "object", "properties": {}},
        }

