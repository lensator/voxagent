"""Sub-agent definition for voxagent.

This module provides SubAgentDefinition, a ToolDefinition subclass that wraps
an Agent and makes it callable as a tool by a parent agent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from voxagent.subagent.context import (
    DEFAULT_MAX_DEPTH,
    MaxDepthExceededError,
    SubAgentContext,
)
from voxagent.tools.context import ToolContext
from voxagent.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from voxagent.agent.core import Agent


class SubAgentDefinition(ToolDefinition):
    """A ToolDefinition that wraps an Agent as a callable tool.

    This allows parent agents to delegate tasks to specialized child agents.
    The child agent inherits context (abort signal, deps, session) from parent.

    Example:
        researcher = Agent(model="...", system_prompt="You research topics...")
        
        parent = Agent(
            model="...",
            sub_agents=[researcher],  # Auto-converted to SubAgentDefinition
        )
    """

    def __init__(
        self,
        agent: "Agent[Any, Any]",
        name: str | None = None,
        description: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """Initialize SubAgentDefinition.

        Args:
            agent: The Agent instance to wrap as a tool.
            name: Tool name (defaults to agent name or 'sub_agent').
            description: Tool description (defaults to agent's system prompt).
            max_depth: Maximum nesting depth for recursive calls.
        """
        # Determine tool name
        tool_name = name or getattr(agent, "name", None) or "sub_agent"
        tool_name = self._sanitize_name(tool_name)

        # Determine description
        tool_desc = description or agent._system_prompt or f"Delegate to {tool_name}"
        # Truncate long descriptions for tool schema
        if len(tool_desc) > 500:
            tool_desc = tool_desc[:497] + "..."

        # Validate name manually (don't call parent __init__ with execute)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            raise ValueError(
                f"Tool name '{tool_name}' is invalid. "
                "Must contain only alphanumeric characters and underscores."
            )

        self.name = tool_name
        self.description = tool_desc
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or query to delegate to this agent.",
                }
            },
            "required": ["task"],
        }
        self.is_async = True

        # Sub-agent specific attributes
        self._agent = agent
        self._max_depth = max_depth

        # Set execute placeholder
        self.execute = self._execute_placeholder

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize agent name to be a valid tool name."""
        # Replace spaces and invalid chars with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Ensure doesn't start with digit
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized or "sub_agent"

    async def _execute_placeholder(self, **params: Any) -> Any:
        """Placeholder - actual execution happens in run()."""
        raise NotImplementedError("Use run() instead")

    async def run(self, params: dict[str, Any], context: ToolContext) -> Any:
        """Execute the sub-agent with the given task.

        Args:
            params: Must contain 'task' key with the prompt for the sub-agent.
            context: The ToolContext from the parent agent.

        Returns:
            The sub-agent's response text.

        Raises:
            MaxDepthExceededError: If nesting depth exceeds max_depth.
        """
        task = params.get("task", "")
        if not task:
            return "Error: No task provided for sub-agent"

        # Convert or get SubAgentContext with depth tracking
        if isinstance(context, SubAgentContext):
            sub_context = context
        else:
            sub_context = SubAgentContext.from_tool_context(
                context, depth=0, max_depth=self._max_depth
            )

        # Check depth before spawning child
        try:
            child_context = sub_context.child_context(
                run_id=None,  # Agent.run() will create new run_id
            )
        except MaxDepthExceededError as e:
            return f"Error: {e}"

        # Run the sub-agent
        try:
            print(f"\n🤖 SUB-AGENT CALLED: {self.name}")
            print(f"   Task: {task[:100]}{'...' if len(task) > 100 else ''}")
            print(f"   Depth: {child_context.depth}/{child_context.max_depth}")

            result = await self._agent.run(
                prompt=task,
                deps=child_context.deps,
                session_key=child_context.session_id,
                # Pass depth info via message_history metadata (optional)
            )

            if result.error:
                return f"Sub-agent error: {result.error}"

            # Return the agent's response
            if result.assistant_texts:
                return result.assistant_texts[-1]  # Last response
            return "Sub-agent completed with no response"

        except Exception as e:
            return f"Sub-agent execution error: {type(e).__name__}: {e}"

    @property
    def agent(self) -> "Agent[Any, Any]":
        """Get the wrapped Agent instance."""
        return self._agent

    @property
    def max_depth(self) -> int:
        """Get the maximum nesting depth."""
        return self._max_depth

