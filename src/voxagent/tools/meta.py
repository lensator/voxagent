"""Meta-tools for controlling the agent itself.

Provides tools for switching models and inspecting agent state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voxagent.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from voxagent.agent.core import Agent


def create_switch_model_tool(agent: Agent) -> ToolDefinition:
    """Create a tool that allows the agent to switch its own model.

    Args:
        agent: The agent instance to control.

    Returns:
        A ToolDefinition for the switch_model tool.
    """
    
    async def switch_model(model_name: str) -> str:
        """Switch the current agent's model to a new provider:model.
        
        Use this when the current model is struggling with a task and a
        more capable model is needed, or when a faster/cheaper model
        is sufficient for the remaining work.
        
        Args:
            model_name: The model string (e.g., 'anthropic:claude-3-5-sonnet', 'openai:gpt-4o').
            
        Returns:
            Confirmation message.
        """
        old_model = agent.model_string
        agent.set_model(model_name)
        return f"Successfully switched model from {old_model} to {model_name}. Next turn will use the new model."

    return ToolDefinition(
        name="switch_model",
        description="Switch the agent's model to a different provider:model.",
        execute=switch_model,
        parameters={
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The new model string (e.g., 'openai:gpt-4o')"
                }
            },
            "required": ["model_name"]
        },
        is_async=True,
    )
