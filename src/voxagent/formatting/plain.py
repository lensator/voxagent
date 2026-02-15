"""Plain text formatter with simple prefixes.

This formatter uses simple text prefixes without any special markup,
suitable for models or contexts that prefer minimal formatting.
"""

import json
from typing import Any

from voxagent.formatting.base import PromptFormat, PromptFormatter


class PlainFormatter(PromptFormatter):
    """Plain text formatter with simple prefixes.

    This formatter uses simple text prefixes:
    - System prompts: "System: content"
    - Thoughts: "Thinking: content"
    - Actions: "Action: tool_name(json_args)"
    - Observations: "Observation: content"
    """

    @property
    def format_type(self) -> PromptFormat:
        """Return the format type this formatter uses.

        Returns:
            PromptFormat: PromptFormat.PLAIN
        """
        return PromptFormat.PLAIN

    def format_system(self, content: str) -> str:
        """Format a system prompt with simple prefix.

        Args:
            content: The raw system prompt content.

        Returns:
            str: Content with "System: " prefix.
        """
        return f"System: {content}"

    def format_thought(self, content: str) -> str:
        """Format agent thinking/reasoning with simple prefix.

        Args:
            content: The raw thought/reasoning content.

        Returns:
            str: Content with "Thinking: " prefix.
        """
        return f"Thinking: {content}"

    def format_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format a tool call action as function call syntax.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments to pass to the tool.

        Returns:
            str: Function-call style format: "Action: tool_name(json_args)"
        """
        args_json = json.dumps(args)
        return f"Action: {tool_name}({args_json})"

    def format_observation(self, content: str) -> str:
        """Format a tool result/observation with simple prefix.

        Args:
            content: The raw observation/result content.

        Returns:
            str: Content with "Observation: " prefix.
        """
        return f"Observation: {content}"

