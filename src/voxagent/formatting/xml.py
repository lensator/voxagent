"""XML-based formatter optimized for Claude/Anthropic models.

This formatter wraps prompt components in XML tags, which is the preferred
format for Claude and other Anthropic models.
"""

import json
from typing import Any

from voxagent.formatting.base import PromptFormat, PromptFormatter


class XMLFormatter(PromptFormatter):
    """XML-based formatter optimized for Claude/Anthropic models.

    This formatter wraps content in XML tags:
    - System prompts: <system>content</system>
    - Thoughts: <thinking>content</thinking>
    - Actions: <action><tool>name</tool><arguments>json</arguments></action>
    - Observations: <observation>content</observation>
    """

    @property
    def format_type(self) -> PromptFormat:
        """Return the format type this formatter uses.

        Returns:
            PromptFormat: PromptFormat.XML
        """
        return PromptFormat.XML

    def format_system(self, content: str) -> str:
        """Format a system prompt with XML tags.

        Args:
            content: The raw system prompt content.

        Returns:
            str: Content wrapped in <system> tags.
        """
        return f"<system>{content}</system>"

    def format_thought(self, content: str) -> str:
        """Format agent thinking/reasoning with XML tags.

        Args:
            content: The raw thought/reasoning content.

        Returns:
            str: Content wrapped in <thinking> tags.
        """
        return f"<thinking>{content}</thinking>"

    def format_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format a tool call action with XML structure.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments to pass to the tool.

        Returns:
            str: Structured XML with tool name and JSON-serialized arguments.
        """
        args_json = json.dumps(args)
        return f"<action><tool>{tool_name}</tool><arguments>{args_json}</arguments></action>"

    def format_observation(self, content: str) -> str:
        """Format a tool result/observation with XML tags.

        Args:
            content: The raw observation/result content.

        Returns:
            str: Content wrapped in <observation> tags.
        """
        return f"<observation>{content}</observation>"

