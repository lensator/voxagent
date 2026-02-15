"""Markdown-based formatter optimized for OpenAI/GPT models.

This formatter uses markdown headers and code blocks, which is a common
format for OpenAI GPT models.
"""

import json
from typing import Any

from voxagent.formatting.base import PromptFormat, PromptFormatter


class MarkdownFormatter(PromptFormatter):
    """Markdown-based formatter optimized for OpenAI/GPT models.

    This formatter uses markdown structure:
    - System prompts: ## System header
    - Thoughts: ## Thinking header
    - Actions: ## Action: name header with JSON code block
    - Observations: ## Observation header
    """

    @property
    def format_type(self) -> PromptFormat:
        """Return the format type this formatter uses.

        Returns:
            PromptFormat: PromptFormat.MARKDOWN
        """
        return PromptFormat.MARKDOWN

    def format_system(self, content: str) -> str:
        """Format a system prompt with markdown header.

        Args:
            content: The raw system prompt content.

        Returns:
            str: Content with ## System header.
        """
        return f"## System\n{content}"

    def format_thought(self, content: str) -> str:
        """Format agent thinking/reasoning with markdown header.

        Args:
            content: The raw thought/reasoning content.

        Returns:
            str: Content with ## Thinking header.
        """
        return f"## Thinking\n{content}"

    def format_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format a tool call action with markdown header and JSON code block.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments to pass to the tool.

        Returns:
            str: Header with tool name and JSON in fenced code block.
        """
        args_json = json.dumps(args)
        return f"## Action: {tool_name}\n```json\n{args_json}\n```"

    def format_observation(self, content: str) -> str:
        """Format a tool result/observation with markdown header.

        Args:
            content: The raw observation/result content.

        Returns:
            str: Content with ## Observation header.
        """
        return f"## Observation\n{content}"

