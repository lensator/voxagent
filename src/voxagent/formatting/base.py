"""Base formatting types for prompt generation.

This module defines the core formatting types used throughout the voxagent
prompt formatting system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class PromptFormat(str, Enum):
    """Enumeration of supported prompt formats for LLM providers.

    The format determines how prompts are structured when sent to different
    LLM providers. Using str as a mixin ensures values are string-compatible
    for serialization and configuration.

    Attributes:
        XML: XML-based formatting, preferred by Claude/Anthropic models.
        MARKDOWN: Markdown formatting, commonly used with OpenAI/GPT models.
        JSON: Structured JSON format for schema-based interactions.
        PLAIN: Plain text with no special formatting.
    """

    XML = "xml"
    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN = "plain"


class PromptFormatter(ABC):
    """Abstract base class for provider-specific prompt formatting.

    This class defines the interface that all prompt formatters must implement.
    Each formatter transforms various prompt components (system messages,
    thoughts, actions, observations) into a format appropriate for a specific
    LLM provider.

    Subclasses must implement all abstract methods to provide concrete
    formatting behavior for their target format type.
    """

    @property
    @abstractmethod
    def format_type(self) -> PromptFormat:
        """Return the format type this formatter uses.

        Returns:
            PromptFormat: The enum value indicating this formatter's format type.
        """
        ...

    @abstractmethod
    def format_system(self, content: str) -> str:
        """Format a system prompt.

        Args:
            content: The raw system prompt content.

        Returns:
            str: The formatted system prompt.
        """
        ...

    @abstractmethod
    def format_thought(self, content: str) -> str:
        """Format agent thinking/reasoning.

        Args:
            content: The raw thought/reasoning content.

        Returns:
            str: The formatted thought.
        """
        ...

    @abstractmethod
    def format_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format a tool call action.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments to pass to the tool.

        Returns:
            str: The formatted action.
        """
        ...

    @abstractmethod
    def format_observation(self, content: str) -> str:
        """Format a tool result/observation.

        Args:
            content: The raw observation/result content.

        Returns:
            str: The formatted observation.
        """
        ...
