"""Provider to formatter registry.

Maps LLM provider names to their optimal prompt formatters.
"""

from typing import Type

from voxagent.formatting.base import PromptFormatter
from voxagent.formatting.markdown import MarkdownFormatter
from voxagent.formatting.plain import PlainFormatter
from voxagent.formatting.xml import XMLFormatter

# Default provider → formatter mappings
# All keys should be lowercase for case-insensitive lookup
PROVIDER_FORMATTERS: dict[str, Type[PromptFormatter]] = {
    # XML formatters (Claude/Anthropic perform best with XML tags)
    "anthropic": XMLFormatter,
    "claude": XMLFormatter,
    # Markdown formatters (OpenAI/GPT perform best with markdown)
    "openai": MarkdownFormatter,
    "gpt": MarkdownFormatter,
    "google": MarkdownFormatter,
    "groq": MarkdownFormatter,
    # Plain formatters (local models, simpler)
    "ollama": PlainFormatter,
}


def get_formatter_for_provider(provider_name: str) -> PromptFormatter:
    """Get the appropriate formatter for a provider.

    Args:
        provider_name: The provider name (case-insensitive).

    Returns:
        An instance of the appropriate PromptFormatter.
        Falls back to PlainFormatter for unknown providers.
    """
    normalized = provider_name.lower()
    formatter_class = PROVIDER_FORMATTERS.get(normalized, PlainFormatter)
    return formatter_class()


def register_provider_formatter(
    provider_name: str,
    formatter_class: Type[PromptFormatter],
) -> None:
    """Register a custom provider→formatter mapping.

    Args:
        provider_name: The provider name (will be normalized to lowercase).
        formatter_class: The formatter class to use for this provider.
    """
    PROVIDER_FORMATTERS[provider_name.lower()] = formatter_class

