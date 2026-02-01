"""Groq provider implementation.

This module implements the Groq provider for chat completions,
using Groq's OpenAI-compatible API format.
"""

from typing import Any

from voxagent.providers.openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq chat completions provider.

    Supports Llama, Mixtral, and Gemma models via Groq's
    OpenAI-compatible API with streaming and tool calling.
    """

    ENV_KEY = "GROQ_API_KEY"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        **kwargs: Any,
    ) -> None:
        """Initialize the Groq provider.

        Args:
            api_key: Groq API key. Falls back to GROQ_API_KEY env var.
            base_url: Custom base URL for API requests.
            model: Model name to use. Defaults to "llama-3.3-70b-versatile".
            **kwargs: Additional provider-specific arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "groq"

    @property
    def models(self) -> list[str]:
        """Get the list of supported model names."""
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

    @property
    def context_limit(self) -> int:
        """Get the maximum context length in tokens."""
        return 131072  # 128K

    def _get_api_url(self) -> str:
        """Get the API URL for chat completions."""
        base = self._base_url or self.DEFAULT_BASE_URL
        return f"{base}/chat/completions"

    def count_tokens(
        self,
        messages: list,
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Uses character-based estimation since tiktoken may not have
        encodings for Groq-specific models.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.

        Returns:
            The estimated token count.
        """
        # Fall back to character-based estimation (roughly 4 chars per token)
        total_chars = 0
        if system:
            total_chars += len(system)
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            else:
                for block in msg.content:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)
        return max(1, total_chars // 4)


__all__ = ["GroqProvider"]

