"""Base provider abstract class and stream chunk types.

This module defines the abstract interface for LLM providers:
- StreamChunk types for streaming responses
- AbortSignal for cancellation
- BaseProvider ABC that all providers must implement
"""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from voxagent.types import Message, ToolCall


# =============================================================================
# Stream Chunk Types
# =============================================================================


class TextDeltaChunk(BaseModel):
    """A text delta chunk from streaming response.

    Attributes:
        type: Discriminator field, always "text_delta".
        delta: The text content delta.
    """

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ToolUseChunk(BaseModel):
    """A tool use chunk from streaming response.

    Attributes:
        type: Discriminator field, always "tool_use".
        tool_call: The tool call request.
    """

    type: Literal["tool_use"] = "tool_use"
    tool_call: ToolCall


class MessageEndChunk(BaseModel):
    """A message end chunk signaling stream completion.

    Attributes:
        type: Discriminator field, always "message_end".
    """

    type: Literal["message_end"] = "message_end"


class ErrorChunk(BaseModel):
    """An error chunk from streaming response.

    Attributes:
        type: Discriminator field, always "error".
        error: The error message.
    """

    type: Literal["error"] = "error"
    error: str


# Union type for all stream chunks
StreamChunk = TextDeltaChunk | ToolUseChunk | MessageEndChunk | ErrorChunk


# =============================================================================
# Abort Signal
# =============================================================================


class AbortSignal:
    """A signal for aborting async operations.

    Attributes:
        _aborted: Internal flag indicating if abort has been requested.
        _reason: The reason for the abort.
    """

    def __init__(self) -> None:
        """Initialize the abort signal."""
        self._aborted = False
        self._reason: str = ""

    @property
    def aborted(self) -> bool:
        """Check if abort has been requested."""
        return self._aborted

    @property
    def reason(self) -> str:
        """Get the reason for the abort."""
        return self._reason

    def abort(self, reason: str = "Aborted") -> None:
        """Request abortion of the operation.

        Args:
            reason: The reason for aborting.
        """
        self._aborted = True
        self._reason = reason


# =============================================================================
# Base Provider Abstract Class
# =============================================================================


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement all abstract properties and methods.
    The ENV_KEY class variable should be set to the environment variable
    name for the API key.
    """

    ENV_KEY: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for authentication. Falls back to ENV_KEY env var.
            base_url: Optional base URL for API requests.
            **kwargs: Additional provider-specific arguments.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._kwargs = kwargs

    @property
    def api_key(self) -> str | None:
        """Get API key from constructor or environment variable."""
        if self._api_key is not None:
            return self._api_key
        return os.environ.get(self.ENV_KEY) if self.ENV_KEY else None

    @property
    def base_url(self) -> str | None:
        """Get the base URL for API requests."""
        return self._base_url

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def models(self) -> list[str]:
        """Get the list of supported model names."""
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling."""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming responses."""
        ...

    @property
    @abstractmethod
    def context_limit(self) -> int:
        """Get the maximum context length in tokens."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the provider.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            abort_signal: Optional signal to abort the stream.

        Yields:
            StreamChunk objects containing response data.
        """
        ...
        # This yield is needed to make this an async generator
        yield  # type: ignore[misc]

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from the provider.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The assistant's response message.
        """
        ...

    @abstractmethod
    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.

        Returns:
            The token count.
        """
        ...

    def get_api_key(self, env_var_name: str) -> str | None:
        """Get API key from constructor or specified environment variable.

        Args:
            env_var_name: The environment variable name to check.

        Returns:
            The API key or None if not found.
        """
        if self._api_key is not None:
            return self._api_key
        return os.environ.get(env_var_name)


__all__ = [
    "AbortSignal",
    "BaseProvider",
    "ErrorChunk",
    "MessageEndChunk",
    "StreamChunk",
    "TextDeltaChunk",
    "ToolUseChunk",
]

