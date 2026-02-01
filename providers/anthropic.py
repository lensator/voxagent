"""Anthropic provider implementation.

This module implements the AnthropicProvider for Claude models,
supporting streaming, tool use, and extended thinking mode.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from voxagent.providers.base import (
    AbortSignal,
    BaseProvider,
    ErrorChunk,
    MessageEndChunk,
    StreamChunk,
    TextDeltaChunk,
    ToolUseChunk,
)
from voxagent.types import Message, ToolCall


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, and other models.
    Implements streaming, tool use, and extended thinking mode.
    """

    ENV_KEY = "ANTHROPIC_API_KEY"
    DEFAULT_BASE_URL = "https://api.anthropic.com"

    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: API key for authentication. Falls back to ANTHROPIC_API_KEY env var.
            base_url: Optional custom base URL for API requests.
            model: Model to use (default: claude-3-5-sonnet-20241022).
            thinking: Enable extended thinking mode.
            **kwargs: Additional provider-specific arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model
        self._thinking = thinking
        # State for SSE parsing
        self._reset_tool_state()

    def _reset_tool_state(self) -> None:
        """Reset tool use state for streaming."""
        self._current_tool_id: str | None = None
        self._current_tool_name: str | None = None
        self._current_tool_input_json: str = ""
        self._current_block_type: str | None = None

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Get the current model."""
        return self._model

    @property
    def models(self) -> list[str]:
        """Get the list of supported model names."""
        return self.SUPPORTED_MODELS.copy()

    @property
    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming responses."""
        return True

    @property
    def context_limit(self) -> int:
        """Get the maximum context length in tokens."""
        return 200000

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for messages."""
        base = self._base_url or self.DEFAULT_BASE_URL
        return f"{base}/v1/messages"

    def _convert_messages_to_anthropic(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert voxagent Messages to Anthropic format.

        Args:
            messages: List of voxagent Messages.

        Returns:
            List of Anthropic-formatted message dicts.
        """
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # System messages are handled separately in Anthropic
                continue

            anthropic_msg: dict[str, Any] = {"role": msg.role}

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.params,
                    })
                anthropic_msg["content"] = content_blocks
            else:
                # Regular text content
                if isinstance(msg.content, str):
                    anthropic_msg["content"] = msg.content
                else:
                    # Convert content blocks
                    anthropic_msg["content"] = self._convert_content_blocks(msg.content)

            anthropic_messages.append(anthropic_msg)

        return anthropic_messages

    def _convert_content_blocks(self, blocks: list[Any]) -> list[dict[str, Any]]:
        """Convert content blocks to Anthropic format."""
        result: list[dict[str, Any]] = []
        for block in blocks:
            if hasattr(block, "model_dump"):
                result.append(block.model_dump())
            elif isinstance(block, dict):
                result.append(block)
        return result

    def _convert_anthropic_response_to_message(
        self, response: dict[str, Any]
    ) -> Message:
        """Convert Anthropic API response to voxagent Message.

        Args:
            response: Anthropic API response dict.

        Returns:
            A voxagent Message.
        """
        role = response.get("role", "assistant")
        content_blocks = response.get("content", [])

        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    params=block.get("input", {}),
                ))

        return Message(
            role=role,
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        stream: bool = False,
        thinking: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for Anthropic API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            stream: Whether to enable streaming.
            thinking: Whether to enable thinking mode.

        Returns:
            Request body dict.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages_to_anthropic(messages),
            "max_tokens": 4096,
        }

        if stream:
            body["stream"] = True

        if system:
            body["system"] = system

        if tools:
            body["tools"] = tools

        if thinking:
            body["thinking"] = {"type": "enabled", "budget_tokens": 4096}

        return body

    def _parse_sse_event(
        self, event_type: str, data: dict[str, Any]
    ) -> StreamChunk | None:
        """Parse an SSE event and return appropriate StreamChunk.

        Args:
            event_type: The SSE event type.
            data: The parsed JSON data.

        Returns:
            A StreamChunk or None if the event doesn't produce a chunk.
        """
        if event_type == "message_start":
            # Initialize message state
            return None

        elif event_type == "content_block_start":
            content_block = data.get("content_block", {})
            block_type = content_block.get("type")
            self._current_block_type = block_type

            if block_type == "tool_use":
                self._current_tool_id = content_block.get("id")
                self._current_tool_name = content_block.get("name")
                self._current_tool_input_json = ""
            return None

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                return TextDeltaChunk(delta=delta.get("text", ""))
            elif delta_type == "input_json_delta":
                self._current_tool_input_json += delta.get("partial_json", "")
                return None
            elif delta_type == "thinking_delta":
                # Extended thinking - we could emit thinking chunks
                return None
            return None

        elif event_type == "content_block_stop":
            # If we have accumulated tool use, yield it
            if self._current_tool_id and self._current_tool_name:
                try:
                    params = json.loads(self._current_tool_input_json) if self._current_tool_input_json else {}
                except json.JSONDecodeError:
                    params = {}

                tool_call = ToolCall(
                    id=self._current_tool_id,
                    name=self._current_tool_name,
                    params=params,
                )
                self._reset_tool_state()
                return ToolUseChunk(tool_call=tool_call)

            self._reset_tool_state()
            return None

        elif event_type == "message_delta":
            return None

        elif event_type == "message_stop":
            return MessageEndChunk()

        elif event_type == "error":
            error_info = data.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            return ErrorChunk(error=error_msg)

        return None


    async def _make_streaming_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        thinking: bool = False,
    ) -> AsyncIterator[str]:
        """Make a streaming request to the Anthropic API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            thinking: Whether to enable thinking mode.

        Yields:
            SSE lines from the response.
        """
        body = self._build_request_body(
            messages, system=system, tools=tools, stream=True, thinking=thinking
        )

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self._get_api_endpoint(),
                headers=self._get_headers(),
                json=body,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the Anthropic API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            abort_signal: Optional signal to abort the stream.
            thinking: Override thinking mode for this request.

        Yields:
            StreamChunk objects containing response data.
        """
        self._reset_tool_state()
        use_thinking = thinking if thinking is not None else self._thinking

        try:
            current_event_type: str | None = None

            async for line in self._make_streaming_request(
                messages, system=system, tools=tools, thinking=use_thinking
            ):
                if abort_signal and abort_signal.aborted:
                    break

                line = line.strip()

                if not line:
                    continue

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str and current_event_type:
                        try:
                            data = json.loads(data_str)
                            chunk = self._parse_sse_event(current_event_type, data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            pass
                        current_event_type = None

        except Exception as e:
            yield ErrorChunk(error=str(e))

    async def _make_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Make a non-streaming request to the Anthropic API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The response JSON dict.
        """
        body = self._build_request_body(
            messages, system=system, tools=tools, stream=False
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_api_endpoint(),
                headers=self._get_headers(),
                json=body,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from the Anthropic API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The assistant's response message.
        """
        response = await self._make_request(messages, system=system, tools=tools)
        return self._convert_anthropic_response_to_message(response)

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Uses a simple estimation based on character count.
        For more accurate counting, use the Anthropic token counting API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.

        Returns:
            Estimated token count.
        """
        # Approximate token counting (roughly 4 chars per token for English)
        char_count = 0

        if system:
            char_count += len(system)

        for msg in messages:
            if isinstance(msg.content, str):
                char_count += len(msg.content)
            else:
                for block in msg.content:
                    if hasattr(block, "text"):
                        char_count += len(block.text)
                    elif hasattr(block, "content"):
                        char_count += len(block.content)

            # Add overhead for role and structure
            char_count += 10

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    char_count += len(tc.name) + len(json.dumps(tc.params))

        # Rough estimate: 4 characters per token
        return max(1, char_count // 4)


__all__ = ["AnthropicProvider"]

