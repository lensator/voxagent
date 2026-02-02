"""OpenAI provider implementation.

This module implements the OpenAI provider for chat completions,
supporting both streaming and non-streaming responses with tool calling.
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


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions provider.

    Supports GPT-4o, GPT-4-turbo, GPT-3.5-turbo, and O1 models
    with streaming, tool calling, and token counting.
    """

    ENV_KEY = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Custom base URL for API requests (e.g., for Azure or proxies).
            model: Model name to use. Defaults to "gpt-4o".
            **kwargs: Additional provider-specific arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def models(self) -> list[str]:
        """Get the list of supported model names."""
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview"]

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
        return 128000

    def _get_api_url(self) -> str:
        """Get the API URL for chat completions."""
        base = self._base_url or self.DEFAULT_BASE_URL
        return f"{base}/chat/completions"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _convert_messages_to_openai(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert voxagent Messages to OpenAI message format.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt to prepend.

        Returns:
            List of OpenAI-format message dictionaries.
        """
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg.role}

            # Handle content
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # Handle content blocks - convert to string for simplicity
                text_parts = [b.text for b in msg.content if hasattr(b, "text")]
                openai_msg["content"] = " ".join(text_parts) if text_parts else ""

            # Handle tool calls
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.params),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            result.append(openai_msg)

        return result

    def _convert_openai_response_to_message(self, response: dict[str, Any]) -> Message:
        """Convert OpenAI response message to voxagent Message.

        Args:
            response: OpenAI message dictionary from API response.

        Returns:
            A voxagent Message object.
        """
        role = response.get("role", "assistant")
        content = response.get("content") or ""

        tool_calls: list[ToolCall] | None = None
        if "tool_calls" in response and response["tool_calls"]:
            tool_calls = []
            for tc in response["tool_calls"]:
                params = {}
                if tc.get("function", {}).get("arguments"):
                    try:
                        params = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        params = {}
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        params=params,
                    )
                )

        return Message(role=role, content=content, tool_calls=tool_calls)

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for OpenAI API.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            stream: Whether to enable streaming.

        Returns:
            Request body dictionary.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages_to_openai(messages, system=system),
            "stream": stream,
        }

        if tools:
            body["tools"] = tools

        return body

    async def _make_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Make a non-streaming request to the OpenAI API.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The JSON response from the API.

        Raises:
            Exception: If the API request fails.
        """
        body = self._build_request_body(messages, system=system, tools=tools, stream=False)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_api_url(),
                headers=self._get_headers(),
                json=body,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

    async def _make_streaming_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Make a streaming request to the OpenAI API.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Yields:
            Parsed JSON chunks from the SSE stream.
        """
        body = self._build_request_body(messages, system=system, tools=tools, stream=True)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self._get_api_url(),
                headers=self._get_headers(),
                json=body,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the OpenAI API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            abort_signal: Optional signal to abort the stream.

        Yields:
            StreamChunk objects containing response data.
        """
        # Track tool calls being built across chunks
        pending_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in self._make_streaming_request(messages, system=system, tools=tools):
                # Check abort signal
                if abort_signal and abort_signal.aborted:
                    return

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Handle content delta
                if "content" in delta and delta["content"]:
                    yield TextDeltaChunk(delta=delta["content"])

                # Handle tool calls
                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        else:
                            if tc_delta.get("id"):
                                pending_tool_calls[idx]["id"] = tc_delta["id"]
                            if tc_delta.get("function", {}).get("name"):
                                pending_tool_calls[idx]["name"] = tc_delta["function"]["name"]

                        # Accumulate arguments
                        if tc_delta.get("function", {}).get("arguments"):
                            pending_tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]

                # Handle finish
                if finish_reason:
                    # Emit any pending tool calls
                    for tc_data in pending_tool_calls.values():
                        params = {}
                        if tc_data["arguments"]:
                            try:
                                params = json.loads(tc_data["arguments"])
                            except json.JSONDecodeError:
                                params = {}
                        yield ToolUseChunk(
                            tool_call=ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                params=params,
                            )
                        )
                    yield MessageEndChunk()
                    return

        except Exception as e:
            yield ErrorChunk(error=str(e))

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from the OpenAI API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The assistant's response message.

        Raises:
            Exception: If the API request fails.
        """
        response = await self._make_request(messages, system=system, tools=tools)
        choices = response.get("choices", [])
        if not choices:
            return Message(role="assistant", content="")

        return self._convert_openai_response_to_message(choices[0]["message"])

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Uses tiktoken if available, otherwise falls back to character-based estimation.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.

        Returns:
            The estimated token count.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self._model)
        except (ImportError, KeyError):
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

        # Use tiktoken for accurate counting
        total_tokens = 0

        if system:
            total_tokens += len(encoding.encode(system))
            total_tokens += 4  # Overhead for system message

        for msg in messages:
            total_tokens += 4  # Message overhead
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            else:
                for block in msg.content:
                    if hasattr(block, "text"):
                        total_tokens += len(encoding.encode(block.text))

        total_tokens += 2  # Response priming
        return total_tokens


__all__ = ["OpenAIProvider"]

