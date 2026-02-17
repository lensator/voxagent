"""Ollama provider implementation.

This module implements the OllamaProvider for local Ollama models,
supporting NDJSON streaming and tool calling.

Ollama is unique:
- No API key required (local server)
- NDJSON streaming (not SSE)
- Dynamic model list from server
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


class OllamaProvider(BaseProvider):
    """Provider for local Ollama models.

    Supports streaming via NDJSON, tool calling, and dynamic model discovery.
    No API key required as Ollama runs locally.
    """

    ENV_KEY = ""  # No API key needed
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        api_key: str | None = None,  # Ignored
        base_url: str | None = None,
        model: str = "llama3.3",
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            api_key: Ignored - Ollama doesn't need authentication.
            base_url: Custom base URL for remote Ollama servers.
            model: Model name to use. Defaults to "llama3.3".
            **kwargs: Additional provider-specific arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "ollama"

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def models(self) -> list[str]:
        """Get the list of supported model names.

        Returns empty list as models are dynamic from server.
        Use list_local_models() to fetch available models.
        """
        return []

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

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for chat."""
        base = self._base_url or self.DEFAULT_BASE_URL
        return f"{base}/api/chat"

    def _get_tags_endpoint(self) -> str:
        """Get the API endpoint for listing models."""
        base = self._base_url or self.DEFAULT_BASE_URL
        return f"{base}/api/tags"

    def _convert_messages_to_ollama(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert voxagent Messages to Ollama message format.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt to prepend.

        Returns:
            List of Ollama-format message dictionaries.
        """
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            ollama_msg: dict[str, Any] = {"role": msg.role}

            # Handle content
            if isinstance(msg.content, str):
                ollama_msg["content"] = msg.content
            else:
                # Handle content blocks - convert to string
                text_parts = [b.text for b in msg.content if hasattr(b, "text")]
                ollama_msg["content"] = " ".join(text_parts) if text_parts else ""

            # Handle tool calls
            if msg.tool_calls:
                ollama_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.params,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            result.append(ollama_msg)

        return result

    def _convert_ollama_response_to_message(self, response: dict[str, Any]) -> Message:
        """Convert Ollama response to voxagent Message.

        Args:
            response: Ollama API response dictionary.

        Returns:
            A voxagent Message object.
        """
        msg_data = response.get("message", {})
        role = msg_data.get("role", "assistant")
        content = msg_data.get("content", "")

        tool_calls: list[ToolCall] | None = None
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                func = tc.get("function", {})
                params = func.get("arguments", {})
                tool_calls.append(
                    ToolCall(id=tc.get("id", ""), name=func.get("name", ""), params=params)
                )

        return Message(role=role, content=content, tool_calls=tool_calls)

    def _parse_ndjson_line(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse an NDJSON line and return appropriate StreamChunk.

        Args:
            data: The parsed JSON data from a line.

        Returns:
            A StreamChunk or None if no chunk should be yielded.
        """
        # Check for error response
        if "error" in data:
            return ErrorChunk(error=data["error"])

        msg_data = data.get("message", {})
        done = data.get("done", False)

        # Check for tool calls first
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            tc = msg_data["tool_calls"][0]  # Process first tool call
            func = tc.get("function", {})
            params = func.get("arguments", {})
            return ToolUseChunk(
                tool_call=ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    params=params,
                )
            )

        # Check for text content
        content = msg_data.get("content", "")
        if content:
            return TextDeltaChunk(delta=content)

        # Check if done
        if done:
            return MessageEndChunk()

        return None

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for Ollama API.

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
            "messages": self._convert_messages_to_ollama(messages, system=system),
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
        _endpoint: str | None = None,
    ) -> dict[str, Any]:
        """Make a non-streaming request to the Ollama API.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            _endpoint: Internal parameter for endpoint selection ("tags" for model list).

        Returns:
            The JSON response from the API.

        Raises:
            Exception: If the API request fails.
        """
        async with httpx.AsyncClient() as client:
            if _endpoint == "tags":
                # GET request for listing models
                response = await client.get(
                    self._get_tags_endpoint(),
                    timeout=30.0,
                )
            else:
                # POST request for chat
                body = self._build_request_body(messages, system=system, tools=tools, stream=False)
                response = await client.post(
                    self._get_api_endpoint(),
                    json=body,
                    timeout=120.0,
                )
            response.raise_for_status()
            return response.json()

    async def _make_streaming_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        """Make a streaming request to the Ollama API.

        Args:
            messages: List of voxagent Message objects.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Yields:
            NDJSON lines from the response.
        """
        body = self._build_request_body(messages, system=system, tools=tools, stream=True)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self._get_api_endpoint(),
                json=body,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        print(f"DEBUG: Ollama raw line: {line}")
                        yield line

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the Ollama API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            abort_signal: Optional signal to abort the stream.

        Yields:
            StreamChunk objects containing response data.
        """
        try:
            async for line in self._make_streaming_request(messages, system=system, tools=tools):
                if abort_signal and abort_signal.aborted:
                    return

                try:
                    data = json.loads(line)
                    chunk = self._parse_ndjson_line(data)
                    if chunk:
                        yield chunk
                        if isinstance(chunk, MessageEndChunk):
                            return
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            yield ErrorChunk(error=str(e))

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from the Ollama API.

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
        return self._convert_ollama_response_to_message(response)

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Uses a simple estimation based on character count.
        Ollama doesn't provide a token counting API.

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

            # Add overhead for role and structure
            char_count += 10

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    char_count += len(tc.name) + len(json.dumps(tc.params))

        # Rough estimate: 4 characters per token
        return max(1, char_count // 4)

    async def list_local_models(self) -> list[str]:
        """List locally available models from Ollama server.

        Returns:
            List of model names available on the server.

        Raises:
            Exception: If the server connection fails.
        """
        response = await self._make_request(messages=[], _endpoint="tags")
        models = response.get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]


__all__ = ["OllamaProvider"]

