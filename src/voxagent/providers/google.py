"""Google (Gemini) provider implementation.

This module implements the GoogleProvider for Gemini models,
supporting streaming, tool use, and multimodal content.
"""

import json
import os
import uuid
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
from voxagent.types.messages import ToolResultBlock


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Supports Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash, and other models.
    Implements streaming, tool use, and large context windows.
    """

    ENV_KEY = "GOOGLE_API_KEY"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-2.0-flash",
        **kwargs: Any,
    ) -> None:
        """Initialize the Google provider.

        Args:
            api_key: API key for authentication. Falls back to GOOGLE_API_KEY,
                     then GEMINI_API_KEY env vars.
            base_url: Optional custom base URL for API requests.
            model: Model to use (default: gemini-2.0-flash).
            **kwargs: Additional provider-specific arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model

    @property
    def api_key(self) -> str | None:
        """Get API key from constructor or environment variables."""
        if self._api_key is not None:
            return self._api_key
        # Check GOOGLE_API_KEY first, then GEMINI_API_KEY
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "google"

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
        return 1000000

    def _get_base_url(self) -> str:
        """Get the base URL for API requests."""
        return self._base_url or self.DEFAULT_BASE_URL

    def _get_stream_endpoint(self) -> str:
        """Get the streaming endpoint URL."""
        return f"{self._get_base_url()}/models/{self._model}:streamGenerateContent"

    def _get_complete_endpoint(self) -> str:
        """Get the non-streaming endpoint URL."""
        return f"{self._get_base_url()}/models/{self._model}:generateContent"

    def _get_request_url(self, action: str) -> str:
        """Get the full request URL with API key."""
        base = self._get_base_url()
        return f"{base}/models/{self._model}:{action}?key={self.api_key}"

    def _convert_messages_to_gemini(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert voxagent Messages to Gemini contents format.

        Args:
            messages: List of voxagent Messages.

        Returns:
            List of Gemini-formatted content dicts.
        """
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # System messages are handled via system_instruction
                continue

            # Map role: user -> user, assistant -> model
            role = "model" if msg.role == "assistant" else "user"

            parts: list[dict[str, Any]] = []

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.params,
                        }
                    })
            elif isinstance(msg.content, str):
                if msg.content:
                    parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                # Handle content blocks (including tool results)
                for block in msg.content:
                    # Handle ToolResultBlock Pydantic model
                    if isinstance(block, ToolResultBlock):
                        # For Gemini, tool results need functionResponse format
                        tool_name = block.tool_name or "unknown"
                        parts.append({
                            "functionResponse": {
                                "name": tool_name,
                                "response": {
                                    "result": block.content,
                                    "is_error": block.is_error,
                                }
                            }
                        })
                    elif isinstance(block, dict):
                        # Handle tool_result dict (fallback)
                        if block.get("type") == "tool_result":
                            tool_name = block.get("tool_name", "unknown")
                            parts.append({
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {
                                        "result": block.get("content", ""),
                                        "is_error": block.get("is_error", False),
                                    }
                                }
                            })
                        elif "text" in block:
                            parts.append({"text": block["text"]})
                    elif hasattr(block, "text"):
                        parts.append({"text": block.text})

            if parts:
                contents.append({"role": role, "parts": parts})

        return contents

    def _convert_gemini_response_to_message(
        self, response: dict[str, Any]
    ) -> Message:
        """Convert Gemini API response to voxagent Message.

        Args:
            response: Gemini API response dict.

        Returns:
            A voxagent Message.
        """
        candidates = response.get("candidates", [])
        if not candidates:
            return Message(role="assistant", content="")

        content_obj = candidates[0].get("content", {})
        parts = content_obj.get("parts", [])

        text_content = ""
        tool_calls: list[ToolCall] = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=str(uuid.uuid4()),
                    name=fc.get("name", ""),
                    params=fc.get("args", {}),
                ))

        return Message(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _sanitize_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Sanitize a JSON schema for Gemini API compatibility.

        Google's Gemini API has specific requirements:
        - Enum values must be strings
        - Enum is only allowed for STRING type properties

        This method recursively sanitizes the schema.

        Args:
            schema: A JSON schema dict.

        Returns:
            Sanitized schema compatible with Gemini API.
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            if key == "enum" and isinstance(value, list):
                # Convert all enum values to strings
                result[key] = [str(v) for v in value]
            elif isinstance(value, dict):
                result[key] = self._sanitize_schema_for_gemini(value)
            elif isinstance(value, list):
                result[key] = [
                    self._sanitize_schema_for_gemini(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        # If this schema has an enum, ensure type is "string" (Gemini requirement)
        if "enum" in result:
            result["type"] = "string"

        return result

    def _convert_tools_to_gemini(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to Gemini format.

        Args:
            tools: List of tool definitions in OpenAI format.
                   Each tool has structure: {"type": "function", "function": {...}}

        Returns:
            Gemini-formatted tool declarations.
        """
        function_declarations = []
        for tool in tools:
            # Handle OpenAI format: {"type": "function", "function": {...}}
            if tool.get("type") == "function" and "function" in tool:
                func_info = tool["function"]
                func_decl: dict[str, Any] = {
                    "name": func_info.get("name", ""),
                    "description": func_info.get("description", ""),
                }
                if "parameters" in func_info:
                    # Sanitize parameters schema for Gemini compatibility
                    func_decl["parameters"] = self._sanitize_schema_for_gemini(
                        func_info["parameters"]
                    )
            else:
                # Fallback: assume flat structure
                func_decl = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                if "parameters" in tool:
                    func_decl["parameters"] = self._sanitize_schema_for_gemini(
                        tool["parameters"]
                    )
            function_declarations.append(func_decl)

        return [{"functionDeclarations": function_declarations}]

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build the request body for Gemini API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            Request body dict.
        """
        body: dict[str, Any] = {
            "contents": self._convert_messages_to_gemini(messages),
        }

        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}

        if tools:
            body["tools"] = self._convert_tools_to_gemini(tools)

        return body

    async def _make_streaming_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Make a streaming request to the Gemini API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Yields:
            Parsed JSON response chunks.
        """
        body = self._build_request_body(messages, system=system, tools=tools)
        url = f"{self._get_stream_endpoint()}?key={self.api_key}&alt=sse"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            ) as response:
                if response.status_code >= 400:
                    # Read the full error response for better error messages
                    error_body = await response.aread()
                    try:
                        error_json = json.loads(error_body)
                        error_msg = error_json.get("error", {}).get("message", error_body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        error_msg = error_body.decode() if error_body else "Unknown error"
                    raise httpx.HTTPStatusError(
                        f"Google API error: {error_msg}",
                        request=response.request,
                        response=response,
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str:
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the Gemini API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.
            abort_signal: Optional signal to abort the stream.

        Yields:
            StreamChunk objects containing response data.
        """
        try:
            async for chunk in self._make_streaming_request(
                messages, system=system, tools=tools
            ):
                if abort_signal and abort_signal.aborted:
                    break

                candidates = chunk.get("candidates", [])
                if not candidates:
                    continue

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                for part in parts:
                    if "text" in part:
                        yield TextDeltaChunk(delta=part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        yield ToolUseChunk(
                            tool_call=ToolCall(
                                id=str(uuid.uuid4()),
                                name=fc.get("name", ""),
                                params=fc.get("args", {}),
                            )
                        )

            yield MessageEndChunk()

        except Exception as e:
            yield ErrorChunk(error=str(e))

    async def _make_request(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Make a non-streaming request to the Gemini API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The response JSON dict.
        """
        body = self._build_request_body(messages, system=system, tools=tools)
        url = f"{self._get_complete_endpoint()}?key={self.api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
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
        """Get a complete response from the Gemini API.

        Args:
            messages: The conversation messages.
            system: Optional system prompt.
            tools: Optional list of tool definitions.

        Returns:
            The assistant's response message.
        """
        response = await self._make_request(messages, system=system, tools=tools)
        return self._convert_gemini_response_to_message(response)

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Count tokens in the messages.

        Uses a simple estimation based on character count.
        For more accurate counting, use the Gemini token counting API.

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


__all__ = ["GoogleProvider"]
