"""ChatGPT Backend API provider.

This provider uses ChatGPT's private backend API (used by Codex CLI) to access
models like gpt-5 and gpt-5-codex-mini using a ChatGPT Plus subscription.

API Endpoint: https://chatgpt.com/backend-api/codex/responses
Auth: OAuth Bearer token (from Codex CLI or voxdomus vault)

Note: This is an unofficial/undocumented API.

Tool Format:
    The ChatGPT backend API uses a flat tool format different from the standard
    OpenAI nested format. This provider automatically converts between formats:

    Standard OpenAI format (what voxagent uses):
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    ChatGPT backend format (flat):
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}

    The conversion is handled transparently in _convert_tools().
"""

from __future__ import annotations

import json
import logging
import os
import ssl
from collections.abc import AsyncIterator
from typing import Any

import certifi

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

logger = logging.getLogger(__name__)

# API endpoint
CHATGPT_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"

# Default instructions
DEFAULT_INSTRUCTIONS = "You are a helpful AI assistant."


class ChatGPTProvider(BaseProvider):
    """Provider for ChatGPT's private backend API.

    Uses OAuth tokens to access ChatGPT Plus models with full tool support.

    The provider automatically converts voxagent's OpenAI-standard tool format
    to the flat format expected by the ChatGPT backend API.
    """

    ENV_KEY = "CHATGPT_ACCESS_TOKEN"

    SUPPORTED_MODELS = [
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-codex-mini",
        "codex-mini-latest",
    ]

    def __init__(
        self,
        model: str = "gpt-5-codex-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatGPT provider.

        Args:
            model: Model name (gpt-5, gpt-5-codex-mini, etc.).
            api_key: OAuth access token. Falls back to CHATGPT_ACCESS_TOKEN env var.
            base_url: Optional override for API endpoint.
            instructions: Custom system instructions for Codex API.
            **kwargs: Additional arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model
        self._instructions = instructions or DEFAULT_INSTRUCTIONS

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "chatgpt"

    @property
    def models(self) -> list[str]:
        """Get supported models."""
        return self.SUPPORTED_MODELS

    @property
    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling.

        Returns:
            True. The ChatGPT backend API supports tools with a flat format.
            Tools are automatically converted from OpenAI's nested format.
        """
        return True

    @property
    def supports_streaming(self) -> bool:
        """ChatGPT backend requires streaming."""
        return True

    @property
    def context_limit(self) -> int:
        """Approximate context limit."""
        return 128000  # GPT-5 models have large context

    def _convert_tools(self, tools: list[Any] | None) -> list[dict[str, Any]]:
        """Convert OpenAI nested tool format to ChatGPT flat format.

        OpenAI standard format (voxagent uses this):
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        ChatGPT backend format (flat):
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Args:
            tools: List of tools in OpenAI standard format.

        Returns:
            List of tools in ChatGPT flat format.
        """
        if not tools:
            return []

        converted: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                # Check if it's the nested OpenAI format
                if "function" in tool and isinstance(tool["function"], dict):
                    func = tool["function"]
                    converted.append({
                        "type": "function",
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                    })
                # Already flat format or unknown - pass through
                elif "name" in tool:
                    converted.append(tool)
                else:
                    logger.warning("Unknown tool format, skipping: %s", tool)
            else:
                logger.warning("Tool is not a dict, skipping: %s", type(tool))

        return converted

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        token = self.api_key
        if not token:
            raise ValueError("No access token. Set CHATGPT_ACCESS_TOKEN or pass api_key.")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _build_input(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build input array for API request.

        Note: System messages in the messages list are skipped because the
        ChatGPT backend uses a separate 'instructions' field for system prompts.
        The voxagent Agent may pass both a system prompt and system messages
        in the list - we handle this by using the system param only.

        Tool results in voxagent come as user messages with content being a list
        of ToolResultBlock objects. These need to be converted to the ChatGPT
        function_call_output format.

        Assistant messages with tool_calls need to be converted to function_call
        items so that the backend can match tool results to tool calls.
        """
        input_msgs: list[dict[str, Any]] = []

        for msg in messages:
            # Skip system messages - ChatGPT API doesn't support them in input
            if msg.role == "system":
                continue

            # Handle assistant messages with tool calls
            if msg.role == "assistant" and msg.tool_calls:
                # Add text content first if present
                if isinstance(msg.content, str) and msg.content:
                    input_msgs.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content}],
                    })
                # Add function_call items for each tool call
                for tc in msg.tool_calls:
                    # Convert params to JSON string if needed
                    args = tc.params
                    if isinstance(args, dict):
                        import json
                        args = json.dumps(args)
                    input_msgs.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": args,
                    })
                continue

            # Handle content that may be a list (tool results or content blocks)
            if isinstance(msg.content, list):
                for block in msg.content:
                    # Handle ToolResultBlock (Pydantic model or dict)
                    if hasattr(block, "type") and getattr(block, "type", None) == "tool_result":
                        # Pydantic ToolResultBlock
                        input_msgs.append({
                            "type": "function_call_output",
                            "call_id": getattr(block, "tool_use_id", ""),
                            "output": getattr(block, "content", ""),
                        })
                    elif isinstance(block, dict) and block.get("type") == "tool_result":
                        # Dict-style tool result
                        input_msgs.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })
                    elif hasattr(block, "text"):
                        # TextBlock - use output_text for assistant, input_text for user
                        content_type = "output_text" if msg.role == "assistant" else "input_text"
                        input_msgs.append({
                            "type": "message",
                            "role": "user" if msg.role == "user" else "assistant",
                            "content": [{"type": content_type, "text": block.text}],
                        })
                    elif isinstance(block, dict) and "text" in block:
                        content_type = "output_text" if msg.role == "assistant" else "input_text"
                        input_msgs.append({
                            "type": "message",
                            "role": "user" if msg.role == "user" else "assistant",
                            "content": [{"type": content_type, "text": block["text"]}],
                        })
            elif isinstance(msg.content, str):
                # Simple string content - use output_text for assistant, input_text for user
                role = "user" if msg.role == "user" else "assistant"
                content_type = "output_text" if msg.role == "assistant" else "input_text"
                input_msgs.append({
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": msg.content}],
                })

        return input_msgs

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build request body for API.

        Args:
            messages: Conversation messages (system messages are filtered out).
            system: System prompt to use as instructions.
            tools: Optional tool definitions in OpenAI standard format.
                   These will be converted to ChatGPT's flat format.

        Returns:
            Request body dict ready to send to the ChatGPT backend API.
        """
        # Use system prompt if provided, otherwise fall back to default instructions
        instructions = system if system else self._instructions

        # Convert tools from OpenAI nested format to ChatGPT flat format
        converted_tools = self._convert_tools(tools)
        has_tools = len(converted_tools) > 0

        body: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": self._build_input(messages, system),
            "tools": converted_tools,
            "tool_choice": "auto" if has_tools else "none",
            "parallel_tool_calls": False,
            "reasoning": {"summary": "auto"},
            "store": False,
            "stream": True,
        }
        return body

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from ChatGPT backend.

        Args:
            messages: Conversation messages.
            system: Optional system prompt.
            tools: Optional tool definitions.
            abort_signal: Optional abort signal.

        Yields:
            StreamChunk objects.
        """
        body = self._build_request_body(messages, system, tools)
        endpoint = self._base_url or CHATGPT_API_ENDPOINT

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    headers=self._get_headers(),
                    json=body,
                    timeout=120.0,
                ) as response:
                    if response.status_code == 401:
                        yield ErrorChunk(error="Authentication failed - token may be expired")
                        return
                    if response.status_code >= 400:
                        # Read error body before raising
                        error_body = await response.aread()
                        error_text = error_body.decode("utf-8", errors="replace")
                        logger.error("ChatGPT API error %d: %s", response.status_code, error_text)
                        yield ErrorChunk(error=f"HTTP {response.status_code}: {error_text[:500]}")
                        return

                    async for line in response.aiter_lines():
                        if abort_signal and abort_signal.aborted:
                            break
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type", "")
                            # Text delta
                            if event_type == "response.output_text.delta":
                                text = data.get("delta", "")
                                if text:
                                    yield TextDeltaChunk(delta=text)
                            # Tool calls - use output_item.done which has all data
                            elif event_type == "response.output_item.done":
                                item = data.get("item", {})
                                if item.get("type") == "function_call":
                                    yield ToolUseChunk(
                                        tool_call=ToolCall(
                                            id=item.get("call_id", ""),
                                            name=item.get("name", ""),
                                            params=json.loads(item.get("arguments", "{}")),
                                        )
                                    )
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            # Can't access response.text on streaming response without read()
            yield ErrorChunk(error=f"HTTP {e.response.status_code}")
        except Exception as e:
            yield ErrorChunk(error=str(e))

        yield MessageEndChunk()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response (collects streamed chunks)."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async for chunk in self.stream(messages, system, tools):
            if isinstance(chunk, TextDeltaChunk):
                text_parts.append(chunk.delta)
            elif isinstance(chunk, ToolUseChunk):
                tool_calls.append(chunk.tool_call)
            elif isinstance(chunk, ErrorChunk):
                raise Exception(chunk.error)

        return Message(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Estimate token count (rough approximation)."""
        text = system or ""
        for msg in messages:
            text += msg.content or ""
        # Rough estimate: ~4 chars per token
        return len(text) // 4


__all__ = ["ChatGPTProvider", "CHATGPT_API_ENDPOINT"]

