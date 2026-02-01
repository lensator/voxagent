"""OpenAI Codex CLI provider.

This provider wraps the OpenAI Codex CLI (codex command).
It requires:
1. The codex CLI to be installed: npm install -g @openai/codex
2. Authentication via: codex login

Models available:
- o3: OpenAI o3
- o4-mini: OpenAI o4-mini (default)
- gpt-4.1: GPT-4.1
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from voxagent.providers.cli_base import CLINotFoundError, CLIProvider
from voxagent.providers.base import (
    AbortSignal,
    ErrorChunk,
    MessageEndChunk,
    StreamChunk,
    TextDeltaChunk,
)
from voxagent.types import Message

logger = logging.getLogger(__name__)


class CodexProvider(CLIProvider):
    """Provider for OpenAI Codex CLI.

    Uses the codex CLI in exec mode with JSON output for non-interactive use.
    """

    CLI_NAME = "codex"
    ENV_KEY = "OPENAI_API_KEY"

    # Models that work with Codex using ChatGPT Plus account
    # Note: Many models (o3, o4-mini, gpt-4.1) require API key, not ChatGPT Plus
    # "default" means don't specify a model and use the CLI's default
    SUPPORTED_MODELS = [
        "default",
    ]

    def __init__(
        self,
        model: str = "default",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Codex provider.

        Args:
            model: Model name (o3, o4-mini, gpt-4.1).
            api_key: Optional OpenAI API key.
            base_url: Optional base URL override.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "codex"

    @property
    def models(self) -> list[str]:
        """Get supported models."""
        return self.SUPPORTED_MODELS

    @property
    def supports_tools(self) -> bool:
        """Codex has tool support but we don't expose it."""
        return False

    @property
    def context_limit(self) -> int:
        """Approximate context limit."""
        return 128000

    def _build_cli_args(
        self,
        prompt: str,
        system: str | None = None,
    ) -> list[str]:
        """Build codex CLI arguments.

        Uses exec mode for non-interactive execution with JSON output.
        """
        args = ["exec", "--json"]

        # Only pass --model if not using "default"
        if self._model and self._model != "default":
            args.extend(["--model", self._model])

        # Add the prompt
        args.append(prompt)

        return args

    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse codex CLI JSON output.

        The --json flag outputs JSONL events. We look for agent_message items.
        Format: {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        """
        # Parse JSONL output and extract text from agent_message items
        text_parts: list[str] = []

        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                # Look for item.completed events with agent_message type
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            text_parts.append(text)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                text_parts.append(line)

        return "\n".join(text_parts) if text_parts else stdout.strip()

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from Codex CLI.

        Note: The codex CLI has its own tool execution capabilities.
        Tools passed from voxDomus are not used.
        """
        if tools:
            logger.debug(
                "Codex CLI has its own tools - ignoring %d passed tools",
                len(tools),
            )

        try:
            prompt = self._messages_to_prompt(messages)
            response = await self._run_cli(prompt, system)
            if response:
                yield TextDeltaChunk(delta=response)
        except CLINotFoundError as e:
            yield ErrorChunk(error=str(e))
        except Exception as e:
            yield ErrorChunk(error=f"Codex CLI error: {e}")

        yield MessageEndChunk()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from Codex CLI."""
        text_parts: list[str] = []

        async for chunk in self.stream(messages, system, tools):
            if isinstance(chunk, TextDeltaChunk):
                text_parts.append(chunk.delta)
            elif isinstance(chunk, ErrorChunk):
                raise Exception(chunk.error)

        return Message(role="assistant", content="".join(text_parts))


__all__ = ["CodexProvider"]

