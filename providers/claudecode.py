"""Claude Code CLI provider.

This provider wraps the Anthropic Claude Code CLI (claude command).
It requires:
1. The claude CLI to be installed
2. Authentication via: claude setup-token

Models available:
- sonnet: Claude Sonnet (latest)
- opus: Claude Opus (latest)
- haiku: Claude Haiku (latest)
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


class ClaudeCodeProvider(CLIProvider):
    """Provider for Claude Code CLI.

    Uses the claude CLI in print mode with text output for non-interactive use.
    """

    CLI_NAME = "claude"
    ENV_KEY = "ANTHROPIC_API_KEY"

    SUPPORTED_MODELS = [
        "sonnet",
        "opus",
        "haiku",
    ]

    def __init__(
        self,
        model: str = "sonnet",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Claude Code provider.

        Args:
            model: Model alias (sonnet, opus, haiku).
            api_key: Optional Anthropic API key.
            base_url: Optional base URL override.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "claudecode"

    @property
    def models(self) -> list[str]:
        """Get supported models."""
        return self.SUPPORTED_MODELS

    @property
    def supports_tools(self) -> bool:
        """Claude Code has tool support but we don't expose it."""
        return False

    @property
    def context_limit(self) -> int:
        """Approximate context limit."""
        return 200000

    def _build_cli_args(
        self,
        prompt: str,
        system: str | None = None,
    ) -> list[str]:
        """Build claude CLI arguments.

        Uses print mode for non-interactive execution.
        """
        args = ["--print", "--output-format", "text"]

        if self._model:
            args.extend(["--model", self._model])

        if system:
            args.extend(["--system-prompt", system])

        # Add the prompt
        args.append(prompt)

        return args

    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse claude CLI output."""
        # claude --print outputs just the response text
        return stdout.strip()

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from Claude Code CLI.

        Note: The claude CLI has its own MCP tool configuration.
        Tools passed from voxDomus are not used.
        """
        if tools:
            logger.debug(
                "Claude CLI has its own MCP tools - ignoring %d passed tools",
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
            yield ErrorChunk(error=f"Claude Code CLI error: {e}")

        yield MessageEndChunk()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from Claude Code CLI."""
        text_parts: list[str] = []

        async for chunk in self.stream(messages, system, tools):
            if isinstance(chunk, TextDeltaChunk):
                text_parts.append(chunk.delta)
            elif isinstance(chunk, ErrorChunk):
                raise Exception(chunk.error)

        return Message(role="assistant", content="".join(text_parts))


__all__ = ["ClaudeCodeProvider"]

