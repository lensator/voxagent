"""Base class for CLI-wrapped LLM providers.

This module provides a base class for providers that wrap CLI tools like
auggie, codex, and claude instead of making direct HTTP API calls.

CLI providers spawn subprocesses and communicate via stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from voxagent.providers.base import (
    AbortSignal,
    BaseProvider,
    ErrorChunk,
    MessageEndChunk,
    StreamChunk,
    TextDeltaChunk,
)
from voxagent.types import Message

logger = logging.getLogger(__name__)


class CLINotFoundError(Exception):
    """Raised when a required CLI tool is not found."""

    pass


class CLIProvider(BaseProvider):
    """Base class for CLI-wrapped providers.

    These providers spawn CLI tools as subprocesses rather than making HTTP calls.
    They require the CLI tools to be installed on the system.

    Subclasses must implement:
    - cli_name: Name of the CLI executable
    - _build_cli_args: Build command line arguments
    - _parse_output: Parse CLI output to extract response
    """

    # Subclasses should set this
    CLI_NAME: str = ""
    ENV_KEY: str = ""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CLI provider.

        Args:
            model: Model name to use.
            api_key: Optional API key (passed to CLI if supported).
            base_url: Optional base URL (unused for most CLIs).
            **kwargs: Additional arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model
        self._cli_path: str | None = None

    def _get_cli_path(self) -> str:
        """Get the path to the CLI executable.

        Returns:
            Path to the CLI executable.

        Raises:
            CLINotFoundError: If the CLI is not found.
        """
        if self._cli_path is None:
            self._cli_path = shutil.which(self.CLI_NAME)
            if not self._cli_path:
                raise CLINotFoundError(
                    f"CLI '{self.CLI_NAME}' not found in PATH. "
                    f"Please install it first."
                )
        return self._cli_path

    @property
    def supports_streaming(self) -> bool:
        """CLI providers typically don't support true streaming."""
        return False

    @abstractmethod
    def _build_cli_args(
        self,
        prompt: str,
        system: str | None = None,
    ) -> list[str]:
        """Build CLI command arguments.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt.

        Returns:
            List of command line arguments.
        """
        ...

    @abstractmethod
    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse CLI output to extract response text.

        Args:
            stdout: Standard output from CLI.
            stderr: Standard error from CLI.

        Returns:
            Extracted response text.
        """
        ...

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert message list to a single prompt string.

        CLI tools typically don't support multi-turn conversations natively,
        so we concatenate messages into a single prompt.

        Args:
            messages: List of conversation messages.

        Returns:
            Combined prompt string.
        """
        parts: list[str] = []
        for msg in messages:
            if msg.role == "user" and isinstance(msg.content, str):
                parts.append(msg.content)
            elif msg.role == "assistant" and isinstance(msg.content, str):
                parts.append(f"[Previous response: {msg.content}]")
        return "\n\n".join(parts)

    async def _run_cli(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Run CLI command and return output.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            Parsed response text.

        Raises:
            Exception: If CLI execution fails.
        """
        cli_path = self._get_cli_path()
        args = [cli_path] + self._build_cli_args(prompt, system)

        logger.debug("Running CLI: %s", " ".join(args))

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            logger.warning("CLI exited with code %d: %s", proc.returncode, stderr)

        return self._parse_output(stdout, stderr)

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the CLI.

        CLI providers don't truly stream - we run the CLI and yield the result.

        Args:
            messages: Conversation messages.
            system: Optional system prompt.
            tools: Tool definitions (not supported by most CLIs).
            abort_signal: Optional abort signal.

        Yields:
            StreamChunk objects.
        """
        if tools:
            logger.warning("Tools not supported by CLI provider %s", self.name)

        try:
            prompt = self._messages_to_prompt(messages)
            response = await self._run_cli(prompt, system)
            if response:
                yield TextDeltaChunk(delta=response)
        except CLINotFoundError as e:
            yield ErrorChunk(error=str(e))
        except Exception as e:
            yield ErrorChunk(error=f"CLI error: {e}")

        yield MessageEndChunk()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from the CLI.

        Args:
            messages: Conversation messages.
            system: Optional system prompt.
            tools: Tool definitions (not supported by most CLIs).

        Returns:
            The assistant's response message.
        """
        if tools:
            logger.warning("Tools not supported by CLI provider %s", self.name)

        prompt = self._messages_to_prompt(messages)
        response = await self._run_cli(prompt, system)
        return Message(role="assistant", content=response)

    def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> int:
        """Estimate token count (rough approximation).

        Args:
            messages: Conversation messages.
            system: Optional system prompt.

        Returns:
            Approximate token count.
        """
        text = system or ""
        for msg in messages:
            if isinstance(msg.content, str):
                text += msg.content
        # Rough estimate: ~4 chars per token
        return len(text) // 4


__all__ = ["CLIProvider", "CLINotFoundError"]

