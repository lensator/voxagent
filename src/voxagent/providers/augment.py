"""Augment (Auggie) CLI provider.

This provider wraps the Auggie CLI directly using subprocess.
It requires:
1. The auggie CLI to be installed: brew install augment-cli
2. Authentication via: auggie login OR a vault service providing tokens

Models available:
- haiku4.5: Fast and efficient responses
- sonnet4.5: Great for everyday tasks
- sonnet4: Legacy model
- opus4.5: Best for complex tasks (Claude Opus 4.5)
- gpt5: OpenAI GPT-5 legacy
- gpt5.1: Strong reasoning and planning

Vault Integration:
When a vault_service is provided (any object implementing VaultProtocol with a
get(key) method), the provider checks for an 'augment_access_token'. If found,
it sets the AUGMENT_SESSION_AUTH environment variable when invoking the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from voxagent.providers.base import (
    AbortSignal,
    ErrorChunk,
    MessageEndChunk,
    StreamChunk,
    TextDeltaChunk,
)
from voxagent.providers.cli_base import CLINotFoundError, CLIProvider
from voxagent.types import Message

if TYPE_CHECKING:
    pass  # No external type imports needed


@runtime_checkable
class VaultProtocol(Protocol):
    """Protocol for vault services that can provide credentials.

    This allows voxagent to work with any vault implementation without
    depending on voxdomus directly.
    """

    def get(self, key: str) -> str:
        """Get a credential value by key.

        Args:
            key: The credential key to retrieve.

        Returns:
            The credential value.

        Raises:
            KeyError: If the credential doesn't exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if a credential exists.

        Args:
            key: The credential key to check.

        Returns:
            True if the credential exists, False otherwise.
        """
        ...

logger = logging.getLogger(__name__)

# Vault key for the augment access token
VAULT_KEY_ACCESS_TOKEN = "augment_access_token"


class AugmentProvider(CLIProvider):
    """Provider for Augment using the auggie CLI directly.

    This provider spawns the auggie CLI as a subprocess to avoid
    issues with the auggie-sdk's async task management.

    Optionally integrates with voxDomus vault for token storage.
    When a vault_service is provided and contains an access token,
    the provider sets AUGMENT_SESSION_AUTH for the CLI subprocess.
    """

    CLI_NAME = "auggie"
    ENV_KEY = "AUGMENT_API_TOKEN"

    SUPPORTED_MODELS = [
        "haiku4.5",
        "sonnet4.5",
        "sonnet4",
        "opus4.5",
        "gpt5",
        "gpt5.1",
    ]

    def __init__(
        self,
        model: str = "sonnet4.5",
        api_key: str | None = None,
        base_url: str | None = None,
        vault_service: VaultProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Augment provider.

        Args:
            model: Model name (haiku4.5, sonnet4.5, opus4.5, etc.).
            api_key: Optional API token (usually from auggie login).
            base_url: Optional API URL override.
            vault_service: Optional vault service (any object with a get(key) method).
            **kwargs: Additional arguments.
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self._vault_service: VaultProtocol | None = vault_service
        self._vault_token: str | None = None
        self._vault_checked = False

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "augment"

    @property
    def models(self) -> list[str]:
        """Get supported models."""
        return self.SUPPORTED_MODELS

    @property
    def supports_tools(self) -> bool:
        """Auggie supports tools but we don't expose them."""
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
        """Build auggie CLI arguments.

        Uses --print mode for non-interactive one-shot execution.
        """
        args = ["--print", "--quiet", "--instruction", prompt]

        if self._model:
            args.extend(["--model", self._model])

        return args

    def _parse_output(self, stdout: str, stderr: str) -> str:
        """Parse auggie CLI output."""
        # auggie chat outputs the response directly
        return stdout.strip()

    def _get_vault_token(self) -> str | None:
        """Get access token from vault if available.

        Returns:
            Access token string if found in vault, None otherwise.

        Note:
            Results are cached after first lookup.
        """
        if self._vault_checked:
            return self._vault_token

        self._vault_checked = True

        if self._vault_service is None:
            return None

        try:
            if self._vault_service.exists(VAULT_KEY_ACCESS_TOKEN):
                self._vault_token = self._vault_service.get(VAULT_KEY_ACCESS_TOKEN)
                logger.debug("Retrieved Augment token from vault")
        except Exception as e:
            logger.debug("Could not retrieve Augment token from vault: %s", e)

        return self._vault_token

    async def _run_cli(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Run CLI command and return output.

        Overrides base class to inject AUGMENT_SESSION_AUTH env var
        if a vault token is available.

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

        # Build environment with vault token if available
        env = os.environ.copy()
        vault_token = self._get_vault_token()
        if vault_token:
            env["AUGMENT_SESSION_AUTH"] = vault_token
            logger.debug("Using vault token for AUGMENT_SESSION_AUTH")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
        """Stream a response from Auggie CLI.

        Note: The auggie CLI uses its own MCP tool configuration.
        Tools passed from voxDomus are not used - auggie will use
        tools from its own ~/.augment/settings.json config.
        """
        if tools:
            logger.debug(
                "Auggie CLI uses its own MCP tools - ignoring %d passed tools",
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
            yield ErrorChunk(error=f"Auggie CLI error: {e}")

        yield MessageEndChunk()

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        """Get a complete response from Auggie CLI."""
        text_parts: list[str] = []

        async for chunk in self.stream(messages, system, tools):
            if isinstance(chunk, TextDeltaChunk):
                text_parts.append(chunk.delta)
            elif isinstance(chunk, ErrorChunk):
                raise Exception(chunk.error)

        return Message(role="assistant", content="".join(text_parts))


__all__ = ["AugmentProvider"]

