"""Abort and error handling for voxagent agent runs.

This module provides:
- AbortController: Controller for aborting agent runs
- TimeoutHandler: Handles timeout for agent runs
- handle_context_overflow: Handle context overflow errors
- FailoverError: Error for failover scenarios (re-exported from auth)
- FailoverReason: Enum for failover reasons (re-exported from auth)
- AllProfilesExhausted: Error when all auth profiles are exhausted
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from voxagent.providers.auth import FailoverError, FailoverReason
from voxagent.providers.base import AbortSignal
from voxagent.session.compaction import compact_context

if TYPE_CHECKING:
    from voxagent.types.messages import Message


# =============================================================================
# Custom Error Types
# =============================================================================


class AllProfilesExhausted(Exception):
    """Error raised when all auth profiles have been exhausted."""

    def __init__(self, message: str = "All authentication profiles exhausted") -> None:
        """Initialize AllProfilesExhausted.

        Args:
            message: Error message.
        """
        super().__init__(message)


# =============================================================================
# AbortController
# =============================================================================


class AbortController:
    """Controller for aborting agent runs.

    Provides a signal that can be checked by tools and providers to abort
    operations, plus cleanup capabilities.
    """

    def __init__(self) -> None:
        """Initialize the abort controller."""
        self._signal = AbortSignal()
        self._cleaned_up = False

    @property
    def signal(self) -> AbortSignal:
        """Get the abort signal."""
        return self._signal

    def abort(self, reason: str = "Aborted") -> None:
        """Trigger abort.

        Args:
            reason: The reason for aborting.
        """
        self._signal._aborted = True
        self._signal._reason = reason

    def cleanup(self) -> None:
        """Cleanup resources. Safe to call multiple times."""
        if not self._cleaned_up:
            self._cleaned_up = True


# =============================================================================
# TimeoutHandler
# =============================================================================


class TimeoutHandler:
    """Handles timeout for agent runs."""

    def __init__(self, timeout_ms: int) -> None:
        """Initialize the timeout handler.

        Args:
            timeout_ms: Timeout in milliseconds.
        """
        self.timeout_ms = timeout_ms
        self._task: asyncio.Task | None = None
        self._expired = False
        self._started = False

    async def start(self, abort_controller: AbortController) -> None:
        """Start timeout timer.

        Args:
            abort_controller: The abort controller to trigger on timeout.
        """
        if self._started:
            return
        self._started = True

        async def _timeout_task() -> None:
            await asyncio.sleep(self.timeout_ms / 1000.0)
            self._expired = True
            abort_controller.abort(f"Timeout after {self.timeout_ms}ms")

        self._task = asyncio.create_task(_timeout_task())

    def cancel(self) -> None:
        """Cancel timeout timer."""
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def expired(self) -> bool:
        """Check if timeout expired."""
        return self._expired


# =============================================================================
# Context Overflow Handler
# =============================================================================


async def handle_context_overflow(
    messages: list[Message],
    error: Exception,
    model: str,
) -> list[Message]:
    """Handle context overflow error.

    Applies aggressive compaction to reduce context size.

    Args:
        messages: The messages that caused the overflow.
        error: The context overflow error.
        model: The model name.

    Returns:
        Compacted list of messages.
    """
    if not messages:
        return []

    # Determine max tokens based on model
    max_tokens = 4000  # Conservative default
    if "gpt-4" in model:
        max_tokens = 128000
    elif "claude" in model:
        max_tokens = 200000
    elif "gpt-3.5" in model:
        max_tokens = 16000

    # Compact to 30% of max for safety margin after overflow
    target_tokens = int(max_tokens * 0.3)

    # Use aggressive compaction with minimal preserve_recent
    return compact_context(messages, target_tokens, preserve_recent=2, model=model, aggressive=True)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AbortController",
    "TimeoutHandler",
    "handle_context_overflow",
    "FailoverError",
    "FailoverReason",
    "AllProfilesExhausted",
]
