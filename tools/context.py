"""Tool context for voxagent."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from voxagent.providers.base import AbortSignal

T = TypeVar("T")


class AbortError(Exception):
    """Raised when an operation is aborted."""

    def __init__(self, message: str = "Operation aborted") -> None:
        super().__init__(message)


class ToolContext(BaseModel, Generic[T]):
    """Runtime context passed to tool functions.

    Replaces PydanticAI's RunContext with a simpler, more focused API.

    Attributes:
        abort_signal: Signal to check for abort requests
        deps: Optional dependencies injected by the agent
        session_id: Current session ID
        run_id: Current run ID
        retry_count: Number of times this tool has been retried
    """

    abort_signal: AbortSignal
    deps: T | None = None
    session_id: str | None = None
    run_id: str | None = None
    retry_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_aborted(self) -> bool:
        """Check if the operation has been aborted."""
        return self.abort_signal.aborted

    def check_abort(self) -> None:
        """Raise AbortError if aborted."""
        if self.abort_signal.aborted:
            raise AbortError("Operation aborted")

