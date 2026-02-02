"""Sub-agent context for depth tracking and propagation.

This module provides SubAgentContext which extends ToolContext with
depth tracking to prevent infinite recursion in nested agent calls.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import ConfigDict, Field

from voxagent.providers.base import AbortSignal
from voxagent.tools.context import ToolContext

T = TypeVar("T")

# Default maximum depth for nested agent calls
DEFAULT_MAX_DEPTH = 5


class MaxDepthExceededError(Exception):
    """Raised when sub-agent call exceeds maximum depth."""

    def __init__(self, depth: int, max_depth: int) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(
            f"Maximum sub-agent depth exceeded: {depth} > {max_depth}. "
            "This may indicate infinite recursion or an overly deep agent hierarchy."
        )


class SubAgentContext(ToolContext[T]):
    """Extended context for sub-agent execution with depth tracking.

    Extends ToolContext with:
    - depth: Current nesting level (0 = root agent)
    - max_depth: Maximum allowed nesting depth
    - parent_run_id: Run ID of the parent agent (for tracing)

    Attributes:
        depth: Current depth in the agent hierarchy (0 = root).
        max_depth: Maximum allowed depth before raising MaxDepthExceededError.
        parent_run_id: The run_id of the parent agent that spawned this sub-agent.
    """

    depth: int = Field(default=0, ge=0)
    max_depth: int = Field(default=DEFAULT_MAX_DEPTH, ge=1)
    parent_run_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def check_depth(self) -> None:
        """Raise MaxDepthExceededError if depth exceeds max_depth."""
        if self.depth >= self.max_depth:
            raise MaxDepthExceededError(self.depth, self.max_depth)

    def child_context(
        self,
        abort_signal: AbortSignal | None = None,
        deps: Any = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> "SubAgentContext[Any]":
        """Create a child context with incremented depth.

        Args:
            abort_signal: Abort signal for the child (uses parent's if None).
            deps: Dependencies for the child (uses parent's if None).
            session_id: Session ID for the child (uses parent's if None).
            run_id: Run ID for the child agent.

        Returns:
            New SubAgentContext with depth + 1.

        Raises:
            MaxDepthExceededError: If the new depth exceeds max_depth.
        """
        new_depth = self.depth + 1
        if new_depth > self.max_depth:
            raise MaxDepthExceededError(new_depth, self.max_depth)

        return SubAgentContext(
            abort_signal=abort_signal or self.abort_signal,
            deps=deps if deps is not None else self.deps,
            session_id=session_id or self.session_id,
            run_id=run_id,
            retry_count=0,  # Reset retry count for child
            depth=new_depth,
            max_depth=self.max_depth,
            parent_run_id=self.run_id,  # Current run becomes parent
        )

    @classmethod
    def from_tool_context(
        cls,
        context: ToolContext,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        parent_run_id: str | None = None,
    ) -> "SubAgentContext[Any]":
        """Create SubAgentContext from a regular ToolContext.

        Args:
            context: The source ToolContext.
            depth: Initial depth (default 0).
            max_depth: Maximum depth (default DEFAULT_MAX_DEPTH).
            parent_run_id: Parent run ID for tracing.

        Returns:
            New SubAgentContext with depth tracking.
        """
        return cls(
            abort_signal=context.abort_signal,
            deps=context.deps,
            session_id=context.session_id,
            run_id=context.run_id,
            retry_count=context.retry_count,
            depth=depth,
            max_depth=max_depth,
            parent_run_id=parent_run_id,
        )

