"""Tool executor for voxagent."""

from __future__ import annotations

import json
from typing import Any

from voxagent.providers.base import AbortSignal
from voxagent.tools.context import AbortError, ToolContext
from voxagent.tools.definition import ToolDefinition
from voxagent.types.messages import ToolResult


async def execute_tool(
    name: str,
    params: dict[str, Any],
    tools: list[ToolDefinition],
    abort_signal: AbortSignal,
    tool_use_id: str,
    deps: Any = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> ToolResult:
    """Execute a tool by name.

    Args:
        name: Tool name to execute
        params: Parameters to pass to the tool
        tools: List of available tools
        abort_signal: Signal to check for abort
        tool_use_id: ID for the tool use (from LLM)
        deps: Optional dependencies to inject
        session_id: Current session ID
        run_id: Current run ID

    Returns:
        ToolResult with content or error
    """
    # Find tool
    tool = next((t for t in tools if t.name == name), None)

    if tool is None:
        return ToolResult(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {name}",
            is_error=True,
        )

    # Check abort before execution
    if abort_signal.aborted:
        return ToolResult(
            tool_use_id=tool_use_id,
            content="Aborted",
            is_error=True,
        )

    # Create context
    context = ToolContext(
        abort_signal=abort_signal,
        deps=deps,
        session_id=session_id,
        run_id=run_id,
    )

    try:
        # Execute tool
        result = await tool.run(params, context)

        # Sanitize result
        content = _sanitize_result(result)

        return ToolResult(
            tool_use_id=tool_use_id,
            content=content,
            is_error=False,
        )

    except AbortError:
        return ToolResult(
            tool_use_id=tool_use_id,
            content="Aborted",
            is_error=True,
        )

    except Exception as e:
        return ToolResult(
            tool_use_id=tool_use_id,
            content=_format_error(e),
            is_error=True,
        )


def _sanitize_result(result: Any) -> str:
    """Convert tool result to string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # JSON serialize dicts, lists, etc.
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


def _format_error(e: Exception) -> str:
    """Format exception for tool result."""
    return f"{type(e).__name__}: {e}"

