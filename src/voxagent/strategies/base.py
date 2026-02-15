"""Strategy system for voxagent agents.

This module provides the foundation for pluggable execution strategies:
- StrategyContext: Context with LLM and tool access helpers
- StrategyResult: Result dataclass from strategy execution
- AgentStrategy: Abstract base class for strategies
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from voxagent.providers.base import AbortSignal, BaseProvider
    from voxagent.streaming.events import StreamEventData
    from voxagent.tools.definition import ToolDefinition
    from voxagent.types.messages import Message, ToolCall, ToolResult
    from voxagent.types.run import ToolMeta


@dataclass
class StrategyResult:
    """Result from strategy execution."""

    messages: list["Message"]
    assistant_texts: list[str]
    tool_metas: list["ToolMeta"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class StrategyContext:
    """Context provided to strategies for LLM and tool access."""

    # Core components
    provider: "BaseProvider"
    tools: list["ToolDefinition"]
    system_prompt: str | None
    abort_signal: "AbortSignal"
    run_id: str

    # Dependencies for tool execution
    deps: Any = None

    async def call_llm(
        self,
        messages: list["Message"],
    ) -> tuple[str, list["ToolCall"]]:
        """Call the LLM and return (response_text, tool_calls)."""
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            TextDeltaChunk,
            ToolUseChunk,
        )

        response_text = ""
        tool_calls: list["ToolCall"] = []

        async for chunk in self.provider.stream(
            messages=messages,
            system=self.system_prompt,
            tools=[t.to_openai_schema() for t in self.tools] if self.tools else None,
            abort_signal=self.abort_signal,
        ):
            if isinstance(chunk, TextDeltaChunk):
                response_text += chunk.delta
            elif isinstance(chunk, ToolUseChunk):
                tool_calls.append(chunk.tool_call)
            elif isinstance(chunk, ErrorChunk):
                raise RuntimeError(chunk.error)
            elif isinstance(chunk, MessageEndChunk):
                break

        return response_text, tool_calls

    async def execute_tool(
        self,
        tool_call: "ToolCall",
    ) -> "ToolResult":
        """Execute a single tool call."""
        from voxagent.tools.executor import execute_tool

        return await execute_tool(
            name=tool_call.name,
            params=tool_call.params,
            tools=self.tools,
            abort_signal=self.abort_signal,
            tool_use_id=tool_call.id,
            deps=self.deps,
            run_id=self.run_id,
        )

    async def run_tool_loop(
        self,
        messages: list["Message"],
        max_iterations: int = 50,
    ) -> StrategyResult:
        """Run the canonical tool loop until no more tool calls."""
        from voxagent.types.messages import Message as Msg
        from voxagent.types.run import ToolMeta

        assistant_texts: list[str] = []
        tool_metas: list[ToolMeta] = []
        working_messages = list(messages)  # Copy to avoid mutation

        for _ in range(max_iterations):
            if self.abort_signal.aborted:
                break

            response_text, tool_calls = await self.call_llm(working_messages)

            if response_text:
                assistant_texts.append(response_text)
                working_messages.append(Msg(role="assistant", content=response_text))

            if not tool_calls:
                break

            # Execute all tool calls
            for tc in tool_calls:
                start_time = time.perf_counter()
                result = await self.execute_tool(tc)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                tool_metas.append(
                    ToolMeta(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        execution_time_ms=elapsed_ms,
                        success=not result.is_error,
                        error=result.content if result.is_error else None,
                    )
                )

                # Add tool result as user message
                working_messages.append(
                    Msg(
                        role="user",
                        content=[
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "tool_name": tc.name,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                        ],
                    )
                )

        return StrategyResult(
            messages=working_messages,
            assistant_texts=assistant_texts,
            tool_metas=tool_metas,
        )

    async def call_llm_stream(
        self,
        messages: list["Message"],
    ) -> AsyncIterator["StreamEventData"]:
        """Stream LLM response, yielding events."""
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            TextDeltaChunk,
        )
        from voxagent.streaming.events import RunErrorEvent, TextDeltaEvent

        async for chunk in self.provider.stream(
            messages=messages,
            system=self.system_prompt,
            tools=[t.to_openai_schema() for t in self.tools] if self.tools else None,
            abort_signal=self.abort_signal,
        ):
            if isinstance(chunk, TextDeltaChunk):
                yield TextDeltaEvent(run_id=self.run_id, delta=chunk.delta)
            elif isinstance(chunk, ErrorChunk):
                yield RunErrorEvent(run_id=self.run_id, error=chunk.error)
                return
            elif isinstance(chunk, MessageEndChunk):
                return

    async def run_tool_loop_stream(
        self,
        messages: list["Message"],
        max_iterations: int = 50,
    ) -> AsyncIterator["StreamEventData"]:
        """Run the tool loop, streaming all events."""
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            TextDeltaChunk,
            ToolUseChunk,
        )
        from voxagent.streaming.events import (
            RunErrorEvent,
            TextDeltaEvent,
            ToolEndEvent,
            ToolOutputEvent,
            ToolStartEvent,
        )
        from voxagent.types.messages import Message as Msg
        from voxagent.types.messages import ToolCall, ToolResult

        working_messages = list(messages)

        for _ in range(max_iterations):
            if self.abort_signal.aborted:
                break

            response_text = ""
            tool_calls: list[ToolCall] = []

            # Stream LLM response
            async for chunk in self.provider.stream(
                messages=working_messages,
                system=self.system_prompt,
                tools=[t.to_openai_schema() for t in self.tools]
                if self.tools
                else None,
                abort_signal=self.abort_signal,
            ):
                if isinstance(chunk, TextDeltaChunk):
                    response_text += chunk.delta
                    yield TextDeltaEvent(run_id=self.run_id, delta=chunk.delta)
                elif isinstance(chunk, ToolUseChunk):
                    tool_calls.append(chunk.tool_call)
                elif isinstance(chunk, ErrorChunk):
                    yield RunErrorEvent(run_id=self.run_id, error=chunk.error)
                    return
                elif isinstance(chunk, MessageEndChunk):
                    break

            if response_text:
                working_messages.append(Msg(role="assistant", content=response_text))

            if not tool_calls:
                break

            # Execute tools and stream events
            for tc in tool_calls:
                yield ToolStartEvent(run_id=self.run_id, tool_call=tc)

                start_time = time.perf_counter()
                result = await self.execute_tool(tc)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                # Stream tool output as delta
                yield ToolOutputEvent(
                    run_id=self.run_id,
                    tool_call_id=tc.id,
                    delta=result.content,
                )

                yield ToolEndEvent(
                    run_id=self.run_id,
                    tool_call_id=tc.id,
                    result=result,
                )

                working_messages.append(
                    Msg(
                        role="user",
                        content=[
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "tool_name": tc.name,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                        ],
                    )
                )


class AgentStrategy(ABC):
    """Abstract base class for agent execution strategies."""

    @property
    def name(self) -> str:
        """Return the strategy name (class name by default)."""
        return self.__class__.__name__

    @abstractmethod
    async def execute(
        self,
        ctx: StrategyContext,
        messages: list["Message"],
    ) -> StrategyResult:
        """Execute the strategy (non-streaming)."""
        ...

    @abstractmethod
    async def execute_stream(
        self,
        ctx: StrategyContext,
        messages: list["Message"],
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the strategy with streaming."""
        ...
        # This yield is needed to make this an async generator
        yield  # type: ignore[misc]


__all__ = [
    "AgentStrategy",
    "StrategyContext",
    "StrategyResult",
]

