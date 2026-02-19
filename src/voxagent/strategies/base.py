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
    from voxagent.agent.abort import AbortController
    from voxagent.providers.base import BaseProvider
    from voxagent.streaming.events import StreamEventData
    from voxagent.tools.definition import ToolDefinition
    from voxagent.types.messages import Message, ToolCall, ToolResult
    from voxagent.types.run import ToolMeta


@dataclass
class StrategyResult:
    """Result from strategy execution.
    
    Attributes:
        messages: All messages from the run (system, user, assistant, tool results).
        assistant_texts: List of assistant response texts (one per LLM call).
        tool_metas: Metadata for each tool execution.
        metadata: Strategy-specific metadata (iterations, steps, etc.).
        error: Domain-level error message if run failed logically.
    """

    messages: list["Message"]
    assistant_texts: list[str]
    tool_metas: list["ToolMeta"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class StrategyContext:
    """Context for strategy execution.
    
    Provides access to LLM, tools, and the canonical tool loop.
    All helper methods handle abort signal internally.
    """

    # Original request
    prompt: str
    deps: Any | None
    session_key: str | None
    message_history: list["Message"] | None
    timeout_ms: int | None

    # Agent internals
    provider: "BaseProvider"
    tools: list["ToolDefinition"]
    system_prompt: str | None
    abort_controller: "AbortController"
    
    # Run tracking
    run_id: str

    # Memory and Sync
    memory_manager: Any | None = None

    # Session storage for persistence
    session_storage: Any | None = None

    async def call_llm(
        self,
        messages: list["Message"],
        tools: list["ToolDefinition"] | None = None,
    ) -> tuple[str, list["ToolCall"]]:
        """Make a single LLM call and return (text, tool_calls).
        
        Collects all streaming chunks internally and returns final result.
        Does NOT emit events - use call_llm_stream() for event emission.
        """
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            TextDeltaChunk,
            ToolUseChunk,
        )

        response_text = ""
        tool_calls: list["ToolCall"] = []

        from voxagent.types.messages import Message as Msg
        
        # Use provided tools or default to context tools
        effective_tools = tools if tools is not None else self.tools
        
        async for chunk in self.provider.stream(
            messages=[m if isinstance(m, Msg) else Msg(**m) for m in messages],
            system=self.system_prompt,
            tools=[t.to_openai_schema() for t in effective_tools] if effective_tools else None,
            abort_signal=self.abort_controller.signal,
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
        """Execute a single tool call.
        
        Does NOT emit events - use execute_tool_stream() for event emission.
        """
        from voxagent.tools.executor import execute_tool

        return await execute_tool(
            name=tool_call.name,
            params=tool_call.params,
            tools=self.tools,
            abort_signal=self.abort_controller.signal,
            tool_use_id=tool_call.id,
            deps=self.deps,
            run_id=self.run_id,
        )

    async def run_tool_loop(
        self,
        messages: list["Message"],
        max_iterations: int = 50,
    ) -> tuple[list["Message"], list[str], list["ToolMeta"]]:
        """Run the canonical tool loop, returning (messages, texts, metas).
        
        This is THE canonical tool loop implementation. DefaultStrategy
        delegates to this method. Other strategies may call this directly
        for their inner loops.
        
        Does NOT emit events - use run_tool_loop_stream() for event emission.
        """
        from voxagent.types.messages import Message as Msg
        from voxagent.types.run import ToolMeta

        assistant_texts: list[str] = []
        tool_metas: list[ToolMeta] = []
        working_messages = list(messages)  # Copy to avoid mutation

        for _ in range(max_iterations):
            if self.abort_controller.signal.aborted:
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

        return working_messages, assistant_texts, tool_metas

    async def call_llm_stream(
        self,
        messages: list["Message"],
        tools: list["ToolDefinition"] | None = None,
    ) -> AsyncIterator["StreamEventData"]:
        """Stream LLM call, yielding events.
        
        Yields TextDeltaEvent and other events.
        """
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            ProviderRequestChunk,
            TextDeltaChunk,
        )
        from voxagent.streaming.events import (
            ProviderRequestEvent,
            RunErrorEvent,
            TextDeltaEvent,
        )

        from voxagent.types.messages import Message as Msg
        
        effective_tools = tools if tools is not None else self.tools
        
        async for chunk in self.provider.stream(
            messages=[m if isinstance(m, Msg) else Msg(**m) for m in messages],
            system=self.system_prompt,
            tools=[t.to_openai_schema() for t in effective_tools] if effective_tools else None,
            abort_signal=self.abort_controller.signal,
        ):
            if isinstance(chunk, TextDeltaChunk):
                yield TextDeltaEvent(run_id=self.run_id, delta=chunk.delta)
            elif isinstance(chunk, ProviderRequestChunk):
                yield ProviderRequestEvent(run_id=self.run_id, body=chunk.body)
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
        """Stream the canonical tool loop, yielding all events.
        
        Yields:
            TextDeltaEvent (LLM streaming)
            ToolStartEvent (before each tool)
            ToolEndEvent (after each tool)
            ProviderRequestEvent
        """
        from voxagent.providers.base import (
            ErrorChunk,
            MessageEndChunk,
            ProviderRequestChunk,
            TextDeltaChunk,
            ToolUseChunk,
        )
        from voxagent.streaming.events import (
            ProviderRequestEvent,
            RunErrorEvent,
            TextDeltaEvent,
            ToolEndEvent,
            ToolOutputEvent,
            ToolStartEvent,
        )
        from voxagent.types.messages import Message as Msg
        from voxagent.types.messages import ToolCall

        working_messages = list(messages)

        for _ in range(max_iterations):
            if self.abort_controller.signal.aborted:
                break

            response_text = ""
            tool_calls: list[ToolCall] = []

            # Stream LLM response
            async for chunk in self.provider.stream(
                messages=[m if isinstance(m, Msg) else Msg(**m) for m in working_messages],
                system=self.system_prompt,
                tools=[t.to_openai_schema() for t in self.tools]
                if self.tools
                else None,
                abort_signal=self.abort_controller.signal,
            ):
                if isinstance(chunk, TextDeltaChunk):
                    response_text += chunk.delta
                    yield TextDeltaEvent(run_id=self.run_id, delta=chunk.delta)
                elif isinstance(chunk, ToolUseChunk):
                    tool_calls.append(chunk.tool_call)
                elif isinstance(chunk, ProviderRequestChunk):
                    yield ProviderRequestEvent(run_id=self.run_id, body=chunk.body)
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
    """Base class for agentic behavior patterns.

    A strategy encapsulates how an agent processes a prompt - whether that's
    a simple tool loop, reflection cycles, planning phases, or other patterns.

    Implementers must:
    - Implement execute() for non-streaming runs
    - Optionally override execute_stream() for true streaming (default wraps execute())
    - Handle abort signal at iteration boundaries
    """

    # Shared session storage for all strategies
    _shared_session_storage: Any = None
    _session_dir: str | None = None

    @classmethod
    def init_session_storage(cls, session_dir: str = "~/.samaritan/sessions") -> None:
        """Initialize shared session storage for all strategies.

        Args:
            session_dir: Directory to store sessions. Defaults to ~/.samaritan/sessions
        """
        from pathlib import Path
        from voxagent.session.storage import FileSessionStorage

        cls._session_dir = session_dir
        path = Path(session_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        cls._shared_session_storage = FileSessionStorage(path)

    @classmethod
    def get_session_storage(cls) -> Any:
        """Get the shared session storage, initializing if needed."""
        if cls._shared_session_storage is None:
            cls.init_session_storage()
        return cls._shared_session_storage

    @property
    def name(self) -> str:
        """Strategy name for logging/debugging."""
        return self.__class__.__name__

    def get_required_tools(self) -> list[Any]:
        """Return tools required by this strategy.

        Override in subclasses to auto-register strategy-specific tools.
        The Agent will call this during initialization and register the returned tools.

        Returns:
            List of ToolDefinition objects required by this strategy.
        """
        return []

    async def save_to_session(
        self,
        session_key: str,
        messages: list["Message"],
    ) -> None:
        """Save messages to session with strategy metadata.

        Args:
            session_key: The session identifier.
            messages: Messages to save.
        """
        from datetime import datetime, timezone
        from voxagent.session.model import Session

        storage = self.get_session_storage()
        if not storage:
            return

        session = await storage.load(session_key)
        if not session:
            session = Session.create(key=session_key)

        # Add strategy metadata to messages
        for msg in messages:
            if not hasattr(msg, 'metadata') or msg.metadata is None:
                msg.metadata = {}
            if 'strategy' not in msg.metadata:
                msg.metadata['strategy'] = self.name
                msg.metadata['timestamp'] = datetime.now(timezone.utc).isoformat()

        session.messages = messages
        await storage.save(session)

    @abstractmethod
    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the strategy behavior pattern.
        
        Must check context.abort_controller.signal.aborted at iteration boundaries.
        Must set StrategyResult.error for domain-level failures.
        May raise exceptions for infrastructure failures.
        """
        ...

    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the strategy with streaming events.
        
        Default implementation calls execute() and yields synthetic events.
        Override for true streaming support.
        """
        from voxagent.streaming.events import TextDeltaEvent
        
        # Default: run non-streaming and yield final events
        result = await self.execute(ctx)
        
        # Yield synthetic text events from collected texts
        for text in result.assistant_texts:
            yield TextDeltaEvent(run_id=ctx.run_id, delta=text)


__all__ = [
    "AgentStrategy",
    "StrategyContext",
    "StrategyResult",
]

