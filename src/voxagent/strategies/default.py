"""Default strategy for voxagent agents.

The DefaultStrategy implements the standard tool-loop behavior:
1. Call LLM with messages
2. If tool calls returned, execute them and add results
3. Repeat until no tool calls
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData
    from voxagent.types.messages import Message


class DefaultStrategy(AgentStrategy):
    """Default strategy that implements the standard tool loop.
    
    This strategy delegates to StrategyContext.run_tool_loop() which is
    the canonical implementation of the tool loop. This ensures 100%
    backwards compatibility with the existing Agent behavior.
    
    Args:
        max_iterations: Maximum number of LLM calls in the tool loop.
            Default is 50.
    """
    
    def __init__(self, max_iterations: int = 50) -> None:
        """Initialize the DefaultStrategy.
        
        Args:
            max_iterations: Maximum tool loop iterations.
        """
        self._max_iterations = max_iterations
    
    async def execute(
        self,
        ctx: StrategyContext,
        messages: list["Message"],
    ) -> StrategyResult:
        """Execute the default tool loop.
        
        Args:
            ctx: Strategy context with LLM and tool access.
            messages: Initial messages for the conversation.
        
        Returns:
            StrategyResult with messages, assistant texts, and tool metadata.
        """
        result = await ctx.run_tool_loop(
            messages=messages,
            max_iterations=self._max_iterations,
        )
        
        # Add strategy metadata
        result.metadata["strategy_name"] = self.name
        result.metadata["max_iterations"] = self._max_iterations
        
        return result
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
        messages: list["Message"],
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the default tool loop with streaming.
        
        Args:
            ctx: Strategy context with LLM and tool access.
            messages: Initial messages for the conversation.
        
        Yields:
            StreamEventData events during execution.
        """
        async for event in ctx.run_tool_loop_stream(
            messages=messages,
            max_iterations=self._max_iterations,
        ):
            yield event


__all__ = ["DefaultStrategy"]

