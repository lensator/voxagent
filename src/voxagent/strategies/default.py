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
    
    def _build_initial_messages(self, ctx: StrategyContext) -> list["Message"]:
        """Build the initial messages list from context."""
        from voxagent.types.messages import Message
        
        messages: list[Message] = []
        
        # message_history is already prepended with summary and system prompt in Agent.run
        # Wait, the plan says the strategy should be responsible for building initial messages.
        # But Agent.run currently does some of it. 
        # For now, let's just use what's in the context.
        
        if ctx.message_history:
            messages.extend(ctx.message_history)
            
        messages.append(Message(role="user", content=ctx.prompt))
        return messages

    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the default tool loop.
        
        Args:
            ctx: Strategy context with LLM and tool access.
        
        Returns:
            StrategyResult with messages, assistant texts, and tool metadata.
        """
        messages = self._build_initial_messages(ctx)
        
        final_messages, texts, metas = await ctx.run_tool_loop(
            messages=messages,
            max_iterations=self._max_iterations,
        )
        
        return StrategyResult(
            messages=final_messages,
            assistant_texts=texts,
            tool_metas=metas,
            metadata={
                "strategy_name": self.name,
                "max_iterations": self._max_iterations,
            }
        )
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the default tool loop with streaming.
        
        Args:
            ctx: Strategy context with LLM and tool access.
        
        Yields:
            StreamEventData events during execution.
        """
        messages = self._build_initial_messages(ctx)
        
        async for event in ctx.run_tool_loop_stream(
            messages=messages,
            max_iterations=self._max_iterations,
        ):
            yield event



__all__ = ["DefaultStrategy"]

