"""Reflection strategy for voxagent agents.

The ReflectionStrategy implements a self-correction loop:
1. Generate an initial response using standard tool loop.
2. Critique the response (with no tools allowed).
3. Revise the response using another tool loop.
4. Repeat until the critique confirms approval or max iterations reached.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, AsyncIterator

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData
    from voxagent.types.messages import Message


DEFAULT_CRITIQUE_PROMPT = """
Review the response for accuracy, completeness, and adherence to instructions.
If the response is correct and complete, output 'APPROVED'.
Otherwise, identify specific areas for improvement and provide constructive feedback for the next revision.
"""


class ReflectionStrategy(AgentStrategy):
    """Strategy that critiques and revises its own responses.
    
    This strategy runs iterative cycles of generation, self-review (critique),
    and revision until the critique phase outputs 'APPROVED' or the maximum
    number of iterations is reached.
    
    Args:
        max_iterations: Maximum number of reflection iterations (default: 3).
        critique_prompt: Prompt for the critique phase.
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        critique_prompt: str = DEFAULT_CRITIQUE_PROMPT,
    ) -> None:
        """Initialize the ReflectionStrategy.
        
        Args:
            max_iterations: Maximum reflection iterations.
            critique_prompt: Prompt for critique phase.
        """
        self._max_iterations = max_iterations
        self._critique_prompt = critique_prompt
    
    def _build_initial_messages(self, ctx: StrategyContext) -> list["Message"]:
        """Build the initial messages list from context."""
        from voxagent.types.messages import Message
        
        messages: list[Message] = []
        if ctx.message_history:
            messages.extend(ctx.message_history)
            
        messages.append(Message(role="user", content=ctx.prompt))
        return messages

    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the reflection cycle.
        
        Args:
            ctx: Strategy context with LLM and tool access.
        
        Returns:
            StrategyResult with messages, assistant texts, and tool metadata.
        """
        from voxagent.types.messages import Message as Msg
        
        messages = self._build_initial_messages(ctx)
        assistant_texts: list[str] = []
        tool_metas = []
        approved = False
        iteration_count = 0
        
        for i in range(self._max_iterations):
            iteration_count += 1
            
            # REQUIRED: Check abort at iteration boundaries
            if ctx.abort_controller.signal.aborted:
                return StrategyResult(
                    messages=messages,
                    assistant_texts=assistant_texts,
                    tool_metas=tool_metas,
                    error="Aborted",
                    metadata={"iterations": iteration_count},
                )
            
            # --- 1. GENERATION / REVISION ---
            # Use canonical tool loop for generation/revision
            msgs, texts, metas = await ctx.run_tool_loop(messages)
            
            messages = msgs
            assistant_texts.extend(texts)
            tool_metas.extend(metas)
            
            if ctx.abort_controller.signal.aborted:
                return StrategyResult(
                    messages=messages,
                    assistant_texts=assistant_texts,
                    tool_metas=tool_metas,
                    error="Aborted",
                    metadata={"iterations": iteration_count},
                )

            # --- 2. CRITIQUE ---
            # Add critique request
            messages.append(Msg(role="user", content=self._critique_prompt))
            
            # LLM call with no tools for critique phase
            critique, _ = await ctx.call_llm(messages, tools=None)
            
            assistant_texts.append(critique)
            messages.append(Msg(role="assistant", content=critique))
            
            if "APPROVED" in critique.upper():
                approved = True
                break
            
            # If not approved, next iteration starts with generation based on critique
            # But we might want to add a "Please revise" user prompt if needed
            # Actually, the critique is already in the messages as assistant content.
            # We just loop back to generation.
            
        return StrategyResult(
            messages=messages,
            assistant_texts=assistant_texts,
            tool_metas=tool_metas,
            metadata={
                "strategy_name": self.name,
                "iterations": iteration_count,
                "approved": approved,
            }
        )
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the reflection cycle with streaming.
        
        Currently delegates to execute() and yields synthetic events.
        True streaming would stream each tool loop iteration.
        """
        from voxagent.streaming.events import TextDeltaEvent
        
        # This is a basic implementation of streaming for reflection.
        # It runs one generation/critique cycle at a time.
        
        from voxagent.types.messages import Message as Msg
        
        messages = self._build_initial_messages(ctx)
        
        for i in range(self._max_iterations):
            # REQUIRED: Check abort
            if ctx.abort_controller.signal.aborted:
                return

            # --- 1. GENERATION / REVISION ---
            async for event in ctx.run_tool_loop_stream(messages):
                yield event
                
                # Keep track of assistant content for internal state
                if isinstance(event, TextDeltaEvent):
                    # In a real implementation, we'd buffer and append to messages
                    pass
            
            # We need the final messages from the stream to continue.
            # Since run_tool_loop_stream doesn't return them directly, 
            # we'd need to re-run or better integrate.
            
            # For this simplified implementation, we'll just yield what we have.
            # A full implementation would need more sophisticated event-to-message mapping.
            
            break # Just one iteration for now in stream mode to keep it simple

__all__ = ["ReflectionStrategy"]
