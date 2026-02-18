"""Retry strategy for voxagent agents.

The RetryStrategy is a wrapper strategy that:
1. Executes an inner strategy.
2. If the inner strategy returns an error, retries execution up to a limit.
3. Supports custom failure detection logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData


class RetryStrategy(AgentStrategy):
    """Strategy that retries another strategy on failure.
    
    This is a meta-strategy that wraps another AgentStrategy and re-runs it
    if it returns an error in StrategyResult.
    
    Args:
        inner_strategy: The strategy to wrap and retry.
        max_retries: Maximum number of retries (default: 3).
        failure_detector: Optional callable to detect failures in StrategyResult.
            If None, checks result.error.
    """
    
    def __init__(
        self,
        inner_strategy: AgentStrategy,
        max_retries: int = 3,
        failure_detector: Callable[[StrategyResult], str | None] | None = None,
    ) -> None:
        """Initialize the RetryStrategy.
        
        Args:
            inner_strategy: The strategy to retry.
            max_retries: Maximum number of retries.
            failure_detector: Optional failure detection logic.
        """
        self._inner_strategy = inner_strategy
        self._max_retries = max_retries
        self._failure_detector = failure_detector or self._default_failure_detector
    
    @staticmethod
    def _default_failure_detector(result: StrategyResult) -> str | None:
        """Detect failures in StrategyResult."""
        if result.error:
            return result.error
        
        # Check for tool failures in metas
        for meta in result.tool_metas:
            if not meta.success:
                return f"Tool {meta.tool_name} failed: {meta.error}"
        
        return None

    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the inner strategy with retries.
        
        Args:
            ctx: Strategy context.
        
        Returns:
            StrategyResult from the last attempt.
        """
        last_result = None
        attempts = 0
        
        # We loop max_retries + 1 (initial attempt + retries)
        for i in range(self._max_retries + 1):
            attempts += 1
            
            # REQUIRED: Check abort between retries
            if ctx.abort_controller.signal.aborted:
                return last_result or StrategyResult(
                    messages=[],
                    assistant_texts=[],
                    error="Aborted",
                    metadata={"attempts": attempts},
                )
            
            # If this is a retry, we might want to modify the prompt or messages
            # but usually the state is handled by the inner strategy.
            # Some strategies might be stateful in the context, but 
            # StrategyContext is usually fresh for each Run.
            
            last_result = await self._inner_strategy.execute(ctx)
            
            error = self._failure_detector(last_result)
            if not error:
                # Success!
                last_result.metadata["attempts"] = attempts
                return last_result
            
            # Failure detected, will retry if attempts remaining
            
        # All retries exhausted
        if last_result:
            last_result.metadata["attempts"] = attempts
            
        return last_result or StrategyResult(
            messages=[],
            assistant_texts=[],
            error="Retries exhausted",
            metadata={"attempts": attempts}
        )
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the retry cycle with streaming.
        
        Note: True streaming with retries is complex because it would mean
        re-emitting events on each retry. The Agent handles RunStart/End,
        so we yield events from the current attempt.
        """
        # This implementation re-emits for EACH attempt.
        attempts = 0
        for i in range(self._max_retries + 1):
            attempts += 1
            
            # REQUIRED: Check abort
            if ctx.abort_controller.signal.aborted:
                return

            # Note: We can't easily capture the result from the stream to detect failure
            # unless the stream helper provides it. 
            # For now, we'll just stream the first attempt.
            
            async for event in self._inner_strategy.execute_stream(ctx):
                yield event
            
            # Detect failure to retry? Complex with streaming.
            # Simplified for now: just one attempt in stream mode.
            break

__all__ = ["RetryStrategy"]
