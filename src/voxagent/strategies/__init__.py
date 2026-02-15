"""Strategy system for voxagent agents.

This module provides pluggable execution strategies for agents:

Strategies:
- DefaultStrategy: Standard tool-loop behavior (backwards compatible)

Base Classes:
- AgentStrategy: Abstract base class for custom strategies
- StrategyContext: Context with LLM and tool access helpers
- StrategyResult: Result dataclass from strategy execution

Usage:
    from voxagent import Agent
    from voxagent.strategies import DefaultStrategy
    
    agent = Agent(
        model="openai:gpt-4o",
        strategy=DefaultStrategy(max_iterations=30),
    )
    result = await agent.run("Hello!")

Creating Custom Strategies:
    from voxagent.strategies import AgentStrategy, StrategyContext, StrategyResult
    
    class MyStrategy(AgentStrategy):
        async def execute(self, ctx, messages):
            # Custom logic here
            return await ctx.run_tool_loop(messages)
        
        async def execute_stream(self, ctx, messages):
            async for event in ctx.run_tool_loop_stream(messages):
                yield event
"""

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult
from voxagent.strategies.default import DefaultStrategy

__all__ = [
    # Base classes
    "AgentStrategy",
    "StrategyContext",
    "StrategyResult",
    # Strategies
    "DefaultStrategy",
]

