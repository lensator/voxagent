"""voxagent - A lightweight, model-agnostic LLM provider abstraction.

voxagent provides:
- Multi-Provider: Unified interface for OpenAI, Anthropic, Google, Groq, Ollama
- Streaming: Typed StreamChunk union (TextDelta, ToolUse, MessageEnd, Error)
- Tool System: @tool decorator, typed definitions, abort signal propagation
- MCP Integration: First-class Model Context Protocol support
- Sub-Agent Support: Hierarchical agent composition with depth-limited delegation
- Session Management: File-based sessions with context compaction

Quick Start:
    >>> from voxagent import Agent
    >>> agent = Agent(model="openai:gpt-4o")
    >>> result = await agent.run("Hello!")

With Tools:
    >>> from voxagent import Agent
    >>> from voxagent.tools import tool
    >>>
    >>> @tool()
    ... def get_weather(city: str) -> str:
    ...     '''Get weather for a city.'''
    ...     return f"Sunny in {city}"
    >>>
    >>> agent = Agent(model="anthropic:claude-3-5-sonnet", tools=[get_weather])
    >>> result = await agent.run("What's the weather in Paris?")

Streaming:
    >>> from voxagent import Agent
    >>> from voxagent.providers import TextDeltaChunk
    >>>
    >>> agent = Agent(model="openai:gpt-4o")
    >>> async for chunk in agent.stream("Tell me a story"):
    ...     if isinstance(chunk, TextDeltaChunk):
    ...         print(chunk.delta, end="")

For more information, see: https://github.com/voxdomus/voxagent
"""

from ._version import __version__, __version_info__

# =============================================================================
# Lazy imports for top-level convenience
# =============================================================================
# We use __getattr__ to avoid importing the full dependency chain on module load.
# This keeps `import voxagent` fast and allows users to import only what they need.


def __getattr__(name: str) -> object:
    """Lazy import for top-level classes."""
    # Core Agent class
    if name == "Agent":
        from .agent import Agent

        return Agent

    # Provider base classes and chunks
    if name in (
        "BaseProvider",
        "StreamChunk",
        "TextDeltaChunk",
        "ToolUseChunk",
        "MessageEndChunk",
        "ErrorChunk",
        "AbortSignal",
    ):
        from . import providers

        return getattr(providers, name)

    # Tool system
    if name in ("tool", "ToolDefinition", "ToolContext"):
        from . import tools

        return getattr(tools, name)

    # Message types
    if name in ("Message", "ToolCall", "ToolResult"):
        from . import types

        return getattr(types, name)

    # Sub-agent support
    if name == "SubAgentDefinition":
        from .subagent import SubAgentDefinition

        return SubAgentDefinition
    if name == "SubAgentContext":
        from .subagent import SubAgentContext

        return SubAgentContext
    if name == "MaxDepthExceededError":
        from .subagent import MaxDepthExceededError

        return MaxDepthExceededError

    # MCP
    if name == "MCPServerManager":
        from .mcp import MCPServerManager

        return MCPServerManager

    # Registry
    if name in ("ProviderRegistry", "get_default_registry"):
        from . import providers

        return getattr(providers, name)

    # Strategies
    if name in (
        "AgentStrategy",
        "StrategyContext",
        "StrategyResult",
        "DefaultStrategy",
        "ReflectionStrategy",
        "PlanAndExecuteStrategy",
        "ReActStrategy",
        "RetryStrategy",
    ):
        from . import strategies

        return getattr(strategies, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Core
    "Agent",
    # Providers
    "BaseProvider",
    "StreamChunk",
    "TextDeltaChunk",
    "ToolUseChunk",
    "MessageEndChunk",
    "ErrorChunk",
    "AbortSignal",
    "ProviderRegistry",
    "get_default_registry",
    # Tools
    "tool",
    "ToolDefinition",
    "ToolContext",
    # Types
    "Message",
    "ToolCall",
    "ToolResult",
    # Sub-agents
    "SubAgentDefinition",
    "SubAgentContext",
    "MaxDepthExceededError",
    # MCP
    "MCPServerManager",
    # Strategies
    "AgentStrategy",
    "StrategyContext",
    "StrategyResult",
    "DefaultStrategy",
    "ReflectionStrategy",
    "PlanAndExecuteStrategy",
    "ReActStrategy",
    "RetryStrategy",
]


