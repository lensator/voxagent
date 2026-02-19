"""Execution strategies for voxagent agents."""

from .base import AgentStrategy, StrategyContext, StrategyResult
from .default import DefaultStrategy
from .home import HomeOrchestratorStrategy
from .planning import PlanAndExecuteStrategy
from .react import ReActStrategy
from .reflection import ReflectionStrategy
from .retry import RetryStrategy

# Registry mapping strategy names to classes
STRATEGY_REGISTRY: dict[str, type[AgentStrategy]] = {
    "HomeOrchestratorStrategy": HomeOrchestratorStrategy,
    "DefaultStrategy": DefaultStrategy,
    "ReflectionStrategy": ReflectionStrategy,
    "PlanAndExecuteStrategy": PlanAndExecuteStrategy,
    "ReActStrategy": ReActStrategy,
    "RetryStrategy": RetryStrategy,
}


def get_strategy_class(name: str) -> type[AgentStrategy] | None:
    """Get strategy class by name from registry."""
    return STRATEGY_REGISTRY.get(name)


__all__ = [
    "AgentStrategy",
    "StrategyContext",
    "StrategyResult",
    "DefaultStrategy",
    "HomeOrchestratorStrategy",
    "ReflectionStrategy",
    "PlanAndExecuteStrategy",
    "ReActStrategy",
    "RetryStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
]
