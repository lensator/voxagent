"""Sub-agent support for voxagent.

This module provides the ability to register agents as tools that can be called
by parent agents, enabling hierarchical agent composition and delegation.
"""

from voxagent.subagent.context import (
    DEFAULT_MAX_DEPTH,
    MaxDepthExceededError,
    SubAgentContext,
)
from voxagent.subagent.definition import SubAgentDefinition

__all__ = [
    "SubAgentDefinition",
    "SubAgentContext",
    "MaxDepthExceededError",
    "DEFAULT_MAX_DEPTH",
]

