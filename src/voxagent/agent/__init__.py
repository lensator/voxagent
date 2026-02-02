"""Agent core module for voxagent.

This subpackage provides:
- Agent class for managing AI agent interactions
- AbortController for abort signal management
- TimeoutHandler for timeout-based abort triggering
- Error recovery and failover handling
"""

from voxagent.agent.abort import (
    AbortController,
    AllProfilesExhausted,
    FailoverError,
    FailoverReason,
    TimeoutHandler,
    handle_context_overflow,
)
from voxagent.agent.core import Agent

# Import providers to ensure they are registered with the default registry
import voxagent.providers  # noqa: F401

__all__ = [
    "AbortController",
    "Agent",
    "AllProfilesExhausted",
    "FailoverError",
    "FailoverReason",
    "TimeoutHandler",
    "handle_context_overflow",
]

