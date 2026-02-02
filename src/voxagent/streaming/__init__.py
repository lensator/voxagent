"""Event streaming infrastructure.

This subpackage provides:
- StreamResult wrapper for streaming responses
- Event emitter for typed events
- Lifecycle events (RUN_START, RUN_END, etc.)
- Tool events (TOOL_START, TOOL_OUTPUT, TOOL_END)
- Typed event data models
"""

from voxagent.streaming.emitter import (
    EventCallback,
    EventEmitter,
    StreamEvent,
    WildcardCallback,
)
from voxagent.streaming.events import (
    AssistantEndEvent,
    AssistantStartEvent,
    BaseEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    RunEndEvent,
    RunErrorEvent,
    RunStartEvent,
    StreamEventData,
    TextDeltaEvent,
    ToolEndEvent,
    ToolOutputEvent,
    ToolStartEvent,
)

__all__ = [
    "AssistantEndEvent",
    "AssistantStartEvent",
    "BaseEvent",
    "CompactionEndEvent",
    "CompactionStartEvent",
    "EventCallback",
    "EventEmitter",
    "RunEndEvent",
    "RunErrorEvent",
    "RunStartEvent",
    "StreamEvent",
    "StreamEventData",
    "TextDeltaEvent",
    "ToolEndEvent",
    "ToolOutputEvent",
    "ToolStartEvent",
    "WildcardCallback",
]

