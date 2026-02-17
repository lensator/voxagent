"""Stream event data types.

This module defines typed event payloads for the streaming event system.
Each event type is a Pydantic model that can be serialized/deserialized.

Event Categories:
- Lifecycle Events: RunStartEvent, RunEndEvent, RunErrorEvent
- Inference Events: AssistantStartEvent, TextDeltaEvent, AssistantEndEvent
- Tool Events: ToolStartEvent, ToolOutputEvent, ToolEndEvent
- Context Events: CompactionStartEvent, CompactionEndEvent
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from voxagent.types import Message, ToolCall, ToolResult


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Base Event
# =============================================================================


class BaseEvent(BaseModel):
    """Base class for all stream events.

    Attributes:
        run_id: Unique identifier for the current agent run.
    """

    run_id: str


# =============================================================================
# Lifecycle Events
# =============================================================================


class RunStartEvent(BaseEvent):
    """Event emitted when an agent run starts.

    Attributes:
        run_id: Unique identifier for the run.
        session_key: Session key for the run.
        timestamp: When the run started.
        event_type: Discriminator field for type narrowing.
    """

    session_key: str
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: Literal["run_start"] = "run_start"


class RunEndEvent(BaseEvent):
    """Event emitted when an agent run ends.

    Attributes:
        run_id: Unique identifier for the run.
        messages: Final list of messages from the run.
        aborted: Whether the run was aborted.
        timed_out: Whether the run timed out.
        timestamp: When the run ended.
        event_type: Discriminator field for type narrowing.
    """

    messages: list[Message]
    aborted: bool = False
    timed_out: bool = False
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: Literal["run_end"] = "run_end"


class RunErrorEvent(BaseEvent):
    """Event emitted when an agent run encounters an error.

    Attributes:
        run_id: Unique identifier for the run.
        error: Error message.
        timestamp: When the error occurred.
        event_type: Discriminator field for type narrowing.
    """

    error: str
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: Literal["run_error"] = "run_error"


# =============================================================================
# Inference Events
# =============================================================================


class AssistantStartEvent(BaseEvent):
    """Event emitted when the assistant starts generating a response.

    Attributes:
        run_id: Unique identifier for the run.
        event_type: Discriminator field for type narrowing.
    """

    event_type: Literal["assistant_start"] = "assistant_start"


class TextDeltaEvent(BaseEvent):
    """Event emitted for each text chunk from the assistant.

    Attributes:
        run_id: Unique identifier for the run.
        delta: The text chunk.
        module: The source module/stage (default: 'assistant').
        event_type: Discriminator field for type narrowing.
    """

    delta: str
    module: str = "assistant"
    event_type: Literal["text_delta"] = "text_delta"


class AssistantEndEvent(BaseEvent):
    """Event emitted when the assistant finishes generating a response.

    Attributes:
        run_id: Unique identifier for the run.
        message: The complete assistant message.
        event_type: Discriminator field for type narrowing.
    """

    message: Message
    event_type: Literal["assistant_end"] = "assistant_end"


class ProviderRequestEvent(BaseEvent):
    """Event emitted when a raw request is sent to the provider.

    Attributes:
        run_id: Unique identifier for the run.
        body: The raw request body (dict or string).
        event_type: Discriminator field for type narrowing.
    """

    body: Any
    event_type: Literal["provider_request"] = "provider_request"


class InternalThoughtEvent(BaseEvent):
    """Event emitted for internal agent reasoning or background processing.

    Attributes:
        run_id: Unique identifier for the run.
        content: The thought or background process output.
        module: The name of the module or stage producing the thought.
        event_type: Discriminator field for type narrowing.
    """

    content: str
    module: str = "agent"
    event_type: Literal["internal_thought"] = "internal_thought"


# =============================================================================
# Tool Events
# =============================================================================


class ToolStartEvent(BaseEvent):
    """Event emitted when a tool execution starts.

    Attributes:
        run_id: Unique identifier for the run.
        tool_call: The tool call being executed.
        event_type: Discriminator field for type narrowing.
    """

    tool_call: ToolCall
    event_type: Literal["tool_start"] = "tool_start"


class ToolOutputEvent(BaseEvent):
    """Event emitted for streaming tool output.

    Attributes:
        run_id: Unique identifier for the run.
        tool_call_id: ID of the tool call.
        delta: Output chunk from the tool.
        event_type: Discriminator field for type narrowing.
    """

    tool_call_id: str
    delta: str
    event_type: Literal["tool_output"] = "tool_output"


class ToolEndEvent(BaseEvent):
    """Event emitted when a tool execution ends.

    Attributes:
        run_id: Unique identifier for the run.
        tool_call_id: ID of the tool call.
        result: The tool result.
        event_type: Discriminator field for type narrowing.
    """

    tool_call_id: str
    result: ToolResult
    event_type: Literal["tool_end"] = "tool_end"


# =============================================================================
# Context Events
# =============================================================================


class CompactionStartEvent(BaseEvent):
    """Event emitted when context compaction starts.

    Attributes:
        run_id: Unique identifier for the run.
        message_count: Number of messages before compaction.
        token_count: Number of tokens before compaction.
        event_type: Discriminator field for type narrowing.
    """

    message_count: int
    token_count: int
    event_type: Literal["compaction_start"] = "compaction_start"


class CompactionEndEvent(BaseEvent):
    """Event emitted when context compaction ends.

    Attributes:
        run_id: Unique identifier for the run.
        messages_removed: Number of messages removed.
        tokens_saved: Number of tokens saved.
        event_type: Discriminator field for type narrowing.
    """

    messages_removed: int
    tokens_saved: int
    event_type: Literal["compaction_end"] = "compaction_end"


# =============================================================================
# Union Type
# =============================================================================


StreamEventData = (
    RunStartEvent
    | RunEndEvent
    | RunErrorEvent
    | AssistantStartEvent
    | TextDeltaEvent
    | AssistantEndEvent
    | ProviderRequestEvent
    | InternalThoughtEvent
    | ToolStartEvent
    | ToolOutputEvent
    | ToolEndEvent
    | CompactionStartEvent
    | CompactionEndEvent
)


__all__ = [
    "AssistantEndEvent",
    "AssistantStartEvent",
    "BaseEvent",
    "CompactionEndEvent",
    "CompactionStartEvent",
    "InternalThoughtEvent",
    "ProviderRequestEvent",
    "RunEndEvent",
    "RunErrorEvent",
    "RunStartEvent",
    "StreamEventData",
    "TextDeltaEvent",
    "ToolEndEvent",
    "ToolOutputEvent",
    "ToolStartEvent",
]

