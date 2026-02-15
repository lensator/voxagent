"""Session model for voxagent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from voxagent.types.messages import Message


class Session(BaseModel):
    """Represents a conversation session."""

    id: str = Field(..., description="Unique session ID (UUID)")
    key: str = Field(..., description="Session key for resolution")
    messages: list[Message] = Field(default_factory=list)
    summary: str | None = Field(None, description="Current summary of the conversation")
    summary_metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata about the summary")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_message(self, message: Message) -> None:
        """Add a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages, optionally limited to last N."""
        if limit is None:
            return list(self.messages)
        return list(self.messages[-limit:])

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "key": self.key,
            "messages": [m.model_dump() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize from dictionary."""
        messages = [Message(**m) for m in data.get("messages", [])]
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            key=data["key"],
            messages=messages,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=updated_at or datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> Session:
        """Create a new session with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            key=key,
            **kwargs,
        )


def resolve_session_key(
    user_id: str | None = None,
    channel: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Generate a session key from components.

    Examples:
        resolve_session_key(user_id="u123") -> "user:u123"
        resolve_session_key(channel="general") -> "channel:general"
        resolve_session_key(user_id="u123", channel="general") -> "user:u123:channel:general"
    """
    parts = []

    if user_id and user_id.strip():
        parts.append(f"user:{user_id}")
    if channel and channel.strip():
        parts.append(f"channel:{channel}")
    if thread_id and thread_id.strip():
        parts.append(f"thread:{thread_id}")

    if not parts:
        raise ValueError("At least one of user_id, channel, or thread_id must be provided")

    return ":".join(parts)

