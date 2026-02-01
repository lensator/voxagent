"""Security events for voxagent."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class SecurityEvent(Enum):
    """Security-related events."""

    SECRET_REDACTED = "secret_redacted"
    CREDENTIAL_ACCESSED = "credential_accessed"
    PATTERN_MATCHED = "pattern_matched"


class SecurityEventEmitter:
    """Emits security events to listeners."""

    def __init__(self) -> None:
        """Initialize an empty event emitter."""
        self._listeners: dict[
            SecurityEvent, list[Callable[[dict[str, Any]], None]]
        ] = {}

    def on(
        self,
        event: SecurityEvent,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        """Register a listener for an event.

        Args:
            event: The security event type to listen for.
            callback: Function to call when the event is emitted.

        Returns:
            An unsubscribe function that removes the listener.
        """
        if event not in self._listeners:
            self._listeners[event] = []

        self._listeners[event].append(callback)

        def unsubscribe() -> None:
            if event in self._listeners and callback in self._listeners[event]:
                self._listeners[event].remove(callback)

        return unsubscribe

    def emit(self, event: SecurityEvent, data: dict[str, Any]) -> None:
        """Emit an event to all registered listeners.

        Args:
            event: The security event type to emit.
            data: Event data to pass to listeners.
        """
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(data)

    def off(
        self,
        event: SecurityEvent,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Remove a specific listener.

        Args:
            event: The security event type.
            callback: The callback to remove.
        """
        if event in self._listeners and callback in self._listeners[event]:
            self._listeners[event].remove(callback)

