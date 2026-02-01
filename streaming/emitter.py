"""Event emitter for streaming events.

Provides typed event emission for the agent lifecycle including
run, inference, tool, and context compaction events.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Awaitable
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StreamEvent(str, Enum):
    """Stream event types for the agent lifecycle.

    Based on algorithm specification from agent_algorithm.md.
    """

    # Lifecycle events
    RUN_START = "run_start"
    RUN_END = "run_end"
    RUN_ERROR = "run_error"

    # Inference events
    ASSISTANT_START = "assistant_start"
    TEXT_DELTA = "text_delta"
    ASSISTANT_END = "assistant_end"

    # Tool events
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"
    TOOL_END = "tool_end"

    # Context events
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"


# Type aliases for callbacks
EventCallback = Callable[[Any], None] | Callable[[Any], Awaitable[None]]
WildcardCallback = (
    Callable[[StreamEvent | str, Any], None]
    | Callable[[StreamEvent | str, Any], Awaitable[None]]
)


class EventEmitter:
    """Event emitter for typed streaming events.

    Thread-safe event emitter supporting both sync and async callbacks.
    Supports wildcard listeners that receive all events.
    """

    def __init__(self) -> None:
        """Initialize the event emitter with empty listener storage."""
        self._listeners: dict[StreamEvent | str, list[EventCallback]] = {}
        self._once_callbacks: set[EventCallback] = set()
        self._wildcard_listeners: list[WildcardCallback] = []
        self._lock = threading.Lock()

    def on(
        self, event_type: StreamEvent | str, callback: EventCallback
    ) -> Callable[[], None]:
        """Register a callback for an event type.

        Args:
            event_type: The event type to listen for.
            callback: The callback function to invoke when event is emitted.

        Returns:
            An unsubscribe function that removes this callback.
        """
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)

        def unsubscribe() -> None:
            self.off(event_type, callback)

        return unsubscribe

    def off(self, event_type: StreamEvent | str, callback: EventCallback) -> None:
        """Unregister a callback for an event type.

        Args:
            event_type: The event type to unregister from.
            callback: The callback to remove.
        """
        with self._lock:
            if event_type in self._listeners:
                try:
                    self._listeners[event_type].remove(callback)
                except ValueError:
                    pass  # Callback was not registered
            # Also remove from once callbacks if present
            self._once_callbacks.discard(callback)

    def once(
        self, event_type: StreamEvent | str, callback: EventCallback
    ) -> Callable[[], None]:
        """Register a one-time callback that auto-removes after first call.

        Args:
            event_type: The event type to listen for.
            callback: The callback function to invoke once.

        Returns:
            An unsubscribe function that removes this callback.
        """
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)
            self._once_callbacks.add(callback)

        def unsubscribe() -> None:
            self.off(event_type, callback)

        return unsubscribe

    def emit(self, event_type: StreamEvent | str, data: Any = None) -> None:
        """Emit an event synchronously to all registered listeners.

        Exceptions in callbacks are caught and logged, but do not prevent
        other callbacks from being called.

        Args:
            event_type: The event type to emit.
            data: Optional data payload to pass to callbacks.
        """
        # Get a snapshot of callbacks to call
        with self._lock:
            callbacks = list(self._listeners.get(event_type, []))
            wildcard_callbacks = list(self._wildcard_listeners)
            once_callbacks = self._once_callbacks.copy()

        # Call specific event callbacks
        for callback in callbacks:
            try:
                callback(data)
            except Exception:
                logger.exception(
                    "Exception in event callback for %s", event_type
                )

            # Remove if it was a once callback
            if callback in once_callbacks:
                self.off(event_type, callback)

        # Call wildcard callbacks
        for callback in wildcard_callbacks:
            try:
                callback(event_type, data)
            except Exception:
                logger.exception(
                    "Exception in wildcard callback for %s", event_type
                )

    async def emit_async(
        self, event_type: StreamEvent | str, data: Any = None
    ) -> None:
        """Emit an event asynchronously, awaiting async callbacks.

        Handles both sync and async callbacks. Exceptions in callbacks are
        caught and logged, but do not prevent other callbacks from being called.

        Args:
            event_type: The event type to emit.
            data: Optional data payload to pass to callbacks.
        """
        # Get a snapshot of callbacks to call
        with self._lock:
            callbacks = list(self._listeners.get(event_type, []))
            wildcard_callbacks = list(self._wildcard_listeners)
            once_callbacks = self._once_callbacks.copy()

        # Call specific event callbacks
        for callback in callbacks:
            try:
                result = callback(data)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Exception in async event callback for %s", event_type
                )

            # Remove if it was a once callback
            if callback in once_callbacks:
                self.off(event_type, callback)

        # Call wildcard callbacks
        for callback in wildcard_callbacks:
            try:
                result = callback(event_type, data)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Exception in async wildcard callback for %s", event_type
                )

    def on_any(self, callback: WildcardCallback) -> Callable[[], None]:
        """Register a wildcard callback that receives all events.

        Args:
            callback: The callback function receiving (event_type, data).

        Returns:
            An unsubscribe function that removes this callback.
        """
        with self._lock:
            self._wildcard_listeners.append(callback)

        def unsubscribe() -> None:
            self.off_any(callback)

        return unsubscribe

    def off_any(self, callback: WildcardCallback) -> None:
        """Unregister a wildcard callback.

        Args:
            callback: The wildcard callback to remove.
        """
        with self._lock:
            try:
                self._wildcard_listeners.remove(callback)
            except ValueError:
                pass  # Callback was not registered

    def clear(self, event_type: StreamEvent | str | None = None) -> None:
        """Clear all listeners, or listeners for a specific event.

        Args:
            event_type: If provided, clear only listeners for this event.
                       If None, clear all listeners including wildcards.
        """
        with self._lock:
            if event_type is None:
                self._listeners.clear()
                self._once_callbacks.clear()
                self._wildcard_listeners.clear()
            else:
                if event_type in self._listeners:
                    # Remove any once callbacks for this event
                    for cb in self._listeners[event_type]:
                        self._once_callbacks.discard(cb)
                    del self._listeners[event_type]

    def listener_count(self, event_type: StreamEvent | str | None = None) -> int:
        """Return the number of listeners for an event type.

        Args:
            event_type: The event type to count listeners for.
                       If None, returns total count of all listeners.

        Returns:
            The number of listeners registered for the event type.
        """
        with self._lock:
            if event_type is None:
                total = sum(len(cbs) for cbs in self._listeners.values())
                total += len(self._wildcard_listeners)
                return total
            return len(self._listeners.get(event_type, []))

    def has_listeners(self, event_type: StreamEvent | str) -> bool:
        """Return True if the event type has any listeners.

        Args:
            event_type: The event type to check.

        Returns:
            True if there are listeners for this event type.
        """
        with self._lock:
            return len(self._listeners.get(event_type, [])) > 0

