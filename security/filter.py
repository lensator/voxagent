"""Redaction filters for voxagent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from voxagent.security.events import SecurityEvent, SecurityEventEmitter
from voxagent.security.registry import SecretRegistry

if TYPE_CHECKING:
    from voxagent.types.messages import Message


class RedactionFilter:
    """Filters secrets from text."""

    def __init__(
        self,
        registry: SecretRegistry,
        placeholder: str = "[REDACTED]",
        event_emitter: SecurityEventEmitter | None = None,
    ) -> None:
        """Initialize a redaction filter.

        Args:
            registry: The secret registry to use for detection.
            placeholder: The text to replace secrets with.
            event_emitter: Optional event emitter for security events.
        """
        self.registry = registry
        self.placeholder = placeholder
        self.event_emitter = event_emitter

    def redact(self, text: str) -> str:
        """Redact all secrets from text.

        Args:
            text: The text to redact secrets from.

        Returns:
            Text with all secrets replaced by the placeholder.
        """
        secrets = self.registry.find_secrets(text)
        if not secrets:
            return text

        # Emit events for each secret found
        if self.event_emitter:
            for match, start, end in secrets:
                # Emit SECRET_REDACTED event
                self.event_emitter.emit(
                    SecurityEvent.SECRET_REDACTED,
                    {"match": match, "start": start, "end": end},
                )
                # Emit PATTERN_MATCHED event
                self.event_emitter.emit(
                    SecurityEvent.PATTERN_MATCHED,
                    {"match": match, "start": start, "end": end},
                )

        # Replace from end to start to preserve positions
        result = text
        for match, start, end in reversed(secrets):
            result = result[:start] + self.placeholder + result[end:]

        return result

    def redact_message(self, message: Message) -> Message:
        """Redact secrets from a message.

        Args:
            message: The message to redact secrets from.

        Returns:
            A new message with redacted content, or the original if unchanged.
        """
        from voxagent.types.messages import Message as MessageClass

        # Handle None or non-string content
        if message.content is None or not isinstance(message.content, str):
            return message

        redacted_content = self.redact(message.content)
        if redacted_content == message.content:
            return message

        # Create new message with redacted content
        return MessageClass(
            role=message.role,
            content=redacted_content,
            tool_calls=message.tool_calls,
        )


class StreamFilter:
    """Streaming-aware redaction with buffering."""

    def __init__(
        self,
        registry: SecretRegistry,
        buffer_size: int = 100,
    ) -> None:
        """Initialize a stream filter.

        Args:
            registry: The secret registry to use for detection.
            buffer_size: Number of characters to buffer for partial match detection.
        """
        self.registry = registry
        self.buffer_size = buffer_size
        self._buffer: str = ""

    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk, buffering for potential secrets.

        Args:
            chunk: The chunk of text to process.

        Returns:
            The safe portion of text that can be output.
        """
        self._buffer += chunk

        # If buffer is smaller than buffer_size, hold everything
        if len(self._buffer) <= self.buffer_size:
            return ""

        # Find safe output point (keep buffer_size chars for potential matches)
        safe_end = len(self._buffer) - self.buffer_size
        output = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]

        # Redact the output portion
        return self._redact_text(output)

    def flush(self) -> str:
        """Flush remaining buffer.

        Returns:
            The remaining buffered text, redacted as needed.
        """
        output = self._buffer
        self._buffer = ""

        # Redact remaining buffer
        return self._redact_text(output)

    def _redact_text(self, text: str) -> str:
        """Redact secrets from text.

        Args:
            text: The text to redact.

        Returns:
            The redacted text.
        """
        if not text:
            return text

        secrets = self.registry.find_secrets(text)
        if not secrets:
            return text

        result = text
        for match, start, end in reversed(secrets):
            result = result[:start] + "[REDACTED]" + result[end:]

        return result

