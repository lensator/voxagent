"""Secret registry for voxagent."""

from __future__ import annotations

import re
import threading
from typing import Pattern


class SecretRegistry:
    """Thread-safe registry for secret patterns and values."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._patterns: list[Pattern[str]] = []
        self._values: dict[str, str] = {}  # name -> value
        self._lock = threading.Lock()

    def register_pattern(self, pattern: str) -> None:
        """Register a regex pattern to detect secrets.

        Args:
            pattern: A regex pattern string to match secrets.
        """
        with self._lock:
            compiled = re.compile(pattern)
            self._patterns.append(compiled)

    def register_value(self, name: str, value: str) -> None:
        """Register a specific value to redact.

        Args:
            name: A name/key for the secret.
            value: The actual secret value to detect.
        """
        with self._lock:
            self._values[name] = value

    def contains_secret(self, text: str) -> bool:
        """Check if text contains any registered secrets.

        Args:
            text: The text to check for secrets.

        Returns:
            True if any registered secret pattern or value is found.
        """
        with self._lock:
            # Check patterns
            for pattern in self._patterns:
                if pattern.search(text):
                    return True
            # Check values
            for value in self._values.values():
                if value in text:
                    return True
            return False

    def find_secrets(self, text: str) -> list[tuple[str, int, int]]:
        """Find all secrets in text.

        Args:
            text: The text to search for secrets.

        Returns:
            List of (match, start, end) tuples for each found secret.
        """
        results: list[tuple[str, int, int]] = []
        with self._lock:
            # Find pattern matches
            for pattern in self._patterns:
                for match in pattern.finditer(text):
                    results.append((match.group(), match.start(), match.end()))
            # Find value matches
            for value in self._values.values():
                start = 0
                while True:
                    idx = text.find(value, start)
                    if idx == -1:
                        break
                    results.append((value, idx, idx + len(value)))
                    start = idx + 1

        # Sort by position
        results.sort(key=lambda x: x[1])
        return results

