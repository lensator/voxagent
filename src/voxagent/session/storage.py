"""Session storage backends for voxagent."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxagent.session.model import Session


class SessionStorage(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    async def load(self, session_key: str) -> "Session | None":
        """Load a session by key. Returns None if not found."""

    @abstractmethod
    async def save(self, session: "Session") -> None:
        """Save a session."""

    @abstractmethod
    async def delete(self, session_key: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""

    @abstractmethod
    async def list_keys(self) -> list[str]:
        """List all session keys."""

    @abstractmethod
    async def exists(self, session_key: str) -> bool:
        """Check if a session exists."""


class InMemorySessionStorage(SessionStorage):
    """In-memory storage for testing."""

    def __init__(self) -> None:
        self._sessions: dict[str, "Session"] = {}

    async def load(self, session_key: str) -> "Session | None":
        return self._sessions.get(session_key)

    async def save(self, session: "Session") -> None:
        self._sessions[session.key] = session

    async def delete(self, session_key: str) -> bool:
        if session_key in self._sessions:
            del self._sessions[session_key]
            return True
        return False

    async def list_keys(self) -> list[str]:
        return list(self._sessions.keys())

    async def exists(self, session_key: str) -> bool:
        return session_key in self._sessions


class FileSessionStorage(SessionStorage):
    """File-based storage using JSONL format."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Map sanitized filename stems to original keys
        self._key_map: dict[str, str] = {}

    def _sanitize_key(self, session_key: str) -> str:
        """Sanitize key for filesystem safety."""
        return session_key.replace(":", "_").replace("/", "_").replace("\\", "_")

    def _get_session_path(self, session_key: str) -> Path:
        """Get file path for a session key."""
        safe_key = self._sanitize_key(session_key)
        return self.base_dir / f"{safe_key}.jsonl"

    async def load(self, session_key: str) -> "Session | None":
        from voxagent.session.model import Session

        path = self._get_session_path(session_key)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return None

        # First line is session metadata
        metadata = json.loads(lines[0])

        # Remaining lines are messages
        messages_data = []
        for line in lines[1:]:
            if line.strip():
                msg_data = json.loads(line)
                messages_data.append(msg_data)

        return Session.from_dict({
            "id": metadata["id"],
            "key": metadata["key"],
            "messages": messages_data,
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "metadata": metadata.get("metadata", {}),
        })

    async def save(self, session: "Session") -> None:
        path = self._get_session_path(session.key)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(
            dir=self.base_dir,
            prefix=".tmp_",
            suffix=".jsonl",
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # First line: session metadata
                metadata = {
                    "id": session.id,
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                }
                f.write(json.dumps(metadata) + "\n")

                # Remaining lines: one message per line
                for msg in session.messages:
                    f.write(json.dumps(msg.model_dump()) + "\n")

            # Atomic rename
            Path(temp_path).rename(path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    async def delete(self, session_key: str) -> bool:
        path = self._get_session_path(session_key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def list_keys(self) -> list[str]:
        keys = []
        for path in self.base_dir.glob("*.jsonl"):
            if not path.name.startswith(".tmp_"):
                # Read the key from the file metadata
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                        if first_line.strip():
                            metadata = json.loads(first_line)
                            keys.append(metadata["key"])
                except (json.JSONDecodeError, KeyError, OSError):
                    # Skip corrupted files
                    pass
        return keys

    async def exists(self, session_key: str) -> bool:
        return self._get_session_path(session_key).exists()


__all__ = [
    "FileSessionStorage",
    "InMemorySessionStorage",
    "SessionStorage",
]

