"""Session management and persistence.

This subpackage provides:
- Session model with messages and metadata
- File-based session storage (JSONL format)
- Session locking for concurrent access
- Context compaction and token management
"""

from voxagent.session.compaction import (
    CompactionStrategy,
    compact_context,
    count_message_tokens,
    count_tokens,
    needs_compaction,
)
from voxagent.session.lock import LockTimeoutError, SessionLock, SessionLockManager
from voxagent.session.model import Session, resolve_session_key
from voxagent.session.storage import (
    FileSessionStorage,
    InMemorySessionStorage,
    SessionStorage,
)

__all__ = [
    "CompactionStrategy",
    "FileSessionStorage",
    "InMemorySessionStorage",
    "LockTimeoutError",
    "Session",
    "SessionLock",
    "SessionLockManager",
    "SessionStorage",
    "compact_context",
    "count_message_tokens",
    "count_tokens",
    "needs_compaction",
    "resolve_session_key",
]
