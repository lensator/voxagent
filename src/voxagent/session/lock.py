"""Session locking for voxagent."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class LockTimeoutError(Exception):
    """Raised when lock acquisition times out."""

    def __init__(self, session_key: str, timeout: float) -> None:
        self.session_key = session_key
        self.timeout = timeout
        super().__init__(f"Lock timeout for session {session_key} after {timeout}s")


# Module-level lock registry keyed by (event_loop_id, session_key)
_lock_registry: dict[tuple[int, str], asyncio.Lock] = {}
_registry_thread_lock = threading.Lock()


def _get_lock_for_session(session_key: str) -> asyncio.Lock:
    """Get or create the asyncio.Lock for a session key in the current event loop."""
    loop = asyncio.get_running_loop()
    key = (id(loop), session_key)

    with _registry_thread_lock:
        if key not in _lock_registry:
            _lock_registry[key] = asyncio.Lock()
        return _lock_registry[key]


class SessionLock:
    """Async lock for session access."""

    def __init__(self, session_key: str, timeout: float = 30.0) -> None:
        self.session_key = session_key
        self.timeout = timeout
        self._acquired: bool = False

    async def acquire(self) -> bool:
        """Acquire the lock. Returns True if acquired, raises LockTimeoutError on timeout."""
        lock = _get_lock_for_session(self.session_key)

        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.timeout)
            self._acquired = True
            return True
        except asyncio.TimeoutError:
            raise LockTimeoutError(self.session_key, self.timeout)

    async def release(self) -> None:
        """Release the lock."""
        if self._acquired:
            lock = _get_lock_for_session(self.session_key)
            try:
                lock.release()
            except RuntimeError:
                # Lock was not held
                pass
            self._acquired = False

    async def __aenter__(self) -> "SessionLock":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release()

    @property
    def is_locked(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._acquired


class SessionLockManager:
    """Manages locks for multiple sessions."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._locks: dict[str, SessionLock] = {}
        self._timeout = timeout

    def get_lock(self, session_key: str) -> SessionLock:
        """Get or create a lock for a session."""
        if session_key not in self._locks:
            self._locks[session_key] = SessionLock(session_key, self._timeout)
        return self._locks[session_key]

    async def acquire(self, session_key: str) -> SessionLock:
        """Acquire lock for a session."""
        lock = self.get_lock(session_key)
        await lock.acquire()
        return lock

    async def release(self, session_key: str) -> None:
        """Release lock for a session."""
        if session_key in self._locks:
            await self._locks[session_key].release()

