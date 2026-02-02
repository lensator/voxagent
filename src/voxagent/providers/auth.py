"""Authentication profile management with failover support.

This module provides API credential profile management with cooldown
and failure tracking for multi-provider LLM failover.
"""

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict


class FailoverReason(str, Enum):
    """Reasons for failover to a different API profile."""

    AUTH_ERROR = "auth_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_ERROR = "model_error"


class FailoverError(Exception):
    """Error that triggers failover to another profile."""

    def __init__(self, reason: FailoverReason, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason.value)


class AuthProfile(BaseModel):
    """API credential profile with cooldown and failure tracking."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    api_key: str
    provider: str
    cooldown_until: datetime | None = None
    failure_count: int = 0
    disabled: bool = False

    def is_available(self) -> bool:
        """Check if profile is available (not in cooldown, not disabled)."""
        if self.disabled:
            return False
        if self.cooldown_until is not None:
            if datetime.now() < self.cooldown_until:
                return False
        return True

    def set_cooldown(self, duration_seconds: int) -> None:
        """Set cooldown until now + duration."""
        self.cooldown_until = datetime.now() + timedelta(seconds=duration_seconds)

    def record_failure(self) -> None:
        """Increment failure count."""
        self.failure_count += 1

    def reset_failures(self) -> None:
        """Reset failure count on success."""
        self.failure_count = 0

    def disable(self) -> None:
        """Disable this profile."""
        self.disabled = True


class AuthProfileManager:
    """Manages multiple API profiles with failover support."""

    DEFAULT_MAX_FAILURES = 3
    DEFAULT_RATE_LIMIT_COOLDOWN = 60  # seconds

    def __init__(
        self,
        profiles: list[AuthProfile],
        max_failures: int = DEFAULT_MAX_FAILURES,
        rate_limit_cooldown: int = DEFAULT_RATE_LIMIT_COOLDOWN,
    ) -> None:
        self._profiles = list(profiles)
        self._max_failures = max_failures
        self._rate_limit_cooldown = rate_limit_cooldown

    @property
    def profiles(self) -> list[AuthProfile]:
        return self._profiles

    @property
    def max_failures(self) -> int:
        return self._max_failures

    @property
    def rate_limit_cooldown(self) -> int:
        return self._rate_limit_cooldown

    def get_available_profiles(self, provider: str | None = None) -> list[AuthProfile]:
        """Get profiles that are not in cooldown and not disabled."""
        available = [p for p in self._profiles if p.is_available()]
        if provider is not None:
            available = [p for p in available if p.provider == provider]
        return available

    def handle_failover(self, profile: AuthProfile, error: FailoverError) -> None:
        """Update profile state based on failover error."""
        if error.reason == FailoverReason.RATE_LIMIT:
            profile.set_cooldown(self._rate_limit_cooldown)
        elif error.reason == FailoverReason.AUTH_ERROR:
            profile.record_failure()
            if profile.failure_count >= self._max_failures:
                profile.disable()

    def record_success(self, profile: AuthProfile) -> None:
        """Record successful use of profile."""
        profile.reset_failures()

