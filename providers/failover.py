"""Failover logic for provider profiles."""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from voxagent.providers.auth import (
    AuthProfile,
    AuthProfileManager,
    FailoverError,
)

T = TypeVar("T")


class NoProfilesAvailableError(Exception):
    """No profiles available for the requested provider."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        msg = (
            f"No profiles available for provider: {provider}"
            if provider
            else "No profiles available"
        )
        super().__init__(msg)


class FailoverExhaustedError(Exception):
    """All available profiles failed."""

    def __init__(self, last_error: FailoverError) -> None:
        self.last_error = last_error
        super().__init__(f"All profiles failed. Last error: {last_error}")


async def run_with_failover(
    manager: AuthProfileManager,
    provider_name: str,
    operation: Callable[[AuthProfile], Awaitable[T]],
    max_retries: int | None = None,
) -> T:
    """
    Run an operation with automatic failover across available profiles.

    Args:
        manager: AuthProfileManager with configured profiles
        provider_name: Provider to filter profiles by (e.g., "openai")
        operation: Async callable that takes an AuthProfile and returns a result
        max_retries: Maximum number of profiles to try (default: all available)

    Returns:
        Result from successful operation

    Raises:
        NoProfilesAvailableError: If no profiles are available
        FailoverExhaustedError: If all profiles failed with FailoverError
        Exception: If operation raises a non-FailoverError exception
    """
    available = manager.get_available_profiles(provider=provider_name)

    if not available:
        raise NoProfilesAvailableError(provider_name)

    # Limit retries if specified
    profiles_to_try = available[:max_retries] if max_retries is not None else available

    last_error: FailoverError | None = None

    for profile in profiles_to_try:
        try:
            result = await operation(profile)
            # Success - record it and return
            manager.record_success(profile)
            return result

        except FailoverError as e:
            # Handle failover based on error type
            manager.handle_failover(profile, e)
            last_error = e
            # Continue to next profile
            continue

    # All profiles exhausted
    if last_error is not None:
        raise FailoverExhaustedError(last_error)
    else:
        # This shouldn't happen, but handle it gracefully
        raise NoProfilesAvailableError(provider_name)

