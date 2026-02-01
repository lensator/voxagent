"""Provider registry for managing and instantiating LLM providers.

This module provides:
- ProviderRegistry for registering and looking up provider classes
- get_default_registry() for accessing a global singleton registry
- Exceptions for registry-related errors
"""

from typing import Any

from voxagent.providers.base import BaseProvider


# =============================================================================
# Exceptions
# =============================================================================


class ProviderNotFoundError(Exception):
    """Raised when a provider is not found in the registry."""

    pass


class InvalidModelStringError(Exception):
    """Raised when a model string is malformed."""

    pass


# =============================================================================
# Provider Registry
# =============================================================================


class ProviderRegistry:
    """Registry for managing LLM provider classes.

    Provides functionality to register, unregister, and look up provider classes
    by name. Also supports instantiating providers from model strings in the
    format "provider:model".
    """

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._providers: dict[str, type[BaseProvider]] = {}

    def register(self, name: str, provider_class: type[BaseProvider]) -> None:
        """Register a provider class by name.

        Args:
            name: The provider name (must be non-empty, non-whitespace).
            provider_class: The provider class (must be a BaseProvider subclass).

        Raises:
            ValueError: If name is empty or whitespace-only.
            TypeError: If provider_class is not a BaseProvider subclass.
        """
        if not name or not name.strip():
            raise ValueError("Provider name cannot be empty or whitespace-only")

        if not isinstance(provider_class, type) or not issubclass(
            provider_class, BaseProvider
        ):
            raise TypeError("provider_class must be a subclass of BaseProvider")

        self._providers[name] = provider_class

    def unregister(self, name: str) -> None:
        """Remove a provider from the registry.

        Args:
            name: The provider name to unregister.

        Raises:
            ProviderNotFoundError: If the provider is not registered.
        """
        if name not in self._providers:
            raise ProviderNotFoundError(name)
        del self._providers[name]

    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: The provider name to check.

        Returns:
            True if the provider is registered, False otherwise.
        """
        return name in self._providers

    def get_provider_class(self, name: str) -> type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: The provider name.

        Returns:
            The registered provider class.

        Raises:
            ProviderNotFoundError: If the provider is not found.
        """
        if name not in self._providers:
            raise ProviderNotFoundError(name)
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            A copy of the list of registered provider names.
        """
        return list(self._providers.keys())

    def get_provider(self, model_string: str, **kwargs: Any) -> BaseProvider:
        """Parse model string and instantiate provider.

        Args:
            model_string: Format "provider:model" (e.g., "openai:gpt-4o").
                Handles multiple colons: first part is provider, rest is model.
                e.g., "ollama:model:latest" → provider="ollama", model="model:latest"
            **kwargs: Additional arguments to pass to provider constructor.

        Returns:
            Instantiated provider.

        Raises:
            InvalidModelStringError: If model_string is malformed.
            ProviderNotFoundError: If provider is not registered.
        """
        if not model_string or ":" not in model_string:
            raise InvalidModelStringError(model_string)

        # Split on first colon only
        parts = model_string.split(":", 1)
        provider_name = parts[0]
        model_name = parts[1] if len(parts) > 1 else ""

        if not provider_name:
            raise InvalidModelStringError(model_string)
        if not model_name:
            raise InvalidModelStringError(model_string)

        provider_class = self.get_provider_class(provider_name)
        return provider_class(model=model_name, **kwargs)


# =============================================================================
# Default Registry Singleton
# =============================================================================

_default_registry: ProviderRegistry | None = None


def get_default_registry() -> ProviderRegistry:
    """Get the default global provider registry.

    Returns:
        The singleton ProviderRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ProviderRegistry()
    return _default_registry


__all__ = [
    "InvalidModelStringError",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "get_default_registry",
]

