"""LLM provider implementations.

This subpackage contains unified interfaces for various LLM providers:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude)
- Google (Gemini)
- Groq (Llama, Mixtral)
- Ollama (local models)
- ChatGPT (ChatGPT Plus backend)
- Augment (Auggie CLI)
- Codex (OpenAI Codex CLI)
- ClaudeCode (Claude Code CLI)
"""

from voxagent.providers.anthropic import AnthropicProvider
from voxagent.providers.augment import AugmentProvider
from voxagent.providers.auth import (
    AuthProfile,
    AuthProfileManager,
    FailoverError,
    FailoverReason,
)
from voxagent.providers.base import (
    AbortSignal,
    BaseProvider,
    ErrorChunk,
    MessageEndChunk,
    StreamChunk,
    TextDeltaChunk,
    ToolUseChunk,
)
from voxagent.providers.chatgpt import ChatGPTProvider
from voxagent.providers.claudecode import ClaudeCodeProvider
from voxagent.providers.cli_base import CLINotFoundError, CLIProvider
from voxagent.providers.codex import CodexProvider
from voxagent.providers.failover import (
    FailoverExhaustedError,
    NoProfilesAvailableError,
    run_with_failover,
)
from voxagent.providers.google import GoogleProvider
from voxagent.providers.groq import GroqProvider
from voxagent.providers.ollama import OllamaProvider
from voxagent.providers.openai import OpenAIProvider
from voxagent.providers.registry import (
    InvalidModelStringError,
    ProviderNotFoundError,
    ProviderRegistry,
    get_default_registry,
)

__all__ = [
    "AbortSignal",
    "AnthropicProvider",
    "AugmentProvider",
    "AuthProfile",
    "AuthProfileManager",
    "BaseProvider",
    "ChatGPTProvider",
    "CLINotFoundError",
    "CLIProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "ErrorChunk",
    "FailoverError",
    "FailoverExhaustedError",
    "FailoverReason",
    "GoogleProvider",
    "GroqProvider",
    "InvalidModelStringError",
    "MessageEndChunk",
    "NoProfilesAvailableError",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderNotFoundError",
    "ProviderRegistry",
    "StreamChunk",
    "TextDeltaChunk",
    "ToolUseChunk",
    "get_default_registry",
    "run_with_failover",
]


# =============================================================================
# Register default providers
# =============================================================================

def _register_default_providers() -> None:
    """Register all built-in providers with the default registry."""
    registry = get_default_registry()

    # Only register if not already registered (idempotent)
    if not registry.list_providers():
        # HTTP API providers
        registry.register("openai", OpenAIProvider)
        registry.register("anthropic", AnthropicProvider)
        registry.register("google", GoogleProvider)
        registry.register("groq", GroqProvider)
        registry.register("ollama", OllamaProvider)
        registry.register("chatgpt", ChatGPTProvider)
        # CLI-wrapped providers
        registry.register("augment", AugmentProvider)
        registry.register("codex", CodexProvider)
        registry.register("claudecode", ClaudeCodeProvider)


# Auto-register providers on module import
_register_default_providers()

