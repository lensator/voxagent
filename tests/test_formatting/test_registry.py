"""Tests for provider→formatter registry.

These tests follow TDD - they are written BEFORE the implementation exists.
They should FAIL initially until the registry is implemented.
"""

import pytest

from voxagent.formatting import MarkdownFormatter, PlainFormatter, XMLFormatter
from voxagent.formatting.registry import (
    PROVIDER_FORMATTERS,
    get_formatter_for_provider,
    register_provider_formatter,
)


class TestProviderFormatterRegistry:
    """Tests for provider→formatter registry."""

    # ========================================================================
    # Anthropic/Claude → XMLFormatter
    # ========================================================================

    def test_anthropic_returns_xml_formatter(self):
        """Provider 'anthropic' maps to XMLFormatter."""
        formatter = get_formatter_for_provider("anthropic")
        assert isinstance(formatter, XMLFormatter)

    def test_claude_returns_xml_formatter(self):
        """Provider 'claude' maps to XMLFormatter."""
        formatter = get_formatter_for_provider("claude")
        assert isinstance(formatter, XMLFormatter)

    # ========================================================================
    # OpenAI/GPT → MarkdownFormatter
    # ========================================================================

    def test_openai_returns_markdown_formatter(self):
        """Provider 'openai' maps to MarkdownFormatter."""
        formatter = get_formatter_for_provider("openai")
        assert isinstance(formatter, MarkdownFormatter)

    def test_gpt_returns_markdown_formatter(self):
        """Provider 'gpt' maps to MarkdownFormatter."""
        formatter = get_formatter_for_provider("gpt")
        assert isinstance(formatter, MarkdownFormatter)

    # ========================================================================
    # Google → MarkdownFormatter
    # ========================================================================

    def test_google_returns_markdown_formatter(self):
        """Provider 'google' maps to MarkdownFormatter."""
        formatter = get_formatter_for_provider("google")
        assert isinstance(formatter, MarkdownFormatter)

    # ========================================================================
    # Groq → MarkdownFormatter
    # ========================================================================

    def test_groq_returns_markdown_formatter(self):
        """Provider 'groq' maps to MarkdownFormatter."""
        formatter = get_formatter_for_provider("groq")
        assert isinstance(formatter, MarkdownFormatter)

    # ========================================================================
    # Ollama → PlainFormatter
    # ========================================================================

    def test_ollama_returns_plain_formatter(self):
        """Provider 'ollama' maps to PlainFormatter."""
        formatter = get_formatter_for_provider("ollama")
        assert isinstance(formatter, PlainFormatter)

    # ========================================================================
    # Default Fallback → PlainFormatter
    # ========================================================================

    def test_unknown_provider_returns_plain_formatter(self):
        """Unknown providers fall back to PlainFormatter."""
        formatter = get_formatter_for_provider("unknown_provider_xyz")
        assert isinstance(formatter, PlainFormatter)

    def test_empty_provider_returns_plain_formatter(self):
        """Empty string provider falls back to PlainFormatter."""
        formatter = get_formatter_for_provider("")
        assert isinstance(formatter, PlainFormatter)

    # ========================================================================
    # Case Insensitivity
    # ========================================================================

    def test_case_insensitive_lookup_uppercase(self):
        """Provider lookup is case-insensitive (uppercase)."""
        formatter = get_formatter_for_provider("ANTHROPIC")
        assert isinstance(formatter, XMLFormatter)

    def test_case_insensitive_lookup_mixed_case(self):
        """Provider lookup is case-insensitive (mixed case)."""
        formatter = get_formatter_for_provider("OpenAI")
        assert isinstance(formatter, MarkdownFormatter)

    def test_case_insensitive_lookup_title_case(self):
        """Provider lookup is case-insensitive (title case)."""
        formatter = get_formatter_for_provider("Google")
        assert isinstance(formatter, MarkdownFormatter)

    # ========================================================================
    # Custom Registration
    # ========================================================================

    def test_register_custom_provider(self):
        """Can register new provider mappings."""
        register_provider_formatter("my_custom_provider", XMLFormatter)
        formatter = get_formatter_for_provider("my_custom_provider")
        assert isinstance(formatter, XMLFormatter)

    def test_register_custom_provider_case_insensitive(self):
        """Custom provider registration is case-insensitive."""
        register_provider_formatter("MyCustomProvider2", MarkdownFormatter)
        formatter = get_formatter_for_provider("mycustomprovider2")
        assert isinstance(formatter, MarkdownFormatter)

    def test_override_existing_provider(self):
        """Can override an existing provider mapping."""
        # Store original for cleanup
        original = get_formatter_for_provider("ollama")
        assert isinstance(original, PlainFormatter)

        # Override
        register_provider_formatter("ollama", XMLFormatter)
        formatter = get_formatter_for_provider("ollama")
        assert isinstance(formatter, XMLFormatter)

        # Restore (cleanup)
        register_provider_formatter("ollama", PlainFormatter)

    # ========================================================================
    # Registry Dictionary Access
    # ========================================================================

    def test_provider_formatters_dict_is_accessible(self):
        """PROVIDER_FORMATTERS dict is accessible."""
        assert isinstance(PROVIDER_FORMATTERS, dict)

    def test_provider_formatters_contains_anthropic(self):
        """PROVIDER_FORMATTERS contains 'anthropic' key."""
        assert "anthropic" in PROVIDER_FORMATTERS

    def test_provider_formatters_contains_openai(self):
        """PROVIDER_FORMATTERS contains 'openai' key."""
        assert "openai" in PROVIDER_FORMATTERS

    def test_provider_formatters_contains_google(self):
        """PROVIDER_FORMATTERS contains 'google' key."""
        assert "google" in PROVIDER_FORMATTERS

    def test_provider_formatters_contains_ollama(self):
        """PROVIDER_FORMATTERS contains 'ollama' key."""
        assert "ollama" in PROVIDER_FORMATTERS

