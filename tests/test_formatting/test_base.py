"""Tests for PromptFormat enum and PromptFormatter ABC in formatting base module.

These tests follow TDD - they are written BEFORE the implementation exists.
They should FAIL initially until voxagent.formatting.base is implemented.
"""

import pytest

from voxagent.formatting.base import PromptFormat, PromptFormatter


class TestPromptFormatEnum:
    """Tests for the PromptFormat enum."""

    def test_prompt_format_has_exactly_four_members(self):
        """PromptFormat enum has exactly 4 members."""
        members = list(PromptFormat)
        assert len(members) == 4

    def test_prompt_format_xml_exists(self):
        """PromptFormat.XML exists and has correct value."""
        assert hasattr(PromptFormat, "XML")
        assert PromptFormat.XML.value == "xml"

    def test_prompt_format_markdown_exists(self):
        """PromptFormat.MARKDOWN exists and has correct value."""
        assert hasattr(PromptFormat, "MARKDOWN")
        assert PromptFormat.MARKDOWN.value == "markdown"

    def test_prompt_format_json_exists(self):
        """PromptFormat.JSON exists and has correct value."""
        assert hasattr(PromptFormat, "JSON")
        assert PromptFormat.JSON.value == "json"

    def test_prompt_format_plain_exists(self):
        """PromptFormat.PLAIN exists and has correct value."""
        assert hasattr(PromptFormat, "PLAIN")
        assert PromptFormat.PLAIN.value == "plain"

    def test_enum_values_are_strings(self):
        """Enum values are strings for serialization compatibility."""
        for member in PromptFormat:
            assert isinstance(member.value, str)

    def test_enum_comparison_same_member(self):
        """Enum comparison works correctly for same members."""
        assert PromptFormat.XML == PromptFormat.XML
        assert PromptFormat.MARKDOWN == PromptFormat.MARKDOWN
        assert PromptFormat.JSON == PromptFormat.JSON
        assert PromptFormat.PLAIN == PromptFormat.PLAIN

    def test_enum_comparison_different_members(self):
        """Enum comparison works correctly for different members."""
        assert PromptFormat.XML != PromptFormat.MARKDOWN
        assert PromptFormat.XML != PromptFormat.JSON
        assert PromptFormat.XML != PromptFormat.PLAIN
        assert PromptFormat.MARKDOWN != PromptFormat.JSON
        assert PromptFormat.MARKDOWN != PromptFormat.PLAIN
        assert PromptFormat.JSON != PromptFormat.PLAIN

    def test_enum_identity(self):
        """Enum identity check works (is operator)."""
        xml1 = PromptFormat.XML
        xml2 = PromptFormat.XML
        assert xml1 is xml2

    def test_enum_can_be_accessed_by_value(self):
        """Enum members can be accessed by their string value."""
        assert PromptFormat("xml") == PromptFormat.XML
        assert PromptFormat("markdown") == PromptFormat.MARKDOWN
        assert PromptFormat("json") == PromptFormat.JSON
        assert PromptFormat("plain") == PromptFormat.PLAIN

    def test_enum_name_attribute(self):
        """Enum members have correct name attribute."""
        assert PromptFormat.XML.name == "XML"
        assert PromptFormat.MARKDOWN.name == "MARKDOWN"
        assert PromptFormat.JSON.name == "JSON"
        assert PromptFormat.PLAIN.name == "PLAIN"


class TestPromptFormatterABC:
    """Tests for the PromptFormatter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """PromptFormatter cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError, match="abstract"):
            PromptFormatter()  # type: ignore

    def test_subclass_must_implement_format_type(self):
        """Subclass must implement format_type property."""

        class IncompleteFormatter(PromptFormatter):
            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        with pytest.raises(TypeError, match="abstract"):
            IncompleteFormatter()

    def test_subclass_must_implement_format_system(self):
        """Subclass must implement format_system method."""

        class IncompleteFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        with pytest.raises(TypeError, match="abstract"):
            IncompleteFormatter()

    def test_subclass_must_implement_format_thought(self):
        """Subclass must implement format_thought method."""

        class IncompleteFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        with pytest.raises(TypeError, match="abstract"):
            IncompleteFormatter()

    def test_subclass_must_implement_format_action(self):
        """Subclass must implement format_action method."""

        class IncompleteFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_observation(self, content: str) -> str:
                return content

        with pytest.raises(TypeError, match="abstract"):
            IncompleteFormatter()

    def test_subclass_must_implement_format_observation(self):
        """Subclass must implement format_observation method."""

        class IncompleteFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

        with pytest.raises(TypeError, match="abstract"):
            IncompleteFormatter()

    def test_properly_implemented_subclass_can_be_instantiated(self):
        """A properly implemented subclass can be instantiated."""

        class CompleteFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return f"[SYSTEM] {content}"

            def format_thought(self, content: str) -> str:
                return f"[THOUGHT] {content}"

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"[ACTION] {tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return f"[OBSERVATION] {content}"

        formatter = CompleteFormatter()
        assert formatter is not None
        assert isinstance(formatter, PromptFormatter)

    def test_format_type_returns_prompt_format(self):
        """format_type property returns a PromptFormat enum value."""

        class TestFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.XML

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        formatter = TestFormatter()
        assert formatter.format_type == PromptFormat.XML
        assert isinstance(formatter.format_type, PromptFormat)

    def test_format_system_signature(self):
        """format_system accepts content string and returns string."""

        class TestFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return f"[SYSTEM] {content}"

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        formatter = TestFormatter()
        result = formatter.format_system("You are a helpful assistant.")
        assert isinstance(result, str)
        assert "You are a helpful assistant." in result

    def test_format_thought_signature(self):
        """format_thought accepts content string and returns string."""

        class TestFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return f"[THOUGHT] {content}"

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return content

        formatter = TestFormatter()
        result = formatter.format_thought("I should analyze this request.")
        assert isinstance(result, str)
        assert "I should analyze this request." in result

    def test_format_action_signature(self):
        """format_action accepts tool_name and args dict, returns string."""

        class TestFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"[ACTION] {tool_name}({args})"

            def format_observation(self, content: str) -> str:
                return content

        formatter = TestFormatter()
        result = formatter.format_action("search", {"query": "weather"})
        assert isinstance(result, str)
        assert "search" in result

    def test_format_observation_signature(self):
        """format_observation accepts content string and returns string."""

        class TestFormatter(PromptFormatter):
            @property
            def format_type(self) -> PromptFormat:
                return PromptFormat.PLAIN

            def format_system(self, content: str) -> str:
                return content

            def format_thought(self, content: str) -> str:
                return content

            def format_action(self, tool_name: str, args: dict) -> str:
                return f"{tool_name}: {args}"

            def format_observation(self, content: str) -> str:
                return f"[OBSERVATION] {content}"

        formatter = TestFormatter()
        result = formatter.format_observation("The temperature is 72°F.")
        assert isinstance(result, str)
        assert "The temperature is 72°F." in result

