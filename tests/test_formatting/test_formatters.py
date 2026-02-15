"""Tests for concrete prompt formatters: XMLFormatter, MarkdownFormatter, PlainFormatter.

These tests follow TDD - they are written BEFORE the implementation exists.
They should FAIL initially until the concrete formatters are implemented.
"""

import json

import pytest

from voxagent.formatting import PromptFormat
from voxagent.formatting.xml import XMLFormatter
from voxagent.formatting.markdown import MarkdownFormatter
from voxagent.formatting.plain import PlainFormatter


class TestXMLFormatter:
    """Tests for XMLFormatter (Claude/Anthropic optimized)."""

    def test_format_type_returns_xml(self):
        """XMLFormatter.format_type returns PromptFormat.XML."""
        formatter = XMLFormatter()
        assert formatter.format_type == PromptFormat.XML

    def test_format_system_wraps_in_system_tags(self):
        """format_system wraps content in <system> tags."""
        formatter = XMLFormatter()
        result = formatter.format_system("You are helpful.")
        assert result == "<system>You are helpful.</system>"

    def test_format_system_with_empty_string(self):
        """format_system handles empty string."""
        formatter = XMLFormatter()
        result = formatter.format_system("")
        assert result == "<system></system>"

    def test_format_system_with_special_characters(self):
        """format_system preserves special characters in content."""
        formatter = XMLFormatter()
        content = "Use <code> and > symbols & quotes \"here\""
        result = formatter.format_system(content)
        assert result == f"<system>{content}</system>"

    def test_format_thought_wraps_in_thinking_tags(self):
        """format_thought wraps content in <thinking> tags."""
        formatter = XMLFormatter()
        result = formatter.format_thought("I should analyze this.")
        assert result == "<thinking>I should analyze this.</thinking>"

    def test_format_thought_with_empty_string(self):
        """format_thought handles empty string."""
        formatter = XMLFormatter()
        result = formatter.format_thought("")
        assert result == "<thinking></thinking>"

    def test_format_thought_with_multiline_content(self):
        """format_thought preserves multiline content."""
        formatter = XMLFormatter()
        content = "Line 1\nLine 2\nLine 3"
        result = formatter.format_thought(content)
        assert result == f"<thinking>{content}</thinking>"

    def test_format_action_wraps_in_action_tags(self):
        """format_action creates proper XML structure with tool and arguments."""
        formatter = XMLFormatter()
        result = formatter.format_action("search", {"query": "weather"})
        expected = '<action><tool>search</tool><arguments>{"query": "weather"}</arguments></action>'
        assert result == expected

    def test_format_action_with_empty_args(self):
        """format_action handles empty arguments dict."""
        formatter = XMLFormatter()
        result = formatter.format_action("get_time", {})
        expected = "<action><tool>get_time</tool><arguments>{}</arguments></action>"
        assert result == expected

    def test_format_action_with_complex_args(self):
        """format_action properly serializes complex nested arguments."""
        formatter = XMLFormatter()
        args = {
            "location": "New York",
            "units": "celsius",
            "details": {"include_humidity": True, "forecast_days": 5},
        }
        result = formatter.format_action("get_weather", args)
        # Verify JSON is parseable
        assert "<action><tool>get_weather</tool><arguments>" in result
        assert "</arguments></action>" in result
        # Extract and parse the JSON
        json_str = result.split("<arguments>")[1].split("</arguments>")[0]
        parsed = json.loads(json_str)
        assert parsed == args

    def test_format_action_args_are_json_serialized(self):
        """format_action serializes args as JSON."""
        formatter = XMLFormatter()
        args = {"key": "value", "number": 42}
        result = formatter.format_action("test_tool", args)
        # Extract JSON from result
        json_str = result.split("<arguments>")[1].split("</arguments>")[0]
        parsed = json.loads(json_str)
        assert parsed == args

    def test_format_observation_wraps_in_observation_tags(self):
        """format_observation wraps content in <observation> tags."""
        formatter = XMLFormatter()
        result = formatter.format_observation("Temperature: 72°F")
        assert result == "<observation>Temperature: 72°F</observation>"

    def test_format_observation_with_empty_string(self):
        """format_observation handles empty string."""
        formatter = XMLFormatter()
        result = formatter.format_observation("")
        assert result == "<observation></observation>"

    def test_format_observation_with_json_content(self):
        """format_observation preserves JSON content."""
        formatter = XMLFormatter()
        json_content = '{"status": "success", "data": [1, 2, 3]}'
        result = formatter.format_observation(json_content)
        assert result == f"<observation>{json_content}</observation>"

    def test_formatter_is_instance_of_prompt_formatter(self):
        """XMLFormatter is a PromptFormatter subclass."""
        from voxagent.formatting.base import PromptFormatter

        formatter = XMLFormatter()
        assert isinstance(formatter, PromptFormatter)


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter (OpenAI/GPT optimized)."""

    def test_format_type_returns_markdown(self):
        """MarkdownFormatter.format_type returns PromptFormat.MARKDOWN."""
        formatter = MarkdownFormatter()
        assert formatter.format_type == PromptFormat.MARKDOWN

    def test_format_system_uses_header(self):
        """format_system formats with ## System header."""
        formatter = MarkdownFormatter()
        result = formatter.format_system("You are helpful.")
        assert result == "## System\nYou are helpful."

    def test_format_system_with_empty_string(self):
        """format_system handles empty string."""
        formatter = MarkdownFormatter()
        result = formatter.format_system("")
        assert result == "## System\n"

    def test_format_system_with_multiline_content(self):
        """format_system preserves multiline content."""
        formatter = MarkdownFormatter()
        content = "Line 1\nLine 2"
        result = formatter.format_system(content)
        assert result == f"## System\n{content}"

    def test_format_thought_uses_header(self):
        """format_thought formats with ## Thinking header."""
        formatter = MarkdownFormatter()
        result = formatter.format_thought("I should analyze this.")
        assert result == "## Thinking\nI should analyze this."

    def test_format_thought_with_empty_string(self):
        """format_thought handles empty string."""
        formatter = MarkdownFormatter()
        result = formatter.format_thought("")
        assert result == "## Thinking\n"

    def test_format_action_uses_header_and_code_block(self):
        """format_action formats with header and JSON code block."""
        formatter = MarkdownFormatter()
        result = formatter.format_action("search", {"query": "weather"})
        expected = '## Action: search\n```json\n{"query": "weather"}\n```'
        assert result == expected

    def test_format_action_with_empty_args(self):
        """format_action handles empty arguments dict."""
        formatter = MarkdownFormatter()
        result = formatter.format_action("get_time", {})
        expected = "## Action: get_time\n```json\n{}\n```"
        assert result == expected

    def test_format_action_with_complex_args(self):
        """format_action properly serializes complex nested arguments."""
        formatter = MarkdownFormatter()
        args = {"location": "NYC", "options": {"detailed": True}}
        result = formatter.format_action("weather", args)
        assert "## Action: weather" in result
        assert "```json" in result
        # Extract and verify JSON is valid
        lines = result.split("\n")
        json_str = lines[2]  # JSON should be on third line
        parsed = json.loads(json_str)
        assert parsed == args

    def test_format_observation_uses_header(self):
        """format_observation formats with ## Observation header."""
        formatter = MarkdownFormatter()
        result = formatter.format_observation("Temperature: 72°F")
        assert result == "## Observation\nTemperature: 72°F"

    def test_format_observation_with_empty_string(self):
        """format_observation handles empty string."""
        formatter = MarkdownFormatter()
        result = formatter.format_observation("")
        assert result == "## Observation\n"

    def test_format_observation_with_multiline_content(self):
        """format_observation preserves multiline content."""
        formatter = MarkdownFormatter()
        content = "Line 1\nLine 2\nLine 3"
        result = formatter.format_observation(content)
        assert result == f"## Observation\n{content}"

    def test_formatter_is_instance_of_prompt_formatter(self):
        """MarkdownFormatter is a PromptFormatter subclass."""
        from voxagent.formatting.base import PromptFormatter

        formatter = MarkdownFormatter()
        assert isinstance(formatter, PromptFormatter)


class TestPlainFormatter:
    """Tests for PlainFormatter (no special formatting)."""

    def test_format_type_returns_plain(self):
        """PlainFormatter.format_type returns PromptFormat.PLAIN."""
        formatter = PlainFormatter()
        assert formatter.format_type == PromptFormat.PLAIN

    def test_format_system_uses_prefix(self):
        """format_system formats with 'System: ' prefix."""
        formatter = PlainFormatter()
        result = formatter.format_system("You are helpful.")
        assert result == "System: You are helpful."

    def test_format_system_with_empty_string(self):
        """format_system handles empty string."""
        formatter = PlainFormatter()
        result = formatter.format_system("")
        assert result == "System: "

    def test_format_system_with_multiline_content(self):
        """format_system preserves multiline content."""
        formatter = PlainFormatter()
        content = "Line 1\nLine 2"
        result = formatter.format_system(content)
        assert result == f"System: {content}"

    def test_format_thought_uses_prefix(self):
        """format_thought formats with 'Thinking: ' prefix."""
        formatter = PlainFormatter()
        result = formatter.format_thought("I should analyze this.")
        assert result == "Thinking: I should analyze this."

    def test_format_thought_with_empty_string(self):
        """format_thought handles empty string."""
        formatter = PlainFormatter()
        result = formatter.format_thought("")
        assert result == "Thinking: "

    def test_format_action_uses_function_call_format(self):
        """format_action formats as tool_name(json_args)."""
        formatter = PlainFormatter()
        result = formatter.format_action("search", {"query": "weather"})
        expected = 'Action: search({"query": "weather"})'
        assert result == expected

    def test_format_action_with_empty_args(self):
        """format_action handles empty arguments dict."""
        formatter = PlainFormatter()
        result = formatter.format_action("get_time", {})
        expected = "Action: get_time({})"
        assert result == expected

    def test_format_action_with_complex_args(self):
        """format_action properly serializes complex nested arguments."""
        formatter = PlainFormatter()
        args = {"key": "value", "nested": {"a": 1}}
        result = formatter.format_action("test_tool", args)
        # Verify structure
        assert result.startswith("Action: test_tool(")
        assert result.endswith(")")
        # Extract and verify JSON
        json_str = result[len("Action: test_tool(") : -1]
        parsed = json.loads(json_str)
        assert parsed == args

    def test_format_observation_uses_prefix(self):
        """format_observation formats with 'Observation: ' prefix."""
        formatter = PlainFormatter()
        result = formatter.format_observation("Temperature: 72°F")
        assert result == "Observation: Temperature: 72°F"

    def test_format_observation_with_empty_string(self):
        """format_observation handles empty string."""
        formatter = PlainFormatter()
        result = formatter.format_observation("")
        assert result == "Observation: "

    def test_format_observation_with_multiline_content(self):
        """format_observation preserves multiline content."""
        formatter = PlainFormatter()
        content = "Line 1\nLine 2\nLine 3"
        result = formatter.format_observation(content)
        assert result == f"Observation: {content}"

    def test_formatter_is_instance_of_prompt_formatter(self):
        """PlainFormatter is a PromptFormatter subclass."""
        from voxagent.formatting.base import PromptFormatter

        formatter = PlainFormatter()
        assert isinstance(formatter, PromptFormatter)

