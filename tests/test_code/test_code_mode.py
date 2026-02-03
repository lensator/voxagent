"""Tests for code execution mode."""

from __future__ import annotations

import pytest

from voxagent.code import (
    CodeModeConfig,
    CodeModeExecutor,
    ToolRegistry,
    CODE_MODE_SYSTEM_PROMPT,
)


class TestCodeModeConfig:
    """Tests for CodeModeConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CodeModeConfig()
        assert config.enabled is True
        assert config.timeout_seconds == 10
        assert config.memory_limit_mb == 128
        assert config.max_output_chars == 10000

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CodeModeConfig(
            timeout_seconds=5,
            memory_limit_mb=64,
            max_output_chars=5000,
            enabled=False,
        )
        assert config.timeout_seconds == 5
        assert config.memory_limit_mb == 64
        assert config.max_output_chars == 5000
        assert config.enabled is False


class TestCodeModeExecutor:
    """Tests for CodeModeExecutor."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create a tool registry with test tools."""
        reg = ToolRegistry()
        reg.register_category(
            "test",
            description="Test tools",
            tools={
                "greet.py": 'def greet(name: str) -> str:\n    """Greet someone."""\n    ...',
                "add.py": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    ...',
            },
        )
        return reg

    @pytest.fixture
    def executor(self, registry: ToolRegistry) -> CodeModeExecutor:
        """Create a CodeModeExecutor for testing."""
        config = CodeModeConfig(timeout_seconds=5)
        return CodeModeExecutor(config, registry)

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, executor: CodeModeExecutor) -> None:
        """Execute simple Python code."""
        result = await executor.execute_code("print('hello')")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_ls_tools(self, executor: CodeModeExecutor) -> None:
        """Execute ls() to list tool categories."""
        result = await executor.execute_code('print(ls("tools/"))')
        assert "test/" in result

    @pytest.mark.asyncio
    async def test_execute_ls_category(self, executor: CodeModeExecutor) -> None:
        """Execute ls() to list tools in a category."""
        result = await executor.execute_code('print(ls("tools/test/"))')
        assert "greet.py" in result
        assert "add.py" in result

    @pytest.mark.asyncio
    async def test_execute_read_tool(self, executor: CodeModeExecutor) -> None:
        """Execute read() to read a tool definition."""
        result = await executor.execute_code('print(read("tools/test/greet.py"))')
        assert "def greet" in result
        assert "name: str" in result

    @pytest.mark.asyncio
    async def test_execute_read_index(self, executor: CodeModeExecutor) -> None:
        """Execute read() to read category index."""
        result = await executor.execute_code('print(read("tools/test/__index__.md"))')
        assert "test" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_error_returns_message(self, executor: CodeModeExecutor) -> None:
        """Errors are returned as messages."""
        result = await executor.execute_code("raise ValueError('oops')")
        assert "Error" in result
        assert "oops" in result

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self, executor: CodeModeExecutor) -> None:
        """Syntax errors are returned as messages."""
        result = await executor.execute_code("if True print('bad')")
        assert "Error" in result
        assert "SyntaxError" in result

    @pytest.mark.asyncio
    async def test_execute_no_output(self, executor: CodeModeExecutor) -> None:
        """Code with no print returns indicator."""
        result = await executor.execute_code("x = 1")
        assert "(no output)" in result

    def test_get_execute_code_tool(self, executor: CodeModeExecutor) -> None:
        """Get the execute_code tool definition."""
        tool = executor.get_execute_code_tool()
        assert tool.name == "execute_code"
        assert tool.is_async is True
        assert "code" in tool.parameters.get("properties", {})


class TestCodeModeSystemPrompt:
    """Tests for code mode system prompt."""

    def test_system_prompt_contains_instructions(self) -> None:
        """System prompt contains key instructions."""
        assert "execute_code" in CODE_MODE_SYSTEM_PROMPT
        assert "ls(" in CODE_MODE_SYSTEM_PROMPT
        assert "read(" in CODE_MODE_SYSTEM_PROMPT
        assert "print(" in CODE_MODE_SYSTEM_PROMPT
        assert "call_tool(" in CODE_MODE_SYSTEM_PROMPT

    def test_system_prompt_contains_example(self) -> None:
        """System prompt contains usage example."""
        assert "call_tool(" in CODE_MODE_SYSTEM_PROMPT
        assert "result" in CODE_MODE_SYSTEM_PROMPT

    def test_system_prompt_contains_workflow(self) -> None:
        """System prompt contains workflow steps."""
        assert "Explore" in CODE_MODE_SYSTEM_PROMPT
        assert "Learn" in CODE_MODE_SYSTEM_PROMPT
        assert "Execute" in CODE_MODE_SYSTEM_PROMPT

