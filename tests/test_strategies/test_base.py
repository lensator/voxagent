"""Tests for strategy base classes."""

import pytest
from unittest.mock import MagicMock

from voxagent.strategies.base import (
    AgentStrategy,
    StrategyContext,
    StrategyResult,
)
from voxagent.types.messages import Message
from voxagent.types.run import ToolMeta


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""

    def test_create_minimal_result(self):
        """StrategyResult can be created with minimal fields."""
        result = StrategyResult(
            messages=[],
            assistant_texts=[],
        )
        assert result.messages == []
        assert result.assistant_texts == []
        assert result.tool_metas == []
        assert result.metadata == {}
        assert result.error is None

    def test_create_full_result(self):
        """StrategyResult can be created with all fields."""
        messages = [Message(role="user", content="Hello")]
        tool_metas = [ToolMeta(tool_name="test", tool_call_id="1")]

        result = StrategyResult(
            messages=messages,
            assistant_texts=["Response"],
            tool_metas=tool_metas,
            metadata={"key": "value"},
            error="Some error",
        )

        assert result.messages == messages
        assert result.assistant_texts == ["Response"]
        assert result.tool_metas == tool_metas
        assert result.metadata == {"key": "value"}
        assert result.error == "Some error"


class TestStrategyContext:
    """Tests for StrategyContext."""

    def test_create_context(self):
        """StrategyContext can be created with required fields."""
        mock_provider = MagicMock()
        mock_signal = MagicMock()

        ctx = StrategyContext(
            provider=mock_provider,
            tools=[],
            system_prompt="You are helpful",
            abort_signal=mock_signal,
            run_id="test-run-id",
        )

        assert ctx.provider == mock_provider
        assert ctx.tools == []
        assert ctx.system_prompt == "You are helpful"
        assert ctx.abort_signal == mock_signal
        assert ctx.run_id == "test-run-id"
        assert ctx.deps is None

    def test_context_with_deps(self):
        """StrategyContext accepts custom deps."""
        mock_provider = MagicMock()
        mock_signal = MagicMock()
        mock_deps = {"session": "abc123"}

        ctx = StrategyContext(
            provider=mock_provider,
            tools=[],
            system_prompt=None,
            abort_signal=mock_signal,
            run_id="test-run-id",
            deps=mock_deps,
        )

        assert ctx.deps == mock_deps


class TestAgentStrategyABC:
    """Tests for AgentStrategy abstract base class."""

    def test_cannot_instantiate_directly(self):
        """AgentStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AgentStrategy()

    def test_name_property_returns_class_name(self):
        """Strategy name property returns class name by default."""

        class MyCustomStrategy(AgentStrategy):
            async def execute(self, ctx, messages):
                return StrategyResult(messages=[], assistant_texts=[])

            async def execute_stream(self, ctx, messages):
                yield  # type: ignore

        strategy = MyCustomStrategy()
        assert strategy.name == "MyCustomStrategy"

    def test_custom_strategy_must_implement_execute(self):
        """Custom strategies must implement execute method."""

        class IncompleteStrategy(AgentStrategy):
            async def execute_stream(self, ctx, messages):
                yield  # type: ignore

        with pytest.raises(TypeError, match="abstract"):
            IncompleteStrategy()

    def test_custom_strategy_must_implement_execute_stream(self):
        """Custom strategies must implement execute_stream method."""

        class IncompleteStrategy(AgentStrategy):
            async def execute(self, ctx, messages):
                return StrategyResult(messages=[], assistant_texts=[])

        with pytest.raises(TypeError, match="abstract"):
            IncompleteStrategy()

