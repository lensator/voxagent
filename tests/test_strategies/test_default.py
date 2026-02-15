"""Tests for DefaultStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from voxagent.strategies.default import DefaultStrategy
from voxagent.strategies.base import StrategyContext, StrategyResult
from voxagent.types.messages import Message, ToolCall, ToolResult
from voxagent.streaming.events import (
    TextDeltaEvent,
    ToolStartEvent,
    ToolEndEvent,
)


class TestDefaultStrategyInit:
    """Tests for DefaultStrategy initialization."""

    def test_default_max_iterations(self):
        """DefaultStrategy has default max_iterations of 50."""
        strategy = DefaultStrategy()
        assert strategy._max_iterations == 50

    def test_custom_max_iterations(self):
        """DefaultStrategy accepts custom max_iterations."""
        strategy = DefaultStrategy(max_iterations=10)
        assert strategy._max_iterations == 10

    def test_strategy_name(self):
        """DefaultStrategy returns correct name."""
        strategy = DefaultStrategy()
        assert strategy.name == "DefaultStrategy"


class TestDefaultStrategyExecute:
    """Tests for DefaultStrategy.execute()."""

    @pytest.mark.asyncio
    async def test_execute_delegates_to_context(self):
        """execute() delegates to ctx.run_tool_loop()."""
        strategy = DefaultStrategy(max_iterations=25)

        # Mock context with run_tool_loop
        mock_result = StrategyResult(
            messages=[Message(role="assistant", content="Hello")],
            assistant_texts=["Hello"],
            metadata={},
        )

        mock_ctx = MagicMock(spec=StrategyContext)
        mock_ctx.run_tool_loop = AsyncMock(return_value=mock_result)

        messages = [Message(role="user", content="Hi")]

        result = await strategy.execute(mock_ctx, messages)

        # Verify delegation
        mock_ctx.run_tool_loop.assert_called_once_with(
            messages=messages,
            max_iterations=25,
        )

        # Verify result has strategy metadata
        assert result.metadata["strategy_name"] == "DefaultStrategy"
        assert result.metadata["max_iterations"] == 25

    @pytest.mark.asyncio
    async def test_execute_preserves_result_data(self):
        """execute() preserves all result data from context."""
        strategy = DefaultStrategy()

        messages = [Message(role="user", content="Test")]
        tool_metas = []

        mock_result = StrategyResult(
            messages=messages,
            assistant_texts=["Response 1", "Response 2"],
            tool_metas=tool_metas,
            metadata={"existing": "data"},
            error=None,
        )

        mock_ctx = MagicMock(spec=StrategyContext)
        mock_ctx.run_tool_loop = AsyncMock(return_value=mock_result)

        result = await strategy.execute(mock_ctx, messages)

        assert result.assistant_texts == ["Response 1", "Response 2"]
        assert result.error is None
        # Strategy adds to existing metadata
        assert result.metadata["existing"] == "data"
        assert result.metadata["strategy_name"] == "DefaultStrategy"


class TestDefaultStrategyExecuteStream:
    """Tests for DefaultStrategy.execute_stream()."""

    @pytest.mark.asyncio
    async def test_execute_stream_delegates_to_context(self):
        """execute_stream() delegates to ctx.run_tool_loop_stream()."""
        strategy = DefaultStrategy(max_iterations=15)

        # Create async generator for mock
        async def mock_stream(*args, **kwargs):
            yield TextDeltaEvent(run_id="test", delta="Hello")
            yield TextDeltaEvent(run_id="test", delta=" World")

        mock_ctx = MagicMock(spec=StrategyContext)
        mock_ctx.run_tool_loop_stream = mock_stream

        messages = [Message(role="user", content="Hi")]

        events = []
        async for event in strategy.execute_stream(mock_ctx, messages):
            events.append(event)

        assert len(events) == 2
        assert events[0].delta == "Hello"
        assert events[1].delta == " World"

    @pytest.mark.asyncio
    async def test_execute_stream_yields_all_events(self):
        """execute_stream() yields all events from context."""
        strategy = DefaultStrategy()

        tc = ToolCall(id="1", name="test_tool", params={})
        tr = ToolResult(tool_use_id="1", content="result")

        async def mock_stream(*args, **kwargs):
            yield TextDeltaEvent(run_id="test", delta="Thinking...")
            yield ToolStartEvent(run_id="test", tool_call=tc)
            yield ToolEndEvent(run_id="test", tool_call_id="1", result=tr)
            yield TextDeltaEvent(run_id="test", delta="Done!")

        mock_ctx = MagicMock(spec=StrategyContext)
        mock_ctx.run_tool_loop_stream = mock_stream

        messages = [Message(role="user", content="Do something")]

        events = []
        async for event in strategy.execute_stream(mock_ctx, messages):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[0], TextDeltaEvent)
        assert isinstance(events[1], ToolStartEvent)
        assert isinstance(events[2], ToolEndEvent)
        assert isinstance(events[3], TextDeltaEvent)

