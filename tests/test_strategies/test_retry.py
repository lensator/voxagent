"""Tests for RetryStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from voxagent.strategies.retry import RetryStrategy
from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult
from voxagent.strategies.default import DefaultStrategy


class TestRetryStrategy:
    """Tests for RetryStrategy."""

    def test_init(self):
        """RetryStrategy initializes with inner strategy and retries."""
        inner = DefaultStrategy()
        strategy = RetryStrategy(inner_strategy=inner, max_retries=3)
        assert strategy._inner_strategy == inner
        assert strategy._max_retries == 3

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self):
        """RetryStrategy succeeds on first try if inner strategy succeeds."""
        inner = MagicMock(spec=AgentStrategy)
        mock_result = StrategyResult(messages=[], assistant_texts=["Success"], error=None)
        inner.execute = AsyncMock(return_value=mock_result)

        strategy = RetryStrategy(inner_strategy=inner)
        mock_ctx = MagicMock()
        mock_ctx.abort_controller.signal.aborted = False

        result = await strategy.execute(mock_ctx)

        assert result.assistant_texts == ["Success"]
        assert result.metadata["attempts"] == 1
        inner.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_retry_on_error(self):
        """RetryStrategy retries if inner strategy returns error."""
        inner = MagicMock(spec=AgentStrategy)
        fail_result = StrategyResult(messages=[], assistant_texts=["Fail"], error="Some error")
        success_result = StrategyResult(messages=[], assistant_texts=["Success"], error=None)
        
        inner.execute = AsyncMock()
        inner.execute.side_effect = [fail_result, success_result]

        strategy = RetryStrategy(inner_strategy=inner, max_retries=2)
        mock_ctx = MagicMock()
        mock_ctx.abort_controller.signal.aborted = False

        result = await strategy.execute(mock_ctx)

        assert result.assistant_texts == ["Success"]
        assert result.metadata["attempts"] == 2
        assert inner.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_max_retries_reached(self):
        """RetryStrategy returns last error if max retries reached."""
        inner = MagicMock(spec=AgentStrategy)
        fail_result = StrategyResult(messages=[], assistant_texts=["Fail"], error="Some error")
        
        inner.execute = AsyncMock(return_value=fail_result)

        strategy = RetryStrategy(inner_strategy=inner, max_retries=2)
        mock_ctx = MagicMock()
        mock_ctx.abort_controller.signal.aborted = False

        result = await strategy.execute(mock_ctx)

        assert result.error == "Some error"
        assert result.metadata["attempts"] == 3 # Initial + 2 retries
        assert inner.execute.call_count == 3
