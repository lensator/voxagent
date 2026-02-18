"""Tests for PlanAndExecuteStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from voxagent.strategies.planning import PlanAndExecuteStrategy
from voxagent.strategies.base import StrategyContext, StrategyResult
from voxagent.types.messages import Message


class TestPlanAndExecuteStrategy:
    """Tests for PlanAndExecuteStrategy."""

    def test_init(self):
        """PlanAndExecuteStrategy initializes with default or custom values."""
        strategy = PlanAndExecuteStrategy()
        assert strategy._max_steps == 10
        assert "plan" in strategy._planner_prompt.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """PlanAndExecuteStrategy creates and executes a plan."""
        strategy = PlanAndExecuteStrategy(max_steps=2)

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Do complex task"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = False

        # 1. Planning phase
        # mock call_llm returns a plan
        plan_text = """Plan:
1. Step one
2. Step two"""
        mock_ctx.call_llm = AsyncMock(return_value=(plan_text, []))

        # 2. Execution phase
        # mock run_tool_loop for each step
        mock_ctx.run_tool_loop = AsyncMock()
        mock_ctx.run_tool_loop.side_effect = [
            ([Message(role="assistant", content="Step 1 done")], ["Step 1 done"], []), # Step 1
            ([Message(role="assistant", content="Step 2 done")], ["Step 2 done"], []), # Step 2
        ]

        result = await strategy.execute(mock_ctx)

        assert result.metadata["plan"] == ["Step one", "Step two"]
        assert result.metadata["steps_executed"] == 2
        assert len(result.assistant_texts) == 3 # Plan + Step1 + Step2
        assert "Step 1 done" in result.assistant_texts
        assert "Step 2 done" in result.assistant_texts

    @pytest.mark.asyncio
    async def test_abort_handling(self):
        """PlanAndExecuteStrategy respects abort signal."""
        strategy = PlanAndExecuteStrategy()

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Do task"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = True

        result = await strategy.execute(mock_ctx)
        assert result.error == "Aborted"
