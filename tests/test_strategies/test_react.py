"""Tests for ReActStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from voxagent.strategies.react import ReActStrategy
from voxagent.strategies.base import StrategyContext, StrategyResult
from voxagent.types.messages import Message, ToolCall, ToolResult


class TestReActStrategy:
    """Tests for ReActStrategy."""

    def test_init(self):
        """ReActStrategy initializes with default or custom values."""
        strategy = ReActStrategy()
        assert strategy._max_steps == 10
        assert "thought" in strategy._system_prompt.lower()

    @pytest.mark.asyncio
    async def test_execute_success_with_tool(self):
        """ReActStrategy executes tools and provides final answer."""
        strategy = ReActStrategy(max_steps=5)

        mock_ctx = MagicMock()
        mock_ctx.prompt = "What is the weather?"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = False

        # Step 1: LLM says Thought + Action
        step1_text = """Thought: I need to check the weather.
Action: get_weather
Action Input: {"location": "London"}"""
        
        # Step 2: Tool execution (mocked by StrategyContext.execute_tool usually, 
        # but ReActStrategy calls it directly)
        mock_tool_result = ToolResult(tool_use_id="react_step_1", content="Sunny, 25C")
        
        # Step 3: LLM says Thought + Final Answer
        step2_text = """Thought: I have the info.
Action: FINISH
Action Input: The weather in London is sunny and 25C."""


        mock_ctx.call_llm = AsyncMock()
        mock_ctx.call_llm.side_effect = [
            (step1_text, []),
            (step2_text, []),
        ]
        
        mock_ctx.execute_tool = AsyncMock(return_value=mock_tool_result)

        result = await strategy.execute(mock_ctx)

        assert result.metadata["steps"] == 2
        assert "The weather in London is sunny and 25C" in result.assistant_texts[-1]
        assert result.metadata["finished"] is True

    @pytest.mark.asyncio
    async def test_abort_handling(self):
        """ReActStrategy respects abort signal."""
        strategy = ReActStrategy()

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Do task"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = True

        result = await strategy.execute(mock_ctx)
        assert result.error == "Aborted"
