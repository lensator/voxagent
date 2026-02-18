"""Tests for ReflectionStrategy."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from voxagent.strategies.reflection import ReflectionStrategy
from voxagent.strategies.base import StrategyContext, StrategyResult
from voxagent.types.messages import Message
from voxagent.streaming.events import TextDeltaEvent


class TestReflectionStrategy:
    """Tests for ReflectionStrategy."""

    def test_init(self):
        """ReflectionStrategy initializes with default or custom values."""
        strategy = ReflectionStrategy()
        assert strategy._max_iterations == 3
        assert "accuracy" in strategy._critique_prompt.lower()

        strategy = ReflectionStrategy(max_iterations=5, critique_prompt="Review this")
        assert strategy._max_iterations == 5
        assert strategy._critique_prompt == "Review this"

    @pytest.mark.asyncio
    async def test_execute_approved_first_try(self):
        """ReflectionStrategy finishes early if approved."""
        strategy = ReflectionStrategy(max_iterations=3)

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Write a poem"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = False

        # 1. Initial generation
        # Original: user message. Generation adds assistant message.
        gen_messages = [
            Message(role="user", content="Write a poem"),
            Message(role="assistant", content="The rose is red")
        ]
        mock_ctx.run_tool_loop = AsyncMock(return_value=(gen_messages, ["The rose is red"], []))

        # 2. Critique returns APPROVED
        mock_ctx.call_llm = AsyncMock(return_value=("APPROVED", []))

        result = await strategy.execute(mock_ctx)

        assert result.assistant_texts == ["The rose is red", "APPROVED"]
        assert len(result.messages) == 4 # user + assistant + critique_request + critique_response
        assert result.metadata["iterations"] == 1
        assert result.metadata["approved"] is True

    @pytest.mark.asyncio
    async def test_execute_multiple_iterations(self):
        """ReflectionStrategy runs multiple iterations until approved or limit reached."""
        strategy = ReflectionStrategy(max_iterations=2)

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Write a poem"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = False

        # Iteration 1
        gen_messages_1 = [
            Message(role="user", content="Write a poem"),
            Message(role="assistant", content="Bad poem")
        ]
        
        # Iteration 2
        # It will contain: user, assistant1, critique1_req, critique1_res, assistant2
        gen_messages_2 = [
            Message(role="user", content="Write a poem"),
            Message(role="assistant", content="Bad poem"),
            Message(role="user", content="CRITIQUE_PROMPT"),
            Message(role="assistant", content="It's bad, fix it."),
            Message(role="assistant", content="Better poem")
        ]

        mock_ctx.run_tool_loop = AsyncMock()
        mock_ctx.run_tool_loop.side_effect = [
            (gen_messages_1, ["Bad poem"], []),
            (gen_messages_2, ["Better poem"], []),
        ]

        # Critique 1: Not approved, Critique 2: Approved (though we reach max_iterations=2)
        mock_ctx.call_llm = AsyncMock()
        mock_ctx.call_llm.side_effect = [
            ("It's bad, fix it.", []), # Critique 1
            ("Still bad.", []), # Critique 2
        ]

        result = await strategy.execute(mock_ctx)

        assert result.metadata["iterations"] == 2
        assert result.metadata["approved"] is False
        assert len(result.assistant_texts) == 4 # gen1, crit1, gen2, crit2

    @pytest.mark.asyncio
    async def test_abort_handling(self):
        """ReflectionStrategy respects abort signal."""
        strategy = ReflectionStrategy()

        mock_ctx = MagicMock()
        mock_ctx.prompt = "Write a poem"
        mock_ctx.message_history = []
        mock_ctx.run_id = "test-run"
        mock_ctx.abort_controller.signal.aborted = True

        result = await strategy.execute(mock_ctx)
        assert result.error == "Aborted"
