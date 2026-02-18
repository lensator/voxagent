"""Plan and execute strategy for voxagent agents.

The PlanAndExecuteStrategy implements a two-phase approach:
1. Planning: Create a step-by-step plan for the user's request.
2. Execution: Execute each step using the canonical tool loop.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, AsyncIterator

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData
    from voxagent.types.messages import Message


DEFAULT_PLANNER_PROMPT = """
Create a detailed step-by-step plan to solve the following request.
Each step should be on a new line starting with a number followed by a period (e.g., '1. ').
Focus only on the plan steps.
"""


class PlanAndExecuteStrategy(AgentStrategy):
    """Strategy that plans first then executes steps.
    
    This strategy uses a planner phase to generate a list of steps,
    then executes each step sequentially using the standard tool loop.
    
    Args:
        max_steps: Maximum number of steps to execute (default: 10).
        planner_prompt: Prompt for the planning phase.
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        planner_prompt: str = DEFAULT_PLANNER_PROMPT,
    ) -> None:
        """Initialize the PlanAndExecuteStrategy.
        
        Args:
            max_steps: Maximum steps to execute.
            planner_prompt: Prompt for planning phase.
        """
        self._max_steps = max_steps
        self._planner_prompt = planner_prompt
    
    def _build_initial_messages(self, ctx: StrategyContext) -> list["Message"]:
        """Build the initial messages list from context."""
        from voxagent.types.messages import Message
        
        messages: list[Message] = []
        if ctx.message_history:
            messages.extend(ctx.message_history)
            
        messages.append(Message(role="user", content=ctx.prompt))
        return messages

    def _parse_plan(self, text: str) -> list[str]:
        """Parse plan steps from text.
        
        Looks for lines starting with '1. ', '2. ', etc.
        """
        steps = []
        lines = text.split("\n")
        for line in lines:
            match = re.match(r"^\s*(\d+)\.\s*(.+)$", line)
            if match:
                steps.append(match.group(2).strip())
        return steps

    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the planning and execution phases.
        
        Args:
            ctx: Strategy context with LLM and tool access.
        
        Returns:
            StrategyResult with messages, assistant texts, and tool metadata.
        """
        from voxagent.types.messages import Message as Msg
        
        # REQUIRED: Check abort at start
        if ctx.abort_controller.signal.aborted:
            return StrategyResult(
                messages=[],
                assistant_texts=[],
                tool_metas=[],
                error="Aborted",
            )
            
        messages = self._build_initial_messages(ctx)
        assistant_texts: list[str] = []
        tool_metas = []
        
        # --- 1. PLANNING PHASE ---
        planning_messages = list(messages)
        planning_messages.append(Msg(role="user", content=self._planner_prompt))
        
        plan_text, _ = await ctx.call_llm(planning_messages, tools=None)
        
        assistant_texts.append(plan_text)
        # We don't necessarily add plan_text to messages if we want to keep it clean,
        # but usually it's helpful context.
        messages.append(Msg(role="assistant", content=plan_text))
        
        plan_steps = self._parse_plan(plan_text)
        
        if not plan_steps:
            # Fallback if no numbered steps found: treat whole text as one step
            plan_steps = [plan_text.strip()]
            
        steps_executed = 0
        
        # --- 2. EXECUTION PHASE ---
        for step in plan_steps[:self._max_steps]:
            steps_executed += 1
            
            # REQUIRED: Check abort between steps
            if ctx.abort_controller.signal.aborted:
                return StrategyResult(
                    messages=messages,
                    assistant_texts=assistant_texts,
                    tool_metas=tool_metas,
                    error="Aborted",
                    metadata={"plan": plan_steps, "steps_executed": steps_executed},
                )
            
            # Request execution of the current step
            messages.append(Msg(role="user", content=f"Now execute Step {steps_executed}: {step}"))
            
            msgs, texts, metas = await ctx.run_tool_loop(messages)
            
            messages = msgs
            assistant_texts.extend(texts)
            tool_metas.extend(metas)
            
        return StrategyResult(
            messages=messages,
            assistant_texts=assistant_texts,
            tool_metas=tool_metas,
            metadata={
                "strategy_name": self.name,
                "plan": plan_steps,
                "steps_executed": steps_executed,
            }
        )
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the planning cycle with streaming.
        
        Delegates to execute() and yields synthetic events.
        """
        from voxagent.streaming.events import TextDeltaEvent
        
        result = await self.execute(ctx)
        for text in result.assistant_texts:
            yield TextDeltaEvent(run_id=ctx.run_id, delta=text)

__all__ = ["PlanAndExecuteStrategy"]
