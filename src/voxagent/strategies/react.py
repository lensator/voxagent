"""ReAct strategy for voxagent agents.

The ReActStrategy implements the Thought-Action-Observation pattern:
1. Thought: Reasoning about the current state.
2. Action: Choosing a tool to execute or FINISH.
3. Observation: Result from the tool execution.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, AsyncIterator

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData
    from voxagent.types.messages import Message


DEFAULT_REACT_PROMPT = """
You operate using the Thought-Action-Observation loop.
For each step, follow this format exactly:

Thought: <your reasoning here>
Action: <tool_name or FINISH>
Action Input: <JSON parameters for the tool or your final answer>

Available tools:
{tool_descriptions}

If you have the final answer, use Action: FINISH and put your answer in Action Input.
"""


class ReActStrategy(AgentStrategy):
    """Strategy that implements the ReAct reasoning loop.
    
    This strategy uses a specialized system prompt and structured output
    parsing to drive a Thought-Action-Observation cycle.
    
    Args:
        max_steps: Maximum number of steps to execute (default: 10).
        system_prompt: Custom system prompt for ReAct.
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        system_prompt: str = DEFAULT_REACT_PROMPT,
    ) -> None:
        """Initialize the ReActStrategy.
        
        Args:
            max_steps: Maximum steps to execute.
            system_prompt: Custom system prompt.
        """
        self._max_steps = max_steps
        self._system_prompt = system_prompt
    
    def _build_initial_messages(self, ctx: StrategyContext) -> list["Message"]:
        """Build the initial messages list from context."""
        from voxagent.types.messages import Message
        
        # Format tool descriptions into system prompt
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in ctx.tools
        ])
        formatted_system = self._system_prompt.format(tool_descriptions=tool_descriptions)
        
        messages: list[Message] = [
            Message(role="system", content=formatted_system)
        ]
        
        if ctx.message_history:
            messages.extend(ctx.message_history)
            
        messages.append(Message(role="user", content=ctx.prompt))
        return messages

    def _parse_output(self, text: str) -> tuple[str | None, str | None, str | None]:
        """Parse Thought, Action, and Action Input from text.
        
        Returns:
            (thought, action, action_input) tuple.
        """
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\S+)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?=Thought:|$)", text, re.DOTALL)
        
        return (
            thought_match.group(1).strip() if thought_match else None,
            action_match.group(1).strip() if action_match else None,
            input_match.group(1).strip() if input_match else None,
        )

    async def execute(
        self,
        ctx: StrategyContext,
    ) -> StrategyResult:
        """Execute the ReAct loop.
        
        Args:
            ctx: Strategy context with LLM and tool access.
        
        Returns:
            StrategyResult with messages, assistant texts, and tool metadata.
        """
        from voxagent.types.messages import Message as Msg
        from voxagent.types.messages import ToolCall
        from voxagent.types.run import ToolMeta
        
        # REQUIRED: Check abort
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
        finished = False
        step_count = 0
        
        for i in range(self._max_steps):
            step_count += 1
            
            # REQUIRED: Check abort
            if ctx.abort_controller.signal.aborted:
                return StrategyResult(
                    messages=messages,
                    assistant_texts=assistant_texts,
                    tool_metas=tool_metas,
                    error="Aborted",
                    metadata={"steps": step_count, "finished": finished},
                )
            
            # Call LLM for the current step
            response_text, _ = await ctx.call_llm(messages, tools=None)
            
            assistant_texts.append(response_text)
            messages.append(Msg(role="assistant", content=response_text))
            
            thought, action, action_input = self._parse_output(response_text)
            
            if not action or action.upper() == "FINISH":
                finished = True
                break
                
            # Execute tool
            try:
                # Parse params from JSON
                params = {}
                if action_input:
                    try:
                        params = json.loads(action_input)
                    except json.JSONDecodeError:
                        # Fallback for simple string input if not JSON
                        params = {"query": action_input}
                
                tool_call_id = f"react_step_{step_count}"
                tc = ToolCall(id=tool_call_id, name=action, params=params)
                
                import time
                start_time = time.perf_counter()
                result = await ctx.execute_tool(tc)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                
                tool_metas.append(ToolMeta(
                    tool_name=action,
                    tool_call_id=tool_call_id,
                    execution_time_ms=elapsed_ms,
                    success=not result.is_error,
                    error=result.content if result.is_error else None,
                ))
                
                # Add observation as user message
                observation = f"Observation: {result.content}"
                messages.append(Msg(role="user", content=observation))
                
            except Exception as e:
                error_msg = f"Observation: Error executing tool {action}: {e}"
                messages.append(Msg(role="user", content=error_msg))
                
        return StrategyResult(
            messages=messages,
            assistant_texts=assistant_texts,
            tool_metas=tool_metas,
            metadata={
                "strategy_name": self.name,
                "steps": step_count,
                "finished": finished,
            }
        )
    
    async def execute_stream(
        self,
        ctx: StrategyContext,
    ) -> AsyncIterator["StreamEventData"]:
        """Execute the ReAct cycle with streaming.
        
        Delegates to execute() and yields synthetic events.
        """
        from voxagent.streaming.events import TextDeltaEvent
        
        result = await self.execute(ctx)
        for text in result.assistant_texts:
            yield TextDeltaEvent(run_id=ctx.run_id, delta=text)

__all__ = ["ReActStrategy"]
