"""Home Orchestrator Strategy for Samaritan.

Strictly follows the Dual-Stage Workflow:
1. Intent Module (Fast Model): Check if it is a device control command.
2. Tool Execution: If it is a device command, execute it using the fast model.
3. Strategy Selection: If not a device command, or after execution, choose a strategy (e.g. Synthesis) to handle the response.
"""

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, List

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult
from voxagent.types.messages import Message as Msg
from voxagent.streaming.events import (
    TextDeltaEvent, ToolEndEvent, ToolStartEvent, ToolOutputEvent,
    StreamEventData, RunErrorEvent
)
from voxagent.providers.registry import get_default_registry

if TYPE_CHECKING:
    from voxagent.agent.core import Agent

logger = logging.getLogger(__name__)

class HomeOrchestratorStrategy(AgentStrategy):
    """Orchestrates Samaritan behavior using a dual-stage intent/synthesis approach."""

    def __init__(self, fast_model: str = "ollama:qwen2.5-coder:3b", max_iterations: int = 5, debug: bool = False):
        super().__init__()
        self._fast_model = fast_model
        self._max_iterations = max_iterations
        self._debug = debug

    async def _get_device_list(self, ctx: StrategyContext) -> str:
        """Fetch list of devices from LanceDB memory with fallback."""
        devices = []
        if ctx.memory_manager:
            try:
                facts = await ctx.memory_manager.search_facts("HARDWARE_INFO", limit=50)
                for f in facts:
                    devices.append(f.get("content", ""))
            except Exception as e:
                logger.error(f"Error fetching device list: {e}")
        
        if not devices:
            # Fallback for testing environments without working LanceDB
            return "HARDWARE_INFO: Name: Kitchen Light. ID: light.kitchen. Status: ON. Type: Light."
            
        return "\n".join(devices)

    async def execute(self, ctx: StrategyContext) -> StrategyResult:
        """Synchronous execution (not optimized for this strategy)."""
        # For simplicity, we'll just run the stream logic and collect it
        assistant_texts = []
        messages = []
        async for event in self.execute_stream(ctx):
            if isinstance(event, TextDeltaEvent):
                assistant_texts.append(event.delta)
        
        return StrategyResult(
            messages=ctx.message_history or [],
            assistant_texts=["".join(assistant_texts)],
        )

    async def execute_stream(self, ctx: StrategyContext) -> AsyncIterator[StreamEventData]:
        """Dual-stage execution flow."""
        if self._debug:
            print("DEBUG: Entering execute_stream")
        # 1. Fetch Context (History + Devices)
        device_list = await self._get_device_list(ctx)
        user_prompt = ctx.prompt
        
        # 2. INTENT MODULE (Fast Model)
        registry = get_default_registry()
        fast_provider = registry.get_provider(self._fast_model)
        
        if not fast_provider:
            yield RunErrorEvent(run_id=ctx.run_id, error=f"Fast model provider {self._fast_model} not found.")
            return

        # Prepare Intent Context
        intent_system = (
            "SYSTEM: You are the Samaritan Intent Module. "
            "Your ONLY job is to determine if the user wants to control hardware (lights, lamps, etc) or check their status. "
            "1. If it IS a hardware command: Choose the correct tool from the list below and use it immediately. "
            "   Example: If user says 'turn off kitchen light' and you see 'ID: light.kitchen', call turn_off_light(entity_id='light.kitchen'). "
            "   After the tool call, output ONLY 'DONE'. "
            "2. If it IS NOT a hardware command: Output ONLY 'NOT_A_DEVICE_COMMAND'. "
            "DO NOT talk to the user. DO NOT explain yourself.\n\n"
            f"AVAILABLE DEVICES:\n{device_list}"
        )
        
        intent_messages = [Msg(role="user", content=user_prompt)]
        
        if self._debug:
            print("\n" + "="*20 + " DEBUG: INTENT STAGE (FAST MODEL) " + "="*20)
            print(f"MODEL: {self._fast_model}")
            print(f"SYSTEM:\n{intent_system}")
            print(f"MESSAGES: {intent_messages}")
            print("="*60 + "\n")

        # Create a local context for the fast model
        local_ctx = StrategyContext(
            prompt=user_prompt,
            deps=ctx.deps,
            session_key=ctx.session_key,
            message_history=ctx.message_history,
            timeout_ms=ctx.timeout_ms,
            provider=fast_provider,
            tools=ctx.tools,
            system_prompt=intent_system,
            abort_controller=ctx.abort_controller,
            run_id=ctx.run_id,
            memory_manager=ctx.memory_manager
        )

        is_device_command = False
        tool_results = []
        intent_output = ""

        # Run Tool Loop with Fast Model (SILENT to user)
        async for event in local_ctx.run_tool_loop_stream(intent_messages, max_iterations=self._max_iterations):
            if isinstance(event, TextDeltaEvent):
                intent_output += event.delta
                # NEVER yield text from the intent module to the user
            elif isinstance(event, (ToolStartEvent, ToolEndEvent, ToolOutputEvent)):
                is_device_command = True
                if isinstance(event, ToolOutputEvent):
                    tool_results.append(event.delta)
                # We yield tool events so the UI shows progress (e.g. "Calling get_light_status")
                yield event
            elif isinstance(event, RunErrorEvent):
                # If the fast model fails, we might still want the power model to try
                logger.error(f"Intent Module Error: {event.error}")
            else:
                # Other events (ProviderRequest, etc) stay internal or follow logic
                pass

        # 3. STRATEGY SELECTION (Synthesis / Handling)
        # Check if the fast model reported not being a device command
        is_not_device = "NOT_A_DEVICE_COMMAND" in intent_output or not is_device_command
        
        if is_not_device:
            # Handle as a general query or complex mission
            synthesis_system = (
                f"You are Samaritan. Follow the Core Protocol.\n"
                f"INTERNAL_SYSTEM_STATE: {device_list}\n"
                f"If the user asked for something related to the home, use the state to answer."
            )
        else:
            # Refine the execution result into a human answer
            synthesis_system = (
                f"You are Samaritan. You just executed a device command.\n"
                f"TOOL_RESULTS: {tool_results}\n"
                f"Confirm the action to the user concisely in their language."
            )

        if self._debug:
            print("\n" + "="*20 + " DEBUG: SYNTHESIS STAGE (POWER MODEL) " + "="*20)
            print(f"MODEL: {ctx.provider}")
            print(f"SYSTEM:\n{synthesis_system}")
            # Build messages for debug view
            debug_msgs = [Msg(role="system", content=synthesis_system)]
            if ctx.message_history:
                debug_msgs.extend(ctx.message_history)
            debug_msgs.append(Msg(role="user", content=ctx.prompt))
            print(f"MESSAGES: {debug_msgs}")
            print("="*60 + "\n")

        from voxagent.providers.base import TextDeltaChunk, ToolUseChunk, ErrorChunk, MessageEndChunk
        
        # Build final messages manually
        final_messages = []
        if ctx.message_history:
            final_messages.extend(ctx.message_history)
        final_messages.append(Msg(role="user", content=ctx.prompt))

        # Final synthesis with Power Model
        async for chunk in ctx.provider.stream(
            messages=final_messages, 
            system=synthesis_system, 
            tools=[t.to_openai_schema() for t in ctx.tools] if ctx.tools else None,
            abort_signal=ctx.abort_controller.signal
        ):
            if self._debug:
                print(f"CHUNK: {type(chunk).__name__}")
            if isinstance(chunk, TextDeltaChunk):
                yield TextDeltaEvent(run_id=ctx.run_id, delta=chunk.delta)
            elif isinstance(chunk, ToolUseChunk):
                yield ToolStartEvent(run_id=ctx.run_id, tool_call=chunk.tool_call)
            elif isinstance(chunk, ErrorChunk):
                yield RunErrorEvent(run_id=ctx.run_id, error=chunk.error)
            elif isinstance(chunk, MessageEndChunk):
                break

__all__ = ["HomeOrchestratorStrategy"]
