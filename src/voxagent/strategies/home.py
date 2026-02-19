"""Home Orchestrator Strategy for Samaritan.

Single-stage workflow using fast model only:
1. Fast Model detects device intent and executes commands
2. Returns simple confirmation or NOT_A_DEVICE_COMMAND for non-device queries
"""

import logging
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, List

import httpx

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

    def __init__(
        self,
        fast_model: str = "ollama:qwen2.5-coder:3b",
        max_iterations: int = 5,
        excluded_devices: list[str] | None = None,
        debug: bool = False
    ):
        super().__init__()
        self._fast_model = fast_model
        self._max_iterations = max_iterations
        self._excluded_devices = set(excluded_devices or [])
        self._debug = debug

    def get_required_tools(self) -> List[Any]:
        """Return Home Assistant tools required by this strategy."""
        from voxagent.tools.home_assistant import (
            hass_list_entities,
            hass_control_device,
            hass_get_state,
        )
        return [hass_list_entities, hass_control_device, hass_get_state]

    async def _get_device_list(self, ctx: StrategyContext) -> str:
        """Fetch list of devices from Home Assistant API directly."""
        url = os.environ.get("HASS_URL", "http://localhost:8123").rstrip("/")
        token = os.environ.get("HASS_TOKEN")

        if not token:
            logger.warning("HASS_TOKEN not set, cannot fetch devices")
            return "No devices available (HASS_TOKEN not set)"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{url}/api/states",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    timeout=10.0
                )
                if response.status_code != 200:
                    logger.error(f"HASS API error: {response.status_code}")
                    return "No devices available (API error)"

                entities = response.json()
                # Filter to only controllable devices, excluding configured ones
                valid_domains = ["light", "switch"]
                result = []
                for e in entities:
                    entity_id = e["entity_id"]
                    # Skip excluded devices
                    if entity_id in self._excluded_devices:
                        continue
                    domain = entity_id.split(".")[0]
                    if domain in valid_domains:
                        name = e.get("attributes", {}).get("friendly_name", entity_id)
                        state = e["state"]
                        device_type = "Light" if domain == "light" else "Switch"
                        result.append(f"HARDWARE_INFO: Name: {name}. ID: {entity_id}. Status: {state.upper()}. Type: {device_type}.")

                return "\n".join(result) if result else "No controllable devices found."
        except Exception as e:
            logger.error(f"Error fetching device list from HASS: {e}")
            return f"No devices available (error: {e})"

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

        # Fast model: minimal technical prompt, no persona
        intent_system = (
            "Detect device intent. "
            "If device command: call tool, output 'DONE'. "
            "If not: output 'NOT_A_DEVICE_COMMAND'. "
            "Use history to resolve 'it', 'that'.\n\n"
            f"DEVICES:\n{device_list}"
        )

        # Include ONLY user/assistant messages from history (no system prompts)
        intent_messages = []
        if ctx.message_history:
            for msg in ctx.message_history:
                if msg.role in ("user", "assistant"):
                    intent_messages.append(msg)
        intent_messages.append(Msg(role="user", content=user_prompt))
        
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
        seen_tool_calls = set()  # Track unique tool calls to detect duplicates

        # Run Tool Loop with Fast Model (SILENT to user)
        async for event in local_ctx.run_tool_loop_stream(intent_messages, max_iterations=self._max_iterations):
            if isinstance(event, TextDeltaEvent):
                intent_output += event.delta
                # NEVER yield text from the intent module to the user
            elif isinstance(event, ToolStartEvent):
                # Create a signature for this tool call to detect duplicates
                tc = event.tool_call
                call_signature = f"{tc.name}:{tc.params}"
                if call_signature in seen_tool_calls:
                    # Duplicate tool call detected - model is looping, stop here
                    logger.debug(f"Duplicate tool call detected: {call_signature}, breaking loop")
                    break
                seen_tool_calls.add(call_signature)
                is_device_command = True
                yield event
            elif isinstance(event, (ToolEndEvent, ToolOutputEvent)):
                is_device_command = True
                if isinstance(event, ToolOutputEvent):
                    tool_results.append(event.delta)
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
        
        # Use system_prompt from config files (personas, rules loaded by client)
        base_system = ctx.system_prompt or ""

        if is_not_device:
            # General query - use full system prompt from config
            synthesis_system = base_system
        else:
            # Device command executed - append result context
            synthesis_system = (
                f"{base_system}\n\n"
                f"[DEVICE_ACTION_RESULT]\n"
                f"Results: {tool_results}\n"
                f"Confirm concisely to the user."
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

        # Track assistant response for session persistence
        assistant_response = ""

        # Final synthesis with Power Model
        # Note: tools=None because fast model already executed device commands
        async for chunk in ctx.provider.stream(
            messages=final_messages,
            system=synthesis_system,
            tools=None,
            abort_signal=ctx.abort_controller.signal
        ):
            if self._debug:
                print(f"CHUNK: {type(chunk).__name__}")
            if isinstance(chunk, TextDeltaChunk):
                assistant_response += chunk.delta
                yield TextDeltaEvent(run_id=ctx.run_id, delta=chunk.delta)
            elif isinstance(chunk, ToolUseChunk):
                yield ToolStartEvent(run_id=ctx.run_id, tool_call=chunk.tool_call)
            elif isinstance(chunk, ErrorChunk):
                yield RunErrorEvent(run_id=ctx.run_id, error=chunk.error)
            elif isinstance(chunk, MessageEndChunk):
                break

        # Save session with the new messages (uses shared session storage from base class)
        if ctx.session_key and assistant_response:
            final_messages.append(Msg(role="assistant", content=assistant_response))
            await self.save_to_session(ctx.session_key, final_messages)

__all__ = ["HomeOrchestratorStrategy"]
