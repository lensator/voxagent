from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, List

from voxagent.strategies.base import AgentStrategy, StrategyContext
from voxagent.types.messages import Message as Msg
from voxagent.streaming.events import (
    TextDeltaEvent, ToolEndEvent, AssistantEndEvent, 
    ProviderRequestEvent, InternalThoughtEvent, ToolStartEvent, ToolOutputEvent
)
from voxagent.providers.registry import get_default_registry

if TYPE_CHECKING:
    from voxagent.agent.core import Agent

logger = logging.getLogger(__name__)

class HomeOrchestratorStrategy(AgentStrategy):
    """Dual-stage strategy for Home Automation.
    
    Stage 1: Fast local model (Ollama) detects intent and executes device tools.
    Stage 2: Power model (Gemini) synthesizes the final response for the user.
    """

    def __init__(self, fast_model: str = "ollama:qwen2.5:1.5b", max_iterations: int = 5):
        super().__init__()
        self._fast_model = fast_model
        self._max_iterations = max_iterations

    async def _assemble_environment_context(self, ctx: StrategyContext, prompt: str) -> str:
        """Assemble the full context for the Power Model."""
        if not ctx.memory_manager:
            return "No environment context available."

        facts = await ctx.memory_manager.search_facts(prompt)
        goals = await ctx.memory_manager.list_active_goals()
        
        # Priority 1: Core Protocol (The 'Constitution' of the Agent)
        core_protocol = ""
        for f in facts:
            if "core_protocol.md" in f.get('source', ''):
                core_protocol = f['content']
                break

        context_parts = [
            "[INTERNAL_ENVIRONMENT_CONTEXT_START]",
            "## CORE_PROTOCOL",
            core_protocol or "Follow the Home Orchestrator rules.",
            "\n## CURRENT_ENVIRONMENT_STATE",
            f"TIME: {datetime.now(timezone.utc).isoformat()}",
        ]
        
        if facts:
            context_parts.append("\nKNOWLEDGE_BASE:")
            for f in facts:
                context_parts.append(f"- {f['content']} (Source: {f['source']})")
        
        if goals:
            context_parts.append("\nACTIVE_MISSIONS:")
            for g in goals:
                context_parts.append(f"- {g['description']} ({g['status']})")
        
        context_parts.append("[INTERNAL_ENVIRONMENT_CONTEXT_END]")
        
        context_parts.append(
            "\nDISTILLATION_INSTRUCTIONS:\n"
            "1. You are receiving a complex 'Environment Context'.\n"
            "2. Reconcile the user's prompt with the facts and goals provided above.\n"
            "3. Provide a simple, high-quality response in the user's current language (the same language the user is speaking).\n"
            "4. Never mention internal IDs, memory sources, or system tags.\n"
            "5. If a mission is updated, explain the next steps clearly."
        )
        
        return "\n".join(context_parts)

    async def execute(
        self,
        ctx: StrategyContext,
        messages: List[Msg],
    ) -> StrategyResult:
        from voxagent.strategies.base import StrategyResult
        
        user_prompt = ""
        for m in reversed(messages):
            role = m.role if hasattr(m, "role") else m.get("role")
            if role == "user":
                user_prompt = str(m.content if hasattr(m, "content") else m.get("content", ""))
                break

        # Keyword Gate
        device_keywords = [
            "light", "lamp", "led", "wled", "switch", "plug", "socket", 
            "temp", "thermostat", "heat", "ac", "air", "fan", "blind", "shutter",
            "on", "off", "open", "close", "status", "state", "room", "kitchen", "bedroom",
            "balcony", "rename", "device", "entity", "prefix", "rainbow", "effect", "item", "thing"
        ]
        has_keyword = any(kw in user_prompt.lower() for kw in device_keywords)

        # 1. Intent Stage: Local model handles devices/tools
        registry = get_default_registry()
        local_provider = None
        try:
            local_provider = registry.get_provider(self._fast_model)
        except Exception:
            pass

        local_results = []
        local_text = ""
        is_generic = not has_keyword

        if local_provider and has_keyword:
            # Build history string for the small model
            history_summary = ""
            if messages:
                history_parts = []
                for m in messages[-5:]:
                    role = m.role if hasattr(m, "role") else m.get("role")
                    content = m.content if hasattr(m, "content") else m.get("content", "")
                    history_parts.append(f"{role.upper()}: {content}")
                history_summary = "\n".join(history_parts)

            intent_prompt = (
                "SYSTEM: You are a hardware command filter for Home Automation (Home Assistant / OpenHAB).\n"
                "TOOLS AVAILABLE:\n"
                "- hass_list_entities / openhab_list_items: List devices and items.\n"
                "- hass_control_device / openhab_send_command: Control a device or send a command to an item.\n"
                "- hass_get_state / openhab_get_item: Get detailed status.\n"
                "- platform_setup_tools: Tools for renaming, mapping, and syncing the environment.\n\n"
                "CATEGORIES:\n"
                "- HARDWARE CONTROL: 'lights on', 'set balcony to rainbow', 'is the AC on?'.\n"
                "- DEVICE MANAGEMENT: 'rename device', 'sync platform', 'map alias'.\n"
                "- GENERAL CHAT: 'hello', 'tell me a joke'.\n\n"
                "RULES:\n"
                "1. If it is a hardware or management command: USE THE TOOLS and respond 'DEVICE_REQUEST_EXECUTED'.\n"
                "2. If it is a command but ambiguous: Respond 'DEVICE_REQUEST_UNCLEAR: [reason]'.\n"
                "3. If it is NOT a device command: Respond ONLY with 'NOT_A_DEVICE_COMMAND'.\n\n"
                f"CONVERSATION HISTORY:\n{history_summary}\n\n"
                f"USER MESSAGE: {user_prompt}"
            )
            
            local_ctx = StrategyContext(
                provider=local_provider,
                tools=ctx.tools,
                system_prompt=None,
                abort_signal=ctx.abort_signal,
                run_id=ctx.run_id,
                memory_manager=ctx.memory_manager,
                deps=ctx.deps
            )

            async for event in local_ctx.run_tool_loop_stream([Msg(role="user", content=intent_prompt)], max_iterations=self._max_iterations):
                if isinstance(event, ProviderRequestEvent):
                    yield event
                elif isinstance(event, ToolStartEvent):
                    yield event
                elif isinstance(event, ToolOutputEvent):
                    local_results.append(f"Tool {event.tool_call_id} result: {event.delta}")
                    yield event
                elif isinstance(event, ToolEndEvent):
                    if not event.result.is_error:
                        local_text = "DEVICE_REQUEST_EXECUTED"
                        # We stop the loop once we have a result to prevent small models from looping
                        break
                elif isinstance(event, TextDeltaEvent):
                    local_text += event.delta
                    if "NOT_A_DEVICE_COMMAND" in local_text:
                        is_generic = True
                    # Yield with 'local' module for live debug
                    yield TextDeltaEvent(run_id=ctx.run_id, delta=event.delta, module="local")
            
            if local_text:
                from voxagent.streaming.events import InternalThoughtEvent
                yield InternalThoughtEvent(run_id=ctx.run_id, content=local_text, module="Local Intent Model")

        # 2. Synthesis Stage: Power model (Gemini)
        action_desc = "Escalated (Non-device query)" if is_generic else "Processed device command"
        findings = local_text if not is_generic else "N/A"
        
        synthesis_context = (
            f"\n[INTERNAL_SYNTHESIS_CONTEXT]\n"
            f"LOCAL_MODEL_ACTION: {action_desc}\n"
            f"LOCAL_MODEL_FINDINGS: {findings}\n"
            f"TOOL_RESULTS: {', '.join(local_results) if local_results else 'None'}\n"
            f"[/INTERNAL_SYNTHESIS_CONTEXT]\n"
        )

        env_context = await self._assemble_environment_context(ctx, user_prompt)
        system_instruction = (
            f"You are 'Samaritan', a proactive and goal-oriented home orchestrator.\n"
            f"{env_context}\n"
            f"{synthesis_context}"
        )

        from voxagent.providers.base import TextDeltaChunk, ToolUseChunk, ErrorChunk
        async for chunk in ctx.provider.stream([m if isinstance(m, Msg) else Msg(**m) for m in messages], system=system_instruction, abort_signal=ctx.abort_signal):
            if isinstance(chunk, TextDeltaChunk):
                yield TextDeltaEvent(run_id=ctx.run_id, delta=chunk.delta)
            elif isinstance(chunk, ToolUseChunk):
                yield ToolStartEvent(run_id=ctx.run_id, tool_call=chunk.tool_call)
            elif isinstance(chunk, ErrorChunk):
                from voxagent.streaming.events import RunErrorEvent
                yield RunErrorEvent(run_id=ctx.run_id, error=chunk.error)
