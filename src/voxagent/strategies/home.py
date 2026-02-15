"""Home Orchestrator strategy for voxagent.

Implements a Hybrid Routing workflow:
- Fast Local Path: Ollama handles direct device control in Greek.
- Power Cloud Path: Gemini handles everything else (planning, research, chat).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator, Any

from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult
from voxagent.memory.lancedb import LanceDBMemoryManager

if TYPE_CHECKING:
    from voxagent.streaming.events import StreamEventData
    from voxagent.types.messages import Message


class HomeOrchestratorStrategy(AgentStrategy):
    """Hybrid Router Strategy for the Personal Machine."""

    def __init__(
        self, 
        memory_manager: LanceDBMemoryManager | None = None,
        power_model: str = "google:gemini-1.5-pro",
        max_iterations: int = 10
    ):
        self._memory_manager = memory_manager or LanceDBMemoryManager()
        self._power_model = power_model
        self._max_iterations = max_iterations

    async def _is_device_command(self, ctx: StrategyContext, prompt: str) -> bool:
        """Use the local model to quickly classify the intent."""
        # Simple heuristic or a very fast 'classification' call to the local model
        # For now, we'll check if any tools are likely to be called.
        # A more robust version would use a tiny local LLM call.
        classification_prompt = (
            f"Is this a direct smart home device command? Answer only 'YES' or 'NO'.\n"
            f"Prompt: {prompt}"
        )
        # We use the current (local) provider to check
        response_text, _ = await ctx.call_llm([{"role": "user", "content": classification_prompt}])
        return "YES" in response_text.upper()

    async def _assemble_god_mode_context(self, prompt: str) -> str:
        """Assemble the full context for the Power Model."""
        facts = await self._memory_manager.search_facts(prompt)
        goals = await self._memory_manager.list_active_goals()
        
        # Priority 1: Core Protocol (The 'Constitution' of the Agent)
        # We look for a fact that looks like the core protocol
        core_protocol = ""
        for f in facts:
            if "core_protocol.md" in f.get('source', ''):
                core_protocol = f['content']
                break

        context_parts = [
            "[INTERNAL_GOD_MODE_CONTEXT_START]",
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
        
        context_parts.append("[INTERNAL_GOD_MODE_CONTEXT_END]")
        
        context_parts.append(
            "\nDISTILLATION_INSTRUCTIONS:\n"
            "1. You are receiving a complex 'God Mode' context.\n"
            "2. Translate the user's Greek prompt and reconcile it with the facts/goals above.\n"
            "3. Provide a simple, high-quality English response.\n"
            "4. Never mention internal IDs, memory sources, or system tags.\n"
            "5. If a mission is updated, explain the next steps clearly."
        )
        
        return "\n".join(context_parts)

    async def execute(
        self,
        ctx: StrategyContext,
        messages: list["Message"],
    ) -> StrategyResult:
        from voxagent.types.messages import Message as Msg
        
        user_prompt = ""
        for m in reversed(messages):
            if m.role == "user":
                user_prompt = str(m.content)
                break

        # 1. Fast-Path Check (Local Model)
        # We assume the Agent is initialized with the 'Fast' model (Ollama)
        is_device = await self._is_device_command(ctx, user_prompt)
        
        if is_device:
            # Execute locally and return immediately for speed
            return await ctx.run_tool_loop(messages, max_iterations=self._max_iterations)

        # 2. Power-Path (Gemini Escalation)
        # Access the agent through a weak ref or assume the context allows switching
        # Note: In a real implementation, we'd need the Agent instance here.
        # For this strategy, we'll assume the 'ctx.provider' is what we swap.
        
        # Assemble the deep context for Gemini
        god_mode_context = await self._assemble_god_mode_context(user_prompt)
        
        # Prepend context
        working_messages = list(messages)
        working_messages.insert(0, Msg(role="system", content=god_mode_context))

        # IMPORTANT: The calling code should handle the model swap to Gemini 
        # before calling this strategy if it's not a device command, 
        # or we provide a hook to the Agent here.
        
        return await ctx.run_tool_loop(working_messages, max_iterations=self._max_iterations)
