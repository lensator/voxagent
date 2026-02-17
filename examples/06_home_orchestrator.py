"""Example of the 'Fat Agent' Home Orchestrator.

Demonstrates the simplified 'Thin Platform' pattern where the Agent
manages its own storage, memory, and background sync.
"""

import asyncio
from voxagent import Agent
from voxagent.strategies.home import HomeOrchestratorStrategy

async def main():
    # 1. Initialize the 'Fat Agent'
    # All sub-systems (LanceDB, SessionStorage, SyncManager) are handled internally
    # based on the 'config_dir' path.
    agent = Agent(
        model="ollama:qwen2.5-coder:3b",
        config_dir="home_config",
        strategy=HomeOrchestratorStrategy(fast_model="ollama:qwen2.5-coder:3b")
    )

    # 2. Start the Agent (Triggering internal service startup)
    async with agent:
        print("--- Samaritan by Cyberdyne: Online ---")
        
        session_key = "admin_session"
        
        # Example 1: Direct device control (Fast-Path)
        print("\nUser: Turn on the kitchen lights.")
        async for event in agent.run_stream("Turn on the kitchen lights.", session_key=session_key):
            if hasattr(event, 'delta'):
                print(event.delta, end="", flush=True)
        print()

        # Example 2: Complex query (Power-Path / Distillation)
        # The agent will auto-escalate to Gemini if configured or perform distillation locally
        print("\nUser: Is the house secure for the night?")
        async for event in agent.run_stream("Is the house secure for the night?", session_key=session_key):
            if hasattr(event, 'delta'):
                print(event.delta, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())
