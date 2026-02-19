import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add src to sys.path to allow imports
sys.path.append(os.path.join(os.getcwd(), "src"))

from voxagent.agent.core import Agent
from voxagent.strategies.home import HomeOrchestratorStrategy
from voxagent.types.messages import Message
from voxagent.streaming.events import InternalThoughtEvent, TextDeltaEvent

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

async def test_dual_stage():
    print("\n--- Initializing Samaritan Dual-Stage Agent ---")
    
    # Initialize strategy with the FAST model (Ollama)
    # The user objective says: qwen2.5-coder:3b
    strategy = HomeOrchestratorStrategy(fast_model="ollama:qwen2.5:0.5b")
    
    # Initialize agent with the POWER model (Antigravity/Gemini)
    # The user objective says: antigravity:gemini-2.0-flash
    agent = Agent(
        model="antigravity:gemini-2.0-flash",
        strategy=strategy,
        name="Samaritan",
        config_dir="./home_config" 
    )
    
    print(f"Agent Model: {agent.model_string}")
    print(f"Strategy Fast Model: {strategy._fast_model}")

    # 1. Test Hardware Intent (Should trigger Fast model + Tool check)
    print("\n--- Stage 1: Testing Hardware Intent (Fast Model) ---")
    print("Prompt: 'Turn on the balcony light'")
    
    async for event in agent.run_stream("Turn on the balcony light"):
        if isinstance(event, InternalThoughtEvent):
            print(f"\n[DEBUG] {event.module}: {event.content}")
        elif isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)

    print("\n\n--- Stage 2: Testing General Query (Power Model escalation) ---")
    print("Prompt: 'Tell me a joke about robots'")
    
    async for event in agent.run_stream("Tell me a joke about robots"):
        if isinstance(event, InternalThoughtEvent):
            print(f"\n[DEBUG] {event.module}: {event.content}")
        elif isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)

    print("\n\n--- Dual-Stage Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_dual_stage())
