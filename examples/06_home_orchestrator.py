"""Example of the Transparent Home Orchestrator.

Demonstrates:
1. Greek language support
2. Dynamic model switching
3. Goal-oriented planning
4. LanceDB memory retrieval
5. Resource-aware reasoning (frugality)
"""

import asyncio
import os
from voxagent import Agent
from voxagent.strategies.home import HomeOrchestratorStrategy
from voxagent.memory.lancedb import LanceDBMemoryManager
from voxagent.session.storage import FileSessionStorage
from voxagent.tools.meta import create_switch_model_tool
from voxagent.tools.decorator import tool

# 1. Setup Memory and Persona
memory = LanceDBMemoryManager(uri="home_config/memory/lancedb")
storage = FileSessionStorage("home_config/sessions")

# Pre-sync the rules and personas into LanceDB
async def init_memory():
    await memory.sync_directory("home_config/agents")
    await memory.sync_directory("home_config/rules")

# 2. Define a home control tool
@tool()
def control_device(room: str, device: str, action: str) -> str:
    """Control a smart home device."""
    return f"SUCCESS: {device} in {room} is now {action}."

# 3. Initialize the Agent
# We start with a fast, cheap model (e.g., Qwen2.5-Coder 3B on local CPU)
agent = Agent(
    model="ollama:qwen2.5-coder:3b",
    name="The Machine",
    session_storage=storage,
    strategy=HomeOrchestratorStrategy(memory_manager=memory),
    tools=[control_device]
)

# Register the meta-tool for model switching
agent.register_tool(create_switch_model_tool(agent))

async def main():
    await init_memory()
    
    session_key = "admin_session"
    
    print("--- Home Orchestrator Started (Greek Language) ---")
    
    # Example 1: Fast Path / Simple Command in Greek
    print("
User: Άναψε τα φώτα στο σαλόνι.")
    result = await agent.run("Άναψε τα φώτα στο σαλόνι.", session_key=session_key)
    print(f"Agent: {result.output}")
    
    # Example 2: Complex Goal with Frugality Rule
    print("
User: Θέλω να αγοράσω ένα server PC. Τι προτείνεις; (Τα οικονομικά μου είναι στενά)")
    result = await agent.run(
        "Θέλω να αγοράσω ένα server PC. Τι προτείνεις; (Τα οικονομικά μου είναι στενά)", 
        session_key=session_key
    )
    print(f"Agent: {result.output}")

    # Example 3: Escalation (Manual or Agent-driven)
    print("
[Admin] Switching to a more capable model for deep planning...")
    agent.set_model("openai:gpt-4o")
    
    print("
User: Φτιάξε ένα πλήρες πλάνο για την εγκατάσταση του server και την ασφάλεια του σπιτιού.")
    result = await agent.run(
        "Φτιάξε ένα πλήρες πλάνο για την εγκατάσταση του server και την ασφάλεια του σπιτιού.",
        session_key=session_key
    )
    print(f"Agent: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())
