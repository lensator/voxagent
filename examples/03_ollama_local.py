#!/usr/bin/env python3
"""Ollama Local Models Example.

This example demonstrates:
- Using local models with Ollama
- No API key required
- Same interface as cloud providers

Requirements:
    pip install voxagent[ollama]
    # Install Ollama: https://ollama.ai
    ollama pull llama3.2
"""

import asyncio

from voxagent import Agent
from voxagent.providers import TextDeltaChunk


async def main() -> None:
    # Create an agent with a local Ollama model
    # No API key needed - Ollama runs locally
    agent = Agent(
        model="ollama:llama3.2",
        system_prompt="You are a helpful coding assistant.",
    )

    prompt = "Write a Python function to calculate fibonacci numbers."
    print(f"Prompt: {prompt}\n")
    print("Response:\n")

    # Stream the response
    async for chunk in agent.stream(prompt):
        if isinstance(chunk, TextDeltaChunk):
            print(chunk.delta, end="", flush=True)

    print("\n\nDone!")


async def list_models() -> None:
    """List available Ollama models."""
    from voxagent.providers import get_default_registry

    registry = get_default_registry()
    provider = registry.get_provider("ollama")

    if provider:
        print("Available Ollama models:")
        for model in provider.models:
            print(f"  - ollama:{model}")


if __name__ == "__main__":
    # Uncomment to list available models:
    # asyncio.run(list_models())

    asyncio.run(main())

