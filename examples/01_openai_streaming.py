#!/usr/bin/env python3
"""OpenAI Streaming Example.

This example demonstrates:
- Basic agent setup with OpenAI
- Streaming responses with typed chunks
- Handling different chunk types

Requirements:
    pip install voxagent[openai]
    export OPENAI_API_KEY="sk-..."
"""

import asyncio

from voxagent import Agent
from voxagent.providers import TextDeltaChunk, MessageEndChunk, ErrorChunk


async def main() -> None:
    # Create an agent with OpenAI
    agent = Agent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant. Be concise.",
    )

    prompt = "Explain quantum computing in 3 sentences."
    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    # Stream the response
    async for chunk in agent.stream(prompt):
        if isinstance(chunk, TextDeltaChunk):
            # Print text as it arrives
            print(chunk.delta, end="", flush=True)
        elif isinstance(chunk, ErrorChunk):
            print(f"\nError: {chunk.error}")
        elif isinstance(chunk, MessageEndChunk):
            print("\n")  # Newline at end

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

