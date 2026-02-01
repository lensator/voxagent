#!/usr/bin/env python3
"""Multi-Provider Example.

This example demonstrates:
- Switching between providers
- Using the provider registry
- Comparing responses from different models

Requirements:
    pip install voxagent[all]
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import asyncio

from voxagent import Agent, get_default_registry


async def compare_providers() -> None:
    """Compare responses from different providers."""
    models = [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022",
        # "google:gemini-1.5-pro",  # Uncomment if you have Google API key
        # "ollama:llama3.2",        # Uncomment if Ollama is running
    ]

    prompt = "What is the capital of France? Answer in one word."

    print("Comparing providers:")
    print("=" * 50)

    for model in models:
        try:
            agent = Agent(model=model)
            result = await agent.run(prompt)
            print(f"{model:40} → {result.output.strip()}")
        except Exception as e:
            print(f"{model:40} → Error: {e}")

    print()


async def list_providers() -> None:
    """List all available providers."""
    registry = get_default_registry()

    print("Available providers:")
    print("=" * 50)

    for name in registry.list_providers():
        provider = registry.get_provider(name)
        if provider:
            print(f"\n{name}:")
            print(f"  Models: {', '.join(provider.models[:3])}...")
            print(f"  Tools: {'Yes' if provider.supports_tools else 'No'}")


async def provider_switching() -> None:
    """Demonstrate switching providers at runtime."""
    print("\nProvider Switching Demo:")
    print("=" * 50)

    # Start with OpenAI
    agent = Agent(model="openai:gpt-4o")
    result = await agent.run("Say 'Hello from OpenAI'")
    print(f"OpenAI: {result.output.strip()}")

    # Switch to Anthropic (create new agent)
    agent = Agent(model="anthropic:claude-3-5-sonnet-20241022")
    result = await agent.run("Say 'Hello from Anthropic'")
    print(f"Anthropic: {result.output.strip()}")


async def main() -> None:
    await list_providers()
    print()
    await compare_providers()
    await provider_switching()


if __name__ == "__main__":
    asyncio.run(main())

