#!/usr/bin/env python3
"""Anthropic Tools Example.

This example demonstrates:
- Using the @tool decorator
- Tool calling with Anthropic Claude
- Handling tool results

Requirements:
    pip install voxagent[anthropic]
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import asyncio

from voxagent import Agent
from voxagent.tools import tool


# Define tools using the @tool decorator
@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.

    Returns:
        A string describing the weather.
    """
    # In a real app, this would call a weather API
    weather_data = {
        "Paris": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 18°C",
        "New York": "Clear, 25°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool()
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone.

    Args:
        timezone: The timezone (e.g., "UTC", "EST", "PST").

    Returns:
        The current time as a string.
    """
    from datetime import datetime

    # Simplified - in production use pytz or zoneinfo
    return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"


async def main() -> None:
    # Create an agent with tools
    agent = Agent(
        model="anthropic:claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful assistant with access to weather and time tools.",
        tools=[get_weather, get_time],
    )

    # Ask a question that requires tool use
    prompt = "What's the weather like in Paris and Tokyo?"
    print(f"Prompt: {prompt}\n")

    result = await agent.run(prompt)
    print(f"Response: {result.output}")

    # Show tool calls that were made
    if result.tool_calls:
        print(f"\nTool calls made: {len(result.tool_calls)}")
        for tc in result.tool_calls:
            print(f"  - {tc.name}({tc.arguments})")


if __name__ == "__main__":
    asyncio.run(main())

