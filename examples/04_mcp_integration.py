#!/usr/bin/env python3
"""MCP Integration Example.

This example demonstrates:
- Loading MCP (Model Context Protocol) servers
- Using MCP tools with any provider
- Managing MCP server lifecycle

Requirements:
    pip install voxagent[mcp,openai]
    export OPENAI_API_KEY="sk-..."
    # Have an MCP server available (e.g., filesystem, git, etc.)
"""

import asyncio

from voxagent import Agent, MCPServerManager


async def main() -> None:
    # Configure MCP servers
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
        # Add more servers as needed:
        # "git": {
        #     "command": "npx",
        #     "args": ["-y", "@anthropic/mcp-server-git"],
        # },
    }

    # Create MCP manager
    async with MCPServerManager(mcp_config) as mcp:
        # Get tools from MCP servers
        mcp_tools = await mcp.get_tools()
        print(f"Loaded {len(mcp_tools)} MCP tools")

        # Create agent with MCP tools
        agent = Agent(
            model="openai:gpt-4o",
            system_prompt="You have access to filesystem tools. Help the user manage files.",
            tools=mcp_tools,
        )

        # Use the agent
        result = await agent.run("List the files in /tmp")
        print(f"\nResponse: {result.output}")


async def simple_example() -> None:
    """Simpler example without actual MCP servers."""
    print("MCP Integration Example")
    print("=" * 40)
    print()
    print("MCP (Model Context Protocol) allows you to:")
    print("  1. Connect to external tool servers")
    print("  2. Use tools from multiple sources")
    print("  3. Extend agent capabilities dynamically")
    print()
    print("Common MCP servers:")
    print("  - @anthropic/mcp-server-filesystem")
    print("  - @anthropic/mcp-server-git")
    print("  - @anthropic/mcp-server-github")
    print("  - @anthropic/mcp-server-postgres")
    print()
    print("See: https://github.com/anthropics/mcp")


if __name__ == "__main__":
    # Run the simple example (no MCP servers needed)
    asyncio.run(simple_example())

    # Uncomment to run with actual MCP servers:
    # asyncio.run(main())

