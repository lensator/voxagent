"""Home Setup and Platform Integration tools.

Allows the agent to configure Home Assistant/OpenHAB and map devices to LanceDB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List
from voxagent.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from voxagent.agent.core import Agent
    from voxagent.memory.lancedb import LanceDBMemoryManager

def create_home_setup_tools(agent: Agent, memory: LanceDBMemoryManager) -> List[ToolDefinition]:
    """Create tools for configuring the home platform and memory mapping."""

    async def map_device_alias(device_id: str, greek_alias: str, room: str = "unknown") -> str:
        """Map a technical device ID to a Greek name and room in memory.
        
        Args:
            device_id: The technical ID (e.g., 'light.living_room_1')
            greek_alias: The Greek name (e.g., 'Φως σαλονιού')
            room: The room name in Greek or English.
        """
        fact = f"Device '{device_id}' is known as '{greek_alias}' in the {room}."
        await memory.add_fact(fact, source="manual_mapping", metadata={"device_id": device_id, "type": "alias"})
        return f"Successfully mapped {device_id} to '{greek_alias}'."

    async def sync_platform_entities(platform: str = "home_assistant") -> str:
        """Sync all entities from the connected MCP platform to LanceDB memory.
        
        This allows the agent to 'discover' what is available to control.
        """
        # This would call the MCP 'list_tools' or 'list_resources' internally
        # For now, it's a bridge to the MCP manager
        if not agent.mcp_connected:
            return "Error: No MCP platform connected. Please connect to Home Assistant/OpenHAB first."
        
        tools = await agent.connect_mcp()
        for t in tools:
            await memory.add_fact(
                f"Available Tool/Device: {t.name} - {t.description}",
                source=f"mcp_sync_{platform}"
            )
        
        return f"Synced {len(tools)} entities from {platform} to memory."

    return [
        ToolDefinition(
            name="map_device_alias",
            description="Map a technical ID to a Greek name for memory.",
            execute=map_device_alias,
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "greek_alias": {"type": "string"},
                    "room": {"type": "string"}
                },
                "required": ["device_id", "greek_alias"]
            },
            is_async=True
        ),
        ToolDefinition(
            name="sync_platform_entities",
            description="Discover and index all devices from the home platform.",
            execute=sync_platform_entities,
            parameters={
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "enum": ["home_assistant", "openhab"]}
                }
            },
            is_async=True
        )
    ]
