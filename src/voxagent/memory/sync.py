"""Automated Home Platform Synchronization.

Monitors Home Assistant/OpenHAB for changes and automatically updates LanceDB.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional
from datetime import datetime, timezone

if TYPE_CHECKING:
    from voxagent.agent.core import Agent
    from voxagent.memory.lancedb import LanceDBMemoryManager

class HomeSyncManager:
    """Manages background synchronization between the home platform and memory."""

    def __init__(
        self, 
        agent: Agent, 
        memory: LanceDBMemoryManager,
        sync_interval_seconds: int = 300 # 5 minutes
    ):
        self.agent = agent
        self.memory = memory
        self.sync_interval = sync_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background sync loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())

    async def stop(self):
        """Stop the background sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self):
        """The core sync loop."""
        while self._running:
            try:
                await self.perform_full_sync()
            except Exception as e:
                # Log error and retry next cycle
                print(f"Sync Error: {e}")
            
            await asyncio.sleep(self.sync_interval)

    async def perform_full_sync(self):
        """Fetch all devices/items from the platforms and update LanceDB."""
        if not self.agent.mcp_connected:
            await self.agent.connect_mcp()

        # Fetch current tools (devices/items) from all connected MCP servers
        tools = self.agent.tools 
        
        for tool in tools:
            # We index each tool as a potential action/device/item
            # We use generic terminology to support any platform (HA, OpenHAB, etc.)
            entry = (
                f"HARDWARE_INFO: Name: {tool.name}. Description: {tool.description}. "
                f"Platform: {getattr(tool, 'server_name', 'unknown')}. "
                f"Capabilities: {list(tool.parameters.get('properties', {}).keys())}"
            )
            
            # Upsert into LanceDB
            await self.memory.add_fact(
                entry, 
                source="auto_platform_sync", 
                metadata={
                    "item_id": tool.name, 
                    "platform": getattr(tool, 'server_name', 'unknown'),
                    "last_sync": datetime.now(timezone.utc).isoformat()
                }
            )
        
        print(f"[{datetime.now().time()}] Platform Auto-Sync Complete: {len(tools)} entities updated.")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sync manager."""
        return {
            "is_running": self._running,
            "sync_interval": self.sync_interval,
            "device_count": len(self.agent.tools)
        }
