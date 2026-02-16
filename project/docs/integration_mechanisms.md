# Integration Mechanisms: Home Orchestrator

## Overview
This document defines the protocols and mechanisms used by the Home Orchestrator to synchronize with external platforms (Home Assistant, OpenHAB) and maintain a consistent "World State" in LanceDB.

Future agents or sessions must follow these guidelines to interact with the home environment.

---

## 1. The Auto-Synchronization Mechanism
**Component**: `src/voxagent/memory/sync.py:HomeSyncManager`

### How it works:
1.  **Polling Loop**: By default, the manager runs a background task (default interval: 5 mins).
2.  **Tool Inspection**: It iterates through all registered tools (including dynamic MCP tools).
3.  **LanceDB Upsert**: For each tool, it creates a structured text entry:
    `DEVICE_INFO: Name: {name}. Description: {desc}. Capabilities: {params}`
4.  **Vector Indexing**: This entry is embedded and stored in the `facts` table of LanceDB.

### For Future Agents:
*   **Startup**: Always instantiate `HomeSyncManager(agent, memory)` and call `await manager.start()`.
*   **Context**: Do not hardcode device lists. Always query `memory.search_facts("DEVICE_INFO")` to get the current state.

---

## 2. The Add-on Architecture (Push Updates)
To eliminate polling latency, external platforms should use an **Event-Driven Push** model.

### Protocol:
1.  **Endpoint**: The agent exposes a webhook or listens to a message queue (MQTT/WebSocket).
2.  **Payload**: The external platform sends a JSON event:
    ```json
    {
      "event_type": "device_registry_updated",
      "device_id": "light.kitchen",
      "new_state": "on",
      "attributes": {"friendly_name": "Kitchen Main Light"}
    }
    ```
3.  **Handler**: The agent receives this event and immediately calls `memory.add_fact(...)` to update the specific record in LanceDB.

### Creating an Add-on:
*   **Home Assistant**: Write a custom integration/automation that calls the agent's webhook on `state_changed`.
*   **OpenHAB**: Use the Rule Engine to post updates to the agent's API.

---

## 3. The Discovery Protocol
When a new agent session starts (e.g., after a restart or on a new device), it does **not** need to rescan the network.

### The "Instant Boot" Sequence:
1.  **Connect to LanceDB**: Open the existing URI (`home_config/memory/lancedb`).
2.  **Load Schema**: Verify the `facts` and `goals` tables exist.
3.  **Hydrate Context**:
    *   Query `facts` for "DEVICE_INFO" to build an internal device map.
    *   Query `goals` for "active" status to resume pending missions.
4.  **Ready State**: The agent is now fully context-aware without making a single external API call.

---

## 4. Troubleshooting
*   **Stale Data**: If the agent hallucinates a device that no longer exists, force a resync: `await sync_manager.perform_full_sync()`.
*   **Missing Aliases**: Use the `map_device_alias` tool to manually inject Greek names if the auto-sync descriptions are insufficient.
