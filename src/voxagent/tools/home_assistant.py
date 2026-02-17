"""Home Assistant Integration tools for voxagent.

Provides direct control of Home Assistant entities via the REST API and WebSocket API.
Required environment variables:
- HASS_URL: e.g., 'http://192.168.1.100:8123'
- HASS_TOKEN: Long-lived access token.
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
import httpx
import websockets

from voxagent.tools.decorator import tool

logger = logging.getLogger(__name__)

def _get_hass_config():
    url = os.environ.get("HASS_URL", "http://localhost:8123")
    token = os.environ.get("HASS_TOKEN")
    if not token:
        logger.warning("HASS_TOKEN not set. Home Assistant tools will fail.")
    return url.rstrip("/"), token

@tool()
async def hass_list_entities(domain: Optional[str] = None) -> str:
    """List available entities from Home Assistant."""
    url, token = _get_hass_config()
    if not token: return "Error: HASS_TOKEN not set."

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{url}/api/states",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10.0
        )
        if response.status_code != 200:
            return f"Error: HASS API returned {response.status_code}: {response.text}"
        
        entities = response.json()
        
        valid_domains = ["light", "switch", "sensor", "binary_sensor", "climate", "cover", "media_player"]
        if domain and domain not in valid_domains:
            if "light" in str(domain).lower(): domain = "light"
            elif "switch" in str(domain).lower(): domain = "switch"
            else: domain = None

        if domain:
            entities = [e for e in entities if e["entity_id"].startswith(f"{domain}.")]
        
        result = []
        for e in entities:
            name = e.get("attributes", {}).get("friendly_name", e["entity_id"])
            result.append(f"- {name} ({e['entity_id']}): {e['state']}")
            
        return "\n".join(result) if result else "No entities found."

@tool()
async def hass_control_device(entity_id: str, action: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Control a Home Assistant device (turn_on, turn_off, toggle, etc.)."""
    url, token = _get_hass_config()
    if not token: return "Error: HASS_TOKEN not set."

    domain = entity_id.split(".")[0]
    service_url = f"{url}/api/services/{domain}/{action}"
    
    body = {"entity_id": entity_id}
    if data:
        body.update(data)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            service_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=10.0
        )
        
        if response.status_code != 200:
            return f"Error: HASS API returned {response.status_code}: {response.text}"
            
        return f"Successfully executed {action} on {entity_id}."

@tool()
async def hass_get_entity_registry_entry(entity_id: str) -> Dict[str, Any]:
    """Get the full registry entry for an entity to find its device_id."""
    url, token = _get_hass_config()
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    async with websockets.connect(ws_url) as websocket:
        await websocket.recv() # auth_required
        await websocket.send(json.dumps({"type": "auth", "access_token": token}))
        await websocket.recv() # auth_ok
        
        await websocket.send(json.dumps({
            "id": 1,
            "type": "config/entity_registry/get",
            "entity_id": entity_id
        }))
        res = json.loads(await websocket.recv())
        return res.get("result", {})

@tool()
async def hass_rename_device(device_id: str, new_name: str) -> str:
    """Rename a physical device in the registry (affects all its entities)."""
    url, token = _get_hass_config()
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    async with websockets.connect(ws_url) as websocket:
        await websocket.recv() # auth_required
        await websocket.send(json.dumps({"type": "auth", "access_token": token}))
        await websocket.recv() # auth_ok
        
        await websocket.send(json.dumps({
            "id": 1,
            "type": "config/device_registry/update",
            "device_id": device_id,
            "name_by_user": new_name
        }))
        res = json.loads(await websocket.recv())
        if not res.get("success"):
            return f"Error: {res.get('error', {}).get('message', 'Unknown error')}"
        return f"Successfully renamed device {device_id} to '{new_name}'"

@tool()
async def hass_rename_all_device_entities(device_id: str, old_prefix: str, new_prefix: str) -> str:
    """Rename all entity IDs belonging to a device by replacing a prefix/keyword."""
    url, token = _get_hass_config()
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            await websocket.recv() # auth_required
            await websocket.send(json.dumps({"type": "auth", "access_token": token}))
            await websocket.recv() # auth_ok

            # 1. Get all entities
            await websocket.send(json.dumps({"id": 1, "type": "config/entity_registry/list"}))
            res = json.loads(await websocket.recv())
            entities = res.get("result", [])
            
            # 2. Filter for this device
            device_entities = [e for e in entities if e.get("device_id") == device_id]
            
            results = []
            msg_id = 2
            for e in device_entities:
                old_id = e["entity_id"]
                if old_prefix in old_id:
                    new_id = old_id.replace(old_prefix, new_prefix)
                    await websocket.send(json.dumps({
                        "id": msg_id,
                        "type": "config/entity_registry/update",
                        "entity_id": old_id,
                        "new_entity_id": new_id
                    }))
                    update_res = json.loads(await websocket.recv())
                    if update_res.get("success"):
                        results.append(f"Renamed {old_id} -> {new_id}")
                    else:
                        results.append(f"Failed {old_id}: {update_res.get('error', {}).get('message')}")
                    msg_id += 1
            
            return "\n".join(results) if results else "No matching entity IDs found to rename."
    except Exception as e:
        return f"WebSocket Error: {e}"

@tool()
async def hass_get_state(entity_id: str) -> str:
    """Get the current state and attributes of a specific entity."""
    url, token = _get_hass_config()
    if not token: return "Error: HASS_TOKEN not set."

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{url}/api/states/{entity_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10.0
        )
        
        if response.status_code != 200:
            return f"Error: HASS API returned {response.status_code}: {response.text}"
            
        data = response.json()
        state = data.get("state", "unknown")
        attrs = json.dumps(data.get("attributes", {}), indent=2)
        
        return f"Entity: {entity_id}\nState: {state}\nAttributes: {attrs}"
