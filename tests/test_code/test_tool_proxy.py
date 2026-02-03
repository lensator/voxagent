"""Tests for tool proxy."""

import pytest
import asyncio
from voxagent.code.tool_proxy import (
    ToolCallRequest,
    ToolCallResponse,
    ToolProxyClient,
    ToolProxyServer,
    create_tool_proxy_pair,
)


class TestToolCallRequest:
    def test_create_request(self):
        req = ToolCallRequest(
            call_id="1",
            category="devices",
            tool_name="list_devices",
            args=(),
            kwargs={"room": "kitchen"},
        )
        assert req.call_id == "1"
        assert req.category == "devices"
        assert req.tool_name == "list_devices"

    def test_create_request_with_defaults(self):
        req = ToolCallRequest(
            call_id="2",
            category="lights",
            tool_name="turn_on",
        )
        assert req.args == ()
        assert req.kwargs == {}


class TestToolCallResponse:
    def test_success_response(self):
        resp = ToolCallResponse(call_id="1", success=True, result={"count": 5})
        assert resp.success
        assert resp.result == {"count": 5}
        assert resp.error is None

    def test_error_response(self):
        resp = ToolCallResponse(call_id="1", success=False, error="Not found")
        assert not resp.success
        assert resp.error == "Not found"
        assert resp.result is None


class TestToolProxyPair:
    def test_create_pair(self):
        client, server = create_tool_proxy_pair()
        assert client is not None
        assert server is not None
        assert client.request_queue is server.request_queue
        assert client.response_queue is server.response_queue


class TestToolProxyServer:
    @pytest.mark.asyncio
    async def test_register_and_call_sync_tool(self):
        client, server = create_tool_proxy_pair()

        # Register a sync tool
        def add(a: int, b: int) -> int:
            return a + b

        server.register_implementation("math", "add", add)

        # Simulate a request
        request = ToolCallRequest(
            call_id="test1",
            category="math",
            tool_name="add",
            args=(2, 3),
            kwargs={},
        )
        client.request_queue.put(request)

        # Small delay to ensure the item is available in the queue
        await asyncio.sleep(0.05)

        # Process it
        processed = await server.process_one_request()
        assert processed

        # Small delay for response queue
        await asyncio.sleep(0.05)

        # Check response
        response = server.response_queue.get_nowait()
        assert response.success
        assert response.result == 5

    @pytest.mark.asyncio
    async def test_register_and_call_async_tool(self):
        client, server = create_tool_proxy_pair()

        # Register an async tool
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        server.register_implementation("math", "async_add", async_add)

        # Simulate a request
        request = ToolCallRequest(
            call_id="test2",
            category="math",
            tool_name="async_add",
            args=(10, 20),
            kwargs={},
        )
        client.request_queue.put(request)

        # Small delay to ensure the item is available in the queue
        await asyncio.sleep(0.05)

        # Process it
        processed = await server.process_one_request()
        assert processed

        # Small delay for response queue
        await asyncio.sleep(0.05)

        # Check response
        response = server.response_queue.get_nowait()
        assert response.success
        assert response.result == 30

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        client, server = create_tool_proxy_pair()

        request = ToolCallRequest(
            call_id="test3",
            category="unknown",
            tool_name="missing",
        )
        client.request_queue.put(request)

        # Small delay to ensure the item is available in the queue
        await asyncio.sleep(0.05)

        await server.process_one_request()

        # Small delay for response queue
        await asyncio.sleep(0.05)

        response = server.response_queue.get_nowait()
        assert not response.success
        assert "not found" in response.error.lower()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        client, server = create_tool_proxy_pair()

        def failing_tool() -> None:
            raise ValueError("Something went wrong")

        server.register_implementation("test", "fail", failing_tool)

        request = ToolCallRequest(
            call_id="test4",
            category="test",
            tool_name="fail",
        )
        client.request_queue.put(request)

        # Small delay to ensure the item is available in the queue
        await asyncio.sleep(0.05)

        await server.process_one_request()

        # Small delay for response queue
        await asyncio.sleep(0.05)

        response = server.response_queue.get_nowait()
        assert not response.success
        assert "ValueError" in response.error
        assert "Something went wrong" in response.error

