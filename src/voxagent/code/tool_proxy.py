"""Tool proxy for routing sandbox calls to real implementations.

The sandbox cannot directly call async MCP tools. This module provides
a queue-based proxy that:
1. Captures tool calls in the sandbox
2. Serializes them to a multiprocessing queue
3. The main process executes real tools and returns results
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any, Callable, Awaitable, TYPE_CHECKING


@dataclass
class ToolCallRequest:
    """A tool call request from the sandbox.

    Attributes:
        call_id: Unique identifier for this call
        category: Tool category (e.g., "devices")
        tool_name: Tool function name (e.g., "list_devices")
        args: Positional arguments
        kwargs: Keyword arguments
    """
    call_id: str
    category: str
    tool_name: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResponse:
    """Response from a tool call.

    Attributes:
        call_id: Matches the request call_id
        success: Whether the call succeeded
        result: The return value (if success)
        error: Error message (if failed)
    """
    call_id: str
    success: bool
    result: Any = None
    error: str | None = None


class ToolProxyClient:
    """Client-side proxy that runs in the sandbox subprocess.

    This creates callable proxies for each tool that:
    1. Serialize the call to the request queue
    2. Wait for the response on the response queue
    3. Return the result or raise an exception
    """

    def __init__(
        self,
        request_queue: Queue,  # type: ignore[type-arg]
        response_queue: Queue,  # type: ignore[type-arg]
        timeout_seconds: float = 30.0,
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout_seconds = timeout_seconds
        self._call_counter = 0

    def create_tool_proxy(self, category: str, tool_name: str) -> Callable[..., Any]:
        """Create a callable proxy for a tool.

        Args:
            category: Tool category
            tool_name: Tool function name

        Returns:
            A callable that proxies to the real tool
        """
        def proxy(*args: Any, **kwargs: Any) -> Any:
            # Generate unique call ID
            self._call_counter += 1
            call_id = f"call_{self._call_counter}"

            # Send request
            request = ToolCallRequest(
                call_id=call_id,
                category=category,
                tool_name=tool_name,
                args=args,
                kwargs=kwargs,
            )
            self.request_queue.put(request)

            # Wait for response (blocking in subprocess)
            start = time.monotonic()
            while True:
                try:
                    response: ToolCallResponse = self.response_queue.get(timeout=0.1)
                    if response.call_id == call_id:
                        if response.success:
                            return response.result
                        else:
                            raise RuntimeError(f"Tool error: {response.error}")
                except Exception:
                    pass

                if time.monotonic() - start > self.timeout_seconds:
                    raise TimeoutError(f"Tool call timed out after {self.timeout_seconds}s")

        return proxy

    def build_tools_namespace(self, categories: dict[str, list[str]]) -> Any:
        """Build a 'tools' namespace with proxy modules.

        Args:
            categories: Dict of category name -> list of tool names

        Returns:
            A namespace object with category submodules
        """
        class ToolModule:
            pass

        class ToolsNamespace:
            pass

        tools = ToolsNamespace()

        for category, tool_names in categories.items():
            module = ToolModule()
            for tool_name in tool_names:
                proxy = self.create_tool_proxy(category, tool_name)
                setattr(module, tool_name, proxy)

            # Replace hyphens with underscores for valid Python identifiers
            attr_name = category.replace("-", "_")
            setattr(tools, attr_name, module)

        return tools


class ToolProxyServer:
    """Server-side proxy that runs in the main process.

    This:
    1. Listens for tool call requests from the sandbox
    2. Executes the real tool implementation
    3. Sends the response back
    """

    def __init__(
        self,
        request_queue: Queue,  # type: ignore[type-arg]
        response_queue: Queue,  # type: ignore[type-arg]
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._implementations: dict[str, Callable[..., Any]] = {}
        self._running = False

    def register_implementation(
        self,
        category: str,
        tool_name: str,
        implementation: Callable[..., Any] | Callable[..., Awaitable[Any]],
    ) -> None:
        """Register a tool implementation.

        Args:
            category: Tool category
            tool_name: Tool function name
            implementation: The actual callable (sync or async)
        """
        key = f"{category}.{tool_name}"
        self._implementations[key] = implementation

    async def process_one_request(self, timeout: float = 0.1) -> bool:
        """Process a single request if available.

        Args:
            timeout: How long to wait for a request

        Returns:
            True if a request was processed, False if queue was empty
        """
        try:
            request: ToolCallRequest = self.request_queue.get_nowait()
        except Exception:
            return False

        key = f"{request.category}.{request.tool_name}"
        impl = self._implementations.get(key)

        if impl is None:
            response = ToolCallResponse(
                call_id=request.call_id,
                success=False,
                error=f"Tool not found: {key}",
            )
        else:
            try:
                # Call the implementation (handle async)
                if asyncio.iscoroutinefunction(impl):
                    result = await impl(*request.args, **request.kwargs)
                else:
                    result = impl(*request.args, **request.kwargs)

                response = ToolCallResponse(
                    call_id=request.call_id,
                    success=True,
                    result=result,
                )
            except Exception as e:
                response = ToolCallResponse(
                    call_id=request.call_id,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )

        self.response_queue.put(response)
        return True

    async def run_until_complete(self, timeout: float = 30.0) -> None:
        """Process requests until timeout or no more requests.

        Args:
            timeout: Maximum time to run
        """
        start = time.monotonic()
        self._running = True

        while self._running and (time.monotonic() - start) < timeout:
            processed = await self.process_one_request()
            if not processed:
                await asyncio.sleep(0.01)  # Small sleep if no request

    def stop(self) -> None:
        """Stop the server loop."""
        self._running = False


def create_tool_proxy_pair() -> tuple[ToolProxyClient, ToolProxyServer]:
    """Create a matched client/server proxy pair.

    Returns:
        Tuple of (client for sandbox, server for main process)
    """
    request_queue: Queue = multiprocessing.Queue()  # type: ignore[type-arg]
    response_queue: Queue = multiprocessing.Queue()  # type: ignore[type-arg]

    client = ToolProxyClient(request_queue, response_queue)
    server = ToolProxyServer(request_queue, response_queue)

    return client, server
