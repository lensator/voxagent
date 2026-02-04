"""Sandboxed Python code execution with security restrictions.

This module provides secure code execution using RestrictedPython
and subprocess isolation with resource limits.
"""

from __future__ import annotations

import multiprocessing
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from voxagent.code.tool_proxy import ToolProxyClient


@dataclass
class SandboxResult:
    """Result of sandboxed code execution.

    Attributes:
        output: Captured stdout from print() calls
        error: Error message if execution failed
        execution_time_ms: Time taken to execute in milliseconds
        success: Whether execution completed without errors (derived from error being None)
    """

    output: str | None = None
    error: str | None = None
    execution_time_ms: float | None = None

    @property
    def success(self) -> bool:
        """Return True if execution completed without errors."""
        return self.error is None


class CodeSandbox(ABC):
    """Abstract base class for code execution sandboxes."""

    @abstractmethod
    async def execute(
        self, code: str, globals_dict: dict[str, Any] | None = None
    ) -> SandboxResult:
        """Execute Python code in a sandboxed environment.

        Args:
            code: Python source code to execute
            globals_dict: Optional globals to inject (e.g., ls, read functions)

        Returns:
            SandboxResult with output or error
        """
        ...


def _inplacevar_(op: str, x: Any, y: Any) -> Any:
    """Handle in-place operators (+=, -=, etc.) in RestrictedPython."""
    if op == "+=":
        return x + y
    elif op == "-=":
        return x - y
    elif op == "*=":
        return x * y
    elif op == "/=":
        return x / y
    elif op == "//=":
        return x // y
    elif op == "%=":
        return x % y
    elif op == "**=":
        return x**y
    elif op == "&=":
        return x & y
    elif op == "|=":
        return x | y
    elif op == "^=":
        return x ^ y
    elif op == ">>=":
        return x >> y
    elif op == "<<=":
        return x << y
    else:
        raise ValueError(f"Unknown operator: {op}")


def _execute_in_subprocess(
    code: str,
    globals_dict: dict[str, Any],
    result_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    memory_limit_mb: int,
    tool_request_queue: "multiprocessing.Queue[Any] | None" = None,
    tool_response_queue: "multiprocessing.Queue[Any] | None" = None,
) -> None:
    """Subprocess entry point for sandboxed execution."""
    # Import here to avoid loading in main process
    import ast
    import warnings

    # Suppress RestrictedPython SyntaxWarnings about print/printed variable
    # These warnings are harmless but noisy (about print transformation internals)
    warnings.filterwarnings(
        "ignore",
        message=".*Prints, but never reads 'printed' variable.*",
        category=SyntaxWarning,
    )

    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        safer_getattr,
    )
    from RestrictedPython.PrintCollector import PrintCollector
    from RestrictedPython.transformer import (
        IOPERATOR_TO_STR,
        RestrictingNodeTransformer,
        copy_locations,
    )

    # Custom policy that allows augmented assignment on attributes
    class PermissiveNodeTransformer(RestrictingNodeTransformer):
        """A more permissive node transformer that allows augmented assignment."""

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            """Allow augmented assignment on attributes and subscripts.

            Transforms 'a.x += 1' to 'a.x = _inplacevar_("+=", a.x, 1)'
            and 'a[i] += 1' to 'a[i] = _inplacevar_("+=", a[i], 1)'
            """
            node = self.node_contents_visit(node)

            if isinstance(node.target, ast.Attribute):
                # Transform a.x += 1 to a.x = _inplacevar_("+=", a.x, 1)
                new_node = ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=node.target.value,
                            attr=node.target.attr,
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Call(
                        func=ast.Name("_inplacevar_", ast.Load()),
                        args=[
                            ast.Constant(IOPERATOR_TO_STR[type(node.op)]),
                            ast.Attribute(
                                value=node.target.value,
                                attr=node.target.attr,
                                ctx=ast.Load(),
                            ),
                            node.value,
                        ],
                        keywords=[],
                    ),
                )
                copy_locations(new_node, node)
                return new_node

            elif isinstance(node.target, ast.Subscript):
                # Transform a[i] += 1 to a[i] = _inplacevar_("+=", a[i], 1)
                new_node = ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=node.target.value,
                            slice=node.target.slice,
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Call(
                        func=ast.Name("_inplacevar_", ast.Load()),
                        args=[
                            ast.Constant(IOPERATOR_TO_STR[type(node.op)]),
                            ast.Subscript(
                                value=node.target.value,
                                slice=node.target.slice,
                                ctx=ast.Load(),
                            ),
                            node.value,
                        ],
                        keywords=[],
                    ),
                )
                copy_locations(new_node, node)
                return new_node

            elif isinstance(node.target, ast.Name):
                new_node = ast.Assign(
                    targets=[node.target],
                    value=ast.Call(
                        func=ast.Name("_inplacevar_", ast.Load()),
                        args=[
                            ast.Constant(IOPERATOR_TO_STR[type(node.op)]),
                            ast.Name(node.target.id, ast.Load()),
                            node.value,
                        ],
                        keywords=[],
                    ),
                )
                copy_locations(new_node, node)
                return new_node
            else:
                raise NotImplementedError(f"Unknown target type: {type(node.target)}")

    # Set memory limit (Unix only)
    try:
        import resource

        limit_bytes = memory_limit_mb * 1024 * 1024
        # Try to set RLIMIT_AS (address space) - works on Linux
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Only set if we can (soft limit can be lowered)
            if limit_bytes <= hard:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
        except (ValueError, OSError):
            pass
        # Also try RLIMIT_DATA
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
            if limit_bytes <= hard:
                resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, hard))
        except (ValueError, OSError):
            pass
    except ImportError:
        pass  # Windows

    # Build safe builtins
    safe_builtins_copy = dict(safe_builtins)
    # Remove dangerous builtins
    for name in ("open", "eval", "exec", "compile", "__import__"):
        safe_builtins_copy.pop(name, None)

    try:
        byte_code = compile_restricted(
            code, "<sandbox>", "exec", policy=PermissiveNodeTransformer
        )
        if byte_code is None:
            result_queue.put(SandboxResult(output="", error="SyntaxError: compilation failed"))
            return

        exec_globals: dict[str, Any] = {
            "__builtins__": safe_builtins_copy,
            "__name__": "__main__",
            "__doc__": None,
            # Required for class definitions in RestrictedPython
            "__metaclass__": type,
            # _print_ is a factory - RestrictedPython will call it to create a collector
            "_print_": PrintCollector,
            # Also provide _getattr_ for attribute access
            "_getattr_": safer_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_inplacevar_": _inplacevar_,
            # Provide write guard for attribute writes
            "_write_": lambda x: x,
            **globals_dict,
        }

        # Create call_tool function using proxy if queues are provided
        if tool_request_queue is not None and tool_response_queue is not None:
            from voxagent.code.tool_proxy import ToolProxyClient
            proxy_client = ToolProxyClient(tool_request_queue, tool_response_queue)

            def call_tool(category: str, tool_name: str, **kwargs: Any) -> Any:
                proxy = proxy_client.create_tool_proxy(category, tool_name)
                return proxy(**kwargs)

            exec_globals["call_tool"] = call_tool

        # Add query functions for efficient data processing
        from voxagent.code.query import QueryResult
        import re as _re

        def query(data: list | dict) -> QueryResult:
            """Wrap data in QueryResult for chaining."""
            if isinstance(data, dict) and "devices" in data:
                # Handle list_devices response format
                return QueryResult(data["devices"])
            if isinstance(data, dict) and "items" in data:
                return QueryResult(data["items"])
            if isinstance(data, list):
                return QueryResult(data)
            return QueryResult([data] if data else [])

        def tree(path: str = "tools") -> str:
            """Show full tool structure with signatures.

            Args:
                path: Starting path (default: "tools")

            Returns:
                Tree-formatted string showing all tools
            """
            lines = []
            path = path.rstrip("/")

            # Get ls function from globals_dict
            ls_func = globals_dict.get("ls")
            if ls_func is None:
                return "(ls function not available)"

            # Get root entries
            entries = ls_func(path)
            if isinstance(entries, str):
                # Handle error message
                return entries
            for entry in entries:
                if entry == "__index__.md":
                    continue
                if entry.endswith("/"):
                    # It's a category
                    cat_name = entry.rstrip("/")
                    lines.append(f"📁 {cat_name}/")
                    # Get tools in category
                    cat_entries = ls_func(f"{path}/{cat_name}")
                    if isinstance(cat_entries, list):
                        for tool in cat_entries:
                            if tool != "__index__.md":
                                lines.append(f"  📄 {tool}")
                else:
                    lines.append(f"📄 {entry}")

            return "\n".join(lines) if lines else "(empty)"

        def search(query_str: str) -> list:
            """Search for tools by keyword.

            Args:
                query_str: Search term (matches tool names)

            Returns:
                List of matching tool paths
            """
            results = []
            pattern = _re.compile(query_str, _re.IGNORECASE)

            # Get ls function from globals_dict
            ls_func = globals_dict.get("ls")
            if ls_func is None:
                return []

            # Search all categories
            categories = ls_func("tools")
            if isinstance(categories, str):
                return []
            for cat in categories:
                if cat == "__index__.md" or not cat.endswith("/"):
                    continue
                cat_name = cat.rstrip("/")
                tools = ls_func(f"tools/{cat_name}")
                if isinstance(tools, str):
                    continue
                for tool in tools:
                    if tool == "__index__.md":
                        continue
                    if pattern.search(tool) or pattern.search(cat_name):
                        results.append(f"tools/{cat_name}/{tool}")

            return results

        exec_globals["query"] = query
        exec_globals["tree"] = tree
        exec_globals["search"] = search
        exec_globals["QueryResult"] = QueryResult

        exec(byte_code, exec_globals)
        # Get the _print object that was created during execution and call it
        _print_obj = exec_globals.get("_print")
        output = _print_obj() if _print_obj else ""
        result_queue.put(
            SandboxResult(
                output=output if output else "",
            )
        )
    except SyntaxError as e:
        result_queue.put(SandboxResult(output="", error=f"SyntaxError: {e}"))
    except Exception as e:
        result_queue.put(SandboxResult(output="", error=f"{type(e).__name__}: {e}"))


class SubprocessSandbox(CodeSandbox):
    """Execute Python code in an isolated subprocess with RestrictedPython.

    Security features:
    - RestrictedPython AST filtering (blocks imports, file access, etc.)
    - Process isolation via multiprocessing
    - Timeout enforcement
    - Memory limits (Unix only)

    Args:
        timeout_seconds: Maximum execution time (default: 10)
        memory_limit_mb: Maximum memory in MB (default: 128, Unix only)
        tool_proxy_client: Optional tool proxy client for routing tool calls
    """

    def __init__(
        self,
        timeout_seconds: int = 10,
        memory_limit_mb: int = 128,
        tool_proxy_client: "ToolProxyClient | None" = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.tool_proxy_client = tool_proxy_client

    async def execute(
        self,
        code: str,
        globals_dict: dict[str, Any] | None = None,
        tool_request_queue: "multiprocessing.Queue[Any] | None" = None,
        tool_response_queue: "multiprocessing.Queue[Any] | None" = None,
    ) -> SandboxResult:
        """Execute code in subprocess with RestrictedPython.

        Args:
            code: Python source code to execute
            globals_dict: Optional globals to inject (e.g., ls, read functions)
            tool_request_queue: Queue for tool call requests from subprocess
            tool_response_queue: Queue for tool call responses to subprocess

        Returns:
            SandboxResult with output or error
        """
        import asyncio

        start_time = time.monotonic()

        result_queue: multiprocessing.Queue[SandboxResult] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_subprocess,
            args=(
                code,
                globals_dict or {},
                result_queue,
                self.memory_limit_mb,
                tool_request_queue,
                tool_response_queue,
            ),
        )
        process.start()

        # Use non-blocking wait to allow event loop to run proxy server
        # Poll the process status instead of blocking on join()
        poll_interval = 0.01  # 10ms
        elapsed = 0.0
        while process.is_alive() and elapsed < self.timeout_seconds:
            await asyncio.sleep(poll_interval)
            elapsed = time.monotonic() - start_time

        execution_time_ms = (time.monotonic() - start_time) * 1000

        if process.is_alive():
            process.kill()
            process.join()
            return SandboxResult(
                output="",
                error="Execution timed out",
                execution_time_ms=execution_time_ms,
            )

        try:
            result = result_queue.get_nowait()
            result.execution_time_ms = execution_time_ms
            return result
        except Exception:
            return SandboxResult(
                output="",
                error="No result from subprocess",
                execution_time_ms=execution_time_ms,
            )

