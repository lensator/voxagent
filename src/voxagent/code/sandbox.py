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
) -> None:
    """Subprocess entry point for sandboxed execution."""
    # Import here to avoid loading in main process
    import ast

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
        self, code: str, globals_dict: dict[str, Any] | None = None
    ) -> SandboxResult:
        """Execute code in subprocess with RestrictedPython."""
        start_time = time.monotonic()

        result_queue: multiprocessing.Queue[SandboxResult] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_subprocess,
            args=(code, globals_dict or {}, result_queue, self.memory_limit_mb),
        )
        process.start()
        process.join(timeout=self.timeout_seconds)

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

