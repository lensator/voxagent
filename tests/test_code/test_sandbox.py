"""Tests for code sandbox infrastructure.

This module tests the CodeSandbox abstract base class and SubprocessSandbox
implementation for secure Python code execution.
"""

from __future__ import annotations

import sys

import pytest

from voxagent.code.sandbox import CodeSandbox, SandboxResult, SubprocessSandbox

# macOS doesn't enforce memory limits via rlimit
IS_MACOS = sys.platform == "darwin"


class TestCodeSandboxABC:
    """Tests for the abstract CodeSandbox interface."""

    def test_sandbox_is_abstract(self) -> None:
        """CodeSandbox cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CodeSandbox()  # type: ignore[abstract]

    def test_sandbox_requires_execute_method(self) -> None:
        """Subclasses must implement execute()."""

        class IncompleteSandbox(CodeSandbox):
            pass

        with pytest.raises(TypeError):
            IncompleteSandbox()  # type: ignore[abstract]


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_success_result(self) -> None:
        """Create successful result with output."""
        result = SandboxResult(output="hello\n", error=None, execution_time_ms=10.5)
        assert result.output == "hello\n"
        assert result.error is None
        assert result.execution_time_ms == 10.5
        assert result.success is True

    def test_error_result(self) -> None:
        """Create error result."""
        result = SandboxResult(output="", error="SyntaxError: invalid syntax", execution_time_ms=5.0)
        assert result.output == ""
        assert result.error == "SyntaxError: invalid syntax"
        assert result.success is False

    def test_result_has_execution_time(self) -> None:
        """Result includes execution time in ms."""
        result = SandboxResult(output="", error=None, execution_time_ms=123.456)
        assert result.execution_time_ms == 123.456


class TestSubprocessSandboxBasicExecution:
    """Tests for subprocess-based Python execution - basic operations."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        """Create a sandbox instance for testing."""
        return SubprocessSandbox(timeout_seconds=5, memory_limit_mb=128)

    async def test_execute_simple_code(self, sandbox: SubprocessSandbox) -> None:
        """Execute simple Python code and capture output."""
        result = await sandbox.execute("print('hello')")
        assert result.output.strip() == "hello"
        assert result.success is True

    async def test_execute_returns_sandbox_result(self, sandbox: SubprocessSandbox) -> None:
        """execute() returns SandboxResult dataclass."""
        result = await sandbox.execute("x = 1")
        assert isinstance(result, SandboxResult)

    async def test_execute_captures_print_output(self, sandbox: SubprocessSandbox) -> None:
        """All print() calls are captured in output."""
        code = "print('line 1')"
        result = await sandbox.execute(code)
        assert "line 1" in result.output

    async def test_execute_multiple_prints(self, sandbox: SubprocessSandbox) -> None:
        """Multiple print statements are captured in order."""
        code = """
print('first')
print('second')
print('third')
"""
        result = await sandbox.execute(code)
        lines = result.output.strip().split("\n")
        assert lines == ["first", "second", "third"]


class TestSubprocessSandboxErrorHandling:
    """Tests for subprocess sandbox error handling."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        """Create a sandbox instance for testing."""
        return SubprocessSandbox(timeout_seconds=5, memory_limit_mb=128)

    async def test_execute_syntax_error(self, sandbox: SubprocessSandbox) -> None:
        """Syntax errors return SandboxResult with error."""
        result = await sandbox.execute("if True print('bad')")
        assert result.success is False
        assert "SyntaxError" in (result.error or "")

    async def test_execute_runtime_error(self, sandbox: SubprocessSandbox) -> None:
        """Runtime errors (NameError, etc.) return error."""
        result = await sandbox.execute("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in (result.error or "")

    async def test_execute_division_by_zero(self, sandbox: SubprocessSandbox) -> None:
        """Division by zero returns error, not crash."""
        result = await sandbox.execute("x = 1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in (result.error or "")


class TestSubprocessSandboxSecurity:
    """Tests for RestrictedPython security enforcement."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        """Create a sandbox instance for testing."""
        return SubprocessSandbox(timeout_seconds=5, memory_limit_mb=128)

    async def test_blocks_import_os(self, sandbox: SubprocessSandbox) -> None:
        """import os is blocked."""
        result = await sandbox.execute("import os")
        assert result.success is False
        assert result.error is not None

    async def test_blocks_import_subprocess(self, sandbox: SubprocessSandbox) -> None:
        """import subprocess is blocked."""
        result = await sandbox.execute("import subprocess")
        assert result.success is False
        assert result.error is not None

    async def test_blocks_open_file(self, sandbox: SubprocessSandbox) -> None:
        """open() is not available."""
        result = await sandbox.execute("f = open('/etc/passwd', 'r')")
        assert result.success is False
        assert result.error is not None

    async def test_blocks_eval(self, sandbox: SubprocessSandbox) -> None:
        """eval() is not available."""
        result = await sandbox.execute("eval('1 + 1')")
        assert result.success is False
        assert result.error is not None

    async def test_blocks_exec(self, sandbox: SubprocessSandbox) -> None:
        """exec() is not available."""
        result = await sandbox.execute("exec('x = 1')")
        assert result.success is False
        assert result.error is not None

    async def test_blocks_getattr_on_private(self, sandbox: SubprocessSandbox) -> None:
        """Cannot access __private__ attributes."""
        code = """
class Foo:
    __secret = 42
f = Foo()
print(f._Foo__secret)
"""
        result = await sandbox.execute(code)
        assert result.success is False
        assert result.error is not None


class TestSubprocessSandboxResourceLimits:
    """Tests for resource limits (timeout, memory)."""

    async def test_timeout_kills_infinite_loop(self) -> None:
        """Infinite loops are killed after timeout."""
        sandbox = SubprocessSandbox(timeout_seconds=1, memory_limit_mb=128)
        code = "while True: pass"
        result = await sandbox.execute(code)
        assert result.success is False
        assert result.error is not None
        # Verify it completed in reasonable time (< 3 seconds)
        assert result.execution_time_ms < 3000

    @pytest.mark.skipif(IS_MACOS, reason="macOS doesn't enforce rlimit memory limits")
    async def test_memory_limit_prevents_exhaustion(self) -> None:
        """Memory allocation beyond limit fails."""
        sandbox = SubprocessSandbox(timeout_seconds=5, memory_limit_mb=64)
        # Try to allocate ~500MB
        code = "x = [0] * (500 * 1024 * 1024 // 8)"
        result = await sandbox.execute(code)
        assert result.success is False
        assert result.error is not None


class TestSubprocessSandboxAllowedOperations:
    """Tests for operations that should be allowed."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        """Create a sandbox instance for testing."""
        return SubprocessSandbox(timeout_seconds=5, memory_limit_mb=128)

    async def test_allows_basic_math(self, sandbox: SubprocessSandbox) -> None:
        """Basic math operations work."""
        code = """
result = (10 + 5) * 2 - 3 / 1
print(result)
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "27" in result.output

    async def test_allows_string_operations(self, sandbox: SubprocessSandbox) -> None:
        """String manipulation works."""
        code = """
s = "hello world"
print(s.upper())
print(s.split())
print(len(s))
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "HELLO WORLD" in result.output

    async def test_allows_list_comprehensions(self, sandbox: SubprocessSandbox) -> None:
        """List comprehensions work."""
        code = """
squares = [x**2 for x in range(5)]
print(squares)
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "[0, 1, 4, 9, 16]" in result.output

    async def test_allows_dict_operations(self, sandbox: SubprocessSandbox) -> None:
        """Dict creation and access works."""
        code = """
d = {"a": 1, "b": 2}
d["c"] = 3
print(d.keys())
print(d["a"])
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "1" in result.output

    async def test_allows_for_loops(self, sandbox: SubprocessSandbox) -> None:
        """For loops work."""
        code = """
total = 0
for i in range(10):
    total += i
print(total)
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "45" in result.output

    async def test_allows_functions(self, sandbox: SubprocessSandbox) -> None:
        """Function definitions work."""
        code = """
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "Hello, World!" in result.output

    async def test_allows_classes(self, sandbox: SubprocessSandbox) -> None:
        """Class definitions work (basic)."""
        code = """
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

c = Counter()
print(c.increment())
print(c.increment())
"""
        result = await sandbox.execute(code)
        assert result.success is True
        assert "1" in result.output
        assert "2" in result.output

