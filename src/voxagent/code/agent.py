"""Code execution mode for Agent.

When execution_mode="code", the agent uses a single execute_code tool
instead of exposing all tools directly. The LLM writes Python code
that calls tools via the virtual filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voxagent.code.sandbox import SubprocessSandbox, SandboxResult
from voxagent.code.virtual_fs import VirtualFilesystem, ToolRegistry
from voxagent.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from voxagent.agent.core import Agent


# System prompt addition for code mode
CODE_MODE_SYSTEM_PROMPT = '''
## Code Execution Mode

You have access to a single tool: `execute_code`. Use it to write Python code that:
1. Explores available tools with `ls("tools/")` and `read("tools/<category>/<tool>.py")`
2. Calls tools using `call_tool(category, tool_name, **kwargs)`
3. Uses `print()` to output results

### Available Functions in Sandbox
- `ls(path)` - List directory contents (e.g., `ls("tools/")`, `ls("tools/devices/")`)
- `read(path)` - Read file contents (e.g., `read("tools/devices/registry.py")`)
- `call_tool(category, tool_name, **kwargs)` - Call a tool with arguments
- `print(*args)` - Output results (captured and returned to you)

### Workflow
1. **Explore**: `print(ls("tools/"))` to see categories
2. **Learn**: `print(read("tools/<category>/<tool>.py"))` to see tool signatures
3. **Execute**: Use `call_tool()` to invoke tools
4. **Report**: Use `print()` to show results

### Example
```python
# Explore available tools
print("Categories:", ls("tools/"))

# Read a tool definition
print(read("tools/devices/registry.py"))

# Call the tool
result = call_tool("devices", "registry.py", device_type="light")
print("Devices:", result)
```

### Rules
1. Always use `print()` to show results
2. Explore before assuming - use `ls()` and `read()` first
3. Handle errors with try/except when appropriate
'''


class CodeModeConfig:
    """Configuration for code execution mode.
    
    Attributes:
        enabled: Whether code mode is active
        timeout_seconds: Max execution time per code block
        memory_limit_mb: Max memory for sandbox
        max_output_chars: Truncate output beyond this
    """
    
    def __init__(
        self,
        enabled: bool = True,
        timeout_seconds: int = 10,
        memory_limit_mb: int = 128,
        max_output_chars: int = 10000,
    ):
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars


class CodeModeExecutor:
    """Executes code for an agent in code mode.
    
    This class:
    1. Manages the sandbox and virtual filesystem
    2. Provides the execute_code tool implementation
    3. Routes tool calls from sandbox to real implementations
    """
    
    def __init__(
        self,
        config: CodeModeConfig,
        tool_registry: ToolRegistry,
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.sandbox = SubprocessSandbox(
            timeout_seconds=config.timeout_seconds,
            memory_limit_mb=config.memory_limit_mb,
        )
        self.virtual_fs = VirtualFilesystem(tool_registry)
        
        # Tool proxy for routing calls
        self._tool_implementations: dict[str, Any] = {}
    
    def register_tool_implementation(
        self,
        category: str,
        tool_name: str,
        implementation: Any,
    ) -> None:
        """Register a real tool implementation for the proxy."""
        key = f"{category}.{tool_name}"
        self._tool_implementations[key] = implementation

    async def execute_code(self, code: str) -> str:
        """Execute Python code in the sandbox.

        This is the tool exposed to the LLM.

        Args:
            code: Python source code to execute

        Returns:
            Captured output or error message
        """
        # Build globals with virtual filesystem functions only
        # Note: call_tool is not passed to sandbox due to pickling constraints
        # The LLM should use ls() and read() to explore, then describe what to call
        globals_dict = self.virtual_fs.get_sandbox_globals()

        # Execute in sandbox
        result = await self.sandbox.execute(code, globals_dict)

        # Format output
        if result.success:
            output = result.output or "(no output)"
            if len(output) > self.config.max_output_chars:
                output = output[:self.config.max_output_chars] + "\n... (truncated)"
            return output
        else:
            return f"Error: {result.error}"

    def call_tool(self, category: str, tool_name: str, **kwargs: Any) -> Any:
        """Call a registered tool implementation.

        This method is called outside the sandbox after the LLM
        has explored tools and decided which one to call.

        Args:
            category: Tool category (e.g., "devices")
            tool_name: Tool name (e.g., "registry.py")
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool result or error message
        """
        key = f"{category}.{tool_name}"
        if key not in self._tool_implementations:
            return f"Error: Tool '{key}' not found"
        try:
            impl = self._tool_implementations[key]
            return impl(**kwargs)
        except Exception as e:
            return f"Error calling {key}: {e}"

    def get_execute_code_tool(self) -> ToolDefinition:
        """Get the execute_code tool definition for the agent."""
        async def execute_code_wrapper(code: str) -> str:
            """Execute Python code to explore and use tools.

            Write Python code that:
            1. Uses ls() to explore available tool categories
            2. Uses read() to see tool signatures and documentation
            3. Imports and calls tools to perform actions
            4. Uses print() to output results

            Args:
                code: Python source code to execute

            Returns:
                Captured print() output or error message
            """
            return await self.execute_code(code)

        return ToolDefinition(
            name="execute_code",
            description="Execute Python code to explore and use tools. Use ls() to see tools, read() to see signatures, and print() to output results.",
            execute=execute_code_wrapper,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute"
                    }
                },
                "required": ["code"]
            },
            is_async=True,
        )


def get_code_mode_system_prompt_addition() -> str:
    """Get the system prompt addition for code mode."""
    return CODE_MODE_SYSTEM_PROMPT


def setup_code_mode_for_agent(
    agent: "Agent[Any, Any]",
    config: CodeModeConfig | None = None,
) -> CodeModeExecutor:
    """Set up code mode for an existing agent.

    This:
    1. Creates a CodeModeExecutor
    2. Registers the execute_code tool
    3. Converts existing tools to virtual filesystem entries

    Args:
        agent: The agent to configure
        config: Optional code mode configuration

    Returns:
        The CodeModeExecutor instance
    """
    config = config or CodeModeConfig()

    # Create tool registry from agent's existing tools
    vf_tool_registry = ToolRegistry()

    # Convert agent's tools to virtual filesystem entries
    for tool_def in agent._tool_registry.list():
        # Create category based on tool name or use "default"
        category = "default"

        # Build tool definition content as Python stub
        params_str = ", ".join(
            f"{k}: {v.get('type', 'Any')}"
            for k, v in tool_def.parameters.get("properties", {}).items()
        )
        tool_content = f'''def {tool_def.name}({params_str}) -> Any:
    """{tool_def.description}"""
    ...
'''

        # Get or create category
        if category not in vf_tool_registry.list_categories():
            vf_tool_registry.register_category(
                category,
                description="Default tool category",
                tools={},
            )

        # Add tool to category
        cat = vf_tool_registry.get_category(category)
        if cat:
            cat.tools[f"{tool_def.name}.py"] = tool_content

    # Create executor
    executor = CodeModeExecutor(config, vf_tool_registry)

    # Register tool implementations
    for tool_def in agent._tool_registry.list():
        executor.register_tool_implementation(
            "default",
            f"{tool_def.name}.py",
            tool_def.execute,
        )

    return executor

