"""Code execution sandbox for token optimization.

This subpackage provides:
- Sandboxed code execution (subprocess, Pyodide)
- Tool file generation from MCP servers
- Skills persistence and discovery
- PII tokenization for privacy
- Virtual filesystem for tool discovery
- Code execution mode for agents
- Tool proxy for routing sandbox calls to real implementations
"""

from voxagent.code.sandbox import CodeSandbox, SandboxResult, SubprocessSandbox
from voxagent.code.virtual_fs import (
    ToolCategory,
    ToolRegistry,
    VirtualFilesystem,
)
from voxagent.code.agent import (
    CodeModeConfig,
    CodeModeExecutor,
    get_code_mode_system_prompt_addition,
    setup_code_mode_for_agent,
    CODE_MODE_SYSTEM_PROMPT,
)
from voxagent.code.tool_proxy import (
    ToolCallRequest,
    ToolCallResponse,
    ToolProxyClient,
    ToolProxyServer,
    create_tool_proxy_pair,
)
from voxagent.code.query import QueryResult

__all__ = [
    # Sandbox
    "CodeSandbox",
    "SandboxResult",
    "SubprocessSandbox",
    # Virtual FS
    "ToolCategory",
    "ToolRegistry",
    "VirtualFilesystem",
    # Code Mode Agent
    "CodeModeConfig",
    "CodeModeExecutor",
    "CODE_MODE_SYSTEM_PROMPT",
    "get_code_mode_system_prompt_addition",
    "setup_code_mode_for_agent",
    # Tool Proxy
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolProxyClient",
    "ToolProxyServer",
    "create_tool_proxy_pair",
    # Query
    "QueryResult",
]
