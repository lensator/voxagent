"""Agent core module for voxagent."""
from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from voxagent.agent.abort import AbortController, TimeoutHandler
from voxagent.mcp import MCPServerManager
from voxagent.providers.base import (
    AbortSignal,
    BaseProvider,
    ErrorChunk,
    MessageEndChunk,
    TextDeltaChunk,
    ToolUseChunk,
)
from voxagent.streaming.events import (
    RunEndEvent,
    RunErrorEvent,
    RunStartEvent,
    StreamEventData,
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from voxagent.subagent.context import DEFAULT_MAX_DEPTH
from voxagent.subagent.definition import SubAgentDefinition
from voxagent.tools.definition import ToolDefinition
from voxagent.tools.executor import execute_tool
from voxagent.tools.registry import ToolRegistry
from voxagent.types.messages import Message, ToolCall, ToolResult
from voxagent.types.run import ModelConfig, RunResult, ToolMeta
from voxagent.strategies.base import AgentStrategy, StrategyContext, StrategyResult
from voxagent.strategies.default import DefaultStrategy

if TYPE_CHECKING:
    pass

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


class Agent(Generic[DepsT, OutputT]):
    """Main agent class for voxagent.

    The Agent is the primary entry point for voxagent. It combines:
    - Model configuration (provider and model)
    - Dependency injection (deps_type)
    - Output type specification (output_type)
    - Tool registration and management
    - Security configuration (secret patterns, redaction)
    """

    def __init__(
        self,
        model: str,  # "provider:model" format
        *,
        name: str | None = None,
        deps_type: type[DepsT] | None = None,
        output_type: type[OutputT] | None = None,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        toolsets: list[Any] | None = None,
        sub_agents: list["Agent[Any, Any]"] | None = None,
        max_sub_agent_depth: int = DEFAULT_MAX_DEPTH,
        retries: int = 1,
        result_retries: int = 1,
        # Strategy
        strategy: AgentStrategy | None = None,
        config_dir: str | None = None,
        session_storage: Any | None = None,
        # Security features
        secret_patterns: list[str] | None = None,
        secrets_to_redact: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Agent.

        Args:
            model: Model string in "provider:model" format (e.g., "openai:gpt-4")
            name: Optional name for this agent (used when registered as sub-agent)
            deps_type: Optional type for dependencies
            output_type: Optional type for structured output
            system_prompt: Optional system prompt for the agent
            tools: Optional list of ToolDefinitions to register
            toolsets: Optional list of toolsets (MCP servers, etc.)
            sub_agents: Optional list of child Agents to register as tools
            max_sub_agent_depth: Maximum nesting depth for sub-agent calls (default: 5)
            retries: Number of retries for failed operations (default: 1)
            result_retries: Number of retries for result validation (default: 1)
            strategy: Optional execution strategy. Defaults to DefaultStrategy.
            config_dir: Optional directory for persistent configuration (memory, rules, agents).
            session_storage: Optional session storage for persistence and hybrid memory.
            secret_patterns: Regex patterns to detect and mask secrets
            secrets_to_redact: Dictionary of named secrets to redact
        """
        # Parse and validate model string
        self._model_config = self._parse_model_string(model)
        self._model_string = model

        # Store configuration
        self._name = name
        self._deps_type = deps_type
        self._output_type = output_type
        self._system_prompt = system_prompt
        self._retries = retries
        self._result_retries = result_retries
        self._max_sub_agent_depth = max_sub_agent_depth

        # Security configuration
        self._secret_patterns = secret_patterns
        self._secrets_to_redact = secrets_to_redact

        # Strategy (default to DefaultStrategy for backwards compatibility)
        self._strategy = strategy or DefaultStrategy()

        # Config directory and internal services
        self._config_dir = config_dir
        self._session_storage = session_storage
        self._memory_manager: Any | None = None
        self._sync_manager: Any | None = None

        if self._config_dir:
            from pathlib import Path
            from voxagent.session.storage import FileSessionStorage
            from voxagent.memory.lancedb import LanceDBMemoryManager
            from voxagent.memory.sync import HomeSyncManager

            config_path = Path(self._config_dir)
            
            # 1. Initialize Storage
            if not self._session_storage:
                self._session_storage = FileSessionStorage(config_path / "sessions")
            
            # 2. Initialize Memory
            self._memory_manager = LanceDBMemoryManager(uri=str(config_path / "memory" / "lancedb"))
            
            # 3. Initialize Sync Manager (Lazy start)
            self._sync_manager = HomeSyncManager(self, self._memory_manager)

        # Tool registry
        self._tool_registry = ToolRegistry()
        if tools:
            for t in tools:
                self._tool_registry.register(t)

        # Register sub-agents as tools
        if sub_agents:
            for sub_agent in sub_agents:
                sub_tool = SubAgentDefinition(
                    agent=sub_agent,
                    name=sub_agent._name,
                    max_depth=max_sub_agent_depth,
                )
                self._tool_registry.register(sub_tool)

        # Toolsets (MCP servers, etc.) - store for later
        self._toolsets = toolsets or []

        # MCP connection caching (persistent across run() calls)
        self._mcp_manager: MCPServerManager | None = None
        self._mcp_tools: list[ToolDefinition] = []
        self._mcp_connected: bool = False

    # -------------------------------------------------------------------------
    # MCP Connection Management
    # -------------------------------------------------------------------------

    async def connect_mcp(self) -> list[ToolDefinition]:
        """Connect to MCP servers and cache the connection.

        This method connects to all MCP servers in toolsets and caches the
        connection for reuse across multiple run() calls. Call this during
        initialization/warmup to avoid connection overhead on first message.

        Returns:
            List of ToolDefinition objects from connected MCP servers.
        """
        if self._mcp_connected:
            return self._mcp_tools

        if self._toolsets:
            self._mcp_manager = MCPServerManager()
            await self._mcp_manager.add_servers(self._toolsets)
            self._mcp_tools = await self._mcp_manager.connect_all()
            self._mcp_connected = True

        return self._mcp_tools

    async def disconnect_mcp(self) -> None:
        """Disconnect from MCP servers.

        Call this when the agent is no longer needed to clean up MCP
        server connections. This is called automatically when using
        the agent as an async context manager.
        """
        if self._mcp_manager and self._mcp_connected:
            await self._mcp_manager.disconnect_all()
            self._mcp_manager = None
            self._mcp_connected = False
            self._mcp_tools = []

    @property
    def mcp_connected(self) -> bool:
        """Check if MCP servers are currently connected."""
        return self._mcp_connected

    async def __aenter__(self) -> "Agent[DepsT, OutputT]":
        """Enter async context manager - connect MCP and start sync."""
        await self.connect_mcp()
        if self._sync_manager:
            await self._sync_manager.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager - disconnect MCP and stop sync."""
        if self._sync_manager:
            await self._sync_manager.stop()
        await self.disconnect_mcp()

    @staticmethod
    def _parse_model_string(model_string: str) -> ModelConfig:
        """Parse 'provider:model' string into ModelConfig.

        Args:
            model_string: String in format "provider:model"

        Returns:
            ModelConfig instance

        Raises:
            ValueError: If the model string is invalid
        """
        if ":" not in model_string:
            raise ValueError(
                f"Invalid model string '{model_string}'. "
                "Expected format: 'provider:model' (e.g., 'openai:gpt-4')"
            )

        # Split on first colon only (model names can contain colons)
        parts = model_string.split(":", 1)
        provider = parts[0].lower()
        model = parts[1]

        if not provider:
            raise ValueError(
                f"Invalid model string '{model_string}'. "
                "provider must be non-empty."
            )

        if not model:
            raise ValueError(
                f"Invalid model string '{model_string}'. "
                "model must be non-empty."
            )

        return ModelConfig(provider=provider, model=model)

    @property
    def name(self) -> str | None:
        """Get the agent name."""
        return self._name

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model_config.model

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._model_config.provider

    @property
    def model_string(self) -> str:
        """Get the full model string."""
        return self._model_string

    @property
    def model_config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._model_config

    def set_model(self, model: str) -> None:
        """Change the model on the fly.

        Args:
            model: New model string in "provider:model" format.
        """
        self._model_config = self._parse_model_string(model)
        self._model_string = model

    @property
    def formatter(self) -> Any:
        """Get the appropriate formatter for the current provider."""
        from voxagent.formatting.registry import get_formatter_for_provider
        return get_formatter_for_provider(self.provider)

    @property
    def deps_type(self) -> type[DepsT] | None:
        """Get the deps type."""
        return self._deps_type

    @property
    def output_type(self) -> type[OutputT] | None:
        """Get the output type."""
        return self._output_type

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @property
    def retries(self) -> int:
        """Get the number of retries."""
        return self._retries

    @property
    def result_retries(self) -> int:
        """Get the number of result retries."""
        return self._result_retries

    @property
    def secret_patterns(self) -> list[str] | None:
        """Get the secret patterns."""
        return self._secret_patterns

    @property
    def secrets_to_redact(self) -> dict[str, str] | None:
        """Get the secrets to redact."""
        return self._secrets_to_redact

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get a copy of the registered tools."""
        return list(self._tool_registry.list())

    @property
    def has_tools(self) -> bool:
        """Check if any tools are registered."""
        return len(self._tool_registry.list()) > 0

    @property
    def toolsets(self) -> list[Any]:
        """Get the toolsets (MCP servers, etc.)."""
        return self._toolsets

    def _get_all_tools(self, mcp_tools: list[ToolDefinition] | None = None) -> list[ToolDefinition]:
        """Get all tools including MCP tools.

        Args:
            mcp_tools: Optional list of MCP tools to include.

        Returns:
            Combined list of native and MCP tools.
        """
        all_tools = list(self._tool_registry.list())
        if mcp_tools:
            all_tools.extend(mcp_tools)
        return all_tools

    def _has_any_tools(self, mcp_tools: list[ToolDefinition] | None = None) -> bool:
        """Check if any tools are available (native or MCP).

        Args:
            mcp_tools: Optional list of MCP tools.

        Returns:
            True if any tools are available.
        """
        if self.has_tools:
            return True
        return bool(mcp_tools)

    def tool(self, fn: Callable[..., Any]) -> ToolDefinition:
        """Decorator to register a tool from a function.

        This method can be used as a decorator to register a function as a tool:

            @agent.tool
            def my_function(x: int) -> str:
                '''Description of the tool.'''
                return str(x)

        Args:
            fn: The function to convert to a ToolDefinition

        Returns:
            ToolDefinition: The created tool definition
        """
        # Build tool definition from function
        tool_name = fn.__name__

        # Validate name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            raise ValueError(f"Invalid tool name: {tool_name}")

        # Get description from docstring
        tool_description = ""
        if fn.__doc__:
            tool_description = fn.__doc__.strip().split("\n")[0].strip()

        # Check if async
        is_async = inspect.iscoroutinefunction(fn)

        # Build parameters schema
        parameters = self._build_parameters_schema(fn)

        # Create ToolDefinition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            execute=fn,
            is_async=is_async,
        )

        # Register the tool
        self._tool_registry.register(tool_def)

        return tool_def

    def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition.

        Args:
            tool_def: The ToolDefinition to register

        Raises:
            ToolAlreadyRegisteredError: If a tool with the same name is already registered
        """
        self._tool_registry.register(tool_def)

    @staticmethod
    def _build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
        """Build JSON Schema from function type hints.

        Args:
            fn: The function to extract parameters from

        Returns:
            JSON Schema dict for the function parameters
        """
        from typing import Union, get_args, get_origin, get_type_hints

        sig = inspect.signature(fn)
        hints: dict[str, Any] = {}
        try:
            hints = get_type_hints(fn)
        except Exception:
            pass

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # Skip 'context' or 'ctx' parameter (ToolContext)
            if param_name in ("context", "ctx"):
                continue

            # Skip *args and **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # Get type hint
            type_hint = hints.get(param_name, Any)

            # Convert type to JSON Schema
            prop_schema = Agent._type_to_json_schema(type_hint)
            properties[param_name] = prop_schema

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                # Add default to schema
                if param.default is not None:
                    properties[param_name]["default"] = param.default

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required
        else:
            schema["required"] = []

        return schema

    @staticmethod
    def _type_to_json_schema(type_hint: Any) -> dict[str, Any]:
        """Convert a Python type hint to JSON Schema."""
        from typing import Union, get_args, get_origin

        # Handle None/NoneType
        if type_hint is type(None):
            return {"type": "null"}

        # Handle basic types
        if type_hint is str:
            return {"type": "string"}
        if type_hint is int:
            return {"type": "integer"}
        if type_hint is float:
            return {"type": "number"}
        if type_hint is bool:
            return {"type": "boolean"}

        # Handle Optional (Union with None)
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # Check if it's Optional (Union[X, None])
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and type(None) in args:
                # It's Optional[X]
                inner_schema = Agent._type_to_json_schema(non_none_args[0])
                inner_schema["nullable"] = True
                return inner_schema
            # General Union - use anyOf
            return {"anyOf": [Agent._type_to_json_schema(a) for a in args]}

        # Handle list
        if origin is list:
            if args:
                return {"type": "array", "items": Agent._type_to_json_schema(args[0])}
            return {"type": "array"}

        # Handle dict
        if origin is dict:
            return {"type": "object"}

        # Handle Any
        if type_hint is Any:
            return {}

        # Default to object for unknown types
        return {"type": "object"}

    # =========================================================================
    # Provider Methods
    # =========================================================================

    def _get_provider(self) -> BaseProvider:
        """Get the provider for this agent's model.

        Returns:
            The provider instance.

        Raises:
            ValueError: If the provider is not found.
        """
        from voxagent.providers.registry import get_default_registry

        registry = get_default_registry()
        return registry.get_provider(self._model_string)

    async def _save_session(self, session_key: str, messages: list[Message]) -> None:
        """Save session messages to storage.

        Args:
            session_key: The session key.
            messages: The messages to save.
        """
        if not self._session_storage:
            return

        from voxagent.session.model import Session
        
        # Load existing or create new
        session = await self._session_storage.load(session_key)
        if not session:
            session = Session.create(key=session_key)
            
        # Update messages
        session.messages = messages
        await self._session_storage.save(session)

    # =========================================================================
    # Run Methods
    # =========================================================================

    async def run(
        self,
        prompt: str | dict[str, Any] | Any,
        *,
        deps: DepsT | None = None,
        session_key: str | None = None,
        message_history: list[Message] | None = None,
        timeout_ms: int | None = None,
        strategy: AgentStrategy | None = None,
    ) -> RunResult:
        """Run the agent with a prompt.

        Args:
            prompt: The user prompt to process (string or structured object).
            deps: Optional dependencies to inject into tools.
            session_key: Optional session key for persistence.
            message_history: Optional message history to prepend.
            timeout_ms: Optional timeout in milliseconds.
            strategy: Optional strategy override for this run.

        Returns:
            RunResult containing messages, outputs, and metadata.
        """
        run_id = str(uuid.uuid4())
        abort_controller = AbortController()
        timeout_handler: TimeoutHandler | None = None
        timed_out = False
        # Track if we connected MCP in this run (for cleanup)
        mcp_connected_in_this_run = False

        if timeout_ms:
            timeout_handler = TimeoutHandler(timeout_ms)
            await timeout_handler.start(abort_controller)

        try:
            # Use cached MCP connection if available, otherwise connect
            if self._mcp_connected:
                mcp_tools = self._mcp_tools
            elif self._toolsets:
                mcp_tools = await self.connect_mcp()
                mcp_connected_in_this_run = True
            else:
                mcp_tools = []

            # Get all tools (native + MCP)
            all_tools = self._get_all_tools(mcp_tools)

            # Build messages list
            messages: list[Message] = []

            # Prepend session summary if available
            summary_content = None
            if session_key and self._session_storage:
                session = await self._session_storage.load(session_key)
                if session and session.summary:
                    summary_content = f"CONTEXT SUMMARY: {session.summary}"
                    messages.append(Message(role="system", content=summary_content))

            # Add system prompt if present
            if self._system_prompt:
                messages.append(Message(role="system", content=self._system_prompt))

            # Add message history if provided
            if message_history:
                messages.extend([
                    m if isinstance(m, Message) else Message(**m) 
                    for m in message_history
                ])

            # Process prompt (handle structured input)
            final_prompt = prompt
            if not isinstance(prompt, str):
                # Use formatter to serialize structured data
                if isinstance(prompt, dict):
                    # Format as key-value blocks using the active formatter
                    formatted_parts = []
                    for key, value in prompt.items():
                        formatted_parts.append(self.formatter.format_observation(f"{key}: {value}"))
                    final_prompt = "\n".join(formatted_parts)
                else:
                    final_prompt = str(prompt)

            # Add user prompt (don't add to messages yet, strategy will do it)
            # messages.append(Message(role="user", content=final_prompt))

            # Get provider
            provider = self._get_provider()

            # Use strategy if provided (per-call override) or fall back to agent default
            active_strategy = strategy or self._strategy

            # Create strategy context
            strategy_ctx = StrategyContext(
                prompt=final_prompt,
                deps=deps,
                session_key=session_key,
                message_history=messages,
                timeout_ms=timeout_ms,
                provider=provider,
                tools=all_tools,
                system_prompt=self._system_prompt,
                abort_controller=abort_controller,
                run_id=run_id,
                memory_manager=self._memory_manager,
            )

            # Execute strategy
            strategy_result = await active_strategy.execute(strategy_ctx)

            # Check timeout
            if timeout_handler and timeout_handler.expired:
                timed_out = True

            # Save session if key provided
            if session_key:
                await self._save_session(session_key, strategy_result.messages)

            # Convert StrategyResult to RunResult
            return RunResult(
                messages=strategy_result.messages,
                assistant_texts=strategy_result.assistant_texts,
                tool_metas=strategy_result.tool_metas,
                aborted=abort_controller.signal.aborted and not timed_out,
                timed_out=timed_out,
                error=strategy_result.error,
                strategy_metadata=strategy_result.metadata,
            )

        except Exception as e:
            return RunResult(
                messages=[],
                assistant_texts=[],
                tool_metas=[],
                aborted=False,
                timed_out=False,
                error=str(e),
            )

        finally:
            # Only disconnect MCP servers if we connected them in this run
            # (not if using cached connection from connect_mcp())
            if mcp_connected_in_this_run and not self._mcp_connected:
                await self.disconnect_mcp()
            if timeout_handler:
                timeout_handler.cancel()
            abort_controller.cleanup()

    async def run_stream(
        self,
        prompt: str | dict[str, Any] | Any,
        *,
        deps: DepsT | None = None,
        session_key: str | None = None,
        message_history: list[Message] | None = None,
        timeout_ms: int | None = None,
        strategy: AgentStrategy | None = None,
    ) -> AsyncIterator[StreamEventData]:
        """Run the agent with streaming events.

        Args:
            prompt: The user prompt to process (string or structured object).
            deps: Optional dependencies to inject into tools.
            session_key: Optional session key for persistence.
            message_history: Optional message history to prepend.
            timeout_ms: Optional timeout in milliseconds.
            strategy: Optional strategy override for this run.

        Yields:
            StreamEventData events for the run lifecycle.
        """
        run_id = str(uuid.uuid4())
        session_id = session_key or f"ephemeral-{run_id}"
        abort_controller = AbortController()
        timeout_handler: TimeoutHandler | None = None
        timed_out = False
        # Track if we connected MCP in this run (for cleanup)
        mcp_connected_in_this_run = False

        if timeout_ms:
            timeout_handler = TimeoutHandler(timeout_ms)
            await timeout_handler.start(abort_controller)

        try:
            # Use cached MCP connection if available, otherwise connect
            if self._mcp_connected:
                mcp_tools = self._mcp_tools
            elif self._toolsets:
                mcp_tools = await self.connect_mcp()
                mcp_connected_in_this_run = True
            else:
                mcp_tools = []

            # Get all tools (native + MCP)
            all_tools = self._get_all_tools(mcp_tools)

            # Emit run start
            yield RunStartEvent(run_id=run_id, session_key=session_id)

            # Build messages list
            messages: list[Message] = []

            # Prepend session summary if available
            if session_key and self._session_storage:
                session = await self._session_storage.load(session_key)
                if session and session.summary:
                    messages.append(Message(role="system", content=f"CONTEXT SUMMARY: {session.summary}"))

            if self._system_prompt:
                messages.append(Message(role="system", content=self._system_prompt))

            if message_history:
                messages.extend([
                    m if isinstance(m, Message) else Message(**m) 
                    for m in message_history
                ])

            # Process prompt
            final_prompt = prompt
            if not isinstance(prompt, str):
                if isinstance(prompt, dict):
                    formatted_parts = []
                    for key, value in prompt.items():
                        formatted_parts.append(self.formatter.format_observation(f"{key}: {value}"))
                    final_prompt = "\n".join(formatted_parts)
                else:
                    final_prompt = str(prompt)

            # Add user prompt (don't add to messages yet, strategy will do it)
            # messages.append(Message(role="user", content=final_prompt))

            # Get provider
            provider = self._get_provider()

            # Use strategy if provided (per-call override) or fall back to agent default
            active_strategy = strategy or self._strategy

            # Create strategy context
            strategy_ctx = StrategyContext(
                prompt=final_prompt,
                deps=deps,
                session_key=session_key,
                message_history=messages,
                timeout_ms=timeout_ms,
                provider=provider,
                tools=all_tools,
                system_prompt=self._system_prompt,
                abort_controller=abort_controller,
                run_id=run_id,
                memory_manager=self._memory_manager,
            )

            # Execute strategy with streaming
            async for event in active_strategy.execute_stream(strategy_ctx):
                yield event

            # Check timeout
            if timeout_handler and timeout_handler.expired:
                timed_out = True

            # Emit run end
            yield RunEndEvent(
                run_id=run_id,
                messages=messages,
                aborted=abort_controller.signal.aborted and not timed_out,
                timed_out=timed_out,
            )

        except Exception as e:
            yield RunErrorEvent(run_id=run_id, error=str(e))
            yield RunEndEvent(
                run_id=run_id,
                messages=[],
                aborted=False,
                timed_out=False,
            )

        finally:
            # Only disconnect MCP servers if we connected them in this run
            # (not if using cached connection from connect_mcp())
            if mcp_connected_in_this_run and not self._mcp_connected:
                await self.disconnect_mcp()
            if timeout_handler:
                timeout_handler.cancel()
            abort_controller.cleanup()

    def run_sync(
        self,
        prompt: str,
        *,
        deps: DepsT | None = None,
        session_key: str | None = None,
        message_history: list[Message] | None = None,
        timeout_ms: int | None = None,
    ) -> RunResult:
        """Synchronous wrapper for run().

        Args:
            prompt: The user prompt to process.
            deps: Optional dependencies to inject into tools.
            session_key: Optional session key for persistence.
            message_history: Optional message history to prepend.
            timeout_ms: Optional timeout in milliseconds.

        Returns:
            RunResult containing messages, outputs, and metadata.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # No event loop running, create one
            return asyncio.run(
                self.run(
                    prompt,
                    deps=deps,
                    session_key=session_key,
                    message_history=message_history,
                    timeout_ms=timeout_ms,
                )
            )
        else:
            # Already in an async context, use nest_asyncio pattern
            # or create new loop in thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.run(
                        prompt,
                        deps=deps,
                        session_key=session_key,
                        message_history=message_history,
                        timeout_ms=timeout_ms,
                    ),
                )
                return future.result()

