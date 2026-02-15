# AgentStrategy System Implementation Plan

## Overview

Add a pluggable strategy system to voxagent where each agentic pattern (Reflection, ReAct, Planning, etc.) is encapsulated as a strategy that controls how the agent's run loop behaves.

### Goals
- Enable different agentic behavior patterns without modifying the core Agent class
- Maintain 100% backwards compatibility with existing code
- Support both `run()` and `run_stream()` methods with true streaming
- Allow strategy composition and wrapping
- Provide clean interfaces for tool access and LLM interaction

### Success Criteria
- All existing tests pass unchanged
- Default behavior is identical to current behavior
- Strategies can be easily swapped via constructor parameter or per-call override
- Each pattern (Reflection, ReAct, Planning, Retry) is independently usable
- Streaming emits the same event sequence as current implementation

### Scope Boundaries
- **Included**: Strategy interface, 5 strategy implementations, Agent integration, streaming support
- **Excluded**: New streaming event types, changes to provider interface, session management changes

---

## Review Questions Answered

### Q1: Per-Agent vs Per-Call Strategy Configuration
**Answer**: Both are supported:
- **Per-Agent (constructor)**: `Agent(model="...", strategy=ReflectionStrategy())`
- **Per-Call override**: `agent.run(prompt, strategy=ReActStrategy())` for single runs

The per-call strategy takes precedence. This allows setting a default strategy at construction time while overriding for specific calls when needed.

### Q2: StrategyResult.metadata Exposure
**Answer**: `StrategyResult.metadata` is exposed via a new `RunResult.strategy_metadata: dict[str, Any]` field:
- This preserves backwards compatibility (new optional field)
- Strategies store iteration counts, plan steps, retry attempts, etc.
- Public API consumers can inspect strategy-specific telemetry

### Q3: Existing Sub-agent/MCP Tests That Must Pass
**Answer**: The following test files contain tests that exercise tool execution and must pass unchanged:
- `tests/test_code/test_tool_proxy.py` - Tool proxy tests (sync/async tool execution, error handling)
- `tests/test_code/test_code_mode.py` - Code mode executor tests (tool execution via code)
- Any future Agent integration tests

Additionally, the sub-agent and MCP flows defined in:
- `src/voxagent/subagent/definition.py` (SubAgentDefinition.run())
- `src/voxagent/mcp/tool.py` (MCPToolDefinition.run())
must continue to work through the `execute_tool()` pipeline.

---

## Prerequisites

### Dependencies
- No new external dependencies required
- All implementations use existing voxagent internals

### Environment Requirements
- Python 3.11+ (existing requirement)
- Pydantic 2.0+ (existing requirement)

---

## File Structure

```
src/voxagent/strategies/
├── __init__.py              # Public exports and lazy imports
├── base.py                  # AgentStrategy ABC + StrategyContext + StrategyResult
├── default.py               # DefaultStrategy (current tool-loop behavior)
├── reflection.py            # ReflectionStrategy (generate → critique → revise)
├── planning.py              # PlanAndExecuteStrategy (plan first, then execute)
├── react.py                 # ReActStrategy (interleaved reasoning + action)
└── retry.py                 # RetryStrategy (self-correction on failure)
```

---

## Streaming Design

### Event Types (from src/voxagent/streaming/events.py)

The strategy system must emit these existing event types in the correct sequence:

| Event Type | When Emitted | Category |
|------------|--------------|----------|
| `RunStartEvent` | At run start | Lifecycle |
| `TextDeltaEvent` | Each LLM text chunk | Inference |
| `ToolStartEvent` | Before tool execution | Tool |
| `ToolOutputEvent` | Streaming tool output (if supported) | Tool |
| `ToolEndEvent` | After tool execution | Tool |
| `RunEndEvent` | At run end | Lifecycle |
| `RunErrorEvent` | On error | Lifecycle |

### Streaming Event Sequence

Normal run with tool calls:
```
RunStartEvent
├─ [LLM Call 1]
│  ├─ TextDeltaEvent (chunk 1)
│  ├─ TextDeltaEvent (chunk 2)
│  └─ ... (multiple TextDeltaEvents)
├─ [Tool Execution Loop]
│  ├─ ToolStartEvent (tool 1)
│  │  └─ ToolEndEvent (tool 1)
│  ├─ ToolStartEvent (tool 2)
│  │  └─ ToolEndEvent (tool 2)
│  └─ ...
├─ [LLM Call 2 - with tool results]
│  ├─ TextDeltaEvent (chunk 1)
│  └─ ...
└─ RunEndEvent
```

### StrategyContext Streaming Methods

The `StrategyContext` will provide **both** non-streaming and streaming helpers:

```python
@dataclass
class StrategyContext:
    """Context for strategy execution."""
    
    # Original request
    prompt: str
    deps: Any | None
    session_key: str | None
    message_history: list[Message] | None
    timeout_ms: int | None
    
    # Agent internals (read-only access)
    provider: BaseProvider
    tools: list[ToolDefinition]
    system_prompt: str | None
    abort_controller: AbortController
    
    # Run tracking
    run_id: str
    
    # --- NON-STREAMING HELPERS ---
    
    async def call_llm(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Make a single LLM call and return (text, tool_calls).
        
        Collects all streaming chunks internally and returns final result.
        Does NOT emit events - use call_llm_stream() for event emission.
        """
        
    async def execute_tool(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Execute a single tool call.
        
        Does NOT emit events - use execute_tool_stream() for event emission.
        """
        
    async def run_tool_loop(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], list[str], list[ToolMeta]]:
        """Run the canonical tool loop, returning (messages, texts, metas).
        
        This is THE canonical tool loop implementation. DefaultStrategy
        delegates to this method. Other strategies may call this directly
        for their inner loops.
        
        Does NOT emit events - use run_tool_loop_stream() for event emission.
        """
    
    # --- STREAMING HELPERS ---
    
    async def call_llm_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[TextDeltaEvent | tuple[str, list[ToolCall]]]:
        """Stream LLM call, yielding TextDeltaEvents.
        
        Yields:
            TextDeltaEvent for each text chunk
            Final tuple[str, list[ToolCall]] as last item
            
        Raises:
            RunErrorEvent is NOT yielded - exceptions are raised
        """
        
    async def execute_tool_stream(
        self,
        tool_call: ToolCall,
    ) -> AsyncIterator[ToolStartEvent | ToolOutputEvent | ToolEndEvent]:
        """Stream tool execution, yielding tool events.
        
        Yields:
            ToolStartEvent at start
            ToolOutputEvent for streaming output (if tool supports it)
            ToolEndEvent at end (contains ToolResult)
        """
        
    async def run_tool_loop_stream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[StreamEventData]:
        """Stream the canonical tool loop, yielding all events.
        
        Yields:
            TextDeltaEvent (LLM streaming)
            ToolStartEvent (before each tool)
            ToolEndEvent (after each tool)
            
        Returns final state via metadata or separate accessor.
        
        This is THE canonical streaming tool loop. DefaultStrategy.execute_stream()
        delegates to this method.
        """
```

### Streaming Behavior per Strategy

| Strategy | `execute_stream()` Behavior |
|----------|---------------------------|
| `DefaultStrategy` | Delegates to `context.run_tool_loop_stream()` - true streaming |
| `ReflectionStrategy` | Streams each iteration: generation → critique → revision events |
| `PlanAndExecuteStrategy` | Streams planning phase, then each step execution |
| `ReActStrategy` | Streams each Thought/Action/Observation cycle |
| `RetryStrategy` | Wraps inner strategy's streaming, re-emits on retry |

---

## Tool Loop Ownership

### Single Source of Truth: `StrategyContext.run_tool_loop()`

The canonical tool loop implementation lives in `StrategyContext.run_tool_loop()` and `run_tool_loop_stream()`. The current tool loop logic from `Agent.run()` will be extracted to this location.

**Why**: This ensures all strategies use exactly the same tool loop semantics, preventing drift.

### DefaultStrategy Delegation

```python
class DefaultStrategy(AgentStrategy):
    """Default tool-loop strategy matching current Agent.run() behavior."""
    
    async def execute(self, context: StrategyContext) -> StrategyResult:
        # Build initial messages
        messages = self._build_initial_messages(context)
        
        # Delegate to canonical tool loop
        final_messages, texts, metas = await context.run_tool_loop(messages)
        
        return StrategyResult(
            messages=final_messages,
            assistant_texts=texts,
            tool_metas=metas,
        )
    
    async def execute_stream(
        self,
        context: StrategyContext,
    ) -> AsyncIterator[StreamEventData]:
        # Build initial messages
        messages = self._build_initial_messages(context)
        
        # Delegate to canonical streaming tool loop
        async for event in context.run_tool_loop_stream(messages):
            yield event
```

### Other Strategies

Other strategies (Reflection, Planning, ReAct) call `context.run_tool_loop()` for their inner tool loops, ensuring consistent behavior:

```python
class ReflectionStrategy(AgentStrategy):
    async def execute(self, context: StrategyContext) -> StrategyResult:
        # Initial generation via tool loop
        messages, texts, metas = await context.run_tool_loop(initial_messages)
        
        for i in range(self.max_iterations):
            # Critique (no tools)
            critique, _ = await context.call_llm(critique_messages, tools=None)
            if "APPROVED" in critique:
                break
            
            # Revision via tool loop
            revision_messages, new_texts, new_metas = await context.run_tool_loop(...)
            # ...
```

---

## Default Strategy Policy

### Single Code Path: Always Use Strategy

After extraction and validation, `Agent.run()` and `Agent.run_stream()` will **always** delegate to a strategy:

```python
class Agent:
    def __init__(
        self,
        model: str,
        *,
        strategy: AgentStrategy | None = None,
        # ...
    ) -> None:
        # Default to DefaultStrategy if none provided
        self._strategy = strategy or DefaultStrategy()
    
    async def run(
        self,
        prompt: str,
        *,
        strategy: AgentStrategy | None = None,  # Per-call override
        # ...
    ) -> RunResult:
        # Use per-call strategy if provided, else agent's default
        effective_strategy = strategy or self._strategy
        
        # Build context
        context = StrategyContext(...)
        
        # ALWAYS delegate to strategy - no legacy code path
        result = await effective_strategy.execute(context)
        
        return RunResult(
            messages=result.messages,
            assistant_texts=result.assistant_texts,
            tool_metas=result.tool_metas,
            error=result.error,
            strategy_metadata=result.metadata,  # New field
        )
```

### Migration Path

1. **Phase 1**: Add strategy support alongside existing inline code (during testing)
2. **Phase 2**: Validate `DefaultStrategy` produces identical results
3. **Phase 3**: Remove legacy inline code path, always use strategy

---

## Sub-agent and MCP Integration

### Sub-agent Pattern

Sub-agents are recursive `Agent.run()` calls with depth tracking:

```python
# From src/voxagent/subagent/definition.py
class SubAgentDefinition(ToolDefinition):
    async def run(self, params: dict[str, Any], context: ToolContext) -> Any:
        # Increment depth
        child_context = sub_context.child_context(run_id=None)
        
        # Recursive Agent.run() - uses its OWN strategy
        result = await self._agent.run(prompt=task, deps=child_context.deps, ...)
```

**Key Insight**: Each sub-agent has its own `Agent._strategy`. When a sub-agent runs, it uses **its own** strategy, not the parent's. This is by design:
- Parent agent with `ReflectionStrategy` spawns sub-agent with `DefaultStrategy`
- Each agent is independently configurable

### MCP Tool Pattern

MCP tools flow through the same `execute_tool()` pipeline:

```python
# From src/voxagent/mcp/tool.py
class MCPToolDefinition(ToolDefinition):
    async def run(self, params: dict[str, Any], context: ToolContext) -> Any:
        result = await self._mcp_server.direct_call_tool(
            self._original_tool_name,
            params,
        )
        return extracted_text
```

**Key Insight**: MCP tools are executed via `StrategyContext.execute_tool()`, which calls `ToolDefinition.run()`. The strategy system doesn't change this - MCP tools work identically.

### Integration Test Requirements

Add explicit integration tests:

```python
class TestSubAgentWithStrategies:
    async def test_sub_agent_uses_own_strategy(self):
        """Sub-agent uses its configured strategy, not parent's."""
        sub_agent = Agent(model="...", strategy=DefaultStrategy())
        parent_agent = Agent(
            model="...",
            strategy=ReflectionStrategy(),
            sub_agents=[sub_agent],
        )
        result = await parent_agent.run("Use the sub-agent to...")
        # Verify sub-agent executed with DefaultStrategy
    
    async def test_sub_agent_depth_tracking_preserved(self):
        """Depth tracking works correctly with strategies."""
        # ...

class TestMCPWithStrategies:
    async def test_mcp_tools_work_with_default_strategy(self):
        """MCP tools execute correctly via DefaultStrategy."""
        # ...
    
    async def test_mcp_tools_work_with_retry_strategy(self):
        """MCP tool failures trigger RetryStrategy correctly."""
        # ...
```

---

## Error Handling Contract

### StrategyResult.error vs Exceptions

**Contract**:

| Scenario | Behavior |
|----------|----------|
| **Infrastructure errors** (provider outage, network failure, serialization bug) | Raise exception |
| **Domain errors** (tool failure, validation error, abort signal) | Set `StrategyResult.error` |
| **Partial success** (some tools succeeded, some failed) | Return result with `error=None`, individual `ToolMeta.success=False` |

### Error Flow

```python
@dataclass
class StrategyResult:
    """Result from strategy execution."""
    
    messages: list[Message]
    assistant_texts: list[str]
    tool_metas: list[ToolMeta]
    
    # Strategy-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Domain-level error (run completed but failed logically)
    error: str | None = None
```

### RetryStrategy Alignment

```python
class RetryStrategy(AgentStrategy):
    @staticmethod
    def _default_failure_detector(result: StrategyResult) -> str | None:
        """Detect domain failures in result."""
        # Check for explicit error
        if result.error:
            return result.error
        
        # Check for tool failures
        for meta in result.tool_metas:
            if not meta.success:
                return f"Tool {meta.tool_name} failed: {meta.error}"
        
        return None  # No failure
```

---

## Abort Handling Requirements

### Per-Strategy Abort Behavior

Each strategy **must** check `context.abort_controller.signal.aborted` and handle abort cleanly:

| Strategy | Abort Handling |
|----------|---------------|
| `DefaultStrategy` | Check at start of each tool loop iteration |
| `ReflectionStrategy` | Check before each iteration (generation, critique, revision) |
| `PlanAndExecuteStrategy` | Check before planning phase and before each step |
| `ReActStrategy` | Check before each Thought/Action/Observation cycle |
| `RetryStrategy` | Check before each retry attempt |

### Abort Result Pattern

```python
async def execute(self, context: StrategyContext) -> StrategyResult:
    messages = []
    texts = []
    metas = []
    
    for iteration in range(self.max_iterations):
        # REQUIRED: Check abort at loop boundaries
        if context.abort_controller.signal.aborted:
            return StrategyResult(
                messages=messages,
                assistant_texts=texts,
                tool_metas=metas,
                error="Aborted",
                metadata={"aborted_at_iteration": iteration},
            )
        
        # ... iteration logic ...
```

### Streaming Abort

For `execute_stream()`, abort should:
1. Stop yielding new events
2. Yield `RunEndEvent(aborted=True)` (handled by Agent, not strategy)
3. Return cleanly

---

## Detailed Interface Design

### 1. StrategyContext (src/voxagent/strategies/base.py)

```python
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voxagent.agent.abort import AbortController
    from voxagent.providers.base import BaseProvider
    from voxagent.streaming.events import StreamEventData
    from voxagent.tools.definition import ToolDefinition
    from voxagent.types import Message, ToolCall, ToolMeta, ToolResult


@dataclass
class StrategyContext:
    """Context for strategy execution.
    
    Provides access to LLM, tools, and the canonical tool loop.
    All helper methods handle abort signal internally.
    """
    
    # Original request
    prompt: str
    deps: Any | None
    session_key: str | None
    message_history: list[Message] | None
    timeout_ms: int | None
    
    # Agent internals
    provider: BaseProvider
    tools: list[ToolDefinition]
    system_prompt: str | None
    abort_controller: AbortController
    
    # Run tracking
    run_id: str
    
    # Internal state (set by Agent, not strategies)
    _agent: Any = field(default=None, repr=False)
    
    # --- NON-STREAMING HELPERS ---
    
    async def call_llm(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Make a single LLM call (non-streaming)."""
        ...
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        ...
    
    async def run_tool_loop(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], list[str], list[ToolMeta]]:
        """Run the canonical tool loop (non-streaming)."""
        ...
    
    # --- STREAMING HELPERS ---
    
    async def call_llm_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamEventData | tuple[str, list[ToolCall]]]:
        """Stream LLM call, yielding TextDeltaEvents."""
        ...
    
    async def execute_tool_stream(
        self,
        tool_call: ToolCall,
    ) -> AsyncIterator[StreamEventData]:
        """Stream tool execution, yielding tool events."""
        ...
    
    async def run_tool_loop_stream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[StreamEventData]:
        """Stream the canonical tool loop."""
        ...
```

### 2. StrategyResult (src/voxagent/strategies/base.py)

```python
@dataclass
class StrategyResult:
    """Result from strategy execution.
    
    Attributes:
        messages: All messages from the run (system, user, assistant, tool results).
        assistant_texts: List of assistant response texts (one per LLM call).
        tool_metas: Metadata for each tool execution.
        metadata: Strategy-specific metadata (iterations, steps, etc.).
        error: Domain-level error message if run failed logically.
    """
    
    messages: list[Message]
    assistant_texts: list[str]
    tool_metas: list[ToolMeta]
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

### 3. AgentStrategy ABC (src/voxagent/strategies/base.py)

```python
from abc import ABC, abstractmethod


class AgentStrategy(ABC):
    """Base class for agentic behavior patterns.
    
    A strategy encapsulates how an agent processes a prompt - whether that's
    a simple tool loop, reflection cycles, planning phases, or other patterns.
    
    Implementers must:
    - Implement execute() for non-streaming runs
    - Optionally override execute_stream() for true streaming (default wraps execute())
    - Handle abort signal at iteration boundaries
    """
    
    @property
    def name(self) -> str:
        """Strategy name for logging/debugging."""
        return self.__class__.__name__
    
    @abstractmethod
    async def execute(self, context: StrategyContext) -> StrategyResult:
        """Execute the strategy's behavior pattern.
        
        Must check context.abort_controller.signal.aborted at iteration boundaries.
        Must set StrategyResult.error for domain-level failures.
        May raise exceptions for infrastructure failures.
        """
        ...
    
    async def execute_stream(
        self,
        context: StrategyContext,
    ) -> AsyncIterator[StreamEventData]:
        """Execute the strategy with streaming events.
        
        Default implementation calls execute() and yields synthetic events.
        Override for true streaming support.
        """
        # Default: run non-streaming and yield final events
        result = await self.execute(context)
        
        # Yield synthetic text events from collected texts
        for text in result.assistant_texts:
            yield TextDeltaEvent(run_id=context.run_id, delta=text)
        
        # Note: RunStartEvent and RunEndEvent are handled by Agent, not strategy
```

---

## Implementation Steps (TDD Order)

### Phase 1: Foundation (Steps 1-3)

#### Step 1.1: Write tests for StrategyContext and StrategyResult

**Tests to write first:**
- `test_strategy_context_call_llm_returns_text_and_tools`
- `test_strategy_context_execute_tool_returns_result`
- `test_strategy_context_execute_tool_handles_mcp_tools`
- `test_strategy_context_run_tool_loop_matches_current_behavior`
- `test_strategy_context_run_tool_loop_checks_abort`
- `test_strategy_result_serialization`
- `test_strategy_result_error_field`

#### Step 1.2: Implement base.py

**Files to create:**
- `src/voxagent/strategies/__init__.py`
- `src/voxagent/strategies/base.py`

**Key implementation:**
- `StrategyContext` with all helper methods
- `StrategyResult` dataclass
- `AgentStrategy` ABC

#### Step 2.1: Write tests for DefaultStrategy

**Tests to write first:**
- `test_default_strategy_basic_prompt_response`
- `test_default_strategy_tool_execution_loop`
- `test_default_strategy_matches_agent_run_output`
- `test_default_strategy_abort_signal`
- `test_default_strategy_empty_tools`
- `test_default_strategy_streaming_events`

#### Step 2.2: Implement DefaultStrategy

**Files to create:**
- `src/voxagent/strategies/default.py`

**Key implementation:**
- `DefaultStrategy` that delegates to `context.run_tool_loop()`

#### Step 3.1: Write backward compatibility tests

**Tests to write first:**
- `test_agent_without_strategy_identical_behavior`
- `test_agent_with_default_strategy_identical_behavior`
- `test_agent_per_call_strategy_override`
- `test_agent_run_stream_with_strategy`
- `test_run_result_includes_strategy_metadata`

#### Step 3.2: Integrate strategy into Agent

**Files to modify:**
- `src/voxagent/agent/core.py`

**Key changes:**
- Add `strategy` parameter to `__init__` and `run()`
- Always delegate to strategy (no legacy code path)
- Add `strategy_metadata` to `RunResult`

---

### Phase 2: Strategies (Steps 4-7)

Each strategy follows TDD: write tests first, then implement.

#### Step 4: ReflectionStrategy

**Tests:**
- `test_reflection_approved_first_iteration`
- `test_reflection_multiple_revisions`
- `test_reflection_max_iterations_limit`
- `test_reflection_custom_critic_prompt`
- `test_reflection_abort_mid_iteration`
- `test_reflection_streaming_events`

**Implementation:**
- Generate → Critique → Revise cycle
- Use `context.run_tool_loop()` for tool-enabled phases
- Use `context.call_llm()` for critique phase (no tools)
- Check abort before each iteration

#### Step 5: PlanAndExecuteStrategy

**Tests:**
- `test_planning_creates_plan`
- `test_planning_executes_steps`
- `test_planning_max_steps_limit`
- `test_planning_parse_variations`
- `test_planning_abort_mid_step`
- `test_planning_streaming_events`

**Implementation:**
- Planning phase with structured output parsing
- Step execution via `context.run_tool_loop()`
- Check abort before planning and before each step

**ReAct Output Parsing:**
```python
def _parse_react_output(self, text: str) -> tuple[str | None, str | None, str | None]:
    """Parse ReAct output structurally.
    
    Looks for:
        Thought: <reasoning>
        Action: <tool_name or FINISH>
        Action Input: <params or final_answer>
    
    Returns:
        (thought, action, action_input) tuple
    """
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    action_match = re.search(r"Action:\s*(\S+)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?=Thought:|$)", text, re.DOTALL)
    
    return (
        thought_match.group(1).strip() if thought_match else None,
        action_match.group(1).strip() if action_match else None,
        input_match.group(1).strip() if input_match else None,
    )
```

#### Step 6: ReActStrategy

**Tests:**
- `test_react_thought_action_observation_cycle`
- `test_react_finish_detection_action_line`
- `test_react_max_steps_limit`
- `test_react_tool_execution`
- `test_react_parse_malformed_output`
- `test_react_abort_mid_cycle`
- `test_react_streaming_events`

**Implementation:**
- Thought → Action → Observation cycle
- Structural parsing of Action: FINISH (not just string contains)
- Check abort before each cycle

#### Step 7: RetryStrategy

**Tests:**
- `test_retry_success_first_attempt`
- `test_retry_success_after_retry`
- `test_retry_max_retries_exhausted`
- `test_retry_custom_failure_detector`
- `test_retry_wraps_reflection_strategy`
- `test_retry_abort_mid_retry`
- `test_retry_streaming_events`

**Implementation:**
- Wrapper pattern around inner strategy
- Custom failure detector support
- Prompt modification on retry
- Check abort before each retry

---

### Phase 3: Polish (Step 8)

#### Step 8.1: Update public exports

**Files to modify:**
- `src/voxagent/__init__.py`
- `src/voxagent/strategies/__init__.py`

#### Step 8.2: Integration tests

**Tests to add:**
- `test_sub_agent_uses_own_strategy`
- `test_sub_agent_depth_tracking_with_strategies`
- `test_mcp_tools_with_default_strategy`
- `test_mcp_tools_with_retry_strategy`
- `test_strategy_composition_retry_reflection`
- `test_concurrent_runs_independent_strategies`

---

## Strategy Composition Patterns

### Supported Compositions

```python
# Retry wrapping Reflection
agent = Agent(
    model="openai:gpt-4o",
    strategy=RetryStrategy(
        inner_strategy=ReflectionStrategy(max_iterations=3),
        max_retries=2,
    ),
)

# Retry wrapping PlanAndExecute
agent = Agent(
    model="anthropic:claude-3-5-sonnet",
    strategy=RetryStrategy(
        inner_strategy=PlanAndExecuteStrategy(max_plan_steps=5),
        max_retries=3,
    ),
)
```

### Thread Safety

Strategies should be treated as **configuration-only** and **stateless**:
- Strategies can be shared across Agents
- Strategies can be used for concurrent runs
- All mutable state lives in `StrategyContext`

```python
# OK: Share strategy across agents
reflection = ReflectionStrategy(max_iterations=3)
agent1 = Agent(model="...", strategy=reflection)
agent2 = Agent(model="...", strategy=reflection)

# OK: Concurrent runs
await asyncio.gather(
    agent1.run("Task 1"),
    agent2.run("Task 2"),
)
```

---

## File Changes Summary

### Files to Create
| File | Description |
|------|-------------|
| `src/voxagent/strategies/__init__.py` | Module exports with lazy loading |
| `src/voxagent/strategies/base.py` | ABC, StrategyContext, StrategyResult |
| `src/voxagent/strategies/default.py` | DefaultStrategy (canonical tool loop) |
| `src/voxagent/strategies/reflection.py` | ReflectionStrategy |
| `src/voxagent/strategies/planning.py` | PlanAndExecuteStrategy |
| `src/voxagent/strategies/react.py` | ReActStrategy |
| `src/voxagent/strategies/retry.py` | RetryStrategy |

### Files to Modify
| File | Changes |
|------|---------|
| `src/voxagent/agent/core.py` | Add `strategy` param, always delegate to strategy |
| `src/voxagent/types/run.py` | Add `strategy_metadata` to `RunResult` |
| `src/voxagent/__init__.py` | Add strategy exports |

### Files to Delete
None.

---

## Testing Strategy (TDD Order)

### Test File Structure

```
tests/
├── test_strategies/
│   ├── __init__.py
│   ├── test_base.py           # StrategyContext, StrategyResult tests
│   ├── test_default.py        # DefaultStrategy tests
│   ├── test_reflection.py     # ReflectionStrategy tests
│   ├── test_planning.py       # PlanAndExecuteStrategy tests
│   ├── test_react.py          # ReActStrategy tests
│   ├── test_retry.py          # RetryStrategy tests
│   └── test_integration.py    # Sub-agent, MCP, composition tests
```

### TDD Execution Order

1. **Write base tests** → Run (FAIL) → Implement base.py → Run (PASS)
2. **Write DefaultStrategy tests** → Run (FAIL) → Implement default.py → Run (PASS)
3. **Write backward compat tests** → Run (FAIL) → Integrate into Agent → Run (PASS)
4. **Write ReflectionStrategy tests** → Run (FAIL) → Implement → Run (PASS)
5. **Write PlanAndExecuteStrategy tests** → Run (FAIL) → Implement → Run (PASS)
6. **Write ReActStrategy tests** → Run (FAIL) → Implement → Run (PASS)
7. **Write RetryStrategy tests** → Run (FAIL) → Implement → Run (PASS)
8. **Write integration tests** → Run (FAIL) → Fix any issues → Run (PASS)

### Backward Compatibility Validation

Before removing the legacy code path:

```python
async def test_default_strategy_identical_to_legacy():
    """DefaultStrategy produces identical output to legacy Agent.run()."""
    # Run with legacy code path
    legacy_agent = Agent(model="...", strategy=None)  # Uses legacy
    legacy_result = await legacy_agent.run("Test prompt")
    
    # Run with DefaultStrategy
    strategy_agent = Agent(model="...", strategy=DefaultStrategy())
    strategy_result = await strategy_agent.run("Test prompt")
    
    # Compare outputs
    assert legacy_result.output == strategy_result.output
    assert legacy_result.messages == strategy_result.messages
    assert legacy_result.tool_calls == strategy_result.tool_calls
```

---

## Rollback Plan

### How to Revert

1. Remove `src/voxagent/strategies/` directory
2. Revert changes to `src/voxagent/agent/core.py`
3. Revert changes to `src/voxagent/types/run.py`
4. Revert changes to `src/voxagent/__init__.py`

### Data Migration

None required - no persistent state changes.

---

## Estimated Effort

### Time Estimate
| Phase | Estimate |
|-------|----------|
| Step 1: Base tests + implementation | 3-4 hours |
| Step 2: DefaultStrategy tests + implementation | 2-3 hours |
| Step 3: Agent integration | 3-4 hours |
| Step 4: ReflectionStrategy | 2-3 hours |
| Step 5: PlanAndExecuteStrategy | 2-3 hours |
| Step 6: ReActStrategy | 2-3 hours |
| Step 7: RetryStrategy | 2-3 hours |
| Step 8: Exports + integration tests | 2-3 hours |
| **Total** | **18-26 hours** |

### Complexity Assessment
**Medium-High**

Key complexity factors:
- Streaming implementation with correct event sequences
- Extracting tool loop to StrategyContext without breaking behavior
- Ensuring all strategies handle abort correctly
- Comprehensive test coverage with TDD
- Sub-agent and MCP integration validation

---

## Implementation Order

Recommended sequence:

1. **Phase 1: Foundation (Steps 1-3)**
   - Write tests for base classes
   - Implement StrategyContext with streaming methods
   - Implement DefaultStrategy
   - Integrate into Agent
   - Verify backwards compatibility
   - Remove legacy code path

2. **Phase 2: Strategies (Steps 4-7)**
   - ReflectionStrategy (with tests first)
   - PlanAndExecuteStrategy (with tests first)
   - ReActStrategy (with tests first)
   - RetryStrategy (with tests first)

3. **Phase 3: Polish (Step 8)**
   - Public exports
   - Integration tests
   - Final validation

Each phase should be independently testable and deployable.

---

## voxDomus Integration

This section describes how the AgentStrategy system integrates with the broader voxDomus architecture as defined in `docs/WORKFLOW-PROPOSED.md`.

### Architecture Context

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         voxDomus PROPOSED ARCHITECTURE                           │
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │Input Addons │───▶│  Transport  │───▶│Core Protocol│───▶│      CORE       │   │
│  │(Voice, CLI) │    │(gRPC, HTTP) │    │  (domubus)  │    │   (ChatAgent)   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └────────┬────────┘   │
│                                                                     │            │
│                                                                     ▼            │
│                                                           ┌─────────────────┐   │
│                                                           │    voxagent     │   │
│                                                           │  ┌───────────┐  │   │
│                                                           │  │   Agent   │  │   │
│                                                           │  │ +Strategy │◀─┼───┤
│                                                           │  └───────────┘  │   │
│                                                           │  ┌───────────┐  │   │
│                                                           │  │ Providers │  │   │
│                                                           │  │(LLM Addons│  │   │
│                                                           │  └───────────┘  │   │
│                                                           └─────────────────┘   │
│                                                                     │            │
│                                                                     ▼            │
│                                                           ┌─────────────────┐   │
│                                                           │   Tool Addons   │   │
│                                                           │ (MCP, HASS, etc)│   │
│                                                           └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points

| voxDomus Component | Strategy Integration |
|-------------------|---------------------|
| **ChatAgent** (Core) | Creates `Agent(strategy=...)` for LLM interactions |
| **LLM Addons** | Already supported via voxagent providers |
| **Tool Addons** | Flow through `StrategyContext.execute_tool()` unchanged |
| **domubus Events** | Strategies can emit events via `StrategyResult.metadata` |
| **Session/Memory Managers** | Passed via `StrategyContext.deps` |

### Strategy Use Cases by Input Type

| Input Source | Recommended Strategy | Rationale |
|-------------|---------------------|-----------|
| **Voice commands** | `DefaultStrategy` | Fast, single-turn responses for real-time TTS |
| **Complex planning** | `PlanAndExecuteStrategy` | "Set up a morning routine for weekdays" |
| **Code/automation** | `ReflectionStrategy` | Self-review before executing automations |
| **Unreliable tools** | `RetryStrategy` | HomeAssistant API timeouts, network issues |
| **Research queries** | `ReActStrategy` | Multi-step reasoning with web search |

### ChatAgent Integration Pattern

```python
# voxdomus/core/chat_agent.py

from voxagent import Agent
from voxagent.strategies import (
    DefaultStrategy,
    ReflectionStrategy,
    PlanAndExecuteStrategy,
    ReActStrategy,
    RetryStrategy,
)

@dataclass
class ChatDeps:
    """Dependencies passed to tools via StrategyContext.deps"""
    session: Session
    memory: MemoryManager
    user_id: str
    speaker_id: str | None = None


class ChatAgent:
    """Core ChatAgent using voxagent with strategy support."""

    def __init__(self, config: ChatAgentConfig):
        # Default strategy from config
        self._default_strategy = self._create_strategy(config.default_strategy)

        # Create agent with tool addons
        self.agent = Agent(
            model=config.model,
            strategy=self._default_strategy,
            tools=self._load_native_tools(),
            toolsets=self._load_tool_addons(),  # MCP, HomeAssistant, etc.
        )

    async def process_message(self, message: CoreMessage) -> CoreResponse:
        """Process incoming message from Core Protocol."""

        # Build dependencies for tools
        deps = ChatDeps(
            session=await self._get_session(message.session_id),
            memory=self._memory_manager,
            user_id=message.user_id,
            speaker_id=message.metadata.get("speaker_id"),
        )

        # Select strategy based on message intent (optional override)
        strategy = self._detect_strategy(message)

        # Run agent with per-call strategy override
        result = await self.agent.run(
            prompt=message.content,
            strategy=strategy,  # None = use default
            deps=deps,
            session_key=message.session_id,
        )

        # Build response
        return CoreResponse(
            request_id=message.request_id,
            content=result.assistant_texts[-1] if result.assistant_texts else "",
            tool_calls=[meta.to_dict() for meta in result.tool_metas],
            metadata={
                "strategy": strategy.name if strategy else self._default_strategy.name,
                "strategy_metadata": result.strategy_metadata,
            },
            error=result.error,
        )

    def _detect_strategy(self, message: CoreMessage) -> AgentStrategy | None:
        """Select strategy based on message analysis."""
        content = message.content.lower()
        metadata = message.metadata

        # Explicit strategy request in metadata
        if "strategy" in metadata:
            return self._create_strategy(metadata["strategy"])

        # Intent-based detection
        if any(word in content for word in ["plan", "schedule", "routine", "steps"]):
            return PlanAndExecuteStrategy()

        if any(word in content for word in ["research", "find out", "look up", "search"]):
            return ReActStrategy(max_steps=10)

        if any(word in content for word in ["code", "script", "automation", "create rule"]):
            return ReflectionStrategy(max_iterations=2)

        # Voice input with unreliable tools - wrap with retry
        if metadata.get("source_addon") == "voice":
            return RetryStrategy(
                inner_strategy=DefaultStrategy(),
                max_retries=2,
            )

        return None  # Use agent's default strategy

    def _create_strategy(self, name: str) -> AgentStrategy:
        """Create strategy by name (for config-driven selection)."""
        strategies = {
            "default": DefaultStrategy(),
            "reflection": ReflectionStrategy(),
            "planning": PlanAndExecuteStrategy(),
            "react": ReActStrategy(),
            "retry": RetryStrategy(),
        }
        return strategies.get(name, DefaultStrategy())
```

### Strategy Registry (Future Enhancement)

For config-driven strategy selection, a registry pattern can be added:

```python
# voxagent/strategies/registry.py

class StrategyRegistry:
    """Registry for strategy lookup by name."""

    _strategies: dict[str, type[AgentStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: type[AgentStrategy]) -> None:
        """Register a strategy class by name."""
        cls._strategies[name] = strategy_class

    @classmethod
    def get(cls, name: str, **kwargs) -> AgentStrategy:
        """Get a strategy instance by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())


# Auto-register built-in strategies
StrategyRegistry.register("default", DefaultStrategy)
StrategyRegistry.register("reflection", ReflectionStrategy)
StrategyRegistry.register("planning", PlanAndExecuteStrategy)
StrategyRegistry.register("react", ReActStrategy)
StrategyRegistry.register("retry", RetryStrategy)
```

### domubus Event Integration

Strategies can emit events for observability via metadata:

```python
# In ChatAgent.process_message()

result = await self.agent.run(prompt=message.content, ...)

# Emit strategy events to domubus
await self.bus.emit("agent.strategy.completed", {
    "request_id": message.request_id,
    "strategy": result.strategy_metadata.get("strategy_name"),
    "iterations": result.strategy_metadata.get("iterations"),
    "steps": result.strategy_metadata.get("plan_steps"),
    "attempts": result.strategy_metadata.get("attempts"),
    "duration_ms": result.strategy_metadata.get("duration_ms"),
})

# Emit tool events
for meta in result.tool_metas:
    await self.bus.emit("agent.tool.executed", {
        "request_id": message.request_id,
        "tool_name": meta.name,
        "success": meta.success,
        "duration_ms": meta.duration_ms,
    })
```

### Streaming with Voice Addon

For real-time voice responses, streaming is critical:

```python
# Voice response with streaming TTS

async def process_message_streaming(
    self,
    message: CoreMessage,
) -> AsyncIterator[CoreResponseChunk]:
    """Process message with streaming for real-time TTS."""

    strategy = self._detect_strategy(message)

    async for event in self.agent.run_stream(
        prompt=message.content,
        strategy=strategy,
        deps=deps,
    ):
        if isinstance(event, TextDeltaEvent):
            # Stream text chunks for TTS
            yield CoreResponseChunk(
                request_id=message.request_id,
                delta=event.delta,
                chunk_type="text",
            )
        elif isinstance(event, ToolStartEvent):
            # Notify voice addon that tool is executing
            yield CoreResponseChunk(
                request_id=message.request_id,
                delta="",
                chunk_type="tool_start",
                metadata={"tool_name": event.tool_name},
            )
        elif isinstance(event, RunEndEvent):
            yield CoreResponseChunk(
                request_id=message.request_id,
                delta="",
                chunk_type="end",
                metadata=event.metadata,
            )
```

### Configuration Example

```yaml
# voxdomus/config.yaml

chat_agent:
  model: "anthropic:claude-sonnet-4-20250514"

  # Default strategy for all requests
  default_strategy: "default"

  # Strategy overrides by input source
  strategy_overrides:
    voice:
      strategy: "retry"
      max_retries: 2
    telegram:
      strategy: "default"
    cli:
      strategy: "react"
      max_steps: 15

  # Intent-based strategy selection
  intent_strategies:
    planning: "planning"
    research: "react"
    automation: "reflection"
```

### Testing voxDomus Integration

```python
# tests/test_voxdomus_integration.py

async def test_chat_agent_with_default_strategy():
    """ChatAgent uses DefaultStrategy for simple queries."""
    agent = ChatAgent(config)

    message = CoreMessage(content="What time is it?")
    response = await agent.process_message(message)

    assert response.metadata["strategy"] == "DefaultStrategy"
    assert response.error is None


async def test_chat_agent_detects_planning_intent():
    """ChatAgent selects PlanAndExecuteStrategy for planning queries."""
    agent = ChatAgent(config)

    message = CoreMessage(content="Plan a morning routine for me")
    response = await agent.process_message(message)

    assert response.metadata["strategy"] == "PlanAndExecuteStrategy"
    assert "plan_steps" in response.metadata["strategy_metadata"]


async def test_voice_input_uses_retry_strategy():
    """Voice input wraps with RetryStrategy for reliability."""
    agent = ChatAgent(config)

    message = CoreMessage(
        content="Turn on the lights",
        metadata={"source_addon": "voice"},
    )
    response = await agent.process_message(message)

    assert response.metadata["strategy"] == "RetryStrategy"


async def test_streaming_with_strategy():
    """Streaming works with strategies for real-time TTS."""
    agent = ChatAgent(config)

    message = CoreMessage(content="Tell me a story")
    chunks = []

    async for chunk in agent.process_message_streaming(message):
        chunks.append(chunk)

    assert any(c.chunk_type == "text" for c in chunks)
    assert chunks[-1].chunk_type == "end"
```

### Summary

The AgentStrategy system integrates seamlessly with voxDomus:

| Aspect | Integration |
|--------|-------------|
| **ChatAgent** | Uses `Agent(strategy=...)` with per-call overrides |
| **Tool Addons** | Unchanged - flow through `StrategyContext.execute_tool()` |
| **Voice Streaming** | Strategies support `execute_stream()` for real-time TTS |
| **Observability** | Strategy metadata emitted to domubus |
| **Configuration** | Config-driven strategy selection by source/intent |
| **Reliability** | `RetryStrategy` wraps unreliable tool addons |

No changes to the core strategy implementation are required. The integration is handled entirely in the voxDomus ChatAgent layer.
