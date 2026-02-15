# Code Review Summary – AgentStrategy Implementation Plan

## Overview
This review covers `project/docs/strategy_implementation_plan.md`, which proposes an `AgentStrategy` system for voxagent, including `StrategyContext`, `StrategyResult`, an `AgentStrategy` ABC, and four concrete behavior strategies plus `RetryStrategy`, integrated into the existing `Agent` class.

The plan is generally clear, well-structured, and aligned with the Strategy pattern. However, there are a few design areas that should be clarified or adjusted before implementation, especially around streaming semantics, default behavior, and integration with existing subsystems.

## Verdict: REQUEST_CHANGES

### Statistics
- Files reviewed: 1 (`project/docs/strategy_implementation_plan.md`)
- Blockers: 1
- Issues: 3
- Suggestions: 6
- Nits: 2

---

## 🔴 Blockers (Must Fix)

### 🔴 [BLOCKER] Streaming semantics and StrategyContext capabilities

**Location:** StrategyContext and AgentStrategy definitions (sections “StrategyContext” and “AgentStrategy ABC”).

**Issue:** The plan defines only non-streaming helpers on `StrategyContext` (`call_llm`, `execute_tool`, `run_tool_loop`) and an `AgentStrategy.execute_stream()` that by default calls `execute()` and then yields events derived from the final `StrategyResult`. There is no concrete design for how strategies or the Agent will integrate with the existing `run_stream()` pipeline and `StreamEventData` model.

Without a clearly specified streaming contract, there is a high risk that:
- `run_stream()` behavior will diverge from the current implementation (e.g., fewer/more/different event types, different ordering).
- Non-default strategies will effectively be non-streaming (only emitting a synthetic “final result” event), which breaks the goal of supporting streaming for all strategies.
- Existing MCP/sub-agent integrations that rely on specific streaming event sequences may regress.

**Why it matters:** Streaming is explicitly in-scope (Goal: “Support both `run()` and `run_stream()`”) and is one of the more subtle integration points. If the streaming story is not nailed down at the design level, it will be difficult to implement strategies in a way that is both correct and backwards compatible.

**Suggested direction:**
- Extend `StrategyContext` with explicit streaming helpers or access to a lower-level event/streaming API, for example:
  - `async def call_llm_stream(...) -> AsyncIterator[StreamEventData]`
  - `async def run_tool_loop_stream(...) -> AsyncIterator[StreamEventData]`
- Clearly specify how the default `execute_stream()` should:
  - Map `StrategyResult` to a canonical sequence of `StreamEventData` when a strategy only implements `execute()`.
  - Preserve the existing event ordering and types for the “default” behavior.
- Add to the plan concrete expectations for each strategy’s streaming behavior (e.g., whether they must override `execute_stream`, when intermediate events are emitted during reflection/planning/steps, and how aborts are handled mid-stream).

Until this contract is explicit, it will be very hard to implement `run_stream()` in a way that is compatible with current behavior.

---

## 🟠 Issues (Should Fix)

### 🟠 [ISSUE] Default strategy usage and backwards-compatibility path are inconsistent

**Location:** Step 2 (DefaultStrategy) and Step 3 (Agent integration).

**Issue:** The comment in `Agent.__init__` says `self._strategy = strategy  # Will use DefaultStrategy if None`, but the `run()` sketch shows:
- Building a `StrategyContext`.
- If `self._strategy` is set, calling `self._strategy.execute(context)`.
- Otherwise, falling back to the *old* inlined implementation (“current implementation (will be extracted to DefaultStrategy)”).

This is internally inconsistent:
- It is unclear whether the long-term design is “always go through a strategy (using `DefaultStrategy` by default)” or “only use strategies when explicitly provided; otherwise keep legacy behavior.”
- Keeping both a legacy code path and a `DefaultStrategy` that duplicates it increases drift risk and weakens the “single source of truth” for the tool loop.

**Why it matters:**
- Backwards compatibility is a primary goal (“Default behavior is identical to current behavior”).
- Having two code paths for the same behavior makes it easy for one path to diverge from the other over time and for tests to cover only one of them.

**Suggested fix:**
- Decide and document a single, explicit policy:
  - Preferred: always use a strategy, with `DefaultStrategy` as the implicit default (e.g., `self._strategy = strategy or DefaultStrategy()` or an equivalent lazy factory), and remove the legacy run-loop path from `Agent.run()`/`run_stream()` after extraction.
- Ensure the plan explicitly calls out that after extraction and validation, `Agent.run()` and `Agent.run_stream()` both delegate exclusively to strategies, to avoid long-term dual paths.

---

### 🟠 [ISSUE] Tool-loop responsibility is split between StrategyContext and DefaultStrategy

**Location:** StrategyContext helper methods and DefaultStrategy pseudocode.

**Issue:**
- `StrategyContext` exposes `run_tool_loop()` described as “Run the default tool loop, returning (messages, texts, metas).”
- `DefaultStrategy.execute()` then re-implements a full tool loop manually (building messages, calling `context.call_llm`, iterating on tool calls, appending tool results), rather than delegating to `context.run_tool_loop()`.

This creates ambiguity about which component is the canonical owner of the core tool loop logic.

**Why it matters:**
- If both `StrategyContext.run_tool_loop()` and `DefaultStrategy.execute()` implement similar but not identical loops, they can easily drift, leading to inconsistent behavior across strategies.
- Other strategies (Reflection/Planning/ReAct) are encouraged to rely on `run_tool_loop()`. If that loop’s behavior doesn’t exactly match what `Agent.run()` historically did, it undermines backwards compatibility.

**Suggested fix:**
- Define a single source of truth for the core tool loop.
  - Either: make `StrategyContext.run_tool_loop()` the canonical implementation, and have `DefaultStrategy.execute()` be a thin wrapper that calls it with the appropriate initial messages; or
  - Keep the canonical loop in `DefaultStrategy`, and have `StrategyContext.run_tool_loop()` delegate to an internal helper that is also used by `DefaultStrategy`.
- Update the plan to state clearly which component owns that logic and how all strategies should interact with it.

---

### 🟠 [ISSUE] Integration plan does not explicitly cover sub-agents and MCP tools

**Location:** Overall scope and testing strategy (goals and “Integration Tests” section).

**Issue:** The plan focuses on integrating strategies with `Agent.run()` and `run_stream()`, but does not explicitly call out:
- How existing sub-agent support (in `src/voxagent/subagent/`) will interact with the new strategy system (e.g., do sub-agents get their own strategies, or rely on the parent Agent’s strategy?).
- How MCP-backed tools are exercised through `StrategyContext.execute_tool` / `run_tool_loop` and whether any additional constraints apply.

**Why it matters:**
- Sub-agents and MCP tools are likely built on top of the existing Agent + tools abstractions. If they rely on precise behavior of the tool loop or streaming events, subtle changes could break them even if core `Agent.run()` tests pass.

**Suggested fix:**
- Extend the testing strategy with explicit integration tests that:
  - Run sub-agent flows (as they exist today) with `DefaultStrategy` and at least one non-trivial strategy (e.g., `RetryStrategy`).
  - Exercise MCP tools via `StrategyContext.execute_tool` / `run_tool_loop` to validate that their behavior (including errors and streaming, where applicable) is unchanged.
- Add a brief design note on how strategies are expected to be configured/used in sub-agent contexts (per-Agent vs per-sub-agent vs per-call).

---

## 🟡 Suggestions (Consider)

### 🟡 [SUGGESTION] Clarify and possibly structure `StrategyContext.deps`

`StrategyContext` includes a very generic `deps: Any | None`. This is convenient but weakly typed and may become a dumping ground for arbitrary state.

Consider:
- Defining a small `StrategyDeps` dataclass or protocol that includes the key extension points you know you need (e.g., logger, telemetry, cache, sub-agent manager) and using that instead of `Any`.
- Documenting what is expected to be passed via `deps` versus what should be first-class fields on the context.

This will help keep strategies maintainable and keep the context surface area clear.

---

### 🟡 [SUGGESTION] Define a precise contract for `StrategyResult.error`

The plan introduces `StrategyResult.error: str | None` but does not describe when it should be set vs when strategies should raise exceptions.

Consider documenting a clear contract, for example:
- `execute()` may raise for unexpected infrastructure errors (e.g., provider outage, serialization bugs).
- `error` is reserved for *logical* or *domain-level* errors where the strategy intentionally decides the run “failed” but still produces a well-formed result.
- `RetryStrategy`’s default `failure_detector` should be aligned with that contract (e.g., examine `error` plus specific `ToolMeta` flags).

This will make error-handling and retries more predictable and easier to reason about.

---

### 🟡 [SUGGESTION] Make abort handling explicit in all strategies

`DefaultStrategy` explicitly checks `context.abort_controller.signal.aborted`, but the higher-level strategies (Reflection, Plan-and-Execute, ReAct, Retry) do not mention abort behavior.

Consider adding to the plan that each strategy must:
- Periodically check `context.abort_controller.signal.aborted` during long-running loops (reflection iterations, plan steps, ReAct steps, retries).
- Short-circuit cleanly when aborted, setting an appropriate `error` or metadata flag.

This will ensure consistent cancellation behavior across all strategies.

---

### 🟡 [SUGGESTION] Parse ReAct outputs structurally instead of via string heuristics

The ReAct strategy currently proposes checking for completion with a simple `"FINISH" in text.upper()` check. This may misfire if “FINISH” appears in another context.

Consider:
- Parsing the ReAct output structure more strictly (e.g., detect an `Action: FINISH` line).
- Encapsulating that parsing logic in a helper method with dedicated tests (success, malformed, ambiguous outputs).

This will make ReAct behavior more robust and easier to debug.

---

### 🟡 [SUGGESTION] Document strategy composition patterns and limitations

The plan introduces `RetryStrategy` as a wrapper around another `AgentStrategy`, which is great, but does not describe recommended composition patterns more generally.

Consider adding short guidance and tests for:
- Nesting strategies (e.g., `RetryStrategy(ReflectionStrategy(...))`).
- Whether sharing a single strategy instance across multiple Agents or concurrent runs is supported or discouraged (i.e., strategies should be treated as configuration-only and thread-safe).

This will help users avoid subtle bugs due to unintended shared state.

---

### 🟡 [SUGGESTION] Align the implementation phases with a TDD-centric testing order

The Testing Strategy section is solid, but you could make the TDD flow more explicit:
- For each phase (Foundation, Strategies, Polish), enumerate which tests should be written first, then implemented against.
- Especially for Step 1–3, call out that backward-compatibility tests (existing Agent tests + new tests comparing legacy vs `DefaultStrategy`) should be in place before refactoring.

This will reduce regression risk as you extract and refactor the tool loop.

---

## 🔵 Nits (Optional)

- The plan sometimes refers to “default tool loop” in both `StrategyContext` and `DefaultStrategy`. A slightly more precise naming (e.g., `run_base_tool_loop`) in the context helper might reduce mental load when reading the code.
- Consider explicitly noting in the plan whether `StrategyResult.metadata` is intended for public consumption (e.g., surfaced on `RunResult`) or primarily for internal debugging/telemetry.

---

## 🟢 What I Liked

- The overall architecture cleanly applies the Strategy pattern and keeps the core `Agent` small while enabling powerful behavior customization.
- The separation between `StrategyContext` (infrastructure access) and `AgentStrategy` (behavior) is well thought-out and should be easy to extend.
- The set of initial strategies (Default, Reflection, Plan-and-Execute, ReAct, Retry) covers a broad range of real-world agentic patterns and encourages composition.
- The testing and rollout plan (including rollback steps and a phased implementation order) shows good attention to risk management and deployability.

---

## Questions

- [ ] Should strategies be configured strictly per-Agent (as in the current constructor design), or do you foresee a need for per-call overrides (e.g., `agent.run(..., strategy=...)` for a single run)?
- [ ] How will `StrategyResult.metadata` be exposed to callers of `Agent.run()` / `run_stream()`? Will it be folded into the existing `RunResult`, and if so, how will that affect backwards compatibility of the public API?
- [ ] Are there any existing sub-agent or MCP-specific tests that should be promoted to “must pass unchanged” as part of the backwards-compatibility validation for this refactor?

