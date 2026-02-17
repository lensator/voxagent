# Samaritan: Personal Home Orchestrator

## Overview
This document tracks the evolution of Samaritan, a proactive, goal-oriented personal agent system built on the **voxDomus platform**. While the agent (voxagent) provides the core intelligence, **voxDomus** acts as the foundational operating environment that enables the system's extensibility and human interface.

---

## 1. The Platform: voxDomus (The Host)
voxDomus acts as the **Initializer and Interface Layer**. Its responsibilities are strictly limited to:
- **Bootstrapping**: Reading the configuration and instantiating the Agent with the necessary paths (`/rules`, `/memory`, `db_uri`) and secrets (`vault`).
- **Communication**: Managing the bridges to the outside world (CLI, Telegram, Signal, Webhooks).
- **Scheduling**: Running the Cron jobs that trigger the Agent for periodic tasks.
- **Handoff**: It simply passes the user's prompt or the system event to the Agent and renders the response.

## 2. The Agent: Samaritan (The System)
The Agent is a **Self-Contained Operating Unit**. Once initialized with a `config_dir`, it manages its own:
- **Lifecycle**: Automatically starts and stops background services (like `HomeSyncManager`) using the `async with agent` pattern.
- **State & Memory**: Internally initializes its `FileSessionStorage` and `LanceDBMemoryManager`.
- **Intelligence**: Executes the Dual-Stage Strategy (Intent/Synthesis), accessing its internal memory directly.
- **Tooling**: Handles MCP connections, discovery, and execution.
- **Logic**: Enforces the Core Protocol and Rules internally.

### Stage 1: Fast Intent (Local)
- **Model**: Ollama (`qwen2.5-coder:3b` or `llama3.2:1b`).
- **Goal**: Sub-second classification and execution of direct Greek hardware commands.
- **Hardware**: CPU-optimized for non-GPU environments (Ubuntu/Mac M2).

### Stage 2: Power Synthesis (Cloud)
- **Model**: Gemini 2.0 Flash (via Antigravity provider).
- **Goal**: Deep reasoning, "Mission" planning, and distillation of complex environment states into simple English answers.
- **Context**: Accesses the full "Environment Context" (LanceDB facts, rules, history).

---

## 2. Memory & Intelligence (LanceDB)
- **LanceDB Manager**: A local, multilingual vector index.
- **Source of Truth**: Plain-text Markdown files in `/home_config/` (User-editable).
- **Auto-Sync**: Background polling of HA/OpenHAB to keep device IDs and statuses fresh.
- **Hybrid Memory**: Summarized long-term history combined with a rolling window of recent turns.

---

## 3. The Rivalry: "The Machine" vs "OpenClaw"
Our goal is to exceed the features of OpenClaw while maintaining superior security and local-first privacy.

| Feature | The Machine | OpenClaw |
| :--- | :--- | :--- |
| **Language** | Native Greek & English | Assumed English Only |
| **Hardware** | Physical Home + Digital | Digital Only |
| **Security** | Sandboxed Code + Protocol | Broad permissions |
| **Transparency** | Distilled human answers | Raw bot output |

---

## 4. Current Status & Integration
- **Direct MCP Connection**: Connecting to Home Assistant via direct URL (`/api/mcp`) or `mcp-proxy` bridge.
- **Voice Capabilities (Planned)**: Local Whisper-based transcription to replace the insecure/limited OpenClaw `voice-transcribe` skill.
- **Rules Engine**: Standardized English rules (`frugality.md`, `transparency.md`) serving as the agent's internal "Constitution."

---

## 5. Operations Manual
- **Launch**: `python examples/06_home_orchestrator.py`
- **Config**: Edit `home_config/rules/core_protocol.md` to change core logic.
- **Refresh**: Use `/refresh` in the Gemini CLI after modifying `settings.json`.

---

## 6. Strategic Objective: Outperform OpenClaw
Our goal is to build a system that surpasses the capabilities of OpenClaw (Moltbot) by combining digital utility with physical grounding and superior security.

### The Rivalry Roadmap
| Feature | OpenClaw | Samaritan (Our Advantage) |
| :--- | :--- | :--- |
| **Interface** | Signal/Telegram/Discord | **Universal API**: CLI + Voice (Whisper) + Chat + HA Assist. |
| **Memory** | Local JSON | **Hybrid Vector Memory**: LanceDB + Human-Readable Files (Transparent). |
| **Automation** | Cron Jobs | **Event-Driven Orchestration**: Reacts to physical state changes, not just time. |
| **Security** | Broad permissions | **Constitution Protocol**: Strict rules, sandboxed tools, and admin override. |
| **Scope** | Digital Only | **Physical + Digital**: Can lock doors based on emails or flash lights for alerts. |

---

## 7. Future Development & TODOs
To reach full "Samaritan" potential, the following behavioral shapes (strategies) must be implemented:

- [ ] **Human-In-The-Loop Strategy**: A wrapper that pauses execution before critical tool calls, requesting explicit Admin approval via the UI.
- [ ] **Multi-Agent Manager Strategy**: A strategy that can dynamically delegate sub-tasks to specialized child agents (Coder, Researcher, Financial Auditor).
- [ ] **Recursive Goal Achievement**: Enabling the agent to break down long-term missions into hierarchical sub-goals stored in LanceDB.
- [ ] **Dynamic Shape-Shifting**: Implementing logic to auto-swap the active strategy based on the detected intent (e.g., swapping from `HomeOrchestrator` to `MultiAgent` for technical projects).
- [ ] **Native Voice Gateway**: Implementing a high-speed, local Whisper-based bridge within voxDomus to provide the agent with native "Ears."


