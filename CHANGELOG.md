# Changelog

All notable changes to voxagent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No changes yet.

## [0.2.0] - 2026-02-03

### Added
- **Code Execution Mode**: New `execution_mode="code"` option for agents
  - `SubprocessSandbox`: Secure Python execution with RestrictedPython and process isolation
  - `VirtualFilesystem`: `ls()` and `read()` functions for progressive tool discovery
  - `ToolRegistry`: Registry for tool categories and definitions
  - `CodeModeExecutor`: Main executor for code mode agents
  - `ToolProxyClient` and `ToolProxyServer`: Queue-based IPC for routing sandbox calls to real tools
- New optional dependency: `RestrictedPython>=7.0` (install with `pip install voxagent[code]`)

## [0.1.0] - 2026-02-01

### Added
- Initial public release
- Multi-provider support: OpenAI, Anthropic, Google, Groq, Ollama
- Streaming with typed `StreamChunk` union types
- `@tool` decorator for easy function-to-tool conversion
- MCP (Model Context Protocol) integration
- Sub-agent support for hierarchical agent composition
- Session management with file-based persistence
- Full type hints with `py.typed` marker
- Comprehensive examples directory

### Provider Support
- `openai:*` - OpenAI models (gpt-4o, gpt-4-turbo, etc.)
- `anthropic:*` - Anthropic Claude models
- `google:*` - Google Gemini models
- `groq:*` - Groq-hosted models
- `ollama:*` - Local Ollama models

### Core Features
- `Agent` class with `run()` and `stream()` methods
- `BaseProvider` abstract class for custom providers
- `ProviderRegistry` for provider management
- `ToolDefinition` and `ToolContext` for tool system
- `Message`, `ToolCall`, `ToolResult` types

[Unreleased]: https://github.com/lensator/voxagent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lensator/voxagent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lensator/voxagent/releases/tag/v0.1.0

