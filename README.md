# voxagent

[![PyPI version](https://badge.fury.io/py/voxagent.svg)](https://badge.fury.io/py/voxagent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typed](https://img.shields.io/badge/typed-yes-green.svg)](https://peps.python.org/pep-0561/)

A lightweight, model-agnostic LLM provider abstraction with streaming and tool support.

## Features

- **Multi-Provider**: Unified interface for OpenAI, Anthropic, Google, Groq, Ollama
- **Streaming**: Typed `StreamChunk` union (TextDelta, ToolUse, MessageEnd, Error)
- **Tool System**: `@tool` decorator for easy function-to-tool conversion
- **MCP Integration**: First-class Model Context Protocol support
- **Type Safe**: Full type hints with `py.typed` marker
- **Minimal Dependencies**: Core requires only `pydantic`, `httpx`, `anyio`

## Installation

```bash
# Core only (no provider SDKs)
pip install voxagent

# With specific providers
pip install voxagent[openai]
pip install voxagent[anthropic]
pip install voxagent[google]
pip install voxagent[ollama]

# All providers
pip install voxagent[all]
```

## Quick Start

```python
import asyncio
from voxagent import Agent

async def main():
    agent = Agent(model="openai:gpt-4o")
    result = await agent.run("Hello, world!")
    print(result.output)

asyncio.run(main())
```

## Streaming

```python
from voxagent import Agent
from voxagent.providers import TextDeltaChunk

agent = Agent(model="anthropic:claude-3-5-sonnet")

async for chunk in agent.stream("Tell me a story"):
    if isinstance(chunk, TextDeltaChunk):
        print(chunk.delta, end="", flush=True)
```

## Tools

```python
from voxagent import Agent
from voxagent.tools import tool

@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = Agent(
    model="openai:gpt-4o",
    tools=[get_weather],
)

result = await agent.run("What's the weather in Paris?")
```

## Supported Providers

| Provider | Model Format | Example |
|----------|--------------|---------|
| OpenAI | `openai:model` | `openai:gpt-4o` |
| Anthropic | `anthropic:model` | `anthropic:claude-3-5-sonnet` |
| Google | `google:model` | `google:gemini-1.5-pro` |
| Groq | `groq:model` | `groq:llama-3.1-70b` |
| Ollama | `ollama:model` | `ollama:llama3.2` |

## API Reference

### Agent

```python
from voxagent import Agent

agent = Agent(
    model="provider:model",      # Required: provider:model string
    system_prompt="...",         # Optional: system instructions
    tools=[...],                 # Optional: list of tools
    temperature=0.7,             # Optional: sampling temperature
)

# Single response
result = await agent.run("prompt")

# Streaming
async for chunk in agent.stream("prompt"):
    ...
```

### StreamChunk Types

```python
from voxagent.providers import (
    TextDeltaChunk,    # Text content
    ToolUseChunk,      # Tool invocation
    MessageEndChunk,   # End of message
    ErrorChunk,        # Error occurred
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

