# voxagent Examples

This directory contains working examples demonstrating voxagent features.

## Prerequisites

```bash
# Install voxagent with all providers
pip install voxagent[all]

# Or install specific providers
pip install voxagent[openai]      # For OpenAI examples
pip install voxagent[anthropic]   # For Anthropic examples
pip install voxagent[ollama]      # For Ollama examples
```

## Examples

| File | Description | Provider |
|------|-------------|----------|
| `01_openai_streaming.py` | Basic streaming with OpenAI | OpenAI |
| `02_anthropic_tools.py` | Tool calling with Anthropic | Anthropic |
| `03_ollama_local.py` | Local models with Ollama | Ollama |
| `04_mcp_integration.py` | MCP server integration | Any |
| `05_multi_provider.py` | Provider switching & failover | Multiple |

## Running Examples

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run an example
python 01_openai_streaming.py
```

## Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI | Yes |
| `ANTHROPIC_API_KEY` | Anthropic | Yes |
| `GOOGLE_API_KEY` | Google | Yes |
| `GROQ_API_KEY` | Groq | Yes |
| (none) | Ollama | No (local) |

