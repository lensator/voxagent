# Contributing to voxagent

Thank you for your interest in contributing to voxagent!

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/lensator/voxagent.git
cd voxagent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode with all extras
pip install -e ".[dev,all]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=voxagent --cov-report=term-missing

# Run specific test file
pytest tests/test_agent.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/voxagent
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Ensure all tests pass**: `pytest`
5. **Format and lint**: `ruff format . && ruff check .`
6. **Commit** with a clear message
7. **Push** and create a Pull Request

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints for all public functions
- Write docstrings for public APIs (Google style)
- Keep functions focused and small
- Prefer explicit over implicit

## Adding a New Provider

1. Create `src/voxagent/providers/yourprovider.py`
2. Inherit from `BaseProvider`
3. Implement required methods: `complete()`, `stream()`
4. Register in `providers/__init__.py`
5. Add tests in `tests/test_providers/`
6. Update documentation

Example:

```python
from voxagent.providers.base import BaseProvider, StreamChunk

class YourProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "yourprovider"

    @property
    def models(self) -> list[str]:
        return ["model-1", "model-2"]

    async def complete(self, messages, **kwargs) -> str:
        # Implementation
        ...

    async def stream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:
        # Implementation
        ...
```

## Reporting Issues

- Use GitHub Issues
- Include Python version, OS, and voxagent version
- Provide minimal reproduction steps
- Include full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

