"""Run parameters and result types for voxagent.

This module defines the types used for agent run configuration and results:
- ModelConfig for model provider and settings
- AgentConfig for agent behavior settings
- ToolPolicy for tool access control
- RunParams for run configuration
- ToolMeta for tool execution metadata
- RunResult for run outcome
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .messages import Message


class ModelConfig(BaseModel):
    """Configuration for the model to use.

    Attributes:
        provider: The model provider (e.g., "anthropic", "openai", "ollama").
        model: The model name (e.g., "claude-3-5-sonnet", "gpt-4o").
        temperature: Optional sampling temperature.
        max_tokens: Optional maximum tokens for generation.
    """

    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_string(cls, model_string: str) -> "ModelConfig":
        """Parse a model string in 'provider:model' format.

        Args:
            model_string: String in format "provider:model"

        Returns:
            ModelConfig instance

        Raises:
            ValueError: If the string doesn't contain a colon separator
        """
        if ":" not in model_string:
            raise ValueError(
                f"Invalid model string format: '{model_string}'. Expected 'provider:model'"
            )
        provider, model = model_string.split(":", 1)
        return cls(provider=provider, model=model)

    def to_string(self) -> str:
        """Convert to 'provider:model' string format.

        Returns:
            String in format "provider:model"
        """
        return f"{self.provider}:{self.model}"


class AgentConfig(BaseModel):
    """Configuration for agent behavior.

    Attributes:
        max_turns: Maximum number of conversation turns. Default 50.
        context_limit: Maximum context size in tokens. Default 100000.
    """

    max_turns: int = 50
    context_limit: int = 100000


class ToolPolicy(BaseModel):
    """Policy for controlling tool access.

    Attributes:
        allow_list: List of allowed tool names. None means allow all.
        deny_list: List of denied tool names. Defaults to empty.
    """

    allow_list: list[str] | None = None
    deny_list: list[str] = Field(default_factory=list)


class RunParams(BaseModel):
    """Parameters for an agent run.

    Attributes:
        session_key: Unique session identifier (required, non-empty).
        prompt: The user prompt to process (required).
        model: Model configuration (required).
        config: Agent configuration (required).
        timeout_ms: Timeout in milliseconds (required, must be > 0).
        workspace_dir: Working directory for the agent (required).
        abort_signal: Optional signal for aborting the run.
        channel: Optional channel identifier.
        policies: List of tool policies to apply. Defaults to empty.
    """

    session_key: str
    prompt: str
    model: ModelConfig
    config: AgentConfig
    timeout_ms: int
    workspace_dir: Path
    abort_signal: Any | None = None
    channel: str | None = None
    policies: list[ToolPolicy] = Field(default_factory=list)

    @field_validator("session_key")
    @classmethod
    def session_key_must_not_be_empty(cls, v: str) -> str:
        """Validate that session_key is not empty."""
        if not v:
            raise ValueError("session_key must not be empty")
        return v

    @field_validator("timeout_ms")
    @classmethod
    def timeout_ms_must_be_positive(cls, v: int) -> int:
        """Validate that timeout_ms is positive."""
        if v <= 0:
            raise ValueError("timeout_ms must be positive")
        return v


class ToolMeta(BaseModel):
    """Metadata about a tool execution.

    Attributes:
        tool_name: Name of the tool that was called.
        tool_call_id: Unique identifier for the tool call.
        execution_time_ms: Time taken to execute in milliseconds.
        success: Whether the execution was successful.
        error: Error message if execution failed.
    """

    tool_name: str
    tool_call_id: str
    execution_time_ms: int = 0
    success: bool = True
    error: str | None = None


class RunResult(BaseModel):
    """Result of an agent run.

    Attributes:
        messages: List of messages from the conversation.
        assistant_texts: List of text responses from the assistant.
        tool_metas: List of tool execution metadata.
        aborted: Whether the run was aborted.
        timed_out: Whether the run timed out.
        error: Error message if the run failed.
    """

    messages: list[Message]
    assistant_texts: list[str]
    tool_metas: list[ToolMeta] = Field(default_factory=list)
    aborted: bool = False
    timed_out: bool = False
    error: str | None = None


__all__ = [
    "AgentConfig",
    "ModelConfig",
    "RunParams",
    "RunResult",
    "ToolMeta",
    "ToolPolicy",
]

