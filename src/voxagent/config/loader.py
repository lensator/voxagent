"""Configuration loader for voxagent.

Provides Pydantic model for config schema and functions to load
configuration from JSON files and assemble system prompts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SamaritanConfig(BaseModel):
    """Configuration schema for Samaritan agent.

    Attributes:
        version: Config schema version.
        model: Main model string (e.g., 'ollama:glm-5:cloud').
        fast_model: Optional fast model for intent classification.
        strategy: Strategy class name to use.
        excluded_devices: List of device IDs to exclude from control.
        debug: Enable debug mode.
        rules_dir: Directory containing rule markdown files.
        personas_dir: Directory containing persona markdown files.
        persona: Active persona filename.
    """

    version: str = "1.0"
    model: str = Field(..., description="Main model string e.g. 'ollama:glm-5:cloud'")
    fast_model: Optional[str] = Field(None, description="Fast intent model")
    strategy: str = Field("HomeOrchestratorStrategy", description="Strategy class name")
    excluded_devices: list[str] = Field(default_factory=list)
    debug: bool = False
    rules_dir: str = "rules"
    personas_dir: str = "personas"
    persona: str = Field("default.md", description="Active persona file")


def get_default_config_path() -> Path:
    """Returns the default configuration path ~/.samaritan/config.json."""
    return Path.home() / ".samaritan" / "config.json"


def load_config(config_path: Path | str | None = None) -> SamaritanConfig:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        SamaritanConfig instance with loaded values.

    Raises:
        FileNotFoundError: If config file does not exist.
        json.JSONDecodeError: If config file contains invalid JSON.
        pydantic.ValidationError: If config values don't match schema.
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(config_path).expanduser()

    config_path = config_path.expanduser().resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SamaritanConfig(**data)


def load_system_prompt(config: SamaritanConfig, config_dir: Path) -> str:
    """Build system prompt from rules and persona files.

    Assembles the system prompt by:
    1. Loading all .md files from the rules directory
    2. Loading the active persona file

    Args:
        config: The SamaritanConfig instance.
        config_dir: Directory containing config.json (for resolving relative paths).

    Returns:
        Assembled system prompt string.
    """
    parts: list[str] = []

    # Resolve directories relative to config_dir
    rules_path = (config_dir / config.rules_dir).resolve()
    personas_path = (config_dir / config.personas_dir).resolve()

    # Load all rule files
    if rules_path.is_dir():
        rule_files = sorted(rules_path.glob("*.md"))
        for rule_file in rule_files:
            try:
                content = rule_file.read_text(encoding="utf-8")
                parts.append(f"## RULE: {rule_file.name}\n{content}")
            except OSError as e:
                warnings.warn(f"Failed to read rule file {rule_file}: {e}")
    else:
        warnings.warn(f"Rules directory not found: {rules_path}")

    # Load active persona
    persona_file = personas_path / config.persona
    if persona_file.is_file():
        try:
            content = persona_file.read_text(encoding="utf-8")
            parts.append(f"## PERSONA: {config.persona}\n{content}")
        except OSError as e:
            warnings.warn(f"Failed to read persona file {persona_file}: {e}")
    else:
        warnings.warn(f"Persona file not found: {persona_file}")

    return "\n\n".join(parts)


__all__ = [
    "SamaritanConfig",
    "get_default_config_path",
    "load_config",
    "load_system_prompt",
]

