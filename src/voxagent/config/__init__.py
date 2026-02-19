"""Configuration loading for voxagent.

This subpackage provides:
- SamaritanConfig: Pydantic model for configuration schema
- Config loading from JSON files
- System prompt assembly from rules and personas
"""

from voxagent.config.loader import (
    SamaritanConfig,
    get_default_config_path,
    load_config,
    load_system_prompt,
)

__all__ = [
    "SamaritanConfig",
    "get_default_config_path",
    "load_config",
    "load_system_prompt",
]

