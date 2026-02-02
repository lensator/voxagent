"""Tool decorator for auto-generating ToolDefinition from functions.

This module provides the @tool decorator that creates a ToolDefinition
from a function's signature, type hints, and docstring.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, TypeVar, Union, get_args, get_origin, get_type_hints

from voxagent.tools.definition import ToolDefinition

R = TypeVar("R")


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., R]], ToolDefinition]:
    """Decorator to create a ToolDefinition from a function.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring first line)

    Returns:
        A decorator that converts a function to a ToolDefinition

    Example:
        @tool()
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny"

        # get_weather is now a ToolDefinition
        assert get_weather.name == "get_weather"
        assert get_weather.description == "Get weather for a city."
    """

    def decorator(fn: Callable[..., R]) -> ToolDefinition:
        # Determine name
        tool_name = name if name is not None else fn.__name__

        # Validate name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            raise ValueError(f"Invalid tool name: {tool_name}")

        # Determine description from docstring
        tool_description = description
        if tool_description is None:
            if fn.__doc__:
                # Use first line of docstring
                tool_description = fn.__doc__.strip().split("\n")[0].strip()
            else:
                tool_description = ""

        # Check if async
        is_async = inspect.iscoroutinefunction(fn)

        # Build parameters schema from type hints
        parameters = _build_parameters_schema(fn)

        return ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            execute=fn,
            is_async=is_async,
        )

    return decorator


def _build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build JSON Schema from function type hints."""
    sig = inspect.signature(fn)
    hints: dict[str, Any] = {}
    try:
        hints = get_type_hints(fn)
    except Exception:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'context' parameter (ToolContext)
        if param_name == "context":
            continue

        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Get type hint
        type_hint = hints.get(param_name, Any)

        # Convert type to JSON Schema
        prop_schema = _type_to_json_schema(type_hint)
        properties[param_name] = prop_schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            # Add default to schema
            if param.default is not None:
                properties[param_name]["default"] = param.default

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required
    else:
        schema["required"] = []

    return schema


def _type_to_json_schema(type_hint: Any) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    # Handle None/NoneType
    if type_hint is type(None):
        return {"type": "null"}

    # Handle basic types
    if type_hint is str:
        return {"type": "string"}
    if type_hint is int:
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}

    # Handle Optional (Union with None)
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Union:
        # Check if it's Optional (Union[X, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1 and type(None) in args:
            # It's Optional[X]
            inner_schema = _type_to_json_schema(non_none_args[0])
            inner_schema["nullable"] = True
            return inner_schema
        # General Union - use anyOf
        return {"anyOf": [_type_to_json_schema(a) for a in args]}

    # Handle list
    if origin is list:
        if args:
            return {"type": "array", "items": _type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle dict
    if origin is dict:
        return {"type": "object"}

    # Handle Any
    if type_hint is Any:
        return {}

    # Default to object for unknown types
    return {"type": "object"}

