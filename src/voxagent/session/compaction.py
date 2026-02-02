"""Context compaction for voxagent.

This module provides functionality to manage token limits by:
- Counting tokens in messages and text
- Detecting when compaction is needed
- Compacting context using various strategies
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from voxagent.types.messages import Message, ToolResultBlock


class CompactionStrategy(Enum):
    """Available compaction strategies."""

    REMOVE_TOOL_RESULTS = "remove_tool_results"
    TRUNCATE_OLDEST = "truncate_oldest"
    SUMMARIZE = "summarize"  # Requires LLM call (placeholder)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text.

    Uses tiktoken for OpenAI models, estimates for others.

    Args:
        text: The text to count tokens for.
        model: The model name for tokenization.

    Returns:
        Number of tokens in the text.
    """
    if not text:
        return 0

    try:
        import tiktoken

        # Map model names to tiktoken encodings
        if "gpt-4" in model or "gpt-3.5" in model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base for other models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except ImportError:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4


def count_message_tokens(messages: list[Message], model: str = "gpt-4") -> int:
    """Count total tokens in a list of messages.

    Args:
        messages: List of messages to count tokens for.
        model: The model name for tokenization.

    Returns:
        Total token count across all messages.
    """
    if not messages:
        return 0

    total = 0
    for msg in messages:
        # Role overhead (tokens for role, separators, message framing, metadata)
        # Include overhead for message structure in API calls
        total += 20

        # Content tokens
        if isinstance(msg.content, str):
            total += count_tokens(msg.content, model)
        elif isinstance(msg.content, list):
            # Content blocks
            for block in msg.content:
                if hasattr(block, "text"):
                    total += count_tokens(block.text, model)
                elif hasattr(block, "content"):
                    total += count_tokens(block.content, model)
                elif isinstance(block, dict):
                    if "text" in block:
                        total += count_tokens(block["text"], model)
                    elif "content" in block:
                        total += count_tokens(block["content"], model)

        # Tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += count_tokens(tc.name, model)
                if isinstance(tc.params, str):
                    total += count_tokens(tc.params, model)
                elif isinstance(tc.params, dict):
                    import json

                    total += count_tokens(json.dumps(tc.params), model)

    return total


def needs_compaction(
    messages: list[Message],
    system_prompt: str,
    max_tokens: int,
    reserve_tokens: int = 0,
    model: str = "gpt-4",
) -> bool:
    """Check if context needs compaction.

    Args:
        messages: List of messages in the conversation.
        system_prompt: The system prompt text.
        max_tokens: Maximum allowed tokens.
        reserve_tokens: Tokens to reserve for response.
        model: The model name for tokenization.

    Returns:
        True if current tokens > (max_tokens - reserve_tokens).
    """
    available = max_tokens - reserve_tokens

    # Count system prompt tokens
    current = count_tokens(system_prompt, model)

    # Count message tokens
    current += count_message_tokens(messages, model)

    return current > available


def _has_tool_result_content(msg: Message) -> bool:
    """Check if a message contains tool result content blocks."""
    if not isinstance(msg.content, list):
        return False
    return any(isinstance(block, ToolResultBlock) for block in msg.content)


def compact_context(
    messages: list[Message],
    target_tokens: int,
    preserve_recent: int = 4,
    model: str = "gpt-4",
    aggressive: bool = False,
) -> list[Message]:
    """Compact messages to fit within target token count.

    Strategies (in order):
    1. Remove messages with tool result content blocks
    2. Truncate from oldest messages
    3. If aggressive: more aggressive truncation

    Always preserves the most recent `preserve_recent` turns.

    Args:
        messages: List of messages to compact.
        target_tokens: Target maximum token count.
        preserve_recent: Number of recent messages to preserve.
        model: The model name for tokenization.
        aggressive: Whether to use aggressive compaction.

    Returns:
        Compacted list of messages.
    """
    if not messages:
        return []

    # Handle negative or zero target with preserve_recent=0
    if target_tokens <= 0 and preserve_recent == 0:
        return []

    # Check if already under target
    current_tokens = count_message_tokens(messages, model)
    if current_tokens <= target_tokens:
        return list(messages)

    result = list(messages)

    # Strategy 1: Remove messages with tool result content blocks from outside preserve window
    if len(result) > preserve_recent:
        if preserve_recent > 0:
            compactable = result[:-preserve_recent]
            preserved = result[-preserve_recent:]
        else:
            compactable = result
            preserved = []

        # Remove messages with tool result content blocks from compactable portion
        compactable = [m for m in compactable if not _has_tool_result_content(m)]
        result = compactable + preserved

        current_tokens = count_message_tokens(result, model)
        if current_tokens <= target_tokens:
            return result

    # If still over target, also remove tool results from preserved portion
    if count_message_tokens(result, model) > target_tokens:
        result = [m for m in result if not _has_tool_result_content(m)]

        current_tokens = count_message_tokens(result, model)
        if current_tokens <= target_tokens:
            return result

    # Strategy 2: Truncate from oldest messages (respecting preserve_recent)
    while len(result) > preserve_recent and count_message_tokens(result, model) > target_tokens:
        # Skip system messages at the beginning
        if result[0].role == "system":
            if len(result) > 1:
                result = [result[0]] + result[2:]
            else:
                break
        else:
            result = result[1:]  # Remove oldest message

    # Strategy 3: If aggressive and still over, truncate more aggressively
    if aggressive:
        while len(result) > 1 and count_message_tokens(result, model) > target_tokens:
            # In aggressive mode, we can remove from preserved messages too
            # But keep at least one message
            if result[0].role == "system":
                if len(result) > 1:
                    result = [result[0]] + result[2:]
                else:
                    break
            else:
                result = result[1:]

        # If still over and only one message left, keep it (graceful handling)

    return result

