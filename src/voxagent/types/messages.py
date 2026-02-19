"""Message types for voxagent.

This module defines the core message types used in agent conversations:
- ContentBlock types (TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock)
- ToolCall and ToolResult for tool interactions
- Message for conversation messages
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A text content block.

    Attributes:
        type: Discriminator field, always "text".
        text: The text content.
    """

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image content block with base64-encoded data.

    Attributes:
        type: Discriminator field, always "image".
        media_type: The MIME type of the image (e.g., "image/png", "image/jpeg").
        data: Base64-encoded image data.
    """

    type: Literal["image"] = "image"
    media_type: str
    data: str


class ToolUseBlock(BaseModel):
    """A tool use block representing a tool call in assistant content.

    Attributes:
        type: Discriminator field, always "tool_use".
        id: Unique identifier for this tool use.
        name: Name of the tool to call.
        input: Dictionary of input parameters for the tool.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """A tool result block representing a tool's output in user content.

    Attributes:
        type: Discriminator field, always "tool_result".
        tool_use_id: The ID of the corresponding ToolUseBlock.
        tool_name: Optional name of the tool (for providers that need it).
        content: The result content from the tool.
        is_error: Whether the result represents an error.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    tool_name: str | None = None
    content: str
    is_error: bool = False


# ContentBlock is a discriminated union of all content block types
# Using Pydantic's discriminator for robust JSON parsing
ContentBlock = Annotated[
    Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]


class ToolCall(BaseModel):
    """A tool call request.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to call.
        params: Dictionary of parameters for the tool.
    """

    id: str
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from a tool execution.

    Attributes:
        tool_use_id: The ID of the corresponding ToolCall.
        content: The result content from the tool.
        is_error: Whether the result represents an error.
    """

    tool_use_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """A message in a conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", or "system").
        content: The message content, either a string or a list of ContentBlocks.
        tool_calls: Optional list of tool calls (only for assistant messages).
        metadata: Optional metadata (strategy name, timestamp, etc.).
    """

    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] | None = None


__all__ = [
    "ContentBlock",
    "ImageBlock",
    "Message",
    "TextBlock",
    "ToolCall",
    "ToolResult",
    "ToolResultBlock",
    "ToolUseBlock",
]

