"""Tool policy for voxagent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from voxagent.tools.definition import ToolDefinition


class ToolPolicy(BaseModel):
    """Policy for filtering available tools.

    Attributes:
        allow_list: List of allowed tool names, or None to allow all.
        deny_list: List of denied tool names.
    """

    allow_list: list[str] | None = None  # None = allow all
    deny_list: list[str] = Field(default_factory=list)

    def allows(self, tool_name: str) -> bool:
        """Check if this policy allows a tool.

        Deny list takes precedence over allow list.

        Args:
            tool_name: The name of the tool to check.

        Returns:
            True if the tool is allowed, False otherwise.
        """
        # Deny list takes precedence
        if tool_name in self.deny_list:
            return False
        # If allow_list is set, tool must be in it
        if self.allow_list is not None and tool_name not in self.allow_list:
            return False
        return True


def apply_tool_policies(
    tools: list[ToolDefinition],
    policies: list[ToolPolicy],
) -> list[ToolDefinition]:
    """Apply layered policies to filter tools.

    Policies are applied in order:
    - Allow lists are intersected (each policy can only restrict further)
    - Deny lists are unioned (each policy can add more denials)

    Args:
        tools: List of tools to filter.
        policies: List of policies to apply in order.

    Returns:
        Filtered list of tools.
    """
    if not policies:
        return tools

    # Compute effective allow and deny lists
    effective_allow: set[str] | None = None
    effective_deny: set[str] = set()

    for policy in policies:
        # Intersect allow lists
        if policy.allow_list is not None:
            policy_allow = set(policy.allow_list)
            if effective_allow is None:
                effective_allow = policy_allow
            else:
                effective_allow = effective_allow.intersection(policy_allow)

        # Union deny lists
        effective_deny = effective_deny.union(policy.deny_list)

    # Filter tools
    result = []
    for tool in tools:
        # Check allow list (if set)
        if effective_allow is not None and tool.name not in effective_allow:
            continue
        # Check deny list
        if tool.name in effective_deny:
            continue
        result.append(tool)

    return result

