"""Virtual filesystem for tool discovery.

Provides ls() and read() functions that allow the LLM to browse
available tools without loading all definitions upfront.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCategory:
    """A category of related tools.
    
    Attributes:
        name: Category name (e.g., "devices", "home-assistant")
        description: Category description
        tools: Dict of tool filename -> tool content/definition
    """
    name: str
    description: str = ""
    tools: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Registry of tool categories and definitions.
    
    Manages tool registration and provides lookup for the virtual filesystem.
    """
    
    def __init__(self) -> None:
        self._categories: dict[str, ToolCategory] = {}
    
    def register_category(
        self,
        name: str,
        description: str = "",
        tools: dict[str, str] | None = None,
    ) -> ToolCategory:
        """Register a new tool category.
        
        Args:
            name: Category name
            description: Category description
            tools: Dict of tool filename -> tool content
            
        Returns:
            The created ToolCategory
        """
        category = ToolCategory(
            name=name,
            description=description,
            tools=tools or {},
        )
        self._categories[name] = category
        return category
    
    def get_category(self, name: str) -> ToolCategory | None:
        """Get a category by name."""
        return self._categories.get(name)
    
    def get_tool_definition(self, category: str, tool_name: str) -> str | None:
        """Get a specific tool's definition content.
        
        Args:
            category: Category name
            tool_name: Tool filename (e.g., "registry.py")
            
        Returns:
            Tool content/definition or None if not found
        """
        cat = self._categories.get(category)
        if cat:
            return cat.tools.get(tool_name)
        return None
    
    def list_categories(self) -> list[str]:
        """List all category names."""
        return list(self._categories.keys())
    
    def list_tools(self, category: str) -> list[str]:
        """List tool filenames in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of tool filenames (e.g., ["registry.py", "control.py"])
        """
        cat = self._categories.get(category)
        if cat:
            return list(cat.tools.keys())
        return []


class VirtualFilesystem:
    """Virtual filesystem for tool discovery.
    
    Provides ls() and read() functions that can be injected into the sandbox.
    
    Directory structure:
        tools/
        ├── __index__.md
        ├── devices/
        │   ├── __index__.md
        │   ├── registry.py
        │   └── control.py
        └── sensors/
            ├── __index__.md
            └── temperature.py
    """
    
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
    
    def ls(self, path: str) -> list[str]:
        """List directory contents.
        
        Args:
            path: Path like "tools/" or "tools/devices/"
            
        Returns:
            List of entries (directories end with /, files don't)
        """
        path = path.rstrip("/")
        
        if path == "tools" or path == "":
            # Root: list categories
            entries = ["__index__.md"]
            for cat in self._registry.list_categories():
                entries.append(f"{cat}/")
            return sorted(entries)
        
        if path.startswith("tools/"):
            category = path[6:]  # Remove "tools/"
            if category in self._registry.list_categories():
                # Category: list tools
                entries = ["__index__.md"]
                for tool in self._registry.list_tools(category):
                    entries.append(tool)
                return sorted(entries)
        
        return []
    
    def read(self, path: str) -> str:
        """Read file contents.
        
        Args:
            path: Path like "tools/__index__.md" or "tools/devices/registry.py"
            
        Returns:
            File contents or error message
        """
        path = path.lstrip("/")
        
        # Root index
        if path == "tools/__index__.md":
            return self._generate_root_index()
        
        # Category index
        if path.startswith("tools/") and path.endswith("/__index__.md"):
            category = path[6:-13]  # Remove "tools/" and "/__index__.md"
            cat = self._registry.get_category(category)
            if cat:
                return self._generate_category_index(cat)
            return f"Error: Category '{category}' not found"
        
        # Tool file
        if path.startswith("tools/") and path.endswith(".py"):
            parts = path[6:].split("/")  # Remove "tools/"
            if len(parts) == 2:
                category, tool_name = parts
                content = self._registry.get_tool_definition(category, tool_name)
                if content is not None:
                    return content
            return f"Error: Tool not found at '{path}'"
        
        return f"Error: File not found at '{path}'"
    
    def _generate_root_index(self) -> str:
        """Generate tools/__index__.md content."""
        lines = [
            "# Available Tool Categories",
            "",
            "Browse these directories to find tools:",
            "",
            "| Category | Description | Tools |",
            "|----------|-------------|-------|",
        ]
        for cat_name in sorted(self._registry.list_categories()):
            cat = self._registry.get_category(cat_name)
            if cat:
                tool_count = len(cat.tools)
                lines.append(
                    f"| `{cat_name}/` | {cat.description} | {tool_count} tools |"
                )
        lines.extend([
            "",
            'Use `ls("tools/<category>/")` to see tools in a category.',
            'Use `read("tools/<category>/<tool>.py")` to see tool details.',
        ])
        return "\n".join(lines)
    
    def _generate_category_index(self, category: ToolCategory) -> str:
        """Generate __index__.md content for a category."""
        lines = [f"# {category.name}", "", category.description, "", "## Tools", ""]
        for tool_name in sorted(category.tools.keys()):
            lines.append(f"- `{tool_name}`")
        return "\n".join(lines)
    
    def get_sandbox_globals(self) -> dict[str, Any]:
        """Get globals dict to inject into sandbox.
        
        Returns:
            Dict with ls and read functions bound to this filesystem
        """
        return {
            "ls": self.ls,
            "read": self.read,
        }

