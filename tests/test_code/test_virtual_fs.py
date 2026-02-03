"""Tests for virtual filesystem and tool registry.

This module tests the virtual filesystem that provides ls() and read()
functions for tool discovery, and the ToolRegistry for managing tool definitions.
"""

from __future__ import annotations

import pytest

from voxagent.code.virtual_fs import VirtualFilesystem, ToolRegistry


class TestVirtualFilesystem:
    """Tests for ls() and read() functions."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create a populated tool registry for testing."""
        reg = ToolRegistry()
        reg.register_category(
            "devices",
            tools={
                "registry.py": "def get_devices(): ...",
                "control.py": "def set_device_state(id, state): ...",
            },
        )
        reg.register_category(
            "sensors",
            tools={
                "temperature.py": "def read_temperature(sensor_id): ...",
                "humidity.py": "def read_humidity(sensor_id): ...",
            },
        )
        return reg

    @pytest.fixture
    def vfs(self, registry: ToolRegistry) -> VirtualFilesystem:
        """Create a virtual filesystem with the test registry."""
        return VirtualFilesystem(registry=registry)

    def test_ls_root_returns_categories(self, vfs: VirtualFilesystem) -> None:
        """ls('tools/') returns tool categories."""
        result = vfs.ls("tools/")
        assert "devices/" in result
        assert "sensors/" in result

    def test_ls_category_returns_tools(self, vfs: VirtualFilesystem) -> None:
        """ls('tools/devices/') returns tool files."""
        result = vfs.ls("tools/devices/")
        assert "registry.py" in result
        assert "control.py" in result

    def test_ls_nonexistent_returns_empty(self, vfs: VirtualFilesystem) -> None:
        """ls('tools/nonexistent/') returns empty list."""
        result = vfs.ls("tools/nonexistent/")
        assert result == []

    def test_read_tool_file(self, vfs: VirtualFilesystem) -> None:
        """read('tools/devices/registry.py') returns tool definition."""
        content = vfs.read("tools/devices/registry.py")
        assert "def get_devices" in content

    def test_read_index_file(self, vfs: VirtualFilesystem) -> None:
        """read('tools/__index__.md') returns index content."""
        content = vfs.read("tools/__index__.md")
        # Index should list available categories
        assert "devices" in content.lower() or "sensors" in content.lower()

    def test_read_nonexistent_returns_error(self, vfs: VirtualFilesystem) -> None:
        """read('tools/nonexistent.py') returns error message."""
        content = vfs.read("tools/nonexistent.py")
        assert "not found" in content.lower() or "error" in content.lower()


class TestToolRegistry:
    """Tests for tool definition loading and management."""

    def test_register_tool_category(self) -> None:
        """Register a tool category with tools."""
        registry = ToolRegistry()
        registry.register_category(
            "devices",
            tools={
                "registry.py": "def get_devices(): ...",
            },
        )
        assert "devices" in registry.list_categories()

    def test_get_tool_definition(self) -> None:
        """Get a specific tool's definition."""
        registry = ToolRegistry()
        registry.register_category(
            "devices",
            tools={
                "registry.py": "def get_devices(): pass",
            },
        )
        definition = registry.get_tool_definition("devices", "registry.py")
        assert "def get_devices" in definition

    def test_get_nonexistent_tool_returns_none(self) -> None:
        """Get nonexistent tool returns None."""
        registry = ToolRegistry()
        definition = registry.get_tool_definition("devices", "nonexistent.py")
        assert definition is None

    def test_list_categories(self) -> None:
        """List all registered categories."""
        registry = ToolRegistry()
        registry.register_category("devices", tools={})
        registry.register_category("sensors", tools={})
        registry.register_category("network", tools={})
        
        categories = registry.list_categories()
        assert set(categories) == {"devices", "sensors", "network"}

    def test_list_tools_in_category(self) -> None:
        """List tools in a category."""
        registry = ToolRegistry()
        registry.register_category(
            "devices",
            tools={
                "registry.py": "...",
                "control.py": "...",
                "status.py": "...",
            },
        )
        
        tools = registry.list_tools("devices")
        assert set(tools) == {"registry.py", "control.py", "status.py"}

    def test_list_tools_nonexistent_category(self) -> None:
        """List tools in nonexistent category returns empty list."""
        registry = ToolRegistry()
        tools = registry.list_tools("nonexistent")
        assert tools == []

    def test_empty_registry(self) -> None:
        """Empty registry has no categories."""
        registry = ToolRegistry()
        assert registry.list_categories() == []

