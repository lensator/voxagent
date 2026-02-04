"""Query utilities for Code Mode sandbox.

Provides functional-style data pipelines for efficient tool result processing.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator


class QueryResult:
    """Chainable wrapper for query results.
    
    Enables functional-style data pipelines:
        filter("tools/devices/list_devices", name="balcony").map(["device_id", "name"]).first()
    """
    
    def __init__(self, data: list[dict[str, Any]]) -> None:
        """Initialize with list of dicts."""
        self._data = data if isinstance(data, list) else [data] if data else []
    
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Allow iteration over results."""
        return iter(self._data)
    
    def __len__(self) -> int:
        """Return number of results."""
        return len(self._data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QueryResult({len(self._data)} items)"
    
    @property
    def data(self) -> list[dict[str, Any]]:
        """Get raw data."""
        return self._data
    
    def filter(self, **patterns: str) -> "QueryResult":
        """Filter results by regex patterns on field values.
        
        Args:
            **patterns: Field name -> regex pattern pairs
            
        Returns:
            New QueryResult with matching items
            
        Example:
            result.filter(name="balcony", room="living.*")
        """
        if not patterns:
            return self
        
        filtered = []
        for item in self._data:
            match = True
            for field, pattern in patterns.items():
                value = str(item.get(field, ""))
                if not re.search(pattern, value, re.IGNORECASE):
                    match = False
                    break
            if match:
                filtered.append(item)
        
        return QueryResult(filtered)
    
    def map(self, fields: list[str] | str) -> "QueryResult":
        """Extract specific fields from each result.
        
        Args:
            fields: Field name or list of field names to extract
            
        Returns:
            New QueryResult with only specified fields
            
        Example:
            result.map(["device_id", "name"])
            result.map("device_id")  # Returns list of values
        """
        if isinstance(fields, str):
            # Single field - return list of values wrapped in dicts
            return QueryResult([{fields: item.get(fields)} for item in self._data])
        
        mapped = []
        for item in self._data:
            mapped.append({f: item.get(f) for f in fields if f in item})
        return QueryResult(mapped)
    
    def reduce(self, by: str | None = None, count: bool = False) -> dict[str, Any]:
        """Aggregate results.
        
        Args:
            by: Field to group by (optional)
            count: If True, return counts per group
            
        Returns:
            Aggregated dict
            
        Example:
            result.reduce(by="room", count=True)  # {"living": 3, "bedroom": 2}
        """
        if by is None:
            if count:
                return {"count": len(self._data)}
            return {"items": self._data}
        
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in self._data:
            key = str(item.get(by, "unknown"))
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        if count:
            return {k: len(v) for k, v in groups.items()}
        return groups
    
    def first(self) -> dict[str, Any] | None:
        """Get first result or None.
        
        Returns:
            First item or None if empty
        """
        return self._data[0] if self._data else None
    
    def last(self) -> dict[str, Any] | None:
        """Get last result or None."""
        return self._data[-1] if self._data else None
    
    def each(self, fn: Callable[[dict[str, Any]], Any]) -> list[Any]:
        """Apply function to each result.
        
        Args:
            fn: Function to apply to each item
            
        Returns:
            List of function results
            
        Example:
            result.each(lambda d: call_tool("control", "turn_on", **d))
        """
        return [fn(item) for item in self._data]
    
    def sort(self, by: str, reverse: bool = False) -> "QueryResult":
        """Sort results by field.
        
        Args:
            by: Field name to sort by
            reverse: If True, sort descending
            
        Returns:
            New QueryResult with sorted items
        """
        sorted_data = sorted(
            self._data,
            key=lambda x: str(x.get(by, "")),
            reverse=reverse
        )
        return QueryResult(sorted_data)
    
    def unique(self, by: str) -> "QueryResult":
        """Remove duplicates based on field value.
        
        Args:
            by: Field to check for uniqueness
            
        Returns:
            New QueryResult with unique items
        """
        seen: set[str] = set()
        unique_items = []
        for item in self._data:
            key = str(item.get(by, ""))
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        return QueryResult(unique_items)

