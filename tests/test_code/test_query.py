"""Tests for QueryResult class and query utilities."""

import pytest
from voxagent.code.query import QueryResult


class TestQueryResultBasic:
    """Test basic QueryResult functionality."""

    def test_init_with_list(self):
        """Test initialization with a list."""
        data = [{"a": 1}, {"b": 2}]
        result = QueryResult(data)
        assert len(result) == 2
        assert result.data == data

    def test_init_with_dict(self):
        """Test initialization with a single dict."""
        data = {"a": 1}
        result = QueryResult(data)
        assert len(result) == 1
        assert result.data == [data]

    def test_init_with_none(self):
        """Test initialization with None."""
        result = QueryResult(None)
        assert len(result) == 0
        assert result.data == []

    def test_init_with_empty_list(self):
        """Test initialization with empty list."""
        result = QueryResult([])
        assert len(result) == 0

    def test_iteration(self):
        """Test iterating over results."""
        data = [{"a": 1}, {"b": 2}]
        result = QueryResult(data)
        items = list(result)
        assert items == data

    def test_repr(self):
        """Test string representation."""
        result = QueryResult([{"a": 1}, {"b": 2}])
        assert repr(result) == "QueryResult(2 items)"


class TestQueryResultFilter:
    """Test QueryResult.filter() method."""

    def test_filter_single_field(self):
        """Test filtering by a single field."""
        data = [
            {"name": "Balcony Light", "room": "balcony"},
            {"name": "Kitchen Light", "room": "kitchen"},
        ]
        result = QueryResult(data).filter(name="balcony")
        assert len(result) == 1
        assert result.first()["name"] == "Balcony Light"

    def test_filter_multiple_fields(self):
        """Test filtering by multiple fields."""
        data = [
            {"name": "Light A", "room": "living", "type": "light"},
            {"name": "Light B", "room": "living", "type": "dimmer"},
            {"name": "Light C", "room": "bedroom", "type": "light"},
        ]
        result = QueryResult(data).filter(room="living", type="light")
        assert len(result) == 1
        assert result.first()["name"] == "Light A"

    def test_filter_regex_pattern(self):
        """Test filtering with regex patterns."""
        data = [
            {"name": "Living Room Light 1"},
            {"name": "Living Room Light 2"},
            {"name": "Bedroom Light"},
        ]
        result = QueryResult(data).filter(name="living.*light")
        assert len(result) == 2

    def test_filter_case_insensitive(self):
        """Test that filter is case insensitive."""
        data = [{"name": "BALCONY"}, {"name": "kitchen"}]
        result = QueryResult(data).filter(name="balcony")
        assert len(result) == 1

    def test_filter_no_patterns(self):
        """Test filter with no patterns returns same data."""
        data = [{"a": 1}, {"b": 2}]
        result = QueryResult(data).filter()
        assert len(result) == 2


class TestQueryResultMap:
    """Test QueryResult.map() method."""

    def test_map_single_field(self):
        """Test mapping to a single field."""
        data = [{"device_id": "1", "name": "Light"}, {"device_id": "2", "name": "Fan"}]
        result = QueryResult(data).map("device_id")
        assert len(result) == 2
        assert result.first() == {"device_id": "1"}

    def test_map_multiple_fields(self):
        """Test mapping to multiple fields."""
        data = [
            {"device_id": "1", "name": "Light A", "room": "living"},
            {"device_id": "2", "name": "Light B", "room": "bedroom"},
        ]
        result = QueryResult(data).map(["device_id", "name"])
        assert len(result) == 2
        assert result.first() == {"device_id": "1", "name": "Light A"}


class TestQueryResultChaining:
    """Test QueryResult method chaining."""

    def test_filter_then_map(self):
        """Test filter followed by map."""
        data = [
            {"device_id": "1", "name": "Light A", "room": "living"},
            {"device_id": "2", "name": "Light B", "room": "living"},
            {"device_id": "3", "name": "Light C", "room": "bedroom"},
        ]
        result = QueryResult(data).filter(room="living").map(["device_id", "name"])
        assert len(result) == 2
        assert result.first() == {"device_id": "1", "name": "Light A"}

    def test_filter_map_first(self):
        """Test full chain with first."""
        data = [
            {"device_id": "1", "name": "Balcony Light", "room": "balcony"},
            {"device_id": "2", "name": "Kitchen Light", "room": "kitchen"},
        ]
        device = QueryResult(data).filter(name="balcony").first()
        assert device["device_id"] == "1"


class TestQueryResultAggregation:
    """Test QueryResult aggregation methods."""

    def test_reduce_count_only(self):
        """Test reduce with count only."""
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = QueryResult(data).reduce(count=True)
        assert result == {"count": 3}

    def test_reduce_group_by(self):
        """Test reduce with grouping."""
        data = [
            {"name": "A", "room": "living"},
            {"name": "B", "room": "living"},
            {"name": "C", "room": "bedroom"},
        ]
        result = QueryResult(data).reduce(by="room", count=True)
        assert result == {"living": 2, "bedroom": 1}

    def test_first_and_last(self):
        """Test first and last methods."""
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = QueryResult(data)
        assert result.first() == {"a": 1}
        assert result.last() == {"c": 3}

    def test_first_empty(self):
        """Test first on empty result."""
        result = QueryResult([])
        assert result.first() is None

    def test_sort(self):
        """Test sorting results."""
        data = [{"name": "C"}, {"name": "A"}, {"name": "B"}]
        result = QueryResult(data).sort(by="name")
        assert [r["name"] for r in result] == ["A", "B", "C"]

    def test_unique(self):
        """Test removing duplicates."""
        data = [{"room": "living"}, {"room": "living"}, {"room": "bedroom"}]
        result = QueryResult(data).unique(by="room")
        assert len(result) == 2


class TestQueryResultEach:
    """Test QueryResult.each() method."""

    def test_each_applies_function(self):
        """Test that each applies function to all items."""
        data = [{"value": 1}, {"value": 2}, {"value": 3}]
        results = QueryResult(data).each(lambda d: d["value"] * 2)
        assert results == [2, 4, 6]

