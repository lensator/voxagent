"""LanceDB-based memory manager for Home Orchestrator.

Handles vector search for facts, rules, and history, as well as 
structured storage for goals and home state.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import lancedb
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single entry in the memory database."""
    id: str
    content: str
    source: str  # File path or "session"
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GoalEntry(BaseModel):
    """A goal tracked by the agent."""
    id: str
    description: str
    status: str  # "active", "completed", "blocked", "cancelled"
    priority: int = 1
    steps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LanceDBMemoryManager:
    """Manages home memory using LanceDB."""

    def __init__(
        self, 
        uri: str = "memory_db", 
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.uri = uri
        self.embedding_model_name = embedding_model_name
        self._db = None
        self._model = None
        
        if not HAS_LANCEDB:
            # We'll allow initialization but warn or fail on use
            pass

    def _ensure_connected(self):
        """Lazy initialization of LanceDB and embedding model."""
        if not HAS_LANCEDB:
            raise ImportError(
                "LanceDB and sentence-transformers are required for Home Orchestrator. "
                "Install with pip install voxagent[home]"
            )
        
        if self._db is None:
            self._db = lancedb.connect(self.uri)
            
        if self._model is None:
            # This model is CPU-friendly and supports 50+ languages including Greek
            self._model = SentenceTransformer(self.embedding_model_name)

    def _get_table(self, name: str):
        self._ensure_connected()
        if name in self._db.table_names():
            return self._db.open_table(name)
        return None

    async def add_fact(self, content: str, source: str, metadata: Optional[dict] = None):
        """Add a fact to the memory index."""
        if not HAS_LANCEDB:
            return
        self._ensure_connected()
        table_name = "facts"
        
        embedding = self._model.encode(content).tolist()
        data = [{
            "vector": embedding,
            "content": content,
            "source": source,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
        
        if table_name not in self._db.table_names():
            self._db.create_table(table_name, data=data)
        else:
            table = self._db.open_table(table_name)
            table.add(data)

    async def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        """Search for relevant facts using vector similarity."""
        if not HAS_LANCEDB:
            return []
        self._ensure_connected()
        table_name = "facts"
        
        if table_name not in self._db.table_names():
            return []
            
        table = self._db.open_table(table_name)
        query_vector = self._model.encode(query).tolist()
        
        results = table.search(query_vector).limit(limit).to_list()
        return results

    async def sync_directory(self, dir_path: str | Path):
        """Sync a directory of Markdown files to LanceDB."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return

        for file in dir_path.glob("**/*.md"):
            content = file.read_text(encoding="utf-8")
            # Simple chunking by paragraph for now
            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
            for chunk in chunks:
                await self.add_fact(chunk, str(file))

    async def update_goal(self, goal: GoalEntry):
        """Update or create a goal in the structured storage."""
        self._ensure_connected()
        table_name = "goals"
        
        data = [{
            "id": goal.id,
            "description": goal.description,
            "status": goal.status,
            "priority": goal.priority,
            "steps": str(goal.steps), # Simple serialization for now
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        
        if table_name not in self._db.table_names():
            self._db.create_table(table_name, data=data)
        else:
            table = self._db.open_table(table_name)
            # In LanceDB, we usually overwrite or append. 
            # For a small goals table, we can just replace the whole thing or filter.
            # Simplified for now: just append
            table.add(data)

    async def list_active_goals(self) -> list[dict]:
        """List all active goals."""
        self._ensure_connected()
        table_name = "goals"
        
        if table_name not in self._db.table_names():
            return []
            
        table = self._db.open_table(table_name)
        # Search with a filter
        results = table.search().where("status = 'active'").to_list()
        return results
