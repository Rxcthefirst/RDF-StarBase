"""
Saved Queries for RDF-StarBase.

Provides functionality to save, organize, and share SPARQL queries:
- Persist queries with metadata (name, description, tags)
- Query history tracking
- Query templates with parameters
- Export/import query collections
- Query stats (execution count, avg time)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Type of SPARQL query."""
    SELECT = "select"
    CONSTRUCT = "construct"
    ASK = "ask"
    DESCRIBE = "describe"
    INSERT = "insert"
    DELETE = "delete"
    INSERT_DATA = "insert_data"
    DELETE_DATA = "delete_data"


@dataclass
class QueryParameter:
    """A parameter placeholder in a query template."""
    name: str
    description: str = ""
    default_value: str | None = None
    required: bool = True
    value_type: str = "string"  # string, uri, literal, integer
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "default_value": self.default_value,
            "required": self.required,
            "value_type": self.value_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> QueryParameter:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            default_value=data.get("default_value"),
            required=data.get("required", True),
            value_type=data.get("value_type", "string")
        )


@dataclass
class QueryExecution:
    """Record of a query execution."""
    query_id: str
    executed_at: datetime
    duration_ms: float
    result_count: int | None = None
    success: bool = True
    error_message: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "executed_at": self.executed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "result_count": self.result_count,
            "success": self.success,
            "error_message": self.error_message,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> QueryExecution:
        return cls(
            query_id=data["query_id"],
            executed_at=datetime.fromisoformat(data["executed_at"]),
            duration_ms=data["duration_ms"],
            result_count=data.get("result_count"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            parameters=data.get("parameters", {})
        )


@dataclass
class SavedQuery:
    """A saved SPARQL query with metadata."""
    query_id: str
    name: str
    sparql: str
    query_type: QueryType
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    repository: str | None = None  # Scoped to specific repo
    is_template: bool = False
    parameters: list[QueryParameter] = field(default_factory=list)
    is_favorite: bool = False
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    last_executed_at: datetime | None = None
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.execution_count
    
    @property
    def query_hash(self) -> str:
        """SHA-256 hash of the query text."""
        return hashlib.sha256(self.sparql.encode()).hexdigest()[:16]
    
    def fill_template(self, params: dict[str, str]) -> str:
        """
        Fill in template parameters.
        
        Parameters are in the form {{param_name}} in the query.
        """
        result = self.sparql
        for param in self.parameters:
            placeholder = f"{{{{{param.name}}}}}"
            value = params.get(param.name, param.default_value)
            if value is None and param.required:
                raise ValueError(f"Required parameter '{param.name}' not provided")
            if value is not None:
                result = result.replace(placeholder, value)
        return result
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "name": self.name,
            "sparql": self.sparql,
            "query_type": self.query_type.value,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "repository": self.repository,
            "is_template": self.is_template,
            "parameters": [p.to_dict() for p in self.parameters],
            "is_favorite": self.is_favorite,
            "execution_count": self.execution_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "last_executed_at": self.last_executed_at.isoformat() if self.last_executed_at else None,
            "version": self.version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SavedQuery:
        return cls(
            query_id=data["query_id"],
            name=data["name"],
            sparql=data["sparql"],
            query_type=QueryType(data["query_type"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            created_by=data.get("created_by", ""),
            repository=data.get("repository"),
            is_template=data.get("is_template", False),
            parameters=[QueryParameter.from_dict(p) for p in data.get("parameters", [])],
            is_favorite=data.get("is_favorite", False),
            execution_count=data.get("execution_count", 0),
            total_execution_time_ms=data.get("total_execution_time_ms", 0.0),
            last_executed_at=datetime.fromisoformat(data["last_executed_at"]) if data.get("last_executed_at") else None,
            version=data.get("version", 1),
            metadata=data.get("metadata", {})
        )


@dataclass
class QueryCollection:
    """A collection of related queries for export/import."""
    name: str
    description: str
    queries: list[SavedQuery]
    created_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "queries": [q.to_dict() for q in self.queries],
            "created_at": self.created_at.isoformat(),
            "author": self.author,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> QueryCollection:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            queries=[SavedQuery.from_dict(q) for q in data.get("queries", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            author=data.get("author", ""),
            version=data.get("version", "1.0")
        )


def detect_query_type(sparql: str) -> QueryType:
    """Detect the type of SPARQL query from its text."""
    normalized = sparql.strip().upper()
    
    # Skip prefixes to find the actual query keyword
    lines = normalized.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('PREFIX') or line.startswith('BASE'):
            continue
        
        if line.startswith('SELECT'):
            return QueryType.SELECT
        elif line.startswith('CONSTRUCT'):
            return QueryType.CONSTRUCT
        elif line.startswith('ASK'):
            return QueryType.ASK
        elif line.startswith('DESCRIBE'):
            return QueryType.DESCRIBE
        elif line.startswith('INSERT DATA'):
            return QueryType.INSERT_DATA
        elif line.startswith('DELETE DATA'):
            return QueryType.DELETE_DATA
        elif line.startswith('INSERT'):
            return QueryType.INSERT
        elif line.startswith('DELETE'):
            return QueryType.DELETE
        else:
            # Found non-prefix line but unrecognized
            break
    
    # Default to SELECT if unclear
    return QueryType.SELECT


def extract_parameters(sparql: str) -> list[QueryParameter]:
    """Extract template parameters from a query."""
    import re
    
    params = []
    seen = set()
    
    # Match {{param_name}} or {{param_name:description}}
    pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)(:[^}]*)?\}\}'
    
    for match in re.finditer(pattern, sparql):
        name = match.group(1)
        description = match.group(2)[1:] if match.group(2) else ""
        
        if name not in seen:
            seen.add(name)
            params.append(QueryParameter(
                name=name,
                description=description
            ))
    
    return params


class SavedQueryManager:
    """
    Manages saved SPARQL queries.
    
    Features:
    - Save/load queries with metadata
    - Query history tracking
    - Search and filter queries
    - Export/import query collections
    
    Usage:
        manager = SavedQueryManager(workspace / "_queries")
        
        # Save a query
        query = manager.save(
            name="Find all people",
            sparql="SELECT ?s WHERE { ?s a foaf:Person }",
            tags=["foaf", "people"]
        )
        
        # Execute with tracking
        result = manager.execute(query.query_id, store)
        
        # Search queries
        matches = manager.search("people", tags=["foaf"])
    """
    
    def __init__(self, queries_dir: Path | str, history_limit: int = 1000):
        self.queries_dir = Path(queries_dir)
        self.queries_dir.mkdir(parents=True, exist_ok=True)
        self.history_limit = history_limit
        
        self._queries: dict[str, SavedQuery] = {}
        self._history: list[QueryExecution] = []
        
        self._load_queries()
        self._load_history()
    
    def save(
        self,
        name: str,
        sparql: str,
        description: str = "",
        tags: list[str] | None = None,
        repository: str | None = None,
        is_template: bool = False,
        created_by: str = "",
        metadata: dict | None = None
    ) -> SavedQuery:
        """
        Save a new query.
        
        Returns the saved query object.
        """
        query_id = f"query-{uuid.uuid4().hex[:12]}"
        query_type = detect_query_type(sparql)
        parameters = extract_parameters(sparql) if is_template else []
        
        query = SavedQuery(
            query_id=query_id,
            name=name,
            sparql=sparql,
            query_type=query_type,
            description=description,
            tags=tags or [],
            repository=repository,
            is_template=is_template,
            parameters=parameters,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self._queries[query_id] = query
        self._save_query(query)
        
        logger.info(f"Saved query '{name}' with ID {query_id}")
        return query
    
    def update(
        self,
        query_id: str,
        name: str | None = None,
        sparql: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        is_favorite: bool | None = None,
        metadata: dict | None = None
    ) -> SavedQuery:
        """
        Update an existing query.
        
        Only provided fields are updated.
        """
        if query_id not in self._queries:
            raise KeyError(f"Query not found: {query_id}")
        
        query = self._queries[query_id]
        
        if name is not None:
            query.name = name
        if sparql is not None:
            query.sparql = sparql
            query.query_type = detect_query_type(sparql)
            if query.is_template:
                query.parameters = extract_parameters(sparql)
        if description is not None:
            query.description = description
        if tags is not None:
            query.tags = tags
        if is_favorite is not None:
            query.is_favorite = is_favorite
        if metadata is not None:
            query.metadata.update(metadata)
        
        query.updated_at = datetime.now()
        query.version += 1
        
        self._save_query(query)
        return query
    
    def get(self, query_id: str) -> SavedQuery | None:
        """Get a query by ID."""
        return self._queries.get(query_id)
    
    def delete(self, query_id: str) -> bool:
        """Delete a query."""
        if query_id not in self._queries:
            return False
        
        del self._queries[query_id]
        query_path = self.queries_dir / f"{query_id}.json"
        if query_path.exists():
            query_path.unlink()
        
        return True
    
    def list(
        self,
        repository: str | None = None,
        tags: list[str] | None = None,
        query_type: QueryType | None = None,
        favorites_only: bool = False,
        limit: int | None = None
    ) -> list[SavedQuery]:
        """
        List saved queries with optional filters.
        """
        results = []
        
        for query in self._queries.values():
            # Apply filters
            if repository is not None and query.repository != repository:
                continue
            if tags:
                if not any(t in query.tags for t in tags):
                    continue
            if query_type is not None and query.query_type != query_type:
                continue
            if favorites_only and not query.is_favorite:
                continue
            
            results.append(query)
        
        # Sort by last updated
        results.sort(key=lambda q: q.updated_at, reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def search(
        self,
        text: str,
        tags: list[str] | None = None,
        repository: str | None = None
    ) -> list[SavedQuery]:
        """
        Search queries by name, description, or query text.
        """
        text_lower = text.lower()
        results = []
        
        for query in self._queries.values():
            # Check repository filter
            if repository is not None and query.repository != repository:
                continue
            
            # Check tag filter
            if tags and not any(t in query.tags for t in tags):
                continue
            
            # Check text match
            if (text_lower in query.name.lower() or
                text_lower in query.description.lower() or
                text_lower in query.sparql.lower()):
                results.append(query)
        
        # Sort by relevance (name matches first)
        def relevance(q):
            if text_lower in q.name.lower():
                return 0
            if text_lower in q.description.lower():
                return 1
            return 2
        
        results.sort(key=relevance)
        return results
    
    def execute(
        self,
        query_id: str,
        store: Any,
        params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Execute a saved query and track statistics.
        
        Returns execution result and stats.
        """
        if query_id not in self._queries:
            raise KeyError(f"Query not found: {query_id}")
        
        query = self._queries[query_id]
        
        # Get SPARQL (fill template if needed)
        if query.is_template:
            sparql = query.fill_template(params or {})
        else:
            sparql = query.sparql
        
        # Execute query
        start_time = time.time()
        try:
            result = store.query(sparql)
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine result count
            if hasattr(result, '__len__'):
                result_count = len(result)
            else:
                result_count = None
            
            # Record execution
            execution = QueryExecution(
                query_id=query_id,
                executed_at=datetime.now(),
                duration_ms=duration_ms,
                result_count=result_count,
                success=True,
                parameters=params or {}
            )
            
            # Update query stats
            query.execution_count += 1
            query.total_execution_time_ms += duration_ms
            query.last_executed_at = execution.executed_at
            self._save_query(query)
            
            # Add to history
            self._add_to_history(execution)
            
            return {
                "result": result,
                "duration_ms": duration_ms,
                "result_count": result_count,
                "query_id": query_id
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            execution = QueryExecution(
                query_id=query_id,
                executed_at=datetime.now(),
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                parameters=params or {}
            )
            self._add_to_history(execution)
            
            raise
    
    def get_history(
        self,
        query_id: str | None = None,
        limit: int = 100
    ) -> list[QueryExecution]:
        """
        Get query execution history.
        
        Optionally filter by query_id.
        """
        if query_id:
            history = [e for e in self._history if e.query_id == query_id]
        else:
            history = list(self._history)
        
        # Most recent first
        history.sort(key=lambda e: e.executed_at, reverse=True)
        return history[:limit]
    
    def export_collection(
        self,
        name: str,
        query_ids: list[str] | None = None,
        tags: list[str] | None = None,
        description: str = "",
        author: str = ""
    ) -> QueryCollection:
        """
        Export a collection of queries.
        """
        if query_ids:
            queries = [self._queries[qid] for qid in query_ids if qid in self._queries]
        elif tags:
            queries = [q for q in self._queries.values() if any(t in q.tags for t in tags)]
        else:
            queries = list(self._queries.values())
        
        return QueryCollection(
            name=name,
            description=description,
            queries=queries,
            author=author
        )
    
    def export_to_file(
        self,
        path: Path | str,
        query_ids: list[str] | None = None,
        tags: list[str] | None = None,
        name: str = "Exported Queries"
    ) -> int:
        """
        Export queries to a JSON file.
        
        Returns number of queries exported.
        """
        collection = self.export_collection(name, query_ids, tags)
        
        with open(path, "w") as f:
            json.dump(collection.to_dict(), f, indent=2)
        
        return len(collection.queries)
    
    def import_collection(
        self,
        collection: QueryCollection,
        overwrite: bool = False
    ) -> dict[str, str]:
        """
        Import a query collection.
        
        Returns mapping of old query IDs to new IDs.
        """
        id_mapping = {}
        
        for query in collection.queries:
            # Check for existing query with same hash
            existing = None
            for q in self._queries.values():
                if q.query_hash == query.query_hash and q.name == query.name:
                    existing = q
                    break
            
            if existing and not overwrite:
                id_mapping[query.query_id] = existing.query_id
                continue
            
            # Create new query with new ID
            new_query = SavedQuery(
                query_id=f"query-{uuid.uuid4().hex[:12]}",
                name=query.name,
                sparql=query.sparql,
                query_type=query.query_type,
                description=query.description,
                tags=query.tags,
                repository=query.repository,
                is_template=query.is_template,
                parameters=query.parameters,
                metadata=query.metadata
            )
            
            self._queries[new_query.query_id] = new_query
            self._save_query(new_query)
            id_mapping[query.query_id] = new_query.query_id
        
        return id_mapping
    
    def import_from_file(
        self,
        path: Path | str,
        overwrite: bool = False
    ) -> dict[str, str]:
        """
        Import queries from a JSON file.
        
        Returns mapping of old query IDs to new IDs.
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        collection = QueryCollection.from_dict(data)
        return self.import_collection(collection, overwrite)
    
    def get_popular_queries(self, limit: int = 10) -> list[SavedQuery]:
        """Get most frequently executed queries."""
        queries = list(self._queries.values())
        queries.sort(key=lambda q: q.execution_count, reverse=True)
        return queries[:limit]
    
    def get_recent_queries(self, limit: int = 10) -> list[SavedQuery]:
        """Get most recently executed queries."""
        queries = [q for q in self._queries.values() if q.last_executed_at]
        queries.sort(key=lambda q: q.last_executed_at, reverse=True)
        return queries[:limit]
    
    def _save_query(self, query: SavedQuery) -> None:
        """Save a query to disk."""
        query_path = self.queries_dir / f"{query.query_id}.json"
        with open(query_path, "w") as f:
            json.dump(query.to_dict(), f, indent=2)
    
    def _load_queries(self) -> None:
        """Load all queries from disk."""
        for path in self.queries_dir.glob("query-*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                query = SavedQuery.from_dict(data)
                self._queries[query.query_id] = query
            except Exception as e:
                logger.warning(f"Failed to load query from {path}: {e}")
    
    def _add_to_history(self, execution: QueryExecution) -> None:
        """Add execution to history."""
        self._history.append(execution)
        
        # Trim history if needed
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]
        
        self._save_history()
    
    def _save_history(self) -> None:
        """Save execution history to disk."""
        history_path = self.queries_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump([e.to_dict() for e in self._history], f)
    
    def _load_history(self) -> None:
        """Load execution history from disk."""
        history_path = self.queries_dir / "history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                self._history = [QueryExecution.from_dict(e) for e in data]
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")


# Convenience functions

def save_query(
    queries_dir: Path | str,
    name: str,
    sparql: str,
    **kwargs
) -> SavedQuery:
    """Quick way to save a query."""
    manager = SavedQueryManager(queries_dir)
    return manager.save(name, sparql, **kwargs)


def load_query(queries_dir: Path | str, query_id: str) -> SavedQuery | None:
    """Quick way to load a query."""
    manager = SavedQueryManager(queries_dir)
    return manager.get(query_id)
