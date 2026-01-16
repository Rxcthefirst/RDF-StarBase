"""
Assertion Registry for RDF-StarBase.

Tracks datasets, APIs, mappings, and materialization runs as first-class
entities with their own provenance. This enables answering questions like:
- Which datasets contributed to this assertion?
- When was this API last synced?
- What mappings transformed this data?
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4
import json

import polars as pl

from rdf_starbase.models import ProvenanceContext


class SourceType(str, Enum):
    """Types of data sources in the registry."""
    DATASET = "dataset"
    API = "api"
    MAPPING = "mapping"
    PROCESS = "process"
    MANUAL = "manual"


class SourceStatus(str, Enum):
    """Status of a registered source."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ERROR = "error"
    SYNCING = "syncing"


@dataclass
class RegisteredSource:
    """
    A data source registered in the Assertion Registry.
    
    Represents datasets, APIs, mappings, or processes that contribute
    assertions to the knowledge graph.
    """
    id: UUID
    name: str
    source_type: SourceType
    uri: Optional[str] = None
    description: Optional[str] = None
    status: SourceStatus = SourceStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_sync: Optional[datetime] = None
    sync_frequency: Optional[str] = None  # e.g., "daily", "hourly", "manual"
    owner: Optional[str] = None
    schema_uri: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "name": self.name,
            "source_type": self.source_type.value,
            "uri": self.uri,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_sync": self.last_sync,
            "sync_frequency": self.sync_frequency,
            "owner": self.owner,
            "schema_uri": self.schema_uri,
            "config": json.dumps(self.config),
            "tags": json.dumps(self.tags),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegisteredSource":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            source_type=SourceType(data["source_type"]),
            uri=data.get("uri"),
            description=data.get("description"),
            status=SourceStatus(data.get("status", "active")),
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            last_sync=data.get("last_sync"),
            sync_frequency=data.get("sync_frequency"),
            owner=data.get("owner"),
            schema_uri=data.get("schema_uri"),
            config=json.loads(data.get("config", "{}")),
            tags=json.loads(data.get("tags", "[]")),
        )


@dataclass
class SyncRun:
    """
    A record of a synchronization run from a source.
    
    Tracks when data was pulled, how many assertions were created,
    and any errors encountered.
    """
    id: UUID
    source_id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, success, failed, partial
    assertions_created: int = 0
    assertions_updated: int = 0
    assertions_deprecated: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "assertions_created": self.assertions_created,
            "assertions_updated": self.assertions_updated,
            "assertions_deprecated": self.assertions_deprecated,
            "errors": json.dumps(self.errors),
            "metadata": json.dumps(self.metadata),
        }


class AssertionRegistry:
    """
    Registry for tracking data sources and their sync history.
    
    The Assertion Registry answers critical governance questions:
    - Where did this data come from?
    - When was it last updated?
    - Which systems contribute to this knowledge graph?
    - What's the lineage of this assertion?
    
    Example:
        >>> registry = AssertionRegistry()
        >>> 
        >>> # Register a CRM API
        >>> crm = registry.register_source(
        ...     name="Salesforce CRM",
        ...     source_type=SourceType.API,
        ...     uri="https://api.salesforce.com/v52",
        ...     owner="sales-team",
        ...     sync_frequency="hourly"
        ... )
        >>> 
        >>> # Start a sync run
        >>> run = registry.start_sync(crm.id)
        >>> # ... perform sync ...
        >>> registry.complete_sync(run.id, assertions_created=150)
        >>> 
        >>> # Query sources
        >>> apis = registry.get_sources(source_type=SourceType.API)
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._sources_df = pl.DataFrame({
            "id": pl.Series([], dtype=pl.Utf8),
            "name": pl.Series([], dtype=pl.Utf8),
            "source_type": pl.Series([], dtype=pl.Utf8),
            "uri": pl.Series([], dtype=pl.Utf8),
            "description": pl.Series([], dtype=pl.Utf8),
            "status": pl.Series([], dtype=pl.Utf8),
            "created_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "last_sync": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "sync_frequency": pl.Series([], dtype=pl.Utf8),
            "owner": pl.Series([], dtype=pl.Utf8),
            "schema_uri": pl.Series([], dtype=pl.Utf8),
            "config": pl.Series([], dtype=pl.Utf8),
            "tags": pl.Series([], dtype=pl.Utf8),
        })
        
        self._syncs_df = pl.DataFrame({
            "id": pl.Series([], dtype=pl.Utf8),
            "source_id": pl.Series([], dtype=pl.Utf8),
            "started_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "completed_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "status": pl.Series([], dtype=pl.Utf8),
            "assertions_created": pl.Series([], dtype=pl.Int64),
            "assertions_updated": pl.Series([], dtype=pl.Int64),
            "assertions_deprecated": pl.Series([], dtype=pl.Int64),
            "errors": pl.Series([], dtype=pl.Utf8),
            "metadata": pl.Series([], dtype=pl.Utf8),
        })
    
    def register_source(
        self,
        name: str,
        source_type: SourceType,
        uri: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        sync_frequency: Optional[str] = None,
        schema_uri: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> RegisteredSource:
        """
        Register a new data source in the registry.
        
        Args:
            name: Human-readable name for the source
            source_type: Type of source (dataset, api, mapping, process)
            uri: URI or connection string for the source
            description: Optional description
            owner: Team or person responsible for this source
            sync_frequency: How often this source syncs (daily, hourly, etc.)
            schema_uri: URI to schema definition
            config: Additional configuration
            tags: Tags for categorization
            
        Returns:
            The registered source with assigned ID
        """
        source = RegisteredSource(
            id=uuid4(),
            name=name,
            source_type=source_type,
            uri=uri,
            description=description,
            owner=owner,
            sync_frequency=sync_frequency,
            schema_uri=schema_uri,
            config=config or {},
            tags=tags or [],
        )
        
        new_row = pl.DataFrame([source.to_dict()])
        self._sources_df = pl.concat([self._sources_df, new_row], how="vertical")
        
        return source
    
    def get_source(self, source_id: UUID) -> Optional[RegisteredSource]:
        """Get a source by ID."""
        filtered = self._sources_df.filter(pl.col("id") == str(source_id))
        if len(filtered) == 0:
            return None
        return RegisteredSource.from_dict(filtered.row(0, named=True))
    
    def get_source_by_name(self, name: str) -> Optional[RegisteredSource]:
        """Get a source by name."""
        filtered = self._sources_df.filter(pl.col("name") == name)
        if len(filtered) == 0:
            return None
        return RegisteredSource.from_dict(filtered.row(0, named=True))
    
    def get_sources(
        self,
        source_type: Optional[SourceType] = None,
        status: Optional[SourceStatus] = None,
        owner: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[RegisteredSource]:
        """
        Query registered sources with optional filters.
        
        Args:
            source_type: Filter by source type
            status: Filter by status
            owner: Filter by owner
            tag: Filter by tag (sources containing this tag)
            
        Returns:
            List of matching sources
        """
        df = self._sources_df
        
        if source_type is not None:
            df = df.filter(pl.col("source_type") == source_type.value)
        
        if status is not None:
            df = df.filter(pl.col("status") == status.value)
        
        if owner is not None:
            df = df.filter(pl.col("owner") == owner)
        
        if tag is not None:
            df = df.filter(pl.col("tags").str.contains(f'"{tag}"'))
        
        return [RegisteredSource.from_dict(row) for row in df.iter_rows(named=True)]
    
    def update_source_status(
        self,
        source_id: UUID,
        status: SourceStatus,
    ) -> None:
        """Update the status of a source."""
        self._sources_df = self._sources_df.with_columns(
            pl.when(pl.col("id") == str(source_id))
            .then(pl.lit(status.value))
            .otherwise(pl.col("status"))
            .alias("status")
        )
    
    def deprecate_source(self, source_id: UUID) -> None:
        """Mark a source as deprecated."""
        self.update_source_status(source_id, SourceStatus.DEPRECATED)
    
    def start_sync(
        self,
        source_id: UUID,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SyncRun:
        """
        Start a new synchronization run for a source.
        
        Args:
            source_id: ID of the source being synced
            metadata: Optional metadata about the sync
            
        Returns:
            The created sync run
        """
        run = SyncRun(
            id=uuid4(),
            source_id=source_id,
            started_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        
        new_row = pl.DataFrame([run.to_dict()])
        self._syncs_df = pl.concat([self._syncs_df, new_row], how="vertical")
        
        # Update source status
        self.update_source_status(source_id, SourceStatus.SYNCING)
        
        return run
    
    def complete_sync(
        self,
        run_id: UUID,
        assertions_created: int = 0,
        assertions_updated: int = 0,
        assertions_deprecated: int = 0,
        errors: Optional[list[str]] = None,
        status: str = "success",
    ) -> None:
        """
        Complete a synchronization run.
        
        Args:
            run_id: ID of the sync run
            assertions_created: Number of new assertions
            assertions_updated: Number of updated assertions
            assertions_deprecated: Number of deprecated assertions
            errors: Any errors encountered
            status: Final status (success, failed, partial)
        """
        now = datetime.now(timezone.utc)
        run_id_str = str(run_id)
        
        # Get source_id before updating
        run_row = self._syncs_df.filter(pl.col("id") == run_id_str)
        if len(run_row) == 0:
            raise ValueError(f"Sync run {run_id} not found")
        
        source_id = run_row["source_id"][0]
        
        # Update sync run
        self._syncs_df = self._syncs_df.with_columns([
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(now))
            .otherwise(pl.col("completed_at"))
            .alias("completed_at"),
            
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(status))
            .otherwise(pl.col("status"))
            .alias("status"),
            
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(assertions_created))
            .otherwise(pl.col("assertions_created"))
            .alias("assertions_created"),
            
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(assertions_updated))
            .otherwise(pl.col("assertions_updated"))
            .alias("assertions_updated"),
            
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(assertions_deprecated))
            .otherwise(pl.col("assertions_deprecated"))
            .alias("assertions_deprecated"),
            
            pl.when(pl.col("id") == run_id_str)
            .then(pl.lit(json.dumps(errors or [])))
            .otherwise(pl.col("errors"))
            .alias("errors"),
        ])
        
        # Update source last_sync and status
        final_status = SourceStatus.ERROR if status == "failed" else SourceStatus.ACTIVE
        
        self._sources_df = self._sources_df.with_columns([
            pl.when(pl.col("id") == source_id)
            .then(pl.lit(now))
            .otherwise(pl.col("last_sync"))
            .alias("last_sync"),
            
            pl.when(pl.col("id") == source_id)
            .then(pl.lit(final_status.value))
            .otherwise(pl.col("status"))
            .alias("status"),
        ])
    
    def get_sync_history(
        self,
        source_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        Get synchronization history.
        
        Args:
            source_id: Filter by source (None for all)
            limit: Maximum number of records
            
        Returns:
            DataFrame with sync history
        """
        df = self._syncs_df
        
        if source_id is not None:
            df = df.filter(pl.col("source_id") == str(source_id))
        
        return df.sort("started_at", descending=True).head(limit)
    
    def get_last_sync(self, source_id: UUID) -> Optional[SyncRun]:
        """Get the most recent sync run for a source."""
        history = self.get_sync_history(source_id, limit=1)
        if len(history) == 0:
            return None
        
        row = history.row(0, named=True)
        return SyncRun(
            id=UUID(row["id"]),
            source_id=UUID(row["source_id"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            status=row["status"],
            assertions_created=row["assertions_created"],
            assertions_updated=row["assertions_updated"],
            assertions_deprecated=row["assertions_deprecated"],
            errors=json.loads(row["errors"]) if row["errors"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def create_provenance_context(
        self,
        source_id: UUID,
        confidence: float = 1.0,
        process: Optional[str] = None,
    ) -> ProvenanceContext:
        """
        Create a ProvenanceContext linked to a registered source.
        
        This bridges the registry with the triple store, allowing
        assertions to reference their source.
        
        Args:
            source_id: ID of the registered source
            confidence: Confidence level for assertions
            process: Optional process name
            
        Returns:
            ProvenanceContext that can be used with TripleStore.add_triple
        """
        source = self.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")
        
        return ProvenanceContext(
            source=source.name,
            confidence=confidence,
            process=process or f"{source.source_type.value}_sync",
            metadata={"source_id": str(source_id), "source_uri": source.uri},
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        sources_by_type = (
            self._sources_df
            .group_by("source_type")
            .agg(pl.len().alias("count"))
        )
        
        sources_by_status = (
            self._sources_df
            .group_by("status")
            .agg(pl.len().alias("count"))
        )
        
        total_syncs = len(self._syncs_df)
        successful_syncs = len(self._syncs_df.filter(pl.col("status") == "success"))
        
        return {
            "total_sources": len(self._sources_df),
            "sources_by_type": {
                row["source_type"]: row["count"]
                for row in sources_by_type.iter_rows(named=True)
            },
            "sources_by_status": {
                row["status"]: row["count"]
                for row in sources_by_status.iter_rows(named=True)
            },
            "total_sync_runs": total_syncs,
            "successful_sync_runs": successful_syncs,
            "sync_success_rate": successful_syncs / total_syncs if total_syncs > 0 else 0,
        }
    
    def save(self, path: str) -> None:
        """
        Save registry to Parquet files.
        
        Creates two files:
        - {path}_sources.parquet
        - {path}_syncs.parquet
        """
        self._sources_df.write_parquet(f"{path}_sources.parquet")
        self._syncs_df.write_parquet(f"{path}_syncs.parquet")
    
    @classmethod
    def load(cls, path: str) -> "AssertionRegistry":
        """Load registry from Parquet files."""
        registry = cls()
        registry._sources_df = pl.read_parquet(f"{path}_sources.parquet")
        registry._syncs_df = pl.read_parquet(f"{path}_syncs.parquet")
        return registry
