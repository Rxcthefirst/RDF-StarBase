"""
Audit and Compliance module for RDF-StarBase.

Provides audit logging, log export, data lineage tracking,
and source health monitoring for enterprise compliance.
"""

from __future__ import annotations

import json
import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
from collections import defaultdict


class AuditAction(Enum):
    """Types of auditable actions."""
    
    # Authentication
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    KEY_CREATED = "key_created"
    KEY_REVOKED = "key_revoked"
    
    # Data operations
    QUERY = "query"
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    LOAD = "load"
    
    # Repository operations
    REPO_CREATED = "repo_created"
    REPO_DELETED = "repo_deleted"
    REPO_CLONED = "repo_cloned"
    
    # Backup/Restore
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Admin
    CONFIG_CHANGED = "config_changed"
    IMPORT_STAGED = "import_staged"
    IMPORT_COMMITTED = "import_committed"
    IMPORT_ROLLED_BACK = "import_rolled_back"


class AuditSeverity(Enum):
    """Severity levels for audit entries."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    
    entry_id: str
    timestamp: datetime
    action: AuditAction
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Actor information
    key_id: str | None = None
    user_name: str | None = None
    ip_address: str | None = None
    
    # Target information
    repository: str | None = None
    graph: str | None = None
    
    # Action details
    details: dict[str, Any] = field(default_factory=dict)
    
    # Outcome
    success: bool = True
    error_message: str | None = None
    
    # Metrics
    duration_ms: float | None = None
    affected_count: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "severity": self.severity.value,
            "key_id": self.key_id,
            "user_name": self.user_name,
            "ip_address": self.ip_address,
            "repository": self.repository,
            "graph": self.graph,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "affected_count": self.affected_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=AuditAction(data["action"]),
            severity=AuditSeverity(data.get("severity", "info")),
            key_id=data.get("key_id"),
            user_name=data.get("user_name"),
            ip_address=data.get("ip_address"),
            repository=data.get("repository"),
            graph=data.get("graph"),
            details=data.get("details", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            duration_ms=data.get("duration_ms"),
            affected_count=data.get("affected_count"),
        )
    
    def to_csv_row(self) -> list[str]:
        """Convert to CSV row."""
        return [
            self.entry_id,
            self.timestamp.isoformat(),
            self.action.value,
            self.severity.value,
            self.key_id or "",
            self.user_name or "",
            self.ip_address or "",
            self.repository or "",
            self.graph or "",
            json.dumps(self.details),
            str(self.success),
            self.error_message or "",
            str(self.duration_ms) if self.duration_ms else "",
            str(self.affected_count) if self.affected_count else "",
        ]
    
    @staticmethod
    def csv_headers() -> list[str]:
        """Get CSV headers."""
        return [
            "entry_id", "timestamp", "action", "severity",
            "key_id", "user_name", "ip_address",
            "repository", "graph", "details",
            "success", "error_message", "duration_ms", "affected_count",
        ]


class AuditLog:
    """Manages audit log entries."""
    
    def __init__(
        self,
        storage_path: Path | None = None,
        max_entries: int = 100000,
        retention_days: int = 90,
    ):
        """Initialize audit log.
        
        Args:
            storage_path: Path to persist logs
            max_entries: Maximum entries to keep in memory
            retention_days: Days to retain entries
        """
        self.storage_path = storage_path
        self.max_entries = max_entries
        self.retention_days = retention_days
        self._entries: list[AuditEntry] = []
        self._entry_counter = 0
        
        if storage_path and storage_path.exists():
            self._load()
    
    def log(
        self,
        action: AuditAction,
        severity: AuditSeverity = AuditSeverity.INFO,
        key_id: str | None = None,
        user_name: str | None = None,
        ip_address: str | None = None,
        repository: str | None = None,
        graph: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        duration_ms: float | None = None,
        affected_count: int | None = None,
    ) -> AuditEntry:
        """Log an audit entry."""
        self._entry_counter += 1
        entry = AuditEntry(
            entry_id=f"audit-{self._entry_counter:08d}",
            timestamp=datetime.now(),
            action=action,
            severity=severity,
            key_id=key_id,
            user_name=user_name,
            ip_address=ip_address,
            repository=repository,
            graph=graph,
            details=details or {},
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
            affected_count=affected_count,
        )
        
        self._entries.append(entry)
        
        # Trim if over limit
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        self._save()
        return entry
    
    def get_entries(
        self,
        action: AuditAction | None = None,
        key_id: str | None = None,
        repository: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        success: bool | None = None,
        severity: AuditSeverity | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        """Query audit log entries with filters."""
        results = []
        
        for entry in reversed(self._entries):
            if action and entry.action != action:
                continue
            if key_id and entry.key_id != key_id:
                continue
            if repository and entry.repository != repository:
                continue
            if since and entry.timestamp < since:
                continue
            if until and entry.timestamp > until:
                continue
            if success is not None and entry.success != success:
                continue
            if severity and entry.severity != severity:
                continue
            
            results.append(entry)
            
            if limit and len(results) >= limit:
                break
        
        return results
    
    def get_by_id(self, entry_id: str) -> AuditEntry | None:
        """Get entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None
    
    def export_json(
        self,
        entries: list[AuditEntry] | None = None,
        pretty: bool = False,
    ) -> str:
        """Export entries to JSON."""
        if entries is None:
            entries = self._entries
        
        data = [e.to_dict() for e in entries]
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)
    
    def export_csv(
        self,
        entries: list[AuditEntry] | None = None,
    ) -> str:
        """Export entries to CSV."""
        if entries is None:
            entries = self._entries
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(AuditEntry.csv_headers())
        for entry in entries:
            writer.writerow(entry.to_csv_row())
        
        return output.getvalue()
    
    def export_to_file(
        self,
        path: Path,
        entries: list[AuditEntry] | None = None,
        format: str = "json",
    ) -> int:
        """Export entries to file. Returns count exported."""
        if entries is None:
            entries = self._entries
        
        if format == "json":
            content = self.export_json(entries, pretty=True)
        elif format == "csv":
            content = self.export_csv(entries)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        path.write_text(content)
        return len(entries)
    
    def get_statistics(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        """Get audit log statistics."""
        entries = self.get_entries(since=since, until=until)
        
        if not entries:
            return {
                "total_entries": 0,
                "by_action": {},
                "by_severity": {},
                "success_rate": 0.0,
            }
        
        by_action: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        by_repo: dict[str, int] = defaultdict(int)
        by_user: dict[str, int] = defaultdict(int)
        success_count = 0
        
        for entry in entries:
            by_action[entry.action.value] += 1
            by_severity[entry.severity.value] += 1
            if entry.repository:
                by_repo[entry.repository] += 1
            if entry.key_id:
                by_user[entry.key_id] += 1
            if entry.success:
                success_count += 1
        
        return {
            "total_entries": len(entries),
            "by_action": dict(by_action),
            "by_severity": dict(by_severity),
            "by_repository": dict(by_repo),
            "by_user": dict(by_user),
            "success_rate": success_count / len(entries) if entries else 0.0,
            "earliest": entries[-1].timestamp.isoformat() if entries else None,
            "latest": entries[0].timestamp.isoformat() if entries else None,
        }
    
    def cleanup_old_entries(self) -> int:
        """Remove entries older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed = original_count - len(self._entries)
        if removed > 0:
            self._save()
        return removed
    
    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        count = len(self._entries)
        self._entries = []
        self._save()
        return count
    
    def _save(self) -> None:
        """Persist entries to storage."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "counter": self._entry_counter,
        }
        self.storage_path.write_text(json.dumps(data))
    
    def _load(self) -> None:
        """Load entries from storage."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        self._entries = [AuditEntry.from_dict(e) for e in data.get("entries", [])]
        self._entry_counter = data.get("counter", len(self._entries))


@dataclass
class LineageNode:
    """A node in the data lineage graph."""
    
    node_id: str
    node_type: str  # "source", "graph", "derived", "export"
    name: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LineageEdge:
    """An edge in the data lineage graph."""
    
    source_id: str
    target_id: str
    relationship: str  # "derived_from", "loaded_to", "exported_from"
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class DataLineage:
    """Tracks data lineage from source to derived facts."""
    
    def __init__(self, storage_path: Path | None = None):
        """Initialize data lineage tracker."""
        self.storage_path = storage_path
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        
        if storage_path and storage_path.exists():
            self._load()
    
    def add_source(
        self,
        source_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Add a data source node."""
        node = LineageNode(
            node_id=source_id,
            node_type="source",
            name=name,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._nodes[source_id] = node
        self._save()
        return node
    
    def add_graph(
        self,
        graph_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Add a graph node."""
        node = LineageNode(
            node_id=graph_id,
            node_type="graph",
            name=name,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._nodes[graph_id] = node
        self._save()
        return node
    
    def add_derived(
        self,
        derived_id: str,
        name: str,
        source_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Add a derived data node with links to sources."""
        node = LineageNode(
            node_id=derived_id,
            node_type="derived",
            name=name,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._nodes[derived_id] = node
        
        # Add edges from sources
        for source_id in source_ids:
            self.add_edge(source_id, derived_id, "derived_from")
        
        self._save()
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEdge:
        """Add a lineage edge."""
        edge = LineageEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        self._edges.append(edge)
        self._save()
        return edge
    
    def get_node(self, node_id: str) -> LineageNode | None:
        """Get a lineage node."""
        return self._nodes.get(node_id)
    
    def get_ancestors(self, node_id: str) -> list[LineageNode]:
        """Get all ancestor nodes (sources this node derives from)."""
        ancestors = []
        visited = set()
        
        def traverse(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self._edges:
                if edge.target_id == nid and edge.source_id in self._nodes:
                    ancestor = self._nodes[edge.source_id]
                    ancestors.append(ancestor)
                    traverse(edge.source_id)
        
        traverse(node_id)
        return ancestors
    
    def get_descendants(self, node_id: str) -> list[LineageNode]:
        """Get all descendant nodes (derived from this node)."""
        descendants = []
        visited = set()
        
        def traverse(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self._edges:
                if edge.source_id == nid and edge.target_id in self._nodes:
                    descendant = self._nodes[edge.target_id]
                    descendants.append(descendant)
                    traverse(edge.target_id)
        
        traverse(node_id)
        return descendants
    
    def get_full_lineage(self, node_id: str) -> dict[str, Any]:
        """Get complete lineage graph for a node."""
        node = self._nodes.get(node_id)
        if not node:
            return {"error": "Node not found"}
        
        ancestors = self.get_ancestors(node_id)
        descendants = self.get_descendants(node_id)
        
        # Get relevant edges
        all_node_ids = {node_id} | {n.node_id for n in ancestors} | {n.node_id for n in descendants}
        relevant_edges = [e for e in self._edges 
                         if e.source_id in all_node_ids and e.target_id in all_node_ids]
        
        return {
            "node": node.to_dict(),
            "ancestors": [n.to_dict() for n in ancestors],
            "descendants": [n.to_dict() for n in descendants],
            "edges": [e.to_dict() for e in relevant_edges],
        }
    
    def list_sources(self) -> list[LineageNode]:
        """List all source nodes."""
        return [n for n in self._nodes.values() if n.node_type == "source"]
    
    def _save(self) -> None:
        """Persist lineage to storage."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load lineage from storage."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        
        for nid, nd in data.get("nodes", {}).items():
            self._nodes[nid] = LineageNode(
                node_id=nd["node_id"],
                node_type=nd["node_type"],
                name=nd["name"],
                created_at=datetime.fromisoformat(nd["created_at"]),
                metadata=nd.get("metadata", {}),
            )
        
        for ed in data.get("edges", []):
            self._edges.append(LineageEdge(
                source_id=ed["source_id"],
                target_id=ed["target_id"],
                relationship=ed["relationship"],
                created_at=datetime.fromisoformat(ed["created_at"]),
                metadata=ed.get("metadata", {}),
            ))


@dataclass
class SourceHealth:
    """Health status for a data source."""
    
    source_id: str
    name: str
    last_load: datetime | None = None
    last_success: datetime | None = None
    last_error: datetime | None = None
    error_message: str | None = None
    
    # Metrics
    total_loads: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    total_triples: int = 0
    
    # Freshness
    expected_update_hours: int | None = None  # How often source should update
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_loads == 0:
            return 0.0
        return self.successful_loads / self.total_loads
    
    @property
    def is_stale(self) -> bool:
        """Check if source is stale."""
        if not self.expected_update_hours or not self.last_success:
            return False
        stale_threshold = datetime.now() - timedelta(hours=self.expected_update_hours)
        return self.last_success < stale_threshold
    
    @property
    def health_status(self) -> str:
        """Get overall health status."""
        if self.last_error and (not self.last_success or self.last_error > self.last_success):
            return "error"
        if self.is_stale:
            return "stale"
        if self.success_rate < 0.9:
            return "degraded"
        return "healthy"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "last_load": self.last_load.isoformat() if self.last_load else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error.isoformat() if self.last_error else None,
            "error_message": self.error_message,
            "total_loads": self.total_loads,
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "total_triples": self.total_triples,
            "success_rate": self.success_rate,
            "expected_update_hours": self.expected_update_hours,
            "is_stale": self.is_stale,
            "health_status": self.health_status,
        }


class SourceHealthMonitor:
    """Monitors health of data sources."""
    
    def __init__(self, storage_path: Path | None = None):
        """Initialize source health monitor."""
        self.storage_path = storage_path
        self._sources: dict[str, SourceHealth] = {}
        
        if storage_path and storage_path.exists():
            self._load()
    
    def register_source(
        self,
        source_id: str,
        name: str,
        expected_update_hours: int | None = None,
    ) -> SourceHealth:
        """Register a data source for monitoring."""
        health = SourceHealth(
            source_id=source_id,
            name=name,
            expected_update_hours=expected_update_hours,
        )
        self._sources[source_id] = health
        self._save()
        return health
    
    def record_load(
        self,
        source_id: str,
        success: bool,
        triple_count: int = 0,
        error_message: str | None = None,
    ) -> SourceHealth | None:
        """Record a load attempt from a source."""
        health = self._sources.get(source_id)
        if not health:
            return None
        
        health.total_loads += 1
        health.last_load = datetime.now()
        
        if success:
            health.successful_loads += 1
            health.last_success = datetime.now()
            health.total_triples += triple_count
            health.error_message = None
        else:
            health.failed_loads += 1
            health.last_error = datetime.now()
            health.error_message = error_message
        
        self._save()
        return health
    
    def get_health(self, source_id: str) -> SourceHealth | None:
        """Get health status for a source."""
        return self._sources.get(source_id)
    
    def list_sources(self) -> list[SourceHealth]:
        """List all monitored sources."""
        return list(self._sources.values())
    
    def get_unhealthy(self) -> list[SourceHealth]:
        """Get sources with health issues."""
        return [s for s in self._sources.values() if s.health_status != "healthy"]
    
    def get_stale(self) -> list[SourceHealth]:
        """Get stale sources."""
        return [s for s in self._sources.values() if s.is_stale]
    
    def get_summary(self) -> dict[str, Any]:
        """Get overall health summary."""
        sources = list(self._sources.values())
        if not sources:
            return {"total": 0, "by_status": {}}
        
        by_status: dict[str, int] = defaultdict(int)
        for s in sources:
            by_status[s.health_status] += 1
        
        return {
            "total": len(sources),
            "by_status": dict(by_status),
            "healthy": by_status.get("healthy", 0),
            "degraded": by_status.get("degraded", 0),
            "stale": by_status.get("stale", 0),
            "error": by_status.get("error", 0),
        }
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a source from monitoring."""
        if source_id not in self._sources:
            return False
        del self._sources[source_id]
        self._save()
        return True
    
    def _save(self) -> None:
        """Persist health data."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: s.to_dict() for sid, s in self._sources.items()}
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load health data."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        for sid, sd in data.items():
            self._sources[sid] = SourceHealth(
                source_id=sd["source_id"],
                name=sd["name"],
                last_load=datetime.fromisoformat(sd["last_load"]) if sd.get("last_load") else None,
                last_success=datetime.fromisoformat(sd["last_success"]) if sd.get("last_success") else None,
                last_error=datetime.fromisoformat(sd["last_error"]) if sd.get("last_error") else None,
                error_message=sd.get("error_message"),
                total_loads=sd.get("total_loads", 0),
                successful_loads=sd.get("successful_loads", 0),
                failed_loads=sd.get("failed_loads", 0),
                total_triples=sd.get("total_triples", 0),
                expected_update_hours=sd.get("expected_update_hours"),
            )


# Convenience functions

def create_audit_log(
    storage_path: Path | None = None,
    max_entries: int = 100000,
    retention_days: int = 90,
) -> AuditLog:
    """Create a new audit log."""
    return AuditLog(storage_path, max_entries, retention_days)


def create_lineage_tracker(storage_path: Path | None = None) -> DataLineage:
    """Create a new data lineage tracker."""
    return DataLineage(storage_path)


def create_health_monitor(storage_path: Path | None = None) -> SourceHealthMonitor:
    """Create a new source health monitor."""
    return SourceHealthMonitor(storage_path)
