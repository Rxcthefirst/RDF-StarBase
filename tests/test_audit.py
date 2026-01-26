"""Tests for the audit module - audit log, lineage, source health."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from rdf_starbase.storage.audit import (
    AuditAction,
    AuditSeverity,
    AuditEntry,
    AuditLog,
    LineageNode,
    LineageEdge,
    DataLineage,
    SourceHealth,
    SourceHealthMonitor,
    create_audit_log,
    create_lineage_tracker,
    create_health_monitor,
)


class TestAuditAction:
    """Tests for AuditAction enum."""
    
    def test_actions(self):
        assert AuditAction.QUERY.value == "query"
        assert AuditAction.INSERT.value == "insert"
        assert AuditAction.BACKUP_CREATED.value == "backup_created"


class TestAuditSeverity:
    """Tests for AuditSeverity enum."""
    
    def test_severities(self):
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.CRITICAL.value == "critical"


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""
    
    def test_creation(self):
        entry = AuditEntry(
            entry_id="audit-00000001",
            timestamp=datetime.now(),
            action=AuditAction.QUERY,
            key_id="key123",
            repository="test_repo",
        )
        assert entry.entry_id == "audit-00000001"
        assert entry.action == AuditAction.QUERY
        assert entry.success
    
    def test_to_dict(self):
        entry = AuditEntry(
            entry_id="audit-00000001",
            timestamp=datetime.now(),
            action=AuditAction.INSERT,
            severity=AuditSeverity.INFO,
            repository="test_repo",
            affected_count=100,
        )
        d = entry.to_dict()
        assert d["entry_id"] == "audit-00000001"
        assert d["action"] == "insert"
        assert d["affected_count"] == 100
    
    def test_from_dict(self):
        data = {
            "entry_id": "audit-00000001",
            "timestamp": datetime.now().isoformat(),
            "action": "query",
            "severity": "info",
            "success": True,
        }
        entry = AuditEntry.from_dict(data)
        assert entry.entry_id == "audit-00000001"
        assert entry.action == AuditAction.QUERY
    
    def test_to_csv_row(self):
        entry = AuditEntry(
            entry_id="audit-00000001",
            timestamp=datetime.now(),
            action=AuditAction.QUERY,
        )
        row = entry.to_csv_row()
        assert len(row) == 14
        assert row[0] == "audit-00000001"
    
    def test_csv_headers(self):
        headers = AuditEntry.csv_headers()
        assert "entry_id" in headers
        assert "action" in headers


class TestAuditLog:
    """Tests for AuditLog."""
    
    def test_log_entry(self):
        log = AuditLog()
        entry = log.log(
            action=AuditAction.QUERY,
            key_id="key123",
            repository="test_repo",
        )
        
        assert entry.entry_id.startswith("audit-")
        assert entry.action == AuditAction.QUERY
    
    def test_get_entries(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY, repository="repo1")
        log.log(action=AuditAction.INSERT, repository="repo2")
        log.log(action=AuditAction.QUERY, repository="repo1")
        
        all_entries = log.get_entries()
        assert len(all_entries) == 3
        
        # Filter by action
        queries = log.get_entries(action=AuditAction.QUERY)
        assert len(queries) == 2
        
        # Filter by repository
        repo1_entries = log.get_entries(repository="repo1")
        assert len(repo1_entries) == 2
    
    def test_get_entries_limit(self):
        log = AuditLog()
        for i in range(10):
            log.log(action=AuditAction.QUERY)
        
        limited = log.get_entries(limit=5)
        assert len(limited) == 5
    
    def test_get_entries_by_success(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY, success=True)
        log.log(action=AuditAction.QUERY, success=False, error_message="Failed")
        
        successful = log.get_entries(success=True)
        assert len(successful) == 1
    
    def test_get_by_id(self):
        log = AuditLog()
        entry = log.log(action=AuditAction.QUERY)
        
        found = log.get_by_id(entry.entry_id)
        assert found is not None
        assert found.entry_id == entry.entry_id
    
    def test_export_json(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY)
        log.log(action=AuditAction.INSERT)
        
        json_str = log.export_json()
        data = json.loads(json_str)
        assert len(data) == 2
    
    def test_export_json_pretty(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY)
        
        json_str = log.export_json(pretty=True)
        assert "\n" in json_str
    
    def test_export_csv(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY)
        log.log(action=AuditAction.INSERT)
        
        csv_str = log.export_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # Header + 2 entries
    
    def test_export_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog()
            log.log(action=AuditAction.QUERY)
            
            path = Path(tmpdir) / "audit.json"
            count = log.export_to_file(path, format="json")
            
            assert count == 1
            assert path.exists()
    
    def test_get_statistics(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY, repository="repo1")
        log.log(action=AuditAction.INSERT, repository="repo1")
        log.log(action=AuditAction.QUERY, repository="repo2", success=False)
        
        stats = log.get_statistics()
        assert stats["total_entries"] == 3
        assert stats["by_action"]["query"] == 2
        assert stats["success_rate"] == 2/3
    
    def test_cleanup_old_entries(self):
        log = AuditLog(retention_days=1)
        
        # Log an entry
        entry = log.log(action=AuditAction.QUERY)
        # Manually set old timestamp
        entry.timestamp = datetime.now() - timedelta(days=2)
        
        removed = log.cleanup_old_entries()
        assert removed == 1
    
    def test_clear(self):
        log = AuditLog()
        log.log(action=AuditAction.QUERY)
        log.log(action=AuditAction.QUERY)
        
        cleared = log.clear()
        assert cleared == 2
        assert len(log.get_entries()) == 0
    
    def test_max_entries(self):
        log = AuditLog(max_entries=5)
        
        for i in range(10):
            log.log(action=AuditAction.QUERY)
        
        assert len(log.get_entries()) == 5
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            
            log1 = AuditLog(storage_path=path)
            log1.log(action=AuditAction.QUERY, repository="test")
            
            log2 = AuditLog(storage_path=path)
            entries = log2.get_entries()
            assert len(entries) == 1
            assert entries[0].repository == "test"


class TestLineageNode:
    """Tests for LineageNode."""
    
    def test_creation(self):
        node = LineageNode(
            node_id="source-1",
            node_type="source",
            name="API Feed",
            created_at=datetime.now(),
        )
        assert node.node_id == "source-1"
        assert node.node_type == "source"
    
    def test_to_dict(self):
        node = LineageNode(
            node_id="source-1",
            node_type="source",
            name="API Feed",
            created_at=datetime.now(),
            metadata={"url": "http://example.com"},
        )
        d = node.to_dict()
        assert d["node_id"] == "source-1"
        assert d["metadata"]["url"] == "http://example.com"


class TestLineageEdge:
    """Tests for LineageEdge."""
    
    def test_creation(self):
        edge = LineageEdge(
            source_id="source-1",
            target_id="graph-1",
            relationship="loaded_to",
            created_at=datetime.now(),
        )
        assert edge.source_id == "source-1"
        assert edge.relationship == "loaded_to"


class TestDataLineage:
    """Tests for DataLineage."""
    
    def test_add_source(self):
        lineage = DataLineage()
        node = lineage.add_source("src-1", "API Feed")
        
        assert node.node_type == "source"
        assert lineage.get_node("src-1") is not None
    
    def test_add_graph(self):
        lineage = DataLineage()
        node = lineage.add_graph("graph-1", "Main Graph")
        
        assert node.node_type == "graph"
    
    def test_add_derived(self):
        lineage = DataLineage()
        lineage.add_source("src-1", "Source 1")
        lineage.add_source("src-2", "Source 2")
        
        derived = lineage.add_derived(
            "derived-1",
            "Combined Data",
            source_ids=["src-1", "src-2"],
        )
        
        assert derived.node_type == "derived"
    
    def test_get_ancestors(self):
        lineage = DataLineage()
        lineage.add_source("src-1", "Source 1")
        lineage.add_graph("graph-1", "Graph 1")
        lineage.add_edge("src-1", "graph-1", "loaded_to")
        
        ancestors = lineage.get_ancestors("graph-1")
        assert len(ancestors) == 1
        assert ancestors[0].node_id == "src-1"
    
    def test_get_descendants(self):
        lineage = DataLineage()
        lineage.add_source("src-1", "Source 1")
        lineage.add_graph("graph-1", "Graph 1")
        lineage.add_edge("src-1", "graph-1", "loaded_to")
        
        descendants = lineage.get_descendants("src-1")
        assert len(descendants) == 1
        assert descendants[0].node_id == "graph-1"
    
    def test_get_full_lineage(self):
        lineage = DataLineage()
        lineage.add_source("src-1", "Source")
        lineage.add_graph("graph-1", "Graph")
        lineage.add_derived("derived-1", "Derived", ["graph-1"])
        lineage.add_edge("src-1", "graph-1", "loaded_to")
        
        full = lineage.get_full_lineage("graph-1")
        assert full["node"]["node_id"] == "graph-1"
        assert len(full["ancestors"]) == 1
        assert len(full["descendants"]) == 1
    
    def test_list_sources(self):
        lineage = DataLineage()
        lineage.add_source("src-1", "Source 1")
        lineage.add_source("src-2", "Source 2")
        lineage.add_graph("graph-1", "Graph")
        
        sources = lineage.list_sources()
        assert len(sources) == 2
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lineage.json"
            
            lineage1 = DataLineage(storage_path=path)
            lineage1.add_source("src-1", "Source 1")
            
            lineage2 = DataLineage(storage_path=path)
            assert lineage2.get_node("src-1") is not None


class TestSourceHealth:
    """Tests for SourceHealth."""
    
    def test_creation(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
        )
        assert health.source_id == "src-1"
        assert health.total_loads == 0
    
    def test_success_rate(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            total_loads=10,
            successful_loads=8,
            failed_loads=2,
        )
        assert health.success_rate == 0.8
    
    def test_is_stale(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            expected_update_hours=24,
            last_success=datetime.now() - timedelta(hours=48),
        )
        assert health.is_stale
    
    def test_not_stale(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            expected_update_hours=24,
            last_success=datetime.now(),
        )
        assert not health.is_stale
    
    def test_health_status_healthy(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            total_loads=10,
            successful_loads=10,
            last_success=datetime.now(),
        )
        assert health.health_status == "healthy"
    
    def test_health_status_error(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            last_error=datetime.now(),
            error_message="Connection failed",
        )
        assert health.health_status == "error"
    
    def test_health_status_degraded(self):
        health = SourceHealth(
            source_id="src-1",
            name="API Feed",
            total_loads=10,
            successful_loads=8,
            failed_loads=2,
            last_success=datetime.now(),
        )
        assert health.health_status == "degraded"


class TestSourceHealthMonitor:
    """Tests for SourceHealthMonitor."""
    
    def test_register_source(self):
        monitor = SourceHealthMonitor()
        health = monitor.register_source("src-1", "API Feed")
        
        assert health.source_id == "src-1"
    
    def test_record_load_success(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "API Feed")
        
        health = monitor.record_load("src-1", success=True, triple_count=100)
        
        assert health.successful_loads == 1
        assert health.total_triples == 100
    
    def test_record_load_failure(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "API Feed")
        
        health = monitor.record_load(
            "src-1",
            success=False,
            error_message="Connection timeout",
        )
        
        assert health.failed_loads == 1
        assert health.error_message == "Connection timeout"
    
    def test_get_health(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "API Feed")
        
        health = monitor.get_health("src-1")
        assert health is not None
        assert health.name == "API Feed"
    
    def test_list_sources(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "Feed 1")
        monitor.register_source("src-2", "Feed 2")
        
        sources = monitor.list_sources()
        assert len(sources) == 2
    
    def test_get_unhealthy(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "Healthy Feed")
        monitor.register_source("src-2", "Error Feed")
        
        monitor.record_load("src-1", success=True)
        monitor.record_load("src-2", success=False, error_message="Error")
        
        unhealthy = monitor.get_unhealthy()
        assert len(unhealthy) == 1
        assert unhealthy[0].source_id == "src-2"
    
    def test_get_stale(self):
        monitor = SourceHealthMonitor()
        health = monitor.register_source("src-1", "Stale Feed", expected_update_hours=1)
        health.last_success = datetime.now() - timedelta(hours=2)
        
        stale = monitor.get_stale()
        assert len(stale) == 1
    
    def test_get_summary(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "Feed 1")
        monitor.register_source("src-2", "Feed 2")
        monitor.record_load("src-1", success=True)
        monitor.record_load("src-2", success=False, error_message="Error")
        
        summary = monitor.get_summary()
        assert summary["total"] == 2
        assert summary["healthy"] == 1
        assert summary["error"] == 1
    
    def test_remove_source(self):
        monitor = SourceHealthMonitor()
        monitor.register_source("src-1", "Feed")
        
        assert monitor.remove_source("src-1")
        assert monitor.get_health("src-1") is None
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "health.json"
            
            monitor1 = SourceHealthMonitor(storage_path=path)
            monitor1.register_source("src-1", "Feed")
            monitor1.record_load("src-1", success=True, triple_count=50)
            
            monitor2 = SourceHealthMonitor(storage_path=path)
            health = monitor2.get_health("src-1")
            
            assert health is not None
            assert health.total_triples == 50


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_audit_log(self):
        log = create_audit_log()
        assert isinstance(log, AuditLog)
    
    def test_create_lineage_tracker(self):
        lineage = create_lineage_tracker()
        assert isinstance(lineage, DataLineage)
    
    def test_create_health_monitor(self):
        monitor = create_health_monitor()
        assert isinstance(monitor, SourceHealthMonitor)
