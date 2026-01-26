"""Tests for import job tracking and undo."""
import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from rdf_starbase.storage.import_jobs import (
    ImportStatus,
    ImportErrorSeverity,
    ImportError,
    ImportProgress,
    ImportJob,
    ImportJobTracker,
    ImportUndoManager,
    create_import_tracker,
    create_undo_manager,
)


# ========== ImportStatus Tests ==========

class TestImportStatus:
    def test_status_values(self):
        assert ImportStatus.PENDING.value == "pending"
        assert ImportStatus.COMPLETED.value == "completed"
        assert ImportStatus.ROLLED_BACK.value == "rolled_back"


# ========== ImportError Tests ==========

class TestImportError:
    def test_creation(self):
        err = ImportError(
            severity=ImportErrorSeverity.ERROR,
            message="Invalid IRI"
        )
        assert err.severity == ImportErrorSeverity.ERROR
        assert err.message == "Invalid IRI"
    
    def test_with_line_number(self):
        err = ImportError(
            severity=ImportErrorSeverity.WARNING,
            message="Unknown prefix",
            line_number=42,
            source="data.ttl"
        )
        assert err.line_number == 42
        assert err.source == "data.ttl"
    
    def test_to_dict(self):
        err = ImportError(
            severity=ImportErrorSeverity.FATAL,
            message="Parse error"
        )
        d = err.to_dict()
        assert d["severity"] == "fatal"
        assert d["message"] == "Parse error"
    
    def test_from_dict(self):
        data = {
            "severity": "warning",
            "message": "Duplicate triple",
            "line_number": 100,
            "timestamp": datetime.now().isoformat()
        }
        err = ImportError.from_dict(data)
        assert err.severity == ImportErrorSeverity.WARNING
        assert err.line_number == 100


# ========== ImportProgress Tests ==========

class TestImportProgress:
    def test_defaults(self):
        progress = ImportProgress()
        assert progress.total_bytes == 0
        assert progress.byte_progress == 0.0
    
    def test_byte_progress(self):
        progress = ImportProgress(total_bytes=1000, processed_bytes=500)
        assert progress.byte_progress == 50.0
    
    def test_triple_progress(self):
        progress = ImportProgress(total_triples=100, imported_triples=75)
        assert progress.triple_progress == 75.0
    
    def test_elapsed_time(self):
        progress = ImportProgress(started_at=datetime.now())
        time.sleep(0.1)
        assert progress.elapsed_seconds >= 0.1
    
    def test_triples_per_second(self):
        started = datetime.now() - timedelta(seconds=10)
        progress = ImportProgress(
            started_at=started,
            updated_at=datetime.now(),
            imported_triples=1000
        )
        assert progress.triples_per_second == pytest.approx(100, rel=0.1)
    
    def test_to_dict(self):
        progress = ImportProgress(
            total_bytes=1000,
            processed_bytes=500,
            current_phase="parsing"
        )
        d = progress.to_dict()
        assert d["total_bytes"] == 1000
        assert d["current_phase"] == "parsing"


# ========== ImportJob Tests ==========

class TestImportJob:
    def test_creation(self):
        job = ImportJob(repository="test-repo")
        assert job.job_id is not None
        assert job.repository == "test-repo"
        assert job.status == ImportStatus.PENDING
    
    def test_add_error(self):
        job = ImportJob()
        job.add_error("Something went wrong")
        assert len(job.errors) == 1
        assert job.errors[0].severity == ImportErrorSeverity.ERROR
    
    def test_has_fatal_errors(self):
        job = ImportJob()
        job.add_error("Warning", ImportErrorSeverity.WARNING)
        assert not job.has_fatal_errors()
        
        job.add_error("Fatal", ImportErrorSeverity.FATAL)
        assert job.has_fatal_errors()
    
    def test_error_count(self):
        job = ImportJob()
        job.add_error("w1", ImportErrorSeverity.WARNING)
        job.add_error("w2", ImportErrorSeverity.WARNING)
        job.add_error("e1", ImportErrorSeverity.ERROR)
        
        assert job.error_count() == 3
        assert job.error_count(ImportErrorSeverity.WARNING) == 2
        assert job.error_count(ImportErrorSeverity.ERROR) == 1
    
    def test_duration(self):
        job = ImportJob()
        job.started_at = datetime.now() - timedelta(seconds=5)
        job.completed_at = datetime.now()
        
        assert job.duration_seconds >= 5.0
    
    def test_can_undo(self):
        job = ImportJob()
        assert not job.can_undo
        
        job.status = ImportStatus.COMPLETED
        job.wal_transaction_id = "tx-123"
        job.wal_start_sequence = 100
        assert job.can_undo
    
    def test_to_dict(self):
        job = ImportJob(
            repository="test",
            source_file="data.ttl",
            triple_count=1000
        )
        d = job.to_dict()
        assert d["repository"] == "test"
        assert d["source_file"] == "data.ttl"
        assert d["triple_count"] == 1000
    
    def test_from_dict(self):
        data = {
            "job_id": "job-123",
            "repository": "my-repo",
            "status": "completed",
            "triple_count": 500,
            "tags": ["test", "sample"]
        }
        job = ImportJob.from_dict(data)
        assert job.job_id == "job-123"
        assert job.status == ImportStatus.COMPLETED
        assert "test" in job.tags


# ========== ImportJobTracker Tests ==========

class TestImportJobTracker:
    def test_create_job(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("test-repo", source_file="data.ttl")
        
        assert job.repository == "test-repo"
        assert job.source_file == "data.ttl"
    
    def test_get_job(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        retrieved = tracker.get_job(job.job_id)
        assert retrieved.job_id == job.job_id
    
    def test_get_nonexistent(self):
        tracker = ImportJobTracker()
        assert tracker.get_job("nonexistent") is None
    
    def test_update_status(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.update_status(job.job_id, ImportStatus.STAGING)
        assert tracker.get_job(job.job_id).status == ImportStatus.STAGING
        assert tracker.get_job(job.job_id).started_at is not None
    
    def test_update_progress(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.update_progress(
            job.job_id,
            processed_bytes=500,
            imported_triples=100,
            current_phase="parsing"
        )
        
        updated = tracker.get_job(job.job_id)
        assert updated.progress.processed_bytes == 500
        assert updated.progress.imported_triples == 100
    
    def test_add_error(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.add_error(job.job_id, "Test error")
        updated = tracker.get_job(job.job_id)
        assert len(updated.errors) == 1
    
    def test_set_wal_info(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.set_wal_info(job.job_id, "tx-123", 100, 150)
        updated = tracker.get_job(job.job_id)
        assert updated.wal_transaction_id == "tx-123"
        assert updated.wal_start_sequence == 100
    
    def test_complete_job(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.complete_job(job.job_id, triple_count=1000)
        updated = tracker.get_job(job.job_id)
        
        assert updated.status == ImportStatus.COMPLETED
        assert updated.triple_count == 1000
        assert updated.completed_at is not None
    
    def test_fail_job(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        tracker.fail_job(job.job_id, "Parse error")
        updated = tracker.get_job(job.job_id)
        
        assert updated.status == ImportStatus.FAILED
        assert updated.has_fatal_errors()
    
    def test_list_jobs(self):
        tracker = ImportJobTracker()
        tracker.create_job("repo1")
        tracker.create_job("repo2")
        tracker.create_job("repo1")
        
        all_jobs = tracker.list_jobs()
        assert len(all_jobs) == 3
        
        repo1_jobs = tracker.list_jobs(repository="repo1")
        assert len(repo1_jobs) == 2
    
    def test_get_active_jobs(self):
        tracker = ImportJobTracker()
        j1 = tracker.create_job("repo")
        j2 = tracker.create_job("repo")
        tracker.complete_job(j2.job_id)
        
        active = tracker.get_active_jobs()
        assert len(active) == 1
        assert active[0].job_id == j1.job_id
    
    def test_get_undoable_jobs(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        tracker.set_wal_info(job.job_id, "tx-1", 100)
        tracker.complete_job(job.job_id)
        
        undoable = tracker.get_undoable_jobs("repo")
        assert len(undoable) == 1
    
    def test_mark_rolled_back(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        tracker.complete_job(job.job_id)
        
        tracker.mark_rolled_back(job.job_id)
        assert tracker.get_job(job.job_id).status == ImportStatus.ROLLED_BACK
    
    def test_delete_job(self):
        tracker = ImportJobTracker()
        job = tracker.create_job("repo")
        
        assert tracker.delete_job(job.job_id)
        assert tracker.get_job(job.job_id) is None
    
    def test_persistence(self, tmp_path):
        tracker = ImportJobTracker(tmp_path)
        job = tracker.create_job("repo", source_file="data.ttl")
        tracker.complete_job(job.job_id, triple_count=500)
        
        # Create new tracker - should load from file
        tracker2 = ImportJobTracker(tmp_path)
        loaded = tracker2.get_job(job.job_id)
        
        assert loaded is not None
        assert loaded.triple_count == 500
    
    def test_callbacks(self):
        tracker = ImportJobTracker()
        updates = []
        
        def callback(job):
            updates.append(job.status)
        
        tracker.register_callback("test", callback)
        job = tracker.create_job("repo")
        tracker.update_status(job.job_id, ImportStatus.STAGING)
        tracker.complete_job(job.job_id)
        
        assert ImportStatus.PENDING in updates
        assert ImportStatus.STAGING in updates
        assert ImportStatus.COMPLETED in updates
    
    def test_cleanup_old_jobs(self, tmp_path):
        tracker = ImportJobTracker(tmp_path)
        
        # Create old job
        job = tracker.create_job("repo")
        job.created_at = datetime.now() - timedelta(days=60)
        tracker.complete_job(job.job_id)
        
        # Create recent job
        recent = tracker.create_job("repo")
        tracker.complete_job(recent.job_id)
        
        cleaned = tracker.cleanup_old_jobs(days=30)
        assert cleaned == 1
        assert tracker.get_job(recent.job_id) is not None


# ========== ImportUndoManager Tests ==========

class TestImportUndoManager:
    def test_can_undo(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        job = tracker.create_job("repo")
        assert not manager.can_undo(job.job_id)
        
        tracker.set_wal_info(job.job_id, "tx-1", 100)
        tracker.complete_job(job.job_id)
        assert manager.can_undo(job.job_id)
    
    def test_get_undo_preview(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        job = tracker.create_job("repo", source_file="data.ttl")
        tracker.set_wal_info(job.job_id, "tx-1", 100, 150)
        tracker.complete_job(job.job_id, triple_count=1000)
        
        preview = manager.get_undo_preview(job.job_id)
        assert preview["triple_count"] == 1000
        assert preview["wal_transaction_id"] == "tx-1"
    
    def test_undo_nonexistent(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        result = manager.undo_import("nonexistent")
        assert not result["success"]
        assert "not found" in result["error"]
    
    def test_undo_not_undoable(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        job = tracker.create_job("repo")
        tracker.complete_job(job.job_id)
        
        result = manager.undo_import(job.job_id)
        assert not result["success"]
        assert "cannot be undone" in result["error"]
    
    def test_soft_rollback(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        job = tracker.create_job("repo")
        tracker.set_wal_info(job.job_id, "tx-1", 100)
        tracker.complete_job(job.job_id, triple_count=500)
        
        result = manager.undo_import(job.job_id)
        assert result["success"]
        assert result.get("soft_rollback")
        assert tracker.get_job(job.job_id).status == ImportStatus.ROLLED_BACK
    
    def test_undo_last_import(self):
        tracker = ImportJobTracker()
        manager = ImportUndoManager(tracker)
        
        result = manager.undo_last_import("empty-repo")
        assert not result["success"]
        assert "No undoable" in result["error"]


# ========== Convenience Functions Tests ==========

class TestConvenienceFunctions:
    def test_create_import_tracker(self):
        tracker = create_import_tracker()
        assert isinstance(tracker, ImportJobTracker)
    
    def test_create_undo_manager(self):
        tracker = ImportJobTracker()
        manager = create_undo_manager(tracker)
        assert isinstance(manager, ImportUndoManager)
