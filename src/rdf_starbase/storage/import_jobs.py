"""
Import Job Tracking and Undo for RDF-StarBase.

Provides:
- Import job tracking with progress
- Per-commit rollback via WAL
- Import history
- Error logging
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class ImportStatus(Enum):
    """Status of an import job."""
    PENDING = "pending"
    STAGING = "staging"
    VALIDATING = "validating"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ImportErrorSeverity(Enum):
    """Severity of import errors."""
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class ImportError:
    """An error encountered during import."""
    severity: ImportErrorSeverity
    message: str
    line_number: Optional[int] = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "line_number": self.line_number,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImportError":
        return cls(
            severity=ImportErrorSeverity(data["severity"]),
            message=data["message"],
            line_number=data.get("line_number"),
            source=data.get("source"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )


@dataclass
class ImportProgress:
    """Progress tracking for an import job."""
    total_bytes: int = 0
    processed_bytes: int = 0
    total_triples: int = 0
    imported_triples: int = 0
    skipped_triples: int = 0
    current_phase: str = ""
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def byte_progress(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.processed_bytes / self.total_bytes) * 100
    
    @property
    def triple_progress(self) -> float:
        if self.total_triples == 0:
            return 0.0
        return (self.imported_triples / self.total_triples) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.updated_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def triples_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.imported_triples / elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "processed_bytes": self.processed_bytes,
            "total_triples": self.total_triples,
            "imported_triples": self.imported_triples,
            "skipped_triples": self.skipped_triples,
            "current_phase": self.current_phase,
            "byte_progress": self.byte_progress,
            "triple_progress": self.triple_progress,
            "elapsed_seconds": self.elapsed_seconds,
            "triples_per_second": self.triples_per_second,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class ImportJob:
    """
    Represents a single import job.
    
    Tracks the full lifecycle of an import from staging to completion.
    """
    # Identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repository: str = ""
    
    # Source info
    source_file: Optional[str] = None
    source_format: str = ""
    source_graph: Optional[str] = None
    
    # Status tracking
    status: ImportStatus = ImportStatus.PENDING
    progress: ImportProgress = field(default_factory=ImportProgress)
    errors: List[ImportError] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # WAL tracking for undo
    wal_transaction_id: Optional[str] = None
    wal_start_sequence: Optional[int] = None
    wal_end_sequence: Optional[int] = None
    
    # Stats
    triple_count: int = 0
    term_count: int = 0
    graph_count: int = 0
    
    # Metadata
    user: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def add_error(
        self,
        message: str,
        severity: ImportErrorSeverity = ImportErrorSeverity.ERROR,
        line_number: Optional[int] = None,
        source: Optional[str] = None
    ) -> None:
        """Add an error to the job."""
        self.errors.append(ImportError(
            severity=severity,
            message=message,
            line_number=line_number,
            source=source or self.source_file
        ))
    
    def has_fatal_errors(self) -> bool:
        """Check if job has any fatal errors."""
        return any(e.severity == ImportErrorSeverity.FATAL for e in self.errors)
    
    def error_count(self, severity: Optional[ImportErrorSeverity] = None) -> int:
        """Count errors, optionally filtered by severity."""
        if severity is None:
            return len(self.errors)
        return sum(1 for e in self.errors if e.severity == severity)
    
    @property
    def duration_seconds(self) -> float:
        """Get job duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def can_undo(self) -> bool:
        """Check if job can be undone."""
        return (
            self.status == ImportStatus.COMPLETED
            and self.wal_transaction_id is not None
            and self.wal_start_sequence is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "repository": self.repository,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "source_graph": self.source_graph,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "errors": [e.to_dict() for e in self.errors],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "wal_transaction_id": self.wal_transaction_id,
            "wal_start_sequence": self.wal_start_sequence,
            "wal_end_sequence": self.wal_end_sequence,
            "triple_count": self.triple_count,
            "term_count": self.term_count,
            "graph_count": self.graph_count,
            "duration_seconds": self.duration_seconds,
            "can_undo": self.can_undo,
            "user": self.user,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImportJob":
        job = cls(
            job_id=data.get("job_id", str(uuid.uuid4())),
            repository=data.get("repository", ""),
            source_file=data.get("source_file"),
            source_format=data.get("source_format", ""),
            source_graph=data.get("source_graph"),
            status=ImportStatus(data.get("status", "pending")),
            wal_transaction_id=data.get("wal_transaction_id"),
            wal_start_sequence=data.get("wal_start_sequence"),
            wal_end_sequence=data.get("wal_end_sequence"),
            triple_count=data.get("triple_count", 0),
            term_count=data.get("term_count", 0),
            graph_count=data.get("graph_count", 0),
            user=data.get("user"),
            tags=data.get("tags", [])
        )
        
        # Parse timestamps
        if data.get("created_at"):
            job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])
        
        # Parse errors
        if "errors" in data:
            job.errors = [ImportError.from_dict(e) for e in data["errors"]]
        
        return job


class ImportJobTracker:
    """
    Tracks import jobs for a repository.
    
    Features:
    - Job creation and status updates
    - Progress tracking
    - Error logging
    - Job persistence
    - History queries
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._jobs: Dict[str, ImportJob] = {}
        self._lock = threading.RLock()
        self._callbacks: Dict[str, Callable[[ImportJob], None]] = {}
        
        if self.storage_path:
            self._load_jobs()
    
    def _load_jobs(self) -> None:
        """Load jobs from storage."""
        if not self.storage_path:
            return
        
        jobs_file = self.storage_path / "import_jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for job_data in data.get("jobs", []):
                    job = ImportJob.from_dict(job_data)
                    self._jobs[job.job_id] = job
                logger.debug(f"Loaded {len(self._jobs)} import jobs")
            except Exception as e:
                logger.warning(f"Error loading import jobs: {e}")
    
    def _save_jobs(self) -> None:
        """Save jobs to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        jobs_file = self.storage_path / "import_jobs.json"
        
        try:
            data = {
                "jobs": [j.to_dict() for j in self._jobs.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(jobs_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving import jobs: {e}")
    
    def create_job(
        self,
        repository: str,
        source_file: Optional[str] = None,
        source_format: str = "",
        source_graph: Optional[str] = None,
        user: Optional[str] = None,
        tags: List[str] = None
    ) -> ImportJob:
        """Create a new import job."""
        with self._lock:
            job = ImportJob(
                repository=repository,
                source_file=source_file,
                source_format=source_format,
                source_graph=source_graph,
                user=user,
                tags=tags or []
            )
            self._jobs[job.job_id] = job
            self._save_jobs()
            self._notify_callbacks(job)
            return job
    
    def get_job(self, job_id: str) -> Optional[ImportJob]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_status(
        self,
        job_id: str,
        status: ImportStatus,
        **kwargs
    ) -> Optional[ImportJob]:
        """Update job status."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.status = status
            
            if status == ImportStatus.STAGING and not job.started_at:
                job.started_at = datetime.now()
                job.progress.started_at = datetime.now()
            
            if status in (ImportStatus.COMPLETED, ImportStatus.FAILED, ImportStatus.ROLLED_BACK):
                job.completed_at = datetime.now()
                job.progress.updated_at = datetime.now()
            
            # Apply additional updates
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            self._save_jobs()
            self._notify_callbacks(job)
            return job
    
    def update_progress(
        self,
        job_id: str,
        processed_bytes: Optional[int] = None,
        imported_triples: Optional[int] = None,
        skipped_triples: Optional[int] = None,
        current_phase: Optional[str] = None
    ) -> Optional[ImportJob]:
        """Update job progress."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            if processed_bytes is not None:
                job.progress.processed_bytes = processed_bytes
            if imported_triples is not None:
                job.progress.imported_triples = imported_triples
            if skipped_triples is not None:
                job.progress.skipped_triples = skipped_triples
            if current_phase is not None:
                job.progress.current_phase = current_phase
            
            job.progress.updated_at = datetime.now()
            
            # Don't save on every progress update (too expensive)
            self._notify_callbacks(job)
            return job
    
    def add_error(
        self,
        job_id: str,
        message: str,
        severity: ImportErrorSeverity = ImportErrorSeverity.ERROR,
        line_number: Optional[int] = None,
        source: Optional[str] = None
    ) -> Optional[ImportJob]:
        """Add an error to a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.add_error(message, severity, line_number, source)
            self._save_jobs()
            return job
    
    def set_wal_info(
        self,
        job_id: str,
        transaction_id: str,
        start_sequence: int,
        end_sequence: Optional[int] = None
    ) -> Optional[ImportJob]:
        """Set WAL tracking info for undo capability."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.wal_transaction_id = transaction_id
            job.wal_start_sequence = start_sequence
            job.wal_end_sequence = end_sequence
            self._save_jobs()
            return job
    
    def complete_job(
        self,
        job_id: str,
        triple_count: int = 0,
        term_count: int = 0,
        graph_count: int = 0
    ) -> Optional[ImportJob]:
        """Mark job as completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.status = ImportStatus.COMPLETED
            job.completed_at = datetime.now()
            job.triple_count = triple_count
            job.term_count = term_count
            job.graph_count = graph_count
            job.progress.updated_at = datetime.now()
            
            self._save_jobs()
            self._notify_callbacks(job)
            return job
    
    def fail_job(self, job_id: str, reason: str) -> Optional[ImportJob]:
        """Mark job as failed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.status = ImportStatus.FAILED
            job.completed_at = datetime.now()
            job.add_error(reason, ImportErrorSeverity.FATAL)
            
            self._save_jobs()
            self._notify_callbacks(job)
            return job
    
    def list_jobs(
        self,
        repository: Optional[str] = None,
        status: Optional[ImportStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ImportJob]:
        """List jobs with optional filtering."""
        with self._lock:
            jobs = list(self._jobs.values())
            
            if repository:
                jobs = [j for j in jobs if j.repository == repository]
            
            if status:
                jobs = [j for j in jobs if j.status == status]
            
            # Sort by created_at descending
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            return jobs[offset:offset + limit]
    
    def get_active_jobs(self, repository: Optional[str] = None) -> List[ImportJob]:
        """Get currently active jobs."""
        active_statuses = {
            ImportStatus.PENDING,
            ImportStatus.STAGING,
            ImportStatus.VALIDATING,
            ImportStatus.COMMITTING
        }
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.status in active_statuses]
            if repository:
                jobs = [j for j in jobs if j.repository == repository]
            return jobs
    
    def get_undoable_jobs(self, repository: str) -> List[ImportJob]:
        """Get jobs that can be undone."""
        with self._lock:
            jobs = [
                j for j in self._jobs.values()
                if j.repository == repository and j.can_undo
            ]
            jobs.sort(key=lambda j: j.completed_at or j.created_at, reverse=True)
            return jobs
    
    def mark_rolled_back(self, job_id: str) -> Optional[ImportJob]:
        """Mark job as rolled back."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.status = ImportStatus.ROLLED_BACK
            self._save_jobs()
            self._notify_callbacks(job)
            return job
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._save_jobs()
                return True
            return False
    
    def register_callback(
        self,
        callback_id: str,
        callback: Callable[[ImportJob], None]
    ) -> None:
        """Register a callback for job updates."""
        self._callbacks[callback_id] = callback
    
    def unregister_callback(self, callback_id: str) -> None:
        """Unregister a callback."""
        self._callbacks.pop(callback_id, None)
    
    def _notify_callbacks(self, job: ImportJob) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks.values():
            try:
                callback(job)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than specified days."""
        with self._lock:
            cutoff = datetime.now()
            from datetime import timedelta
            cutoff = cutoff - timedelta(days=days)
            
            old_jobs = [
                j.job_id for j in self._jobs.values()
                if j.created_at < cutoff and j.status in (
                    ImportStatus.COMPLETED,
                    ImportStatus.FAILED,
                    ImportStatus.ROLLED_BACK
                )
            ]
            
            for job_id in old_jobs:
                del self._jobs[job_id]
            
            if old_jobs:
                self._save_jobs()
            
            return len(old_jobs)


class ImportUndoManager:
    """
    Manages undo operations for imports.
    
    Uses WAL transaction IDs to roll back imports.
    """
    
    def __init__(
        self,
        job_tracker: ImportJobTracker,
        wal_provider: Optional[Callable[[str], Any]] = None
    ):
        self.job_tracker = job_tracker
        self._wal_provider = wal_provider
    
    def can_undo(self, job_id: str) -> bool:
        """Check if a job can be undone."""
        job = self.job_tracker.get_job(job_id)
        return job.can_undo if job else False
    
    def get_undo_preview(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get preview of what undo would do."""
        job = self.job_tracker.get_job(job_id)
        if not job or not job.can_undo:
            return None
        
        return {
            "job_id": job.job_id,
            "source_file": job.source_file,
            "triple_count": job.triple_count,
            "imported_at": job.completed_at.isoformat() if job.completed_at else None,
            "wal_transaction_id": job.wal_transaction_id,
            "operations_to_rollback": job.wal_end_sequence - job.wal_start_sequence + 1
                if job.wal_end_sequence else "unknown"
        }
    
    def undo_import(self, job_id: str, store: Any = None) -> Dict[str, Any]:
        """
        Undo an import by rolling back WAL entries.
        
        Args:
            job_id: The job to undo
            store: Optional store instance (for direct rollback)
        
        Returns:
            Result dictionary with status and details
        """
        job = self.job_tracker.get_job(job_id)
        
        if not job:
            return {
                "success": False,
                "error": f"Job not found: {job_id}"
            }
        
        if not job.can_undo:
            return {
                "success": False,
                "error": "Job cannot be undone (missing WAL info or not completed)"
            }
        
        try:
            # Get WAL instance
            wal = None
            if self._wal_provider and job.repository:
                wal = self._wal_provider(job.repository)
            elif store and hasattr(store, '_wal'):
                wal = store._wal
            
            if wal is None:
                # No WAL available - can't actually rollback
                # Mark as rolled back anyway (soft undo)
                self.job_tracker.mark_rolled_back(job_id)
                return {
                    "success": True,
                    "soft_rollback": True,
                    "message": "Job marked as rolled back (no WAL available for hard rollback)",
                    "triples_affected": job.triple_count
                }
            
            # Perform actual rollback using WAL
            rollback_count = 0
            
            if hasattr(wal, 'rollback_to_sequence'):
                # Ideal: WAL supports sequence-based rollback
                rollback_count = wal.rollback_to_sequence(job.wal_start_sequence - 1)
            elif hasattr(wal, 'rollback_transaction'):
                # Alternative: transaction-based rollback
                wal.rollback_transaction(job.wal_transaction_id)
                rollback_count = job.triple_count
            else:
                # Fallback: read WAL and apply inverse operations
                rollback_count = self._manual_rollback(wal, job, store)
            
            # Mark job as rolled back
            self.job_tracker.mark_rolled_back(job_id)
            
            return {
                "success": True,
                "job_id": job_id,
                "triples_rolled_back": rollback_count,
                "wal_sequence_restored": job.wal_start_sequence - 1 if job.wal_start_sequence else None
            }
            
        except Exception as e:
            logger.error(f"Undo failed for job {job_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _manual_rollback(
        self,
        wal: Any,
        job: ImportJob,
        store: Any
    ) -> int:
        """Perform manual rollback by reading WAL entries."""
        if not store:
            raise ValueError("Store required for manual rollback")
        
        rollback_count = 0
        
        # Read WAL entries for this job
        if hasattr(wal, 'read_entries'):
            entries = wal.read_entries(
                from_sequence=job.wal_start_sequence,
                to_sequence=job.wal_end_sequence
            )
            
            # Apply inverse operations in reverse order
            for entry in reversed(list(entries)):
                if hasattr(entry, 'entry_type'):
                    if entry.entry_type.value == 'insert':
                        # Undo insert = delete
                        if hasattr(store, 'remove_triple'):
                            store.remove_triple(
                                entry.payload.subject,
                                entry.payload.predicate,
                                entry.payload.object
                            )
                            rollback_count += 1
                    elif entry.entry_type.value == 'delete':
                        # Undo delete = insert
                        if hasattr(store, 'add_triple'):
                            store.add_triple(
                                entry.payload.subject,
                                entry.payload.predicate,
                                entry.payload.object
                            )
                            rollback_count += 1
        
        return rollback_count
    
    def undo_last_import(
        self,
        repository: str,
        store: Any = None
    ) -> Dict[str, Any]:
        """Undo the most recent import for a repository."""
        undoable = self.job_tracker.get_undoable_jobs(repository)
        
        if not undoable:
            return {
                "success": False,
                "error": "No undoable imports found"
            }
        
        return self.undo_import(undoable[0].job_id, store)


# Convenience functions
def create_import_tracker(storage_path: Optional[Path] = None) -> ImportJobTracker:
    """Create an import job tracker."""
    return ImportJobTracker(storage_path)


def create_undo_manager(
    tracker: ImportJobTracker,
    wal_provider: Optional[Callable[[str], Any]] = None
) -> ImportUndoManager:
    """Create an import undo manager."""
    return ImportUndoManager(tracker, wal_provider)
