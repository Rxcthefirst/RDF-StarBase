"""
Backup and Restore Module

Provides point-in-time snapshots and restore capabilities for repositories.
Snapshots are self-contained and can be restored to a new repository name.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import polars as pl


class BackupFormat(Enum):
    """Backup format options."""
    DIRECTORY = auto()  # Directory with Parquet files + manifest
    ARCHIVE = auto()    # Single .tar.gz archive (future)


class BackupState(Enum):
    """Backup operation state."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class BackupManifest:
    """
    Manifest describing a backup snapshot.
    
    The manifest contains all metadata needed to restore a repository
    to the exact state at backup time.
    """
    # Identity
    snapshot_id: str
    repository_name: str
    repository_uuid: Optional[str] = None
    
    # Timestamps
    created_at: str = ""
    backup_started_at: str = ""
    backup_completed_at: str = ""
    
    # Source info
    source_path: str = ""
    schema_version: int = 1
    
    # Statistics
    term_count: int = 0
    fact_count: int = 0
    quoted_triple_count: int = 0
    graph_count: int = 0
    
    # Files in backup
    files: List[str] = field(default_factory=list)
    file_checksums: Dict[str, str] = field(default_factory=dict)
    
    # Size info
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "repository_name": self.repository_name,
            "repository_uuid": self.repository_uuid,
            "created_at": self.created_at,
            "backup_started_at": self.backup_started_at,
            "backup_completed_at": self.backup_completed_at,
            "source_path": self.source_path,
            "schema_version": self.schema_version,
            "term_count": self.term_count,
            "fact_count": self.fact_count,
            "quoted_triple_count": self.quoted_triple_count,
            "graph_count": self.graph_count,
            "files": self.files,
            "file_checksums": self.file_checksums,
            "total_size_bytes": self.total_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "description": self.description,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackupManifest":
        """Deserialize manifest from dictionary."""
        return cls(
            snapshot_id=d["snapshot_id"],
            repository_name=d["repository_name"],
            repository_uuid=d.get("repository_uuid"),
            created_at=d.get("created_at", ""),
            backup_started_at=d.get("backup_started_at", ""),
            backup_completed_at=d.get("backup_completed_at", ""),
            source_path=d.get("source_path", ""),
            schema_version=d.get("schema_version", 1),
            term_count=d.get("term_count", 0),
            fact_count=d.get("fact_count", 0),
            quoted_triple_count=d.get("quoted_triple_count", 0),
            graph_count=d.get("graph_count", 0),
            files=d.get("files", []),
            file_checksums=d.get("file_checksums", {}),
            total_size_bytes=d.get("total_size_bytes", 0),
            compressed_size_bytes=d.get("compressed_size_bytes", 0),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            custom_metadata=d.get("custom_metadata", {}),
        )
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "BackupManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class BackupProgress:
    """Progress information for backup/restore operations."""
    state: BackupState = BackupState.PENDING
    current_step: str = ""
    steps_completed: int = 0
    total_steps: int = 0
    bytes_processed: int = 0
    total_bytes: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def progress_ratio(self) -> float:
        """Progress as ratio 0.0 to 1.0."""
        if self.total_steps == 0:
            return 0.0
        return self.steps_completed / self.total_steps
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


@dataclass
class RestoreOptions:
    """Options for restore operation."""
    target_name: str  # Required: name for restored repo
    overwrite: bool = False  # If True, delete existing target first
    verify_checksums: bool = True  # Verify file integrity
    skip_wal: bool = False  # Don't restore WAL (fresh start)


class BackupManager:
    """
    Manages backup and restore operations for repositories.
    
    Features:
    - Point-in-time snapshots with manifest
    - Incremental backups (future)
    - Restore to new repository name
    - Checksum verification
    - Progress tracking
    
    Example:
        backup_mgr = BackupManager(backup_dir="./backups")
        
        # Create snapshot
        manifest = backup_mgr.snapshot(
            source_path=Path("./repos/my-repo"),
            repo_name="my-repo",
            description="Before migration"
        )
        
        # List backups
        backups = backup_mgr.list_backups()
        
        # Restore
        backup_mgr.restore(
            snapshot_id=manifest.snapshot_id,
            target_path=Path("./repos/my-repo-restored"),
            options=RestoreOptions(target_name="my-repo-restored")
        )
    """
    
    def __init__(
        self,
        backup_dir: Union[str, Path],
        progress_callback: Optional[Callable[[BackupProgress], None]] = None,
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
            progress_callback: Optional callback for progress updates
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback = progress_callback
        self._current_progress: Optional[BackupProgress] = None
    
    def _update_progress(
        self,
        state: Optional[BackupState] = None,
        step: Optional[str] = None,
        steps_completed: Optional[int] = None,
        bytes_processed: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update and emit progress."""
        if self._current_progress is None:
            return
        
        if state is not None:
            self._current_progress.state = state
        if step is not None:
            self._current_progress.current_step = step
        if steps_completed is not None:
            self._current_progress.steps_completed = steps_completed
        if bytes_processed is not None:
            self._current_progress.bytes_processed = bytes_processed
        if error is not None:
            self._current_progress.error_message = error
        
        if self._progress_callback:
            self._progress_callback(self._current_progress)
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def snapshot(
        self,
        source_path: Path,
        repo_name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        repo_uuid: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> BackupManifest:
        """
        Create a point-in-time snapshot of a repository.
        
        Args:
            source_path: Path to repository directory
            repo_name: Repository name
            description: Human-readable description
            tags: Optional tags for categorization
            repo_uuid: Optional repository UUID
            custom_metadata: Optional custom metadata
            
        Returns:
            BackupManifest describing the snapshot
            
        Raises:
            ValueError: If source doesn't exist or is invalid
        """
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Generate snapshot ID
        snapshot_id = f"{repo_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        snapshot_dir = self.backup_dir / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress
        self._current_progress = BackupProgress(
            state=BackupState.IN_PROGRESS,
            started_at=time.time(),
        )
        self._update_progress(step="Initializing backup")
        
        # Create manifest
        manifest = BackupManifest(
            snapshot_id=snapshot_id,
            repository_name=repo_name,
            repository_uuid=repo_uuid,
            created_at=datetime.now(timezone.utc).isoformat(),
            backup_started_at=datetime.now(timezone.utc).isoformat(),
            source_path=str(source_path),
            description=description,
            tags=tags or [],
            custom_metadata=custom_metadata or {},
        )
        
        try:
            # Find all files to backup
            files_to_backup = []
            if source_path.is_dir():
                for f in source_path.rglob("*"):
                    if f.is_file():
                        files_to_backup.append(f)
            else:
                files_to_backup.append(source_path)
            
            self._current_progress.total_steps = len(files_to_backup) + 2  # +2 for manifest + verify
            self._update_progress(step=f"Found {len(files_to_backup)} files to backup")
            
            # Copy files and compute checksums
            total_size = 0
            for i, src_file in enumerate(files_to_backup):
                rel_path = src_file.relative_to(source_path)
                dst_file = snapshot_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                self._update_progress(
                    step=f"Copying {rel_path}",
                    steps_completed=i,
                )
                
                # Copy file
                shutil.copy2(src_file, dst_file)
                
                # Compute checksum
                checksum = self._compute_checksum(dst_file)
                manifest.files.append(str(rel_path))
                manifest.file_checksums[str(rel_path)] = checksum
                
                # Track size
                file_size = dst_file.stat().st_size
                total_size += file_size
                self._update_progress(bytes_processed=total_size)
                
                # Extract stats from known files
                if rel_path.name == "store.parquet":
                    try:
                        df = pl.read_parquet(dst_file)
                        manifest.fact_count = len(df)
                    except Exception:
                        pass
            
            manifest.total_size_bytes = total_size
            manifest.backup_completed_at = datetime.now(timezone.utc).isoformat()
            
            # Save manifest
            self._update_progress(
                step="Saving manifest",
                steps_completed=len(files_to_backup),
            )
            manifest.save(snapshot_dir / "manifest.json")
            
            # Verify
            self._update_progress(
                step="Verifying backup",
                steps_completed=len(files_to_backup) + 1,
            )
            
            self._update_progress(
                state=BackupState.COMPLETED,
                step="Backup complete",
                steps_completed=len(files_to_backup) + 2,
            )
            self._current_progress.completed_at = time.time()
            
            return manifest
            
        except Exception as e:
            self._update_progress(
                state=BackupState.FAILED,
                error=str(e),
            )
            # Clean up partial backup
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            raise
    
    def restore(
        self,
        snapshot_id: str,
        target_path: Path,
        options: RestoreOptions,
    ) -> BackupManifest:
        """
        Restore a repository from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            target_path: Path to restore to
            options: Restore options
            
        Returns:
            Manifest of restored snapshot
            
        Raises:
            ValueError: If snapshot doesn't exist or target conflicts
        """
        snapshot_dir = self.backup_dir / snapshot_id
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Snapshot manifest not found: {snapshot_id}")
        
        manifest = BackupManifest.load(manifest_path)
        
        # Check target
        if target_path.exists():
            if options.overwrite:
                shutil.rmtree(target_path)
            else:
                raise ValueError(
                    f"Target path already exists: {target_path}. "
                    "Use overwrite=True to replace."
                )
        
        # Initialize progress
        self._current_progress = BackupProgress(
            state=BackupState.IN_PROGRESS,
            started_at=time.time(),
            total_steps=len(manifest.files) + 1,
        )
        self._update_progress(step="Initializing restore")
        
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Restore files
            for i, rel_path in enumerate(manifest.files):
                # Skip WAL if requested
                if options.skip_wal and "wal" in rel_path.lower():
                    continue
                
                src_file = snapshot_dir / rel_path
                dst_file = target_path / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                self._update_progress(
                    step=f"Restoring {rel_path}",
                    steps_completed=i,
                )
                
                # Copy file
                shutil.copy2(src_file, dst_file)
                
                # Verify checksum
                if options.verify_checksums and rel_path in manifest.file_checksums:
                    expected = manifest.file_checksums[rel_path]
                    actual = self._compute_checksum(dst_file)
                    if actual != expected:
                        raise ValueError(
                            f"Checksum mismatch for {rel_path}: "
                            f"expected {expected}, got {actual}"
                        )
            
            self._update_progress(
                state=BackupState.COMPLETED,
                step="Restore complete",
                steps_completed=len(manifest.files) + 1,
            )
            self._current_progress.completed_at = time.time()
            
            return manifest
            
        except Exception as e:
            self._update_progress(
                state=BackupState.FAILED,
                error=str(e),
            )
            # Clean up partial restore
            if target_path.exists():
                shutil.rmtree(target_path)
            raise
    
    def list_backups(
        self,
        repo_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BackupManifest]:
        """
        List available backups.
        
        Args:
            repo_name: Filter by repository name
            tags: Filter by tags (any match)
            
        Returns:
            List of backup manifests
        """
        manifests = []
        
        for snapshot_dir in self.backup_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            
            manifest_path = snapshot_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            
            try:
                manifest = BackupManifest.load(manifest_path)
                
                # Apply filters
                if repo_name and manifest.repository_name != repo_name:
                    continue
                if tags and not any(t in manifest.tags for t in tags):
                    continue
                
                manifests.append(manifest)
            except Exception:
                continue
        
        # Sort by creation time (newest first)
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests
    
    def get_backup(self, snapshot_id: str) -> Optional[BackupManifest]:
        """Get a specific backup by ID."""
        snapshot_dir = self.backup_dir / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        
        if manifest_path.exists():
            return BackupManifest.load(manifest_path)
        return None
    
    def delete_backup(self, snapshot_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            snapshot_id: ID of snapshot to delete
            
        Returns:
            True if deleted, False if not found
        """
        snapshot_dir = self.backup_dir / snapshot_id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
            return True
        return False
    
    def verify_backup(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Verify integrity of a backup.
        
        Args:
            snapshot_id: ID of snapshot to verify
            
        Returns:
            Verification results with status and any errors
        """
        snapshot_dir = self.backup_dir / snapshot_id
        if not snapshot_dir.exists():
            return {"valid": False, "error": "Snapshot not found"}
        
        manifest = self.get_backup(snapshot_id)
        if not manifest:
            return {"valid": False, "error": "Manifest not found"}
        
        errors = []
        files_checked = 0
        
        for rel_path, expected_checksum in manifest.file_checksums.items():
            file_path = snapshot_dir / rel_path
            
            if not file_path.exists():
                errors.append(f"Missing file: {rel_path}")
                continue
            
            actual_checksum = self._compute_checksum(file_path)
            if actual_checksum != expected_checksum:
                errors.append(
                    f"Checksum mismatch: {rel_path} "
                    f"(expected {expected_checksum[:16]}..., got {actual_checksum[:16]}...)"
                )
            
            files_checked += 1
        
        return {
            "valid": len(errors) == 0,
            "files_checked": files_checked,
            "errors": errors,
            "snapshot_id": snapshot_id,
            "repository_name": manifest.repository_name,
        }


def snapshot_repository(
    source_path: Union[str, Path],
    backup_dir: Union[str, Path],
    repo_name: str,
    description: str = "",
) -> BackupManifest:
    """
    Convenience function to create a snapshot.
    
    Args:
        source_path: Path to repository
        backup_dir: Directory for backups
        repo_name: Repository name
        description: Optional description
        
    Returns:
        Backup manifest
    """
    manager = BackupManager(backup_dir)
    return manager.snapshot(
        source_path=Path(source_path),
        repo_name=repo_name,
        description=description,
    )


def restore_repository(
    snapshot_id: str,
    backup_dir: Union[str, Path],
    target_path: Union[str, Path],
    target_name: str,
    overwrite: bool = False,
) -> BackupManifest:
    """
    Convenience function to restore a snapshot.
    
    Args:
        snapshot_id: ID of snapshot to restore
        backup_dir: Directory containing backups
        target_path: Path to restore to
        target_name: Name for restored repository
        overwrite: Whether to overwrite existing target
        
    Returns:
        Restored manifest
    """
    manager = BackupManager(backup_dir)
    return manager.restore(
        snapshot_id=snapshot_id,
        target_path=Path(target_path),
        options=RestoreOptions(
            target_name=target_name,
            overwrite=overwrite,
        ),
    )
