"""Tests for backup and restore functionality."""

import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from rdf_starbase.storage.backup import (
    BackupFormat,
    BackupManager,
    BackupManifest,
    BackupProgress,
    BackupState,
    RestoreOptions,
    restore_repository,
    snapshot_repository,
)


class TestBackupManifest:
    """Tests for BackupManifest dataclass."""
    
    def test_to_dict(self):
        """Test manifest serialization."""
        manifest = BackupManifest(
            snapshot_id="test_20260126_120000_abc12345",
            repository_name="test-repo",
            repository_uuid="uuid-123",
            created_at="2026-01-26T12:00:00Z",
            term_count=1000,
            fact_count=5000,
            description="Test backup",
            tags=["test", "daily"],
        )
        
        d = manifest.to_dict()
        
        assert d["snapshot_id"] == "test_20260126_120000_abc12345"
        assert d["repository_name"] == "test-repo"
        assert d["term_count"] == 1000
        assert d["fact_count"] == 5000
        assert d["tags"] == ["test", "daily"]
    
    def test_from_dict(self):
        """Test manifest deserialization."""
        d = {
            "snapshot_id": "test_snap",
            "repository_name": "my-repo",
            "term_count": 500,
            "fact_count": 2000,
            "files": ["store.parquet", "metadata.json"],
            "file_checksums": {"store.parquet": "abc123"},
        }
        
        manifest = BackupManifest.from_dict(d)
        
        assert manifest.snapshot_id == "test_snap"
        assert manifest.repository_name == "my-repo"
        assert manifest.term_count == 500
        assert len(manifest.files) == 2
        assert manifest.file_checksums["store.parquet"] == "abc123"
    
    def test_save_and_load(self, tmp_path):
        """Test manifest persistence."""
        manifest = BackupManifest(
            snapshot_id="persist_test",
            repository_name="persist-repo",
            fact_count=100,
        )
        
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)
        
        loaded = BackupManifest.load(manifest_path)
        
        assert loaded.snapshot_id == "persist_test"
        assert loaded.repository_name == "persist-repo"
        assert loaded.fact_count == 100


class TestBackupProgress:
    """Tests for BackupProgress tracking."""
    
    def test_progress_ratio(self):
        """Test progress ratio calculation."""
        progress = BackupProgress(
            steps_completed=3,
            total_steps=10,
        )
        
        assert progress.progress_ratio == 0.3
    
    def test_progress_ratio_zero_steps(self):
        """Test progress ratio with zero total steps."""
        progress = BackupProgress(total_steps=0)
        
        assert progress.progress_ratio == 0.0
    
    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        start = time.time()
        progress = BackupProgress(started_at=start)
        
        time.sleep(0.1)
        
        assert progress.elapsed_seconds >= 0.1
    
    def test_elapsed_seconds_completed(self):
        """Test elapsed time for completed operation."""
        start = time.time()
        end = start + 5.0
        
        progress = BackupProgress(
            started_at=start,
            completed_at=end,
        )
        
        assert progress.elapsed_seconds == 5.0


class TestBackupManager:
    """Tests for BackupManager."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        source = Path(tempfile.mkdtemp(prefix="source_"))
        backup = Path(tempfile.mkdtemp(prefix="backup_"))
        restore = Path(tempfile.mkdtemp(prefix="restore_"))
        
        yield {"source": source, "backup": backup, "restore": restore}
        
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(restore, ignore_errors=True)
    
    @pytest.fixture
    def populated_repo(self, temp_dirs):
        """Create a repository with some data."""
        source = temp_dirs["source"]
        
        # Create repository structure
        (source / "store.parquet").write_bytes(b"dummy parquet data")
        (source / "metadata.json").write_text('{"name": "test-repo"}')
        (source / "terms").mkdir()
        (source / "terms" / "iris.parquet").write_bytes(b"iri data")
        (source / "terms" / "literals.parquet").write_bytes(b"literal data")
        
        return source
    
    def test_snapshot_creates_backup(self, temp_dirs, populated_repo):
        """Test creating a snapshot."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
            description="Test snapshot",
        )
        
        assert manifest.snapshot_id.startswith("test-repo_")
        assert manifest.repository_name == "test-repo"
        assert manifest.description == "Test snapshot"
        assert len(manifest.files) == 4
        assert manifest.total_size_bytes > 0
    
    def test_snapshot_computes_checksums(self, temp_dirs, populated_repo):
        """Test that snapshot computes file checksums."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        assert len(manifest.file_checksums) == len(manifest.files)
        for rel_path, checksum in manifest.file_checksums.items():
            assert len(checksum) == 64  # SHA-256 hex
    
    def test_snapshot_saves_manifest(self, temp_dirs, populated_repo):
        """Test that snapshot saves manifest file."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        manifest_path = temp_dirs["backup"] / manifest.snapshot_id / "manifest.json"
        assert manifest_path.exists()
        
        loaded = BackupManifest.load(manifest_path)
        assert loaded.snapshot_id == manifest.snapshot_id
    
    def test_snapshot_with_tags_and_metadata(self, temp_dirs, populated_repo):
        """Test snapshot with custom tags and metadata."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
            tags=["daily", "production"],
            custom_metadata={"version": "1.0", "author": "test"},
        )
        
        assert manifest.tags == ["daily", "production"]
        assert manifest.custom_metadata["version"] == "1.0"
    
    def test_snapshot_invalid_source(self, temp_dirs):
        """Test snapshot with invalid source path."""
        manager = BackupManager(temp_dirs["backup"])
        
        with pytest.raises(ValueError, match="does not exist"):
            manager.snapshot(
                source_path=Path("/nonexistent/path"),
                repo_name="test-repo",
            )
    
    def test_snapshot_progress_callback(self, temp_dirs, populated_repo):
        """Test progress callback during snapshot."""
        progress_updates = []
        
        def on_progress(progress: BackupProgress):
            progress_updates.append(
                (progress.state, progress.current_step, progress.steps_completed)
            )
        
        manager = BackupManager(temp_dirs["backup"], progress_callback=on_progress)
        
        manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        assert len(progress_updates) > 0
        assert any(s == BackupState.COMPLETED for s, _, _ in progress_updates)
    
    def test_restore_creates_repository(self, temp_dirs, populated_repo):
        """Test restoring a snapshot."""
        manager = BackupManager(temp_dirs["backup"])
        
        # Create snapshot
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        # Restore
        target = temp_dirs["restore"] / "restored-repo"
        restored = manager.restore(
            snapshot_id=manifest.snapshot_id,
            target_path=target,
            options=RestoreOptions(target_name="restored-repo"),
        )
        
        assert target.exists()
        assert (target / "store.parquet").exists()
        assert (target / "metadata.json").exists()
        assert (target / "terms" / "iris.parquet").exists()
    
    def test_restore_verifies_checksums(self, temp_dirs, populated_repo):
        """Test that restore verifies checksums."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        # Corrupt a file in the backup
        backup_file = temp_dirs["backup"] / manifest.snapshot_id / "store.parquet"
        backup_file.write_bytes(b"corrupted data!")
        
        target = temp_dirs["restore"] / "restored-repo"
        
        with pytest.raises(ValueError, match="Checksum mismatch"):
            manager.restore(
                snapshot_id=manifest.snapshot_id,
                target_path=target,
                options=RestoreOptions(
                    target_name="restored-repo",
                    verify_checksums=True,
                ),
            )
    
    def test_restore_skip_checksum_verification(self, temp_dirs, populated_repo):
        """Test restore without checksum verification."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        # Corrupt a file
        backup_file = temp_dirs["backup"] / manifest.snapshot_id / "store.parquet"
        backup_file.write_bytes(b"corrupted data!")
        
        target = temp_dirs["restore"] / "restored-repo"
        
        # Should succeed without verification
        manager.restore(
            snapshot_id=manifest.snapshot_id,
            target_path=target,
            options=RestoreOptions(
                target_name="restored-repo",
                verify_checksums=False,
            ),
        )
        
        assert target.exists()
    
    def test_restore_fails_if_target_exists(self, temp_dirs, populated_repo):
        """Test restore fails when target exists."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        target = temp_dirs["restore"] / "existing-repo"
        target.mkdir()
        (target / "data.txt").write_text("existing data")
        
        with pytest.raises(ValueError, match="already exists"):
            manager.restore(
                snapshot_id=manifest.snapshot_id,
                target_path=target,
                options=RestoreOptions(target_name="existing-repo"),
            )
    
    def test_restore_with_overwrite(self, temp_dirs, populated_repo):
        """Test restore with overwrite option."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(
            source_path=populated_repo,
            repo_name="test-repo",
        )
        
        target = temp_dirs["restore"] / "existing-repo"
        target.mkdir()
        (target / "data.txt").write_text("existing data")
        
        manager.restore(
            snapshot_id=manifest.snapshot_id,
            target_path=target,
            options=RestoreOptions(
                target_name="existing-repo",
                overwrite=True,
            ),
        )
        
        assert target.exists()
        assert not (target / "data.txt").exists()
        assert (target / "store.parquet").exists()
    
    def test_restore_nonexistent_snapshot(self, temp_dirs):
        """Test restore with invalid snapshot ID."""
        manager = BackupManager(temp_dirs["backup"])
        
        with pytest.raises(ValueError, match="not found"):
            manager.restore(
                snapshot_id="nonexistent_snapshot",
                target_path=temp_dirs["restore"] / "target",
                options=RestoreOptions(target_name="target"),
            )
    
    def test_list_backups_empty(self, temp_dirs):
        """Test listing backups when none exist."""
        manager = BackupManager(temp_dirs["backup"])
        
        backups = manager.list_backups()
        
        assert backups == []
    
    def test_list_backups(self, temp_dirs, populated_repo):
        """Test listing backups."""
        manager = BackupManager(temp_dirs["backup"])
        
        manager.snapshot(populated_repo, "repo-1", description="First")
        time.sleep(0.1)
        manager.snapshot(populated_repo, "repo-2", description="Second")
        time.sleep(0.1)
        manager.snapshot(populated_repo, "repo-1", description="Third")
        
        backups = manager.list_backups()
        
        assert len(backups) == 3
        # Should be sorted newest first
        assert backups[0].description == "Third"
    
    def test_list_backups_filter_by_name(self, temp_dirs, populated_repo):
        """Test filtering backups by repository name."""
        manager = BackupManager(temp_dirs["backup"])
        
        manager.snapshot(populated_repo, "repo-1")
        manager.snapshot(populated_repo, "repo-2")
        manager.snapshot(populated_repo, "repo-1")
        
        backups = manager.list_backups(repo_name="repo-1")
        
        assert len(backups) == 2
        assert all(b.repository_name == "repo-1" for b in backups)
    
    def test_list_backups_filter_by_tags(self, temp_dirs, populated_repo):
        """Test filtering backups by tags."""
        manager = BackupManager(temp_dirs["backup"])
        
        manager.snapshot(populated_repo, "repo-1", tags=["daily"])
        manager.snapshot(populated_repo, "repo-2", tags=["weekly"])
        manager.snapshot(populated_repo, "repo-3", tags=["daily", "production"])
        
        backups = manager.list_backups(tags=["daily"])
        
        assert len(backups) == 2
    
    def test_get_backup(self, temp_dirs, populated_repo):
        """Test getting a specific backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(populated_repo, "test-repo")
        
        retrieved = manager.get_backup(manifest.snapshot_id)
        
        assert retrieved is not None
        assert retrieved.snapshot_id == manifest.snapshot_id
    
    def test_get_backup_nonexistent(self, temp_dirs):
        """Test getting nonexistent backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        result = manager.get_backup("nonexistent")
        
        assert result is None
    
    def test_delete_backup(self, temp_dirs, populated_repo):
        """Test deleting a backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(populated_repo, "test-repo")
        
        assert manager.get_backup(manifest.snapshot_id) is not None
        
        result = manager.delete_backup(manifest.snapshot_id)
        
        assert result is True
        assert manager.get_backup(manifest.snapshot_id) is None
    
    def test_delete_backup_nonexistent(self, temp_dirs):
        """Test deleting nonexistent backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        result = manager.delete_backup("nonexistent")
        
        assert result is False
    
    def test_verify_backup_valid(self, temp_dirs, populated_repo):
        """Test verifying a valid backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(populated_repo, "test-repo")
        
        result = manager.verify_backup(manifest.snapshot_id)
        
        assert result["valid"] is True
        assert result["files_checked"] == len(manifest.files)
        assert result["errors"] == []
    
    def test_verify_backup_corrupted(self, temp_dirs, populated_repo):
        """Test verifying a corrupted backup."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(populated_repo, "test-repo")
        
        # Corrupt a file
        backup_file = temp_dirs["backup"] / manifest.snapshot_id / "store.parquet"
        backup_file.write_bytes(b"corrupted!")
        
        result = manager.verify_backup(manifest.snapshot_id)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "Checksum mismatch" in result["errors"][0]
    
    def test_verify_backup_missing_file(self, temp_dirs, populated_repo):
        """Test verifying backup with missing file."""
        manager = BackupManager(temp_dirs["backup"])
        
        manifest = manager.snapshot(populated_repo, "test-repo")
        
        # Delete a file
        backup_file = temp_dirs["backup"] / manifest.snapshot_id / "store.parquet"
        backup_file.unlink()
        
        result = manager.verify_backup(manifest.snapshot_id)
        
        assert result["valid"] is False
        assert "Missing file" in result["errors"][0]


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        source = Path(tempfile.mkdtemp(prefix="source_"))
        backup = Path(tempfile.mkdtemp(prefix="backup_"))
        restore = Path(tempfile.mkdtemp(prefix="restore_"))
        
        # Create sample data
        (source / "data.txt").write_text("sample data")
        
        yield {"source": source, "backup": backup, "restore": restore}
        
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(restore, ignore_errors=True)
    
    def test_snapshot_repository(self, temp_dirs):
        """Test snapshot_repository convenience function."""
        manifest = snapshot_repository(
            source_path=temp_dirs["source"],
            backup_dir=temp_dirs["backup"],
            repo_name="my-repo",
            description="Quick snapshot",
        )
        
        assert manifest.repository_name == "my-repo"
        assert manifest.description == "Quick snapshot"
    
    def test_restore_repository(self, temp_dirs):
        """Test restore_repository convenience function."""
        # Create snapshot
        manifest = snapshot_repository(
            source_path=temp_dirs["source"],
            backup_dir=temp_dirs["backup"],
            repo_name="my-repo",
        )
        
        # Restore
        target = temp_dirs["restore"] / "restored"
        restored = restore_repository(
            snapshot_id=manifest.snapshot_id,
            backup_dir=temp_dirs["backup"],
            target_path=target,
            target_name="restored",
        )
        
        assert target.exists()
        assert (target / "data.txt").read_text() == "sample data"


class TestBackupWithRealData:
    """Tests with realistic repository data."""
    
    @pytest.fixture
    def repo_with_parquet(self, tmp_path):
        """Create a repository with actual Parquet files."""
        source = tmp_path / "source"
        source.mkdir()
        
        # Create a proper Parquet file
        df = pl.DataFrame({
            "subject": [1, 2, 3, 4, 5],
            "predicate": [10, 10, 11, 11, 12],
            "object": [100, 101, 102, 103, 104],
        })
        df.write_parquet(source / "store.parquet")
        
        # Create metadata
        metadata = {
            "name": "test-repo",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "triple_count": 5,
        }
        (source / "repository.json").write_text(json.dumps(metadata))
        
        return source
    
    def test_snapshot_captures_fact_count(self, repo_with_parquet, tmp_path):
        """Test that snapshot captures fact count from Parquet."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(backup_dir)
        
        manifest = manager.snapshot(
            source_path=repo_with_parquet,
            repo_name="parquet-repo",
        )
        
        assert manifest.fact_count == 5
    
    def test_full_round_trip(self, repo_with_parquet, tmp_path):
        """Test complete backup and restore cycle."""
        backup_dir = tmp_path / "backups"
        restore_dir = tmp_path / "restored"
        manager = BackupManager(backup_dir)
        
        # Snapshot
        manifest = manager.snapshot(
            source_path=repo_with_parquet,
            repo_name="round-trip-repo",
            description="Round trip test",
        )
        
        # Verify
        verify_result = manager.verify_backup(manifest.snapshot_id)
        assert verify_result["valid"]
        
        # Restore
        manager.restore(
            snapshot_id=manifest.snapshot_id,
            target_path=restore_dir,
            options=RestoreOptions(target_name="restored"),
        )
        
        # Verify restored data matches
        original_df = pl.read_parquet(repo_with_parquet / "store.parquet")
        restored_df = pl.read_parquet(restore_dir / "store.parquet")
        
        assert original_df.equals(restored_df)
