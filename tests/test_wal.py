"""
Tests for Write-Ahead Log (WAL).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import struct

from rdf_starbase.storage.wal import (
    WriteAheadLog,
    WALEntry,
    WALEntryType,
    InsertPayload,
    DeletePayload,
    CheckpointState,
)


class TestWALEntry:
    """Test WALEntry serialization."""
    
    def test_serialize_deserialize_insert(self):
        """Test roundtrip for insert entry."""
        payload = InsertPayload(
            graph_id=0,
            subject_id=100,
            predicate_id=200,
            object_id=300,
            flags=1,
            source_id=400,
            confidence=0.95,
            process_id=500,
        )
        
        entry = WALEntry(
            entry_type=WALEntryType.INSERT,
            sequence_number=42,
            transaction_id=7,
            timestamp=1234567890,
            payload=payload.serialize(),
        )
        
        data = entry.serialize()
        restored, consumed = WALEntry.deserialize(data)
        
        assert restored.entry_type == WALEntryType.INSERT
        assert restored.sequence_number == 42
        assert restored.transaction_id == 7
        assert restored.timestamp == 1234567890
        
        restored_payload = InsertPayload.deserialize(restored.payload)
        assert restored_payload.subject_id == 100
        assert restored_payload.predicate_id == 200
        assert restored_payload.object_id == 300
        assert restored_payload.confidence == 0.95
    
    def test_checksum_validation(self):
        """Test that corrupted entries are detected."""
        payload = InsertPayload(0, 1, 2, 3, 0, 0, 1.0, 0)
        entry = WALEntry(
            entry_type=WALEntryType.INSERT,
            sequence_number=1,
            transaction_id=0,
            timestamp=0,
            payload=payload.serialize(),
        )
        
        data = bytearray(entry.serialize())
        # Corrupt a byte
        data[20] ^= 0xFF
        
        with pytest.raises(ValueError, match="checksum mismatch"):
            WALEntry.deserialize(bytes(data))


class TestWriteAheadLog:
    """Test WriteAheadLog operations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_create_wal(self, temp_dir):
        """Test WAL creation."""
        wal = WriteAheadLog(temp_dir / "wal")
        
        assert (temp_dir / "wal").exists()
        
        stats = wal.stats()
        assert stats["sequence"] == 0
        assert stats["current_segment"] == 0
        
        wal.close()
    
    def test_log_insert(self, temp_dir):
        """Test logging inserts."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        seq1 = wal.log_insert(0, 100, 200, 300)
        seq2 = wal.log_insert(0, 101, 201, 301, confidence=0.8)
        
        assert seq1 == 1
        assert seq2 == 2
        
        stats = wal.stats()
        assert stats["sequence"] == 2
        
        wal.close()
    
    def test_log_delete(self, temp_dir):
        """Test logging deletes."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        seq = wal.log_delete(subject_id=100)
        
        assert seq == 1
        wal.close()
    
    def test_transaction_commit(self, temp_dir):
        """Test transaction commit flow."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        txn_id = wal.begin_transaction()
        wal.log_insert(0, 100, 200, 300, txn_id=txn_id)
        wal.log_insert(0, 101, 201, 301, txn_id=txn_id)
        wal.commit_transaction(txn_id)
        
        assert wal._active_txn is None
        wal.close()
    
    def test_transaction_abort(self, temp_dir):
        """Test transaction abort flow."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        txn_id = wal.begin_transaction()
        wal.log_insert(0, 100, 200, 300, txn_id=txn_id)
        wal.abort_transaction(txn_id)
        
        assert wal._active_txn is None
        wal.close()
    
    def test_checkpoint(self, temp_dir):
        """Test checkpoint creation."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        wal.log_insert(0, 100, 200, 300)
        wal.log_insert(0, 101, 201, 301)
        
        state = wal.checkpoint()
        
        assert state.last_sequence == 3  # 2 inserts + 1 checkpoint
        assert (temp_dir / "wal" / "checkpoint.json").exists()
        
        wal.close()
    
    def test_replay_committed(self, temp_dir):
        """Test replay returns only committed entries."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        # Auto-commit insert
        wal.log_insert(0, 100, 200, 300)
        
        # Committed transaction
        txn1 = wal.begin_transaction()
        wal.log_insert(0, 101, 201, 301, txn_id=txn1)
        wal.commit_transaction(txn1)
        
        # Aborted transaction
        txn2 = wal.begin_transaction()
        wal.log_insert(0, 102, 202, 302, txn_id=txn2)
        wal.abort_transaction(txn2)
        
        wal.close()
        
        # Replay
        wal2 = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        entries = list(wal2.replay(from_sequence=0))
        
        # Should have 2 entries (auto-commit + committed txn)
        # The aborted transaction's insert should be skipped
        assert len(entries) == 2
        
        subjects = [e[1].subject_id for e in entries]
        assert 100 in subjects
        assert 101 in subjects
        assert 102 not in subjects  # Aborted
        
        wal2.close()
    
    def test_replay_from_checkpoint(self, temp_dir):
        """Test replay starts from checkpoint."""
        wal = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        
        wal.log_insert(0, 100, 200, 300)
        wal.checkpoint()
        wal.log_insert(0, 101, 201, 301)
        
        wal.close()
        
        # Reopen and replay
        wal2 = WriteAheadLog(temp_dir / "wal", sync_mode="off")
        entries = list(wal2.replay())
        
        # Should only have 1 entry (after checkpoint)
        assert len(entries) == 1
        assert entries[0][1].subject_id == 101
        
        wal2.close()
    
    def test_segment_rotation(self, temp_dir):
        """Test WAL segment rotation."""
        # Small segment size for testing
        wal = WriteAheadLog(
            temp_dir / "wal",
            segment_max_size=500,  # Very small
            sync_mode="off"
        )
        
        # Write enough to trigger rotation
        for i in range(100):
            wal.log_insert(0, i, i + 1, i + 2)
        
        stats = wal.stats()
        assert stats["segment_count"] > 1
        
        wal.close()
    
    def test_recovery_after_crash(self, temp_dir):
        """Test recovery simulation."""
        wal_dir = temp_dir / "wal"
        
        # First session - write some data
        wal1 = WriteAheadLog(wal_dir, sync_mode="off")
        wal1.log_insert(0, 100, 200, 300)
        wal1.log_insert(0, 101, 201, 301)
        wal1.checkpoint()
        wal1.log_insert(0, 102, 202, 302)  # After checkpoint
        wal1.close()
        
        # Simulate crash - reopen
        wal2 = WriteAheadLog(wal_dir, sync_mode="off")
        
        # Should resume from last sequence
        seq = wal2.log_insert(0, 103, 203, 303)
        assert seq > 3  # Should continue from where we left off
        
        # Replay should give us the post-checkpoint entry
        entries = list(wal2.replay())
        subjects = [e[1].subject_id for e in entries]
        assert 102 in subjects
        
        wal2.close()
    
    def test_truncate_segments(self, temp_dir):
        """Test segment truncation."""
        wal = WriteAheadLog(
            temp_dir / "wal",
            segment_max_size=500,
            sync_mode="off"
        )
        
        # Write enough to create multiple segments
        for i in range(100):
            wal.log_insert(0, i, i + 1, i + 2)
        
        stats = wal.stats()
        initial_segments = stats["segment_count"]
        
        # Truncate old segments
        current = wal._current_segment
        removed = wal.truncate_before(current)
        
        assert removed == initial_segments - 1
        
        wal.close()


class TestInsertPayload:
    """Test InsertPayload serialization."""
    
    def test_roundtrip(self):
        """Test serialize/deserialize roundtrip."""
        payload = InsertPayload(
            graph_id=1,
            subject_id=2,
            predicate_id=3,
            object_id=4,
            flags=5,
            source_id=6,
            confidence=0.99,
            process_id=7,
        )
        
        data = payload.serialize()
        restored = InsertPayload.deserialize(data)
        
        assert restored.graph_id == 1
        assert restored.subject_id == 2
        assert restored.predicate_id == 3
        assert restored.object_id == 4
        assert restored.flags == 5
        assert restored.source_id == 6
        assert restored.confidence == 0.99
        assert restored.process_id == 7


class TestDeletePayload:
    """Test DeletePayload serialization."""
    
    def test_roundtrip_with_none(self):
        """Test serialize/deserialize with None values."""
        payload = DeletePayload(
            graph_id=None,
            subject_id=100,
            predicate_id=None,
            object_id=200,
        )
        
        data = payload.serialize()
        restored = DeletePayload.deserialize(data)
        
        assert restored.graph_id is None
        assert restored.subject_id == 100
        assert restored.predicate_id is None
        assert restored.object_id == 200
