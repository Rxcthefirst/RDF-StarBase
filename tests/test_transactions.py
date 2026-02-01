"""
Tests for ACID transactions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from rdf_starbase.store import TripleStore
from rdf_starbase.storage.transactions import (
    Transaction,
    TransactionManager,
    TransactionState,
    IsolationLevel,
)


class TestTransaction:
    """Test individual transactions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    @pytest.fixture
    def store_with_txn(self, temp_dir):
        """Create a store with transaction manager."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off"  # Faster for tests
        )
        yield store, txn_mgr
        txn_mgr.close()
    
    def test_transaction_commit(self, store_with_txn):
        """Test basic transaction commit."""
        store, txn_mgr = store_with_txn
        
        with txn_mgr.transaction() as txn:
            txn.add_triple(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
                source="test",
            )
        
        # Triple should be in store after commit
        triples = store.get_triples().to_dicts()
        assert len(triples) == 1
        assert triples[0]["subject"] == "http://example.org/alice"
    
    def test_transaction_rollback_on_exception(self, store_with_txn):
        """Test automatic rollback on exception."""
        store, txn_mgr = store_with_txn
        
        try:
            with txn_mgr.transaction() as txn:
                txn.add_triple(
                    "http://example.org/alice",
                    "http://example.org/knows",
                    "http://example.org/bob",
                    source="test",
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Triple should NOT be in store after rollback
        triples = store.get_triples().to_dicts()
        assert len(triples) == 0
    
    def test_transaction_explicit_rollback(self, store_with_txn):
        """Test explicit rollback."""
        store, txn_mgr = store_with_txn
        
        txn = txn_mgr.begin()  # Already active after begin()
        txn.add_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
            source="test",
        )
        txn.rollback()
        
        # Triple should NOT be in store
        triples = store.get_triples().to_dicts()
        assert len(triples) == 0
        assert txn.state == TransactionState.ABORTED
    
    def test_transaction_multiple_inserts(self, store_with_txn):
        """Test multiple inserts in one transaction."""
        store, txn_mgr = store_with_txn
        
        with txn_mgr.transaction() as txn:
            for i in range(10):
                txn.add_triple(
                    f"http://example.org/s{i}",
                    "http://example.org/p",
                    f"http://example.org/o{i}",
                    source="test",
                )
        
        triples = store.get_triples().to_dicts()
        assert len(triples) == 10
    
    def test_transaction_with_delete(self, store_with_txn):
        """Test transaction with delete."""
        store, txn_mgr = store_with_txn
        
        # First add some triples
        with txn_mgr.transaction() as txn:
            txn.add_triple(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
                source="test",
            )
            txn.add_triple(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/charlie",
                source="test",
            )
        
        assert len(store.get_triples().to_dicts()) == 2
        
        # Delete one in a new transaction
        with txn_mgr.transaction() as txn:
            txn.delete_triples(
                subject="http://example.org/alice",
                obj="http://example.org/bob",
            )
        
        # Should have 1 active triple
        triples = store.get_triples().to_dicts()
        assert len(triples) == 1
        assert triples[0]["object"] == "http://example.org/charlie"
    
    def test_transaction_stats(self, store_with_txn):
        """Test transaction statistics."""
        store, txn_mgr = store_with_txn
        
        with txn_mgr.transaction() as txn:
            txn.add_triple("http://s", "http://p", "http://o", source="test")
            txn.add_triple("http://s2", "http://p", "http://o2", source="test")
            txn.delete_triples(subject="http://nonexistent")
            
            stats = txn.stats
            assert stats.inserts == 2
            assert stats.deletes == 1
        
        # After commit
        assert txn.stats.state == TransactionState.COMMITTED
        assert txn.stats.duration_ms > 0


class TestTransactionManager:
    """Test transaction manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_manager_stats(self, temp_dir):
        """Test manager statistics."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off",
        )
        
        stats = txn_mgr.stats()
        assert stats["wal_enabled"] is True
        assert stats["active_transactions"] == 0
        assert stats["next_txn_id"] == 1
        
        txn_mgr.close()
    
    def test_multiple_transactions(self, temp_dir):
        """Test sequential transactions."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off",
        )
        
        with txn_mgr.transaction() as txn:
            txn.add_triple("http://s1", "http://p", "http://o1", source="test")
        
        with txn_mgr.transaction() as txn:
            txn.add_triple("http://s2", "http://p", "http://o2", source="test")
        
        triples = store.get_triples().to_dicts()
        assert len(triples) == 2
        
        txn_mgr.close()
    
    def test_checkpoint(self, temp_dir):
        """Test manual checkpoint."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off",
            auto_checkpoint=False,
        )
        
        with txn_mgr.transaction() as txn:
            txn.add_triple("http://s", "http://p", "http://o", source="test")
        
        txn_mgr.checkpoint()
        
        # Checkpoint file should exist
        assert (temp_dir / "wal" / "checkpoint.json").exists()
        
        txn_mgr.close()


class TestCrashRecovery:
    """Test crash recovery with WAL.
    
    Note: Full crash recovery requires persisting the term dictionary
    alongside the WAL. These tests demonstrate the WAL replay mechanism,
    but a complete recovery scenario needs save/load of the full store.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_recovery_within_session(self, temp_dir):
        """Test that transactions are durable within a session.
        
        This simulates the case where we add data via transactions
        and can query it immediately after commit.
        """
        wal_dir = temp_dir / "wal"
        
        store = TripleStore()
        txn_mgr = TransactionManager(store, wal_dir=wal_dir, sync_mode="off")
        
        with txn_mgr.transaction() as txn:
            txn.add_triple("http://s1", "http://p", "http://o1", source="test")
        
        # Should be queryable immediately after commit
        triples = store.get_triples().to_dicts()
        assert len(triples) == 1
        assert triples[0]["subject"] == "http://s1"
        
        txn_mgr.close()
    
    def test_recovery_wal_replay_structure(self, temp_dir):
        """Test WAL replay returns correct entries (not full store recovery)."""
        wal_dir = temp_dir / "wal"
        
        # Session 1: Write to WAL
        store1 = TripleStore()
        txn_mgr1 = TransactionManager(store1, wal_dir=wal_dir, sync_mode="off")
        
        with txn_mgr1.transaction() as txn:
            txn.add_triple("http://s1", "http://p", "http://o1", source="test")
        
        txn_mgr1.close()
        
        # Session 2: Verify WAL has entries to replay
        from rdf_starbase.storage.wal import WriteAheadLog
        wal = WriteAheadLog(wal_dir, sync_mode="off")
        entries = list(wal.replay(from_sequence=0))
        
        # Should have 1 insert entry
        assert len(entries) == 1
        
        wal.close()
    
    def test_checkpoint_clears_replay(self, temp_dir):
        """Test that checkpoint marks the replay starting point."""
        wal_dir = temp_dir / "wal"
        
        # Session 1: Write data, checkpoint, then write more
        store1 = TripleStore()
        txn_mgr1 = TransactionManager(store1, wal_dir=wal_dir, sync_mode="off")
        
        with txn_mgr1.transaction() as txn:
            txn.add_triple("http://s1", "http://p", "http://o1", source="test")
        
        txn_mgr1.checkpoint()
        
        with txn_mgr1.transaction() as txn:
            txn.add_triple("http://s2", "http://p", "http://o2", source="test")
        
        txn_mgr1.close()
        
        # Session 2: Verify replay starts after checkpoint
        from rdf_starbase.storage.wal import WriteAheadLog
        wal = WriteAheadLog(wal_dir, sync_mode="off")
        entries = list(wal.replay())  # Uses checkpoint as starting point
        
        # Should only have 1 entry (the one after checkpoint)
        assert len(entries) == 1
        
        wal.close()


class TestAtomicity:
    """Test atomicity guarantees."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_partial_failure_rolls_back(self, temp_dir):
        """Test that partial failures roll back all changes."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off",
        )
        
        try:
            with txn_mgr.transaction() as txn:
                txn.add_triple("http://s1", "http://p", "http://o1", source="test")
                txn.add_triple("http://s2", "http://p", "http://o2", source="test")
                raise RuntimeError("Simulated crash mid-transaction")
        except RuntimeError:
            pass
        
        # ALL changes should be rolled back
        triples = store.get_triples().to_dicts()
        assert len(triples) == 0
        
        txn_mgr.close()
    
    def test_cannot_use_committed_transaction(self, temp_dir):
        """Test that committed transactions can't be reused."""
        store = TripleStore()
        txn_mgr = TransactionManager(
            store,
            wal_dir=temp_dir / "wal",
            sync_mode="off",
        )
        
        txn = txn_mgr.begin()  # Already active after begin()
        txn.add_triple("http://s", "http://p", "http://o", source="test")
        txn.commit()
        
        with pytest.raises(RuntimeError, match="not ACTIVE"):
            txn.add_triple("http://s2", "http://p", "http://o2", source="test")
        
        txn_mgr.close()
