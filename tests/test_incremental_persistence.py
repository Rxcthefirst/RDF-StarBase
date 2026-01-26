"""
Tests for incremental persistence.

Tests the delta file + compaction approach for efficient saves.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from rdf_starbase import TripleStore
from rdf_starbase.models import ProvenanceContext


def add_test_triple(store, s, p, o, graph=None):
    """Helper to add a triple with default provenance."""
    prov = ProvenanceContext(source="test", confidence=1.0)
    return store.add_triple(s, p, o, provenance=prov, graph=graph)


class TestIncrementalPersistence:
    """Tests for incremental save/load functionality."""
    
    @pytest.fixture
    def store(self):
        """Create a store with test data."""
        store = TripleStore()
        # Add initial data
        add_test_triple(
            store,
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob"
        )
        add_test_triple(
            store,
            "http://example.org/Alice",
            "http://example.org/name",
            '"Alice"'
        )
        return store
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_first_save_creates_base(self, store, temp_dir):
        """First save should create base files."""
        storage_path = temp_dir / "test_store"
        
        result = store.save_incremental(storage_path)
        
        assert result["status"] == "full_save"
        assert (storage_path / "manifest.json").exists()
        assert (storage_path / "base" / "facts.parquet").exists()
        assert (storage_path / "base" / "terms.parquet").exists()
    
    def test_save_no_changes(self, store, temp_dir):
        """Save with no changes should report no_changes."""
        storage_path = temp_dir / "test_store"
        
        # First save
        store.save_incremental(storage_path)
        
        # Second save without changes
        result = store.save_incremental(storage_path)
        
        assert result["status"] == "no_changes"
        assert result["delta_facts"] == 0
        assert result["was_compacted"] is False
    
    def test_save_delta_after_add(self, store, temp_dir):
        """Adding data after save should create delta file."""
        storage_path = temp_dir / "test_store"
        
        # First save
        store.save_incremental(storage_path)
        
        # Add new data
        add_test_triple(
            store,
            "http://example.org/Bob",
            "http://example.org/knows",
            "http://example.org/Charlie"
        )
        
        # Second save
        result = store.save_incremental(storage_path)
        
        assert result["status"] == "delta_saved"
        assert result["delta_facts"] == 1
        assert result["delta_num"] == 1
        
        # Check delta file exists
        deltas_dir = storage_path / "deltas"
        assert deltas_dir.exists()
        assert (deltas_dir / "delta_0001_facts.parquet").exists()
    
    def test_load_incremental_with_deltas(self, store, temp_dir):
        """Load should merge base + deltas."""
        storage_path = temp_dir / "test_store"
        
        # Save initial data
        store.save_incremental(storage_path)
        
        # Add more data and save as delta
        add_test_triple(
            store,
            "http://example.org/Bob",
            "http://example.org/knows",
            "http://example.org/Charlie"
        )
        store.save_incremental(storage_path)
        
        # Add even more data and save as another delta
        add_test_triple(
            store,
            "http://example.org/Charlie",
            "http://example.org/name",
            '"Charlie"'
        )
        store.save_incremental(storage_path)
        
        # Load fresh store
        loaded = TripleStore.load_incremental(storage_path)
        
        # Should have all 4 triples
        assert len(loaded) == 4
    
    def test_manifest_tracks_state(self, store, temp_dir):
        """Manifest should track delta count and last txn."""
        storage_path = temp_dir / "test_store"
        
        # Save and add data multiple times
        store.save_incremental(storage_path)
        
        for i in range(3):
            add_test_triple(
                store,
                f"http://example.org/Person{i}",
                "http://example.org/id",
                f'"{i}"'
            )
            store.save_incremental(storage_path)
        
        # Check manifest
        with open(storage_path / "manifest.json") as f:
            manifest = json.load(f)
        
        assert manifest["delta_count"] == 3
        assert manifest["last_save_at"] is not None
    
    def test_auto_compaction(self, temp_dir):
        """Should auto-compact when threshold exceeded."""
        from rdf_starbase.storage.persistence import IncrementalPersistence
        
        store = TripleStore()
        storage_path = temp_dir / "test_store"
        
        # Use low threshold for testing
        persistence = IncrementalPersistence(storage_path, compaction_threshold=3)
        
        # Initial save
        persistence.save(store._term_dict, store._fact_store, store._qt_dict)
        
        # Add and save until threshold
        for i in range(3):
            add_test_triple(
                store,
                f"http://example.org/Entity{i}",
                "http://example.org/value",
                f'"{i}"'
            )
            result = persistence.save(store._term_dict, store._fact_store, store._qt_dict)
        
        # Last save should trigger compaction
        assert result["was_compacted"] is True
        
        # Delta files should be cleaned up
        deltas_dir = storage_path / "deltas"
        if deltas_dir.exists():
            delta_files = list(deltas_dir.glob("delta_*.parquet"))
            assert len(delta_files) == 0
    
    def test_force_compact(self, store, temp_dir):
        """Force compact should merge all data."""
        storage_path = temp_dir / "test_store"
        
        # Save and add deltas
        store.save_incremental(storage_path)
        
        for i in range(2):
            add_test_triple(
                store,
                f"http://example.org/Item{i}",
                "http://example.org/index",
                f'"{i}"'
            )
            store.save_incremental(storage_path)
        
        # Force compact
        result = store.compact(storage_path)
        
        assert result["status"] == "full_save"
        assert result["was_compacted"] is True
        
        # Should still load correctly
        loaded = TripleStore.load_incremental(storage_path)
        assert len(loaded) == 4
    
    def test_roundtrip_with_literals(self, temp_dir):
        """Test roundtrip with various literal types."""
        store = TripleStore()
        
        # Add literals with datatypes
        add_test_triple(
            store,
            "http://example.org/Thing",
            "http://example.org/count",
            '"42"^^<http://www.w3.org/2001/XMLSchema#integer>'
        )
        add_test_triple(
            store,
            "http://example.org/Thing",
            "http://example.org/label",
            '"Hello"@en'
        )
        add_test_triple(
            store,
            "http://example.org/Thing",
            "http://example.org/value",
            '"3.14"^^<http://www.w3.org/2001/XMLSchema#decimal>'
        )
        
        storage_path = temp_dir / "literals_store"
        store.save_incremental(storage_path)
        
        loaded = TripleStore.load_incremental(storage_path)
        
        # Should have all 3 triples
        assert len(loaded) == 3
        
        # Check stats show correct term count
        stats = loaded.stats()
        assert stats["active_assertions"] == 3
    
    def test_stats_method(self, store, temp_dir):
        """Test getting storage statistics."""
        from rdf_starbase.storage.persistence import IncrementalPersistence
        
        storage_path = temp_dir / "stats_store"
        
        # Check stats before save
        persistence = IncrementalPersistence(storage_path)
        stats = persistence.get_stats()
        assert stats["exists"] is False
        
        # Save and check stats
        persistence.save(store._term_dict, store._fact_store, store._qt_dict)
        stats = persistence.get_stats()
        
        assert stats["exists"] is True
        assert stats["delta_count"] == 0
        assert stats["created_at"] is not None


class TestIncrementalPersistenceEdgeCases:
    """Edge case tests for incremental persistence."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        path = tempfile.mkdtemp()
        yield Path(path)
        shutil.rmtree(path)
    
    def test_empty_store_save(self, temp_dir):
        """Should handle empty store save."""
        store = TripleStore()
        storage_path = temp_dir / "empty_store"
        
        result = store.save_incremental(storage_path)
        
        assert result["status"] == "full_save"
        assert result["total_facts"] == 0
    
    def test_load_nonexistent_raises(self, temp_dir):
        """Loading from nonexistent path should raise."""
        storage_path = temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            TripleStore.load_incremental(storage_path)
    
    def test_multiple_save_load_cycles(self, temp_dir):
        """Test multiple save/load cycles maintain data."""
        storage_path = temp_dir / "multi_cycle"
        
        # Cycle 1: Create and save
        store1 = TripleStore()
        add_test_triple(store1, "http://ex/a", "http://ex/p", "http://ex/b")
        store1.save_incremental(storage_path)
        
        # Cycle 2: Load, add, save
        store2 = TripleStore.load_incremental(storage_path)
        add_test_triple(store2, "http://ex/b", "http://ex/p", "http://ex/c")
        store2.save_incremental(storage_path)
        
        # Cycle 3: Load, add, save
        store3 = TripleStore.load_incremental(storage_path)
        add_test_triple(store3, "http://ex/c", "http://ex/p", "http://ex/d")
        store3.save_incremental(storage_path)
        
        # Cycle 4: Final load and verify
        final = TripleStore.load_incremental(storage_path)
        
        # Should have 3 triples from 3 cycles
        assert len(final) == 3
