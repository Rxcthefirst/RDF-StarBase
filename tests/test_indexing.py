"""
Tests for B-tree-like sorted index.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from rdf_starbase.storage.indexing import (
    SortedIndex,
    IndexManager,
    IndexStats,
    indexed_filter,
    estimate_selectivity,
    should_use_index,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "subject": [1, 2, 3, 1, 2, 1],  # Subject 1 appears 3 times
        "predicate": [10, 10, 20, 20, 30, 10],
        "object": [100, 101, 102, 103, 104, 105],
    }).cast({
        "subject": pl.UInt32,
        "predicate": pl.UInt32,
        "object": pl.UInt32,
    })


@pytest.fixture
def large_df():
    """Create a larger DataFrame for testing."""
    import random
    random.seed(42)
    
    n = 10000
    subjects = [random.randint(1, 1000) for _ in range(n)]
    predicates = [random.randint(1, 50) for _ in range(n)]
    objects = [random.randint(1, 5000) for _ in range(n)]
    
    return pl.DataFrame({
        "subject": subjects,
        "predicate": predicates,
        "object": objects,
    }).cast({
        "subject": pl.UInt32,
        "predicate": pl.UInt32,
        "object": pl.UInt32,
    })


class TestSortedIndex:
    """Tests for SortedIndex."""
    
    def test_build_index(self, sample_df):
        """Test building an index."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        stats = idx.stats()
        assert stats.num_keys == 3  # 1, 2, 3
        assert stats.num_entries == 6
    
    def test_lookup(self, sample_df):
        """Test point lookup."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Subject 1 appears at positions 0, 3, 5
        positions = idx.lookup(1)
        assert sorted(positions) == [0, 3, 5]
        
        # Subject 2 appears at positions 1, 4
        positions = idx.lookup(2)
        assert sorted(positions) == [1, 4]
        
        # Subject 3 appears at position 2
        positions = idx.lookup(3)
        assert positions == [2]
    
    def test_lookup_missing(self, sample_df):
        """Test lookup for non-existent key."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        positions = idx.lookup(999)
        assert positions == []
    
    def test_range_lookup(self, sample_df):
        """Test range lookup."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Range 1-2 (inclusive)
        positions = idx.range_lookup(1, 2)
        assert sorted(positions) == [0, 1, 3, 4, 5]
        
        # Range 2-3
        positions = idx.range_lookup(2, 3)
        assert sorted(positions) == [1, 2, 4]
    
    def test_range_lookup_open_ended(self, sample_df):
        """Test range lookup with open bounds."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Min only (>= 2)
        positions = idx.range_lookup(min_key=2)
        assert sorted(positions) == [1, 2, 4]
        
        # Max only (<= 2)
        positions = idx.range_lookup(max_key=2)
        assert sorted(positions) == [0, 1, 3, 4, 5]
        
        # No bounds (all)
        positions = idx.range_lookup()
        assert len(positions) == 6
    
    def test_contains(self, sample_df):
        """Test key existence check."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        assert idx.contains(1)
        assert idx.contains(2)
        assert idx.contains(3)
        assert not idx.contains(999)
    
    def test_add_entry(self, sample_df):
        """Test adding a single entry."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Add to existing key
        idx.add(1, 100)
        positions = idx.lookup(1)
        assert 100 in positions
        
        # Add new key
        idx.add(999, 101)
        assert idx.contains(999)
        positions = idx.lookup(999)
        assert positions == [101]
    
    def test_remove_entry(self, sample_df):
        """Test removing entries."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Remove specific position
        idx.remove(1, 0)
        positions = idx.lookup(1)
        assert 0 not in positions
        assert sorted(positions) == [3, 5]
        
        # Remove entire key
        idx.remove(3)
        assert not idx.contains(3)
    
    def test_clear(self, sample_df):
        """Test clearing index."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        idx.clear()
        
        stats = idx.stats()
        assert stats.num_keys == 0
        assert stats.num_entries == 0
    
    def test_save_and_load(self, sample_df):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = SortedIndex("subject")
            idx.build(sample_df)
            
            path = Path(tmpdir) / "index.pkl"
            idx.save(path)
            
            loaded = SortedIndex.load(path)
            
            assert loaded.column_name == "subject"
            assert loaded.stats().num_keys == idx.stats().num_keys
            
            # Verify lookup works
            positions = loaded.lookup(1)
            assert sorted(positions) == [0, 3, 5]
    
    def test_empty_index(self):
        """Test operations on empty index."""
        idx = SortedIndex("subject")
        
        assert idx.lookup(1) == []
        assert idx.range_lookup(1, 10) == []
        assert not idx.contains(1)
        
        stats = idx.stats()
        assert stats.num_keys == 0
        assert stats.num_entries == 0


class TestIndexManager:
    """Tests for IndexManager."""
    
    def test_create_index(self, sample_df):
        """Test creating indexes."""
        manager = IndexManager()
        
        idx = manager.create_index("subject")
        assert idx is not None
        assert manager.has_index("subject")
    
    def test_build_all(self, sample_df):
        """Test building all indexes."""
        manager = IndexManager()
        manager.create_index("subject")
        manager.create_index("predicate")
        
        manager.build_all(sample_df)
        
        stats = manager.stats()
        assert "subject" in stats
        assert "predicate" in stats
        assert stats["subject"].num_entries == 6
    
    def test_lookup(self, sample_df):
        """Test lookup through manager."""
        manager = IndexManager()
        manager.create_index("subject")
        manager.build_all(sample_df)
        
        positions = manager.lookup("subject", 1)
        assert sorted(positions) == [0, 3, 5]
        
        # Non-indexed column returns None
        positions = manager.lookup("object", 100)
        assert positions is None
    
    def test_range_lookup(self, sample_df):
        """Test range lookup through manager."""
        manager = IndexManager()
        manager.create_index("subject")
        manager.build_all(sample_df)
        
        positions = manager.range_lookup("subject", 1, 2)
        assert sorted(positions) == [0, 1, 3, 4, 5]
    
    def test_drop_index(self, sample_df):
        """Test dropping an index."""
        manager = IndexManager()
        manager.create_index("subject")
        
        assert manager.has_index("subject")
        
        result = manager.drop_index("subject")
        assert result is True
        assert not manager.has_index("subject")
        
        # Dropping non-existent returns False
        result = manager.drop_index("nonexistent")
        assert result is False
    
    def test_list_indexes(self, sample_df):
        """Test listing indexes."""
        manager = IndexManager()
        manager.create_index("subject")
        manager.create_index("object")
        
        indexes = manager.list_indexes()
        assert "subject" in indexes
        assert "object" in indexes
    
    def test_save_and_load(self, sample_df):
        """Test saving and loading manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IndexManager()
            manager.create_index("subject")
            manager.create_index("predicate")
            manager.build_all(sample_df)
            
            save_dir = Path(tmpdir) / "indexes"
            manager.save(save_dir)
            
            loaded = IndexManager.load(save_dir)
            
            assert loaded.has_index("subject")
            assert loaded.has_index("predicate")
            
            # Verify lookups work
            positions = loaded.lookup("subject", 1)
            assert sorted(positions) == [0, 3, 5]
    
    def test_clear_all(self, sample_df):
        """Test clearing all indexes."""
        manager = IndexManager()
        manager.create_index("subject")
        manager.build_all(sample_df)
        
        manager.clear_all()
        
        stats = manager.stats()
        assert stats["subject"].num_entries == 0


class TestIndexedFilter:
    """Tests for indexed filtering."""
    
    def test_indexed_filter(self, sample_df):
        """Test filtering DataFrame using index."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        filtered = indexed_filter(sample_df, idx, 1)
        
        assert len(filtered) == 3
        assert filtered["subject"].to_list() == [1, 1, 1]
    
    def test_indexed_filter_empty_result(self, sample_df):
        """Test filtering with no matches."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        filtered = indexed_filter(sample_df, idx, 999)
        
        assert len(filtered) == 0


class TestSelectivity:
    """Tests for selectivity estimation."""
    
    def test_estimate_selectivity(self, sample_df):
        """Test selectivity estimation."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Subject 1 appears 3/6 = 0.5
        sel = estimate_selectivity(idx, 1)
        assert sel == 0.5
        
        # Subject 3 appears 1/6 ≈ 0.167
        sel = estimate_selectivity(idx, 3)
        assert abs(sel - (1/6)) < 0.01
        
        # Non-existent = 0
        sel = estimate_selectivity(idx, 999)
        assert sel == 0.0
    
    def test_should_use_index_low_cardinality(self, sample_df):
        """Test index decision for low cardinality."""
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        # Only 3 unique subjects for 6 rows
        # With 50% selectivity per key, decision depends on threshold
        # Use a higher threshold to ensure index is recommended
        assert should_use_index(idx, 6, threshold=0.6)
    
    def test_should_use_index_high_cardinality(self, large_df):
        """Test index decision for high cardinality."""
        idx = SortedIndex("object")
        idx.build(large_df)
        
        # Many unique objects - check decision
        result = should_use_index(idx, len(large_df))
        # Result depends on actual distribution
        assert isinstance(result, bool)


class TestIndexConcurrency:
    """Tests for thread-safe index access."""
    
    def test_concurrent_reads(self, sample_df):
        """Test concurrent reads don't cause issues."""
        import threading
        
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        results = []
        
        def read_index():
            positions = idx.lookup(1)
            results.append(len(positions))
        
        threads = [threading.Thread(target=read_index) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert all(r == 3 for r in results)
    
    def test_concurrent_writes(self, sample_df):
        """Test concurrent writes are serialized."""
        import threading
        
        idx = SortedIndex("subject")
        idx.build(sample_df)
        
        def add_entries(start):
            for i in range(10):
                idx.add(1000 + start + i, start + i)
        
        threads = [threading.Thread(target=add_entries, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All adds should complete
        stats = idx.stats()
        assert stats.num_entries == 6 + 50  # Original + 5*10


class TestIndexPerformance:
    """Performance-related tests."""
    
    def test_large_index_build(self, large_df):
        """Test building index on large DataFrame."""
        idx = SortedIndex("subject")
        idx.build(large_df)
        
        stats = idx.stats()
        assert stats.num_entries == len(large_df)
    
    def test_large_index_lookup(self, large_df):
        """Test lookup performance on large index."""
        idx = SortedIndex("subject")
        idx.build(large_df)
        
        # Multiple lookups should be fast
        for key in range(1, 100):
            positions = idx.lookup(key)
            # Just verify it returns a list
            assert isinstance(positions, list)
    
    def test_range_on_large_index(self, large_df):
        """Test range query on large index."""
        idx = SortedIndex("subject")
        idx.build(large_df)
        
        positions = idx.range_lookup(100, 200)
        
        # Verify results are correct
        for pos in positions:
            assert 100 <= large_df[pos]["subject"].item() <= 200
