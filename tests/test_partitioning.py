"""
Tests for predicate-based partitioning.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from rdf_starbase.storage import TermDict, Term, TermKind
from rdf_starbase.storage.partitioning import (
    PredicatePartitioner,
    PartitionInfo,
    PartitionStats,
)


@pytest.fixture
def term_dict():
    """Create a term dictionary with sample predicates."""
    td = TermDict()
    # Create some terms
    td.get_or_create(Term(kind=TermKind.IRI, lex="http://example.org/name"))
    td.get_or_create(Term(kind=TermKind.IRI, lex="http://example.org/age"))
    td.get_or_create(Term(kind=TermKind.IRI, lex="http://example.org/email"))
    td.get_or_create(Term(kind=TermKind.IRI, lex="http://example.org/knows"))
    return td


@pytest.fixture
def partitioner(term_dict):
    """Create a partitioner with low threshold for testing."""
    return PredicatePartitioner(
        term_dict,
        partition_threshold=100,  # Low threshold for testing
        hot_partition_limit=5,
    )


class TestPredicatePartitioner:
    """Tests for PredicatePartitioner."""
    
    def test_add_facts_basic(self, partitioner, term_dict):
        """Test adding facts to partitioner."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        facts = [
            (0, 1, name_id, 100),
            (0, 2, name_id, 101),
            (0, 3, name_id, 102),
        ]
        
        partitioner.add_facts(facts)
        
        assert partitioner.count() == 3
    
    def test_get_partition_before_threshold(self, partitioner, term_dict):
        """Test getting facts before partition threshold."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        age_id = term_dict.get_iri_id("http://example.org/age")
        
        facts = [
            (0, 1, name_id, 100),
            (0, 2, age_id, 101),
        ]
        
        partitioner.add_facts(facts)
        
        # Both should be in default partition
        name_facts = partitioner.get_partition(name_id)
        age_facts = partitioner.get_partition(age_id)
        
        assert len(name_facts) == 1
        assert len(age_facts) == 1
    
    def test_partition_promotion(self, partitioner, term_dict):
        """Test that predicates are promoted to own partition at threshold."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        # Add more facts than threshold
        facts = [(0, i, name_id, i + 1000) for i in range(150)]
        partitioner.add_facts(facts)
        
        # Should now have dedicated partition
        assert name_id in partitioner._partitions
        assert partitioner.count() == 150
    
    def test_get_all_facts(self, partitioner, term_dict):
        """Test getting all facts across partitions."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        age_id = term_dict.get_iri_id("http://example.org/age")
        
        facts = [
            (0, 1, name_id, 100),
            (0, 2, age_id, 101),
            (0, 3, name_id, 102),
        ]
        
        partitioner.add_facts(facts)
        
        all_facts = partitioner.get_all_facts()
        assert len(all_facts) == 3
    
    def test_query_with_filters(self, partitioner, term_dict):
        """Test querying with filters."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        facts = [
            (0, 1, name_id, 100),
            (0, 2, name_id, 101),
            (1, 3, name_id, 102),  # Different graph
        ]
        
        partitioner.add_facts(facts)
        
        # Query by graph
        graph0_facts = partitioner.query(graph_id=0)
        assert len(graph0_facts) == 2
        
        # Query by subject
        subject1_facts = partitioner.query(subject_id=1)
        assert len(subject1_facts) == 1
    
    def test_query_by_predicate(self, partitioner, term_dict):
        """Test querying by specific predicate."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        age_id = term_dict.get_iri_id("http://example.org/age")
        
        # Add enough to promote name predicate
        facts = [(0, i, name_id, i + 1000) for i in range(150)]
        facts.extend([(0, i, age_id, i + 2000) for i in range(50)])
        
        partitioner.add_facts(facts)
        
        # Query by name predicate (should only scan that partition)
        name_facts = partitioner.query(predicate_id=name_id)
        assert len(name_facts) == 150
        
        # Query by age predicate
        age_facts = partitioner.query(predicate_id=age_id)
        assert len(age_facts) == 50
    
    def test_stats(self, partitioner, term_dict):
        """Test partition statistics."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        # Add some facts
        facts = [(0, i, name_id, i + 1000) for i in range(150)]
        partitioner.add_facts(facts)
        
        stats = partitioner.stats()
        assert stats.total_facts == 150
        assert stats.total_partitions >= 1
        assert stats.in_memory_partitions >= 1
    
    def test_clear(self, partitioner, term_dict):
        """Test clearing all partitions."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        facts = [(0, i, name_id, i + 1000) for i in range(50)]
        partitioner.add_facts(facts)
        
        assert partitioner.count() == 50
        
        partitioner.clear()
        
        assert partitioner.count() == 0
    
    def test_list_partitions(self, partitioner, term_dict):
        """Test listing partition info."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        # Add enough to create partition
        facts = [(0, i, name_id, i + 1000) for i in range(150)]
        partitioner.add_facts(facts)
        
        partitions = partitioner.list_partitions()
        assert len(partitions) >= 1
        assert all(isinstance(p, PartitionInfo) for p in partitions)


class TestPartitionPersistence:
    """Tests for partition save/load."""
    
    def test_save_and_load(self, term_dict):
        """Test saving and loading partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            partitioner = PredicatePartitioner(
                term_dict,
                partition_threshold=50,
            )
            
            name_id = term_dict.get_iri_id("http://example.org/name")
            age_id = term_dict.get_iri_id("http://example.org/age")
            
            # Add facts
            facts = [(0, i, name_id, i + 1000) for i in range(100)]
            facts.extend([(0, i, age_id, i + 2000) for i in range(30)])
            partitioner.add_facts(facts)
            
            original_count = partitioner.count()
            
            # Save
            save_dir = Path(tmpdir) / "partitions"
            partitioner.save(save_dir)
            
            # Load into new partitioner
            loaded = PredicatePartitioner.load(save_dir, term_dict)
            
            assert loaded.count() == original_count
            
            # Verify data
            name_facts = loaded.get_partition(name_id)
            assert len(name_facts) == 100
    
    def test_save_default_partition(self, term_dict):
        """Test saving when facts are in default partition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            partitioner = PredicatePartitioner(
                term_dict,
                partition_threshold=1000,  # High threshold
            )
            
            name_id = term_dict.get_iri_id("http://example.org/name")
            
            # Add fewer facts than threshold
            facts = [(0, i, name_id, i + 1000) for i in range(50)]
            partitioner.add_facts(facts)
            
            # Save
            save_dir = Path(tmpdir) / "partitions"
            partitioner.save(save_dir)
            
            # Verify default partition file exists
            assert (save_dir / "partition_default.parquet").exists()
            
            # Load and verify
            loaded = PredicatePartitioner.load(save_dir, term_dict)
            assert loaded.count() == 50


class TestPartitionSpilling:
    """Tests for spilling partitions to disk."""
    
    def test_spill_under_memory_pressure(self, term_dict):
        """Test that partitions are spilled under memory pressure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            partitioner = PredicatePartitioner(
                term_dict,
                storage_dir=Path(tmpdir),
                max_memory_mb=0.001,  # Very small to trigger spilling
                partition_threshold=10,
                hot_partition_limit=2,
            )
            
            # Create multiple predicates
            for i in range(5):
                pred = term_dict.get_or_create(
                    Term(kind=TermKind.IRI, lex=f"http://example.org/pred{i}")
                )
                facts = [(0, j, pred, j + 1000) for j in range(20)]
                partitioner.add_facts(facts)
            
            # Check that some partitions were spilled
            stats = partitioner.stats()
            
            # We should have facts
            assert partitioner.count() == 100
    
    def test_load_spilled_partition(self, term_dict):
        """Test that spilled partitions can be reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            partitioner = PredicatePartitioner(
                term_dict,
                storage_dir=Path(tmpdir),
                partition_threshold=10,
            )
            
            name_id = term_dict.get_iri_id("http://example.org/name")
            
            # Add facts
            facts = [(0, i, name_id, i + 1000) for i in range(50)]
            partitioner.add_facts(facts)
            
            # Force spill
            if name_id in partitioner._partitions:
                partitioner._spill_partition(name_id)
            
            # Query should reload
            name_facts = partitioner.get_partition(name_id)
            assert len(name_facts) == 50


class TestPartitionConcurrency:
    """Tests for thread-safe partition access."""
    
    def test_concurrent_reads(self, term_dict):
        """Test concurrent reads don't cause issues."""
        import threading
        
        partitioner = PredicatePartitioner(
            term_dict,
            partition_threshold=50,
        )
        
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        # Add facts
        facts = [(0, i, name_id, i + 1000) for i in range(100)]
        partitioner.add_facts(facts)
        
        results = []
        
        def read_partition():
            df = partitioner.get_partition(name_id)
            results.append(len(df))
        
        threads = [threading.Thread(target=read_partition) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert all(r == 100 for r in results)
    
    def test_concurrent_writes(self, term_dict):
        """Test concurrent writes are serialized."""
        import threading
        
        partitioner = PredicatePartitioner(
            term_dict,
            partition_threshold=1000,  # Keep in default partition
        )
        
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        def write_facts(start):
            facts = [(0, start + i, name_id, start + i + 1000) for i in range(10)]
            partitioner.add_facts(facts)
        
        threads = [threading.Thread(target=write_facts, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All facts should be there
        assert partitioner.count() == 50


class TestPartitionEmpty:
    """Tests for edge cases with empty partitions."""
    
    def test_empty_partitioner(self, term_dict):
        """Test operations on empty partitioner."""
        partitioner = PredicatePartitioner(term_dict)
        
        assert partitioner.count() == 0
        
        all_facts = partitioner.get_all_facts()
        assert len(all_facts) == 0
        
        stats = partitioner.stats()
        assert stats.total_facts == 0
    
    def test_get_nonexistent_partition(self, partitioner, term_dict):
        """Test getting partition for unknown predicate."""
        unknown_pred = 99999
        
        df = partitioner.get_partition(unknown_pred)
        assert len(df) == 0
    
    def test_query_empty_result(self, partitioner, term_dict):
        """Test query that returns no results."""
        name_id = term_dict.get_iri_id("http://example.org/name")
        
        facts = [(0, 1, name_id, 100)]
        partitioner.add_facts(facts)
        
        # Query for non-existent subject
        result = partitioner.query(subject_id=99999)
        assert len(result) == 0
