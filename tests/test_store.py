"""
Tests for the TripleStore implementation.
"""

import pytest
from datetime import datetime
from rdf_starbase import TripleStore, ProvenanceContext


class TestTripleStore:
    
    def test_create_empty_store(self):
        """Test creating an empty triple store."""
        store = TripleStore()
        assert len(store) == 0
        stats = store.stats()
        assert stats["total_assertions"] == 0
    
    def test_add_simple_triple(self):
        """Test adding a simple triple."""
        store = TripleStore()
        provenance = ProvenanceContext(source="test_system", confidence=1.0)
        
        assertion_id = store.add_triple(
            subject="http://example.org/person/1",
            predicate="http://xmlns.com/foaf/0.1/name",
            obj="Alice",
            provenance=provenance
        )
        
        assert len(store) == 1
        assert assertion_id is not None
    
    def test_query_by_subject(self):
        """Test querying triples by subject."""
        store = TripleStore()
        provenance = ProvenanceContext(source="test_system")
        
        subject = "http://example.org/person/1"
        store.add_triple(subject, "http://xmlns.com/foaf/0.1/name", "Alice", provenance)
        store.add_triple(subject, "http://xmlns.com/foaf/0.1/age", 30, provenance)
        store.add_triple("http://example.org/person/2", "http://xmlns.com/foaf/0.1/name", "Bob", provenance)
        
        results = store.get_triples(subject=subject)
        assert len(results) == 2
    
    def test_query_by_predicate(self):
        """Test querying triples by predicate."""
        store = TripleStore()
        provenance = ProvenanceContext(source="test_system")
        
        predicate = "http://xmlns.com/foaf/0.1/name"
        store.add_triple("http://example.org/person/1", predicate, "Alice", provenance)
        store.add_triple("http://example.org/person/2", predicate, "Bob", provenance)
        store.add_triple("http://example.org/person/1", "http://xmlns.com/foaf/0.1/age", 30, provenance)
        
        results = store.get_triples(predicate=predicate)
        assert len(results) == 2
    
    def test_provenance_filtering(self):
        """Test filtering by provenance source."""
        store = TripleStore()
        
        prov1 = ProvenanceContext(source="system_a")
        prov2 = ProvenanceContext(source="system_b")
        
        store.add_triple("http://example.org/1", "http://example.org/prop", "value1", prov1)
        store.add_triple("http://example.org/2", "http://example.org/prop", "value2", prov2)
        
        results = store.get_triples(source="system_a")
        assert len(results) == 1
        assert results["source"][0] == "system_a"
    
    def test_confidence_filtering(self):
        """Test filtering by confidence threshold."""
        store = TripleStore()
        
        high_conf = ProvenanceContext(source="system", confidence=0.9)
        low_conf = ProvenanceContext(source="system", confidence=0.3)
        
        store.add_triple("http://example.org/1", "http://example.org/prop", "high", high_conf)
        store.add_triple("http://example.org/2", "http://example.org/prop", "low", low_conf)
        
        results = store.get_triples(min_confidence=0.5)
        assert len(results) == 1
        assert results["object"][0] == "high"
    
    def test_competing_claims(self):
        """Test detecting competing claims about the same subject-predicate."""
        store = TripleStore()
        
        subject = "http://example.org/customer/123"
        predicate = "http://example.org/age"
        
        # CRM says age is 34
        prov_crm = ProvenanceContext(source="CRM", confidence=0.8)
        store.add_triple(subject, predicate, 34, prov_crm)
        
        # Data Lake says age is 36
        prov_lake = ProvenanceContext(source="DataLake", confidence=0.9)
        store.add_triple(subject, predicate, 36, prov_lake)
        
        # Get competing claims
        claims = store.get_competing_claims(subject, predicate)
        
        assert len(claims) == 2
        # Should be sorted by confidence (desc)
        assert claims["confidence"][0] == 0.9
        assert claims["object"][0] == "36"
    
    def test_deprecate_assertion(self):
        """Test deprecating an assertion."""
        store = TripleStore()
        provenance = ProvenanceContext(source="system")
        
        assertion_id = store.add_triple(
            "http://example.org/1",
            "http://example.org/prop",
            "old_value",
            provenance
        )
        
        assert len(store) == 1
        
        store.deprecate_assertion(assertion_id)
        
        # Should not appear in default queries
        assert len(store) == 0
        
        # But should appear when including deprecated
        results = store.get_triples(include_deprecated=True)
        assert len(results) == 1
        assert results["deprecated"][0] is True
    
    def test_provenance_timeline(self):
        """Test getting the provenance timeline for a subject-predicate."""
        store = TripleStore()
        
        subject = "http://example.org/doc/1"
        predicate = "http://example.org/status"
        
        # Add assertions over time
        from datetime import timedelta, timezone
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        
        for i, status in enumerate(["draft", "review", "approved"]):
            prov = ProvenanceContext(
                source="workflow",
                timestamp=base_time + timedelta(days=i)
            )
            store.add_triple(subject, predicate, status, prov)
        
        timeline = store.get_provenance_timeline(subject, predicate)
        
        assert len(timeline) == 3
        # Should be sorted by time
        assert timeline["object"][0] == "draft"
        assert timeline["object"][2] == "approved"
    
    def test_stats(self):
        """Test store statistics."""
        store = TripleStore()
        prov = ProvenanceContext(source="test")
        
        store.add_triple("http://example.org/1", "http://example.org/p1", "v1", prov)
        store.add_triple("http://example.org/1", "http://example.org/p2", "v2", prov)
        store.add_triple("http://example.org/2", "http://example.org/p1", "v3", prov)
        
        stats = store.stats()
        
        assert stats["total_assertions"] == 3
        assert stats["active_assertions"] == 3
        assert stats["unique_subjects"] == 2
        assert stats["unique_predicates"] == 2
        assert stats["unique_sources"] == 1
