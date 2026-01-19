"""
Tests for Time-Travel queries with AS OF clause.
"""

import pytest
from datetime import datetime, timezone, timedelta
import time

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.sparql import parse_query, SPARQLExecutor
from rdf_starbase.sparql.ast import SelectQuery, AskQuery
from rdf_starbase.sparql.executor import execute_sparql


class TestAsOfParsing:
    """Tests for AS OF clause parsing."""
    
    def test_parse_select_with_as_of_datetime(self):
        """Test parsing SELECT with full ISO datetime."""
        query_str = '''
        SELECT ?s ?p ?o WHERE { ?s ?p ?o }
        AS OF "2025-01-15T00:00:00Z"
        '''
        query = parse_query(query_str)
        
        assert isinstance(query, SelectQuery)
        assert query.as_of is not None
        assert query.as_of.year == 2025
        assert query.as_of.month == 1
        assert query.as_of.day == 15
    
    def test_parse_select_with_as_of_date_only(self):
        """Test parsing SELECT with date only (no time)."""
        query_str = '''
        SELECT ?s WHERE { ?s ?p ?o }
        AS OF "2025-06-01"
        '''
        query = parse_query(query_str)
        
        assert isinstance(query, SelectQuery)
        assert query.as_of is not None
        assert query.as_of.year == 2025
        assert query.as_of.month == 6
        assert query.as_of.day == 1
    
    def test_parse_ask_with_as_of(self):
        """Test parsing ASK with AS OF clause."""
        query_str = '''
        ASK WHERE { ?s ?p ?o }
        AS OF "2024-12-31T23:59:59Z"
        '''
        query = parse_query(query_str)
        
        assert isinstance(query, AskQuery)
        assert query.as_of is not None
        assert query.as_of.year == 2024
        assert query.as_of.month == 12
        assert query.as_of.day == 31
    
    def test_parse_select_without_as_of(self):
        """Test that queries without AS OF have None."""
        query_str = '''
        SELECT ?s WHERE { ?s ?p ?o }
        '''
        query = parse_query(query_str)
        
        assert query.as_of is None
    
    def test_parse_select_with_as_of_and_limit(self):
        """Test AS OF combined with LIMIT/OFFSET."""
        query_str = '''
        SELECT ?s WHERE { ?s ?p ?o }
        LIMIT 10
        OFFSET 5
        AS OF "2025-01-15"
        '''
        query = parse_query(query_str)
        
        assert query.as_of is not None
        assert query.limit == 10
        assert query.offset == 5
    
    def test_parse_select_with_as_of_and_order_by(self):
        """Test AS OF combined with ORDER BY."""
        query_str = '''
        SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        ORDER BY ?name
        AS OF "2025-01-15"
        '''
        query = parse_query(query_str)
        
        assert query.as_of is not None
        assert len(query.order_by) == 1


class TestAsOfExecution:
    """Tests for AS OF clause execution (time-travel queries)."""
    
    @pytest.fixture
    def store_with_history(self):
        """Create store with triples added at different times."""
        store = TripleStore()
        
        # Add a triple at t0 (now - 2 hours)
        t0 = datetime.now(timezone.utc) - timedelta(hours=2)
        prov_t0 = ProvenanceContext(
            source="old_system", 
            confidence=0.8,
            timestamp=t0
        )
        store.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name",
            "Alice (old)",
            prov_t0
        )
        
        # Add another triple at t1 (now - 1 hour)
        t1 = datetime.now(timezone.utc) - timedelta(hours=1)
        prov_t1 = ProvenanceContext(
            source="new_system",
            confidence=0.9,
            timestamp=t1
        )
        store.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name",
            "Alice (new)",
            prov_t1
        )
        
        # Add a recent triple (now)
        prov_now = ProvenanceContext(
            source="current_system",
            confidence=1.0
        )
        store.add_triple(
            "http://example.org/bob",
            "http://xmlns.com/foaf/0.1/name",
            "Bob",
            prov_now
        )
        
        return store, t0, t1
    
    def test_as_of_returns_historical_state(self, store_with_history):
        """Test that AS OF returns data as it existed at that time."""
        store, t0, t1 = store_with_history
        
        # Query as of 90 minutes ago (should see Alice old, not Alice new or Bob)
        as_of_time = datetime.now(timezone.utc) - timedelta(minutes=90)
        as_of_str = as_of_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_str = f'''
        SELECT ?name WHERE {{
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name .
        }}
        AS OF "{as_of_str}"
        '''
        
        result = execute_sparql(store, query_str)
        
        # Should only see the old Alice name
        assert len(result) == 1
        assert result["name"][0] == "Alice (old)"
    
    def test_as_of_excludes_future_data(self, store_with_history):
        """Test that AS OF excludes data added after the specified time."""
        store, t0, t1 = store_with_history
        
        # Query as of 30 minutes ago (Bob was added now, should not appear)
        as_of_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        as_of_str = as_of_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_str = f'''
        SELECT ?s ?name WHERE {{
            ?s <http://xmlns.com/foaf/0.1/name> ?name .
        }}
        AS OF "{as_of_str}"
        '''
        
        result = execute_sparql(store, query_str)
        
        # Should see both Alice entries (old and new) but not Bob
        names = result["name"].to_list()
        assert "Bob" not in names
        assert len(names) == 2  # Both Alice entries
    
    def test_as_of_ask_returns_correct_answer(self, store_with_history):
        """Test ASK with AS OF returns correct boolean."""
        store, t0, t1 = store_with_history
        
        # Ask about Bob at a time before he was added
        as_of_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        as_of_str = as_of_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_str = f'''
        ASK WHERE {{
            <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> ?name .
        }}
        AS OF "{as_of_str}"
        '''
        
        result = execute_sparql(store, query_str)
        
        # Bob should not exist at that time
        assert result is False
        
        # Ask about Alice - should exist
        query_str2 = f'''
        ASK WHERE {{
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name .
        }}
        AS OF "{as_of_str}"
        '''
        
        result2 = execute_sparql(store, query_str2)
        assert result2 is True
    
    def test_as_of_without_clause_returns_current(self, store_with_history):
        """Test that query without AS OF returns current state."""
        store, t0, t1 = store_with_history
        
        query_str = '''
        SELECT ?s ?name WHERE {
            ?s <http://xmlns.com/foaf/0.1/name> ?name .
        }
        '''
        
        result = execute_sparql(store, query_str)
        
        # Should see all current entries (2 Alice + 1 Bob = 3)
        assert len(result) == 3
        names = result["name"].to_list()
        assert "Bob" in names


class TestAsOfEdgeCases:
    """Tests for edge cases with AS OF queries."""
    
    @pytest.fixture
    def store(self):
        """Create store with test data."""
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple("http://example.org/a", "http://example.org/p", "value", prov)
        
        return store
    
    def test_as_of_before_any_data(self, store):
        """Test AS OF at a time before any data existed."""
        # Query from 10 years ago
        old_time = datetime.now(timezone.utc) - timedelta(days=3650)
        as_of_str = old_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_str = f'''
        SELECT ?s WHERE {{ ?s ?p ?o }}
        AS OF "{as_of_str}"
        '''
        
        result = execute_sparql(store, query_str)
        
        # Should return empty
        assert len(result) == 0
    
    def test_as_of_future_time(self, store):
        """Test AS OF with a future time returns all current data."""
        # Query from 1 year in the future
        future_time = datetime.now(timezone.utc) + timedelta(days=365)
        as_of_str = future_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_str = f'''
        SELECT ?s WHERE {{ ?s ?p ?o }}
        AS OF "{as_of_str}"
        '''
        
        result = execute_sparql(store, query_str)
        
        # Should return all current data
        assert len(result) == 1
