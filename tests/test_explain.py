"""
Tests for EXPLAIN query plans.
"""

import pytest

from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.storage.query_context import ExplainPlan
from rdf_starbase.sparql.parser import parse_query
from rdf_starbase.sparql.executor import SPARQLExecutor


class TestExplainQueryPlan:
    """Test EXPLAIN functionality."""
    
    @pytest.fixture
    def store_with_data(self):
        """Create a store with some test data."""
        store = TripleStore()
        
        # Add some triples
        prov = ProvenanceContext(source="test")
        
        for i in range(100):
            store.add_triple(
                f"http://example.org/person{i}",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://example.org/Person",
                prov,
            )
            store.add_triple(
                f"http://example.org/person{i}",
                "http://example.org/name",
                f"Person {i}",
                prov,
            )
            store.add_triple(
                f"http://example.org/person{i}",
                "http://example.org/age",
                str(20 + i),
                prov,
            )
        
        return store
    
    def test_explain_simple_select(self, store_with_data):
        """Test EXPLAIN for simple SELECT query."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person ?name
            WHERE {
                ?person a ex:Person .
                ?person ex:name ?name .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert isinstance(plan, ExplainPlan)
        assert plan.query_type == "SELECT"
        assert len(plan.patterns) == 2
        
        # First pattern should be type pattern
        assert "Person" in plan.patterns[0]["description"]
        
        # Second pattern joins on ?person
        assert "person" in plan.patterns[1]["join_columns"]
        
        # Should have estimated cost
        assert plan.estimated_cost > 0
    
    def test_explain_with_filter(self, store_with_data):
        """Test EXPLAIN with FILTER clause."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person ?age
            WHERE {
                ?person ex:age ?age .
                FILTER (?age > 50)
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert len(plan.filters) > 0
        assert "50" in plan.filters[0] or "age" in plan.filters[0]
    
    def test_explain_with_limit(self, store_with_data):
        """Test EXPLAIN with LIMIT clause."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person
            WHERE {
                ?person a ex:Person .
            }
            LIMIT 10
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert plan.limit == 10
    
    def test_explain_with_order_by(self, store_with_data):
        """Test EXPLAIN with ORDER BY clause."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person ?name
            WHERE {
                ?person ex:name ?name .
            }
            ORDER BY ?name
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert len(plan.order_by) > 0
        assert "name" in plan.order_by[0]
    
    def test_explain_with_distinct(self, store_with_data):
        """Test EXPLAIN with DISTINCT."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT DISTINCT ?type
            WHERE {
                ?s a ?type .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert plan.distinct is True
    
    def test_explain_ask_query(self, store_with_data):
        """Test EXPLAIN for ASK query."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            ASK {
                ?person a ex:Person .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        assert plan.query_type == "ASK"
        assert plan.limit == 1  # ASK only needs one result
    
    def test_explain_selectivity_estimation(self, store_with_data):
        """Test that selectivity is estimated based on bound positions."""
        # Query with bound subject (more selective)
        query1 = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?p ?o
            WHERE {
                ex:person1 ?p ?o .
            }
        """)
        
        # Query with all variables (less selective)
        query2 = parse_query("""
            SELECT ?s ?p ?o
            WHERE {
                ?s ?p ?o .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan1 = executor.explain(query1)
        plan2 = executor.explain(query2)
        
        # Bound subject should have lower selectivity
        assert plan1.patterns[0]["selectivity"] < plan2.patterns[0]["selectivity"]
    
    def test_explain_join_columns(self, store_with_data):
        """Test that join columns are identified correctly."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person ?name ?age
            WHERE {
                ?person ex:name ?name .
                ?person ex:age ?age .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        # Second pattern should join on ?person
        assert len(plan.joins) > 0
        assert "person" in plan.joins[0]["columns"]
    
    def test_explain_to_string(self, store_with_data):
        """Test EXPLAIN plan string representation."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person ?name
            WHERE {
                ?person a ex:Person .
                ?person ex:name ?name .
            }
            ORDER BY ?name
            LIMIT 10
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        plan_str = str(plan)
        
        assert "SELECT" in plan_str
        assert "TriplePattern" in plan_str
        assert "Limit: 10" in plan_str
        assert "Order By" in plan_str
    
    def test_explain_to_dict(self, store_with_data):
        """Test EXPLAIN plan conversion to dict."""
        query = parse_query("""
            PREFIX ex: <http://example.org/>
            SELECT ?person
            WHERE {
                ?person a ex:Person .
            }
        """)
        
        executor = SPARQLExecutor(store_with_data)
        plan = executor.explain(query)
        
        d = plan.to_dict()
        
        assert d["query_type"] == "SELECT"
        assert "patterns" in d
        assert "estimated_cost" in d
