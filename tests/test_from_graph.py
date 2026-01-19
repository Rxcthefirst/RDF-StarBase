"""
Tests for FROM clause and GRAPH pattern support.

Tests querying specific named graphs and the default graph.
"""

import pytest
from datetime import datetime, timezone

from rdf_starbase.store import TripleStore
from rdf_starbase.sparql.parser import parse_query
from rdf_starbase.sparql.executor import execute_sparql
from rdf_starbase.sparql.ast import SelectQuery, GraphPattern, IRI, Variable
from rdf_starbase.models import ProvenanceContext


@pytest.fixture
def multi_graph_store():
    """Create a store with data in multiple graphs."""
    store = TripleStore()
    prov = ProvenanceContext(source="test", confidence=0.9)
    
    # Default graph data
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/name",
        "Alice",
        prov
    )
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/age",
        "30",
        prov
    )
    
    # Graph 1: People
    store.add_triple(
        "http://example.org/bob",
        "http://xmlns.com/foaf/0.1/name",
        "Bob",
        prov,
        graph="http://example.org/graphs/people"
    )
    store.add_triple(
        "http://example.org/bob",
        "http://xmlns.com/foaf/0.1/age",
        "25",
        prov,
        graph="http://example.org/graphs/people"
    )
    
    # Graph 2: Organizations
    store.add_triple(
        "http://example.org/acme",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "ACME Corp",
        prov,
        graph="http://example.org/graphs/orgs"
    )
    store.add_triple(
        "http://example.org/acme",
        "http://schema.org/foundingDate",
        "2010-01-15",
        prov,
        graph="http://example.org/graphs/orgs"
    )
    
    return store


class TestFromClauseParsing:
    """Test parsing of FROM clauses."""
    
    def test_parse_select_with_from(self):
        """Test parsing SELECT with single FROM."""
        query = parse_query("""
            SELECT ?s ?name
            FROM <http://example.org/graphs/people>
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        assert isinstance(query, SelectQuery)
        assert len(query.from_graphs) == 1
        assert query.from_graphs[0].value == "http://example.org/graphs/people"
    
    def test_parse_select_with_multiple_from(self):
        """Test parsing SELECT with multiple FROM clauses."""
        query = parse_query("""
            SELECT ?s ?name
            FROM <http://example.org/graphs/people>
            FROM <http://example.org/graphs/orgs>
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        assert isinstance(query, SelectQuery)
        assert len(query.from_graphs) == 2
    
    def test_parse_select_with_from_named(self):
        """Test parsing SELECT with FROM NAMED."""
        query = parse_query("""
            SELECT ?g ?s ?name
            FROM NAMED <http://example.org/graphs/people>
            WHERE { GRAPH ?g { ?s <http://xmlns.com/foaf/0.1/name> ?name } }
        """)
        assert isinstance(query, SelectQuery)
        assert len(query.from_named_graphs) == 1
        assert query.from_named_graphs[0].value == "http://example.org/graphs/people"


class TestGraphPatternParsing:
    """Test parsing of GRAPH patterns."""
    
    def test_parse_graph_pattern_iri(self):
        """Test parsing GRAPH pattern with IRI."""
        query = parse_query("""
            SELECT ?s ?name WHERE {
                GRAPH <http://example.org/graphs/people> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        assert isinstance(query, SelectQuery)
        assert len(query.where.graph_patterns) == 1
        gp = query.where.graph_patterns[0]
        assert isinstance(gp.graph, IRI)
        assert gp.graph.value == "http://example.org/graphs/people"
    
    def test_parse_graph_pattern_variable(self):
        """Test parsing GRAPH pattern with variable."""
        query = parse_query("""
            SELECT ?g ?s ?name WHERE {
                GRAPH ?g {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        assert isinstance(query, SelectQuery)
        assert len(query.where.graph_patterns) == 1
        gp = query.where.graph_patterns[0]
        assert isinstance(gp.graph, Variable)
        assert gp.graph.name == "g"


class TestFromClauseExecution:
    """Test execution with FROM clauses."""
    
    def test_from_restricts_to_graph(self, multi_graph_store):
        """Test that FROM restricts query to specified graph."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name
            FROM <http://example.org/graphs/people>
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        # Should only find Bob from people graph, not Alice from default
        assert len(result) == 1
        names = result["name"].to_list()
        assert "Bob" in names
        assert "Alice" not in names
    
    def test_from_multiple_graphs(self, multi_graph_store):
        """Test FROM with multiple graphs."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name
            FROM <http://example.org/graphs/people>
            FROM <http://example.org/graphs/orgs>
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        # Should only find Bob (orgs graph doesn't have foaf:name)
        assert len(result) == 1
    
    def test_no_from_queries_all(self, multi_graph_store):
        """Test that without FROM, all graphs are queried."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name WHERE {
                ?s <http://xmlns.com/foaf/0.1/name> ?name
            }
        """)
        
        # Should find both Alice (default) and Bob (people)
        assert len(result) >= 2
        names = result["name"].to_list()
        assert "Alice" in names
        assert "Bob" in names


class TestGraphPatternExecution:
    """Test execution of GRAPH patterns."""
    
    def test_graph_pattern_with_iri(self, multi_graph_store):
        """Test GRAPH pattern with specific IRI."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name WHERE {
                GRAPH <http://example.org/graphs/people> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        
        assert len(result) == 1
        assert result["name"][0] == "Bob"
    
    def test_graph_pattern_with_variable(self, multi_graph_store):
        """Test GRAPH pattern with variable binds graph URI."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?g ?s ?name WHERE {
                GRAPH ?g {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        
        # Should only match named graphs, not default
        assert len(result) >= 1
        graphs = result["g"].to_list()
        assert "http://example.org/graphs/people" in graphs
    
    def test_graph_pattern_nonexistent_graph(self, multi_graph_store):
        """Test GRAPH pattern with non-existent graph."""
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name WHERE {
                GRAPH <http://example.org/nonexistent> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        
        assert len(result) == 0
    
    def test_combined_patterns_and_graph(self, multi_graph_store):
        """Test combining regular patterns with GRAPH patterns."""
        # Add a connecting triple
        prov = ProvenanceContext(source="test", confidence=0.9)
        multi_graph_store.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/knows",
            "http://example.org/bob",
            prov
        )
        
        result = execute_sparql(multi_graph_store, """
            SELECT ?person ?friend ?friendName WHERE {
                ?person <http://xmlns.com/foaf/0.1/knows> ?friend .
                GRAPH <http://example.org/graphs/people> {
                    ?friend <http://xmlns.com/foaf/0.1/name> ?friendName
                }
            }
        """)
        
        assert len(result) >= 1
        assert "Bob" in result["friendName"].to_list()


class TestGraphManagementIntegration:
    """Integration tests combining graph management with query."""
    
    def test_load_and_query_graph(self, multi_graph_store):
        """Test loading data and querying specific graph."""
        # First verify data in orgs graph
        result = execute_sparql(multi_graph_store, """
            SELECT ?org ?label WHERE {
                GRAPH <http://example.org/graphs/orgs> {
                    ?org <http://www.w3.org/2000/01/rdf-schema#label> ?label
                }
            }
        """)
        
        assert len(result) == 1
        assert result["label"][0] == "ACME Corp"
    
    def test_copy_graph_and_query(self, multi_graph_store):
        """Test copying a graph and querying the copy."""
        # Copy people to backup
        execute_sparql(
            multi_graph_store,
            "COPY GRAPH <http://example.org/graphs/people> TO GRAPH <http://example.org/graphs/backup>"
        )
        
        # Query the backup
        result = execute_sparql(multi_graph_store, """
            SELECT ?s ?name WHERE {
                GRAPH <http://example.org/graphs/backup> {
                    ?s <http://xmlns.com/foaf/0.1/name> ?name
                }
            }
        """)
        
        assert len(result) == 1
        assert result["name"][0] == "Bob"
