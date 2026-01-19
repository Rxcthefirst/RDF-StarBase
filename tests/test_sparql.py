"""
Tests for the SPARQL-Star parser and executor.
"""

import pytest
from datetime import datetime

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.sparql import (
    parse_query,
    SPARQLExecutor,
    SelectQuery,
    AskQuery,
    TriplePattern,
    QuotedTriplePattern,
    Variable,
    IRI,
    Literal,
)
from rdf_starbase.sparql.executor import execute_sparql


class TestSPARQLParser:
    """Tests for SPARQL-Star parsing."""
    
    def test_parse_simple_select(self):
        """Test parsing a simple SELECT query."""
        query_str = """
        SELECT ?s ?p ?o
        WHERE { ?s ?p ?o }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, SelectQuery)
        assert len(query.variables) == 3
        assert query.variables[0].name == "s"
        assert query.variables[1].name == "p"
        assert query.variables[2].name == "o"
    
    def test_parse_select_star(self):
        """Test parsing SELECT * query."""
        query_str = """
        SELECT *
        WHERE { ?s ?p ?o }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, SelectQuery)
        assert query.is_select_all()
    
    def test_parse_with_iri(self):
        """Test parsing query with IRI values."""
        query_str = """
        SELECT ?name
        WHERE { <http://example.org/person/1> <http://xmlns.com/foaf/0.1/name> ?name }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, SelectQuery)
        pattern = query.where.patterns[0]
        assert isinstance(pattern.subject, IRI)
        assert pattern.subject.value == "http://example.org/person/1"
    
    def test_parse_with_literal(self):
        """Test parsing query with literal values."""
        query_str = """
        SELECT ?s
        WHERE { ?s <http://xmlns.com/foaf/0.1/name> "Alice" }
        """
        query = parse_query(query_str)
        
        pattern = query.where.patterns[0]
        assert isinstance(pattern.object, Literal)
        assert pattern.object.value == "Alice"
    
    def test_parse_with_filter(self):
        """Test parsing query with FILTER clause."""
        query_str = """
        SELECT ?s ?age
        WHERE {
            ?s <http://example.org/age> ?age
            FILTER(?age > 30)
        }
        """
        query = parse_query(query_str)
        
        assert len(query.where.filters) == 1
    
    def test_parse_distinct(self):
        """Test parsing SELECT DISTINCT."""
        query_str = """
        SELECT DISTINCT ?s
        WHERE { ?s ?p ?o }
        """
        query = parse_query(query_str)
        
        assert query.distinct is True
    
    def test_parse_limit_offset(self):
        """Test parsing LIMIT and OFFSET."""
        query_str = """
        SELECT ?s ?p ?o
        WHERE { ?s ?p ?o }
        LIMIT 10
        OFFSET 5
        """
        query = parse_query(query_str)
        
        assert query.limit == 10
        assert query.offset == 5
    
    def test_parse_order_by(self):
        """Test parsing ORDER BY."""
        query_str = """
        SELECT ?s ?name
        WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        ORDER BY ?name
        """
        query = parse_query(query_str)
        
        assert len(query.order_by) == 1
        assert query.order_by[0][0].name == "name"
        assert query.order_by[0][1] is True  # ascending
    
    def test_parse_order_by_desc(self):
        """Test parsing ORDER BY DESC."""
        query_str = """
        SELECT ?s ?age
        WHERE { ?s <http://example.org/age> ?age }
        ORDER BY DESC(?age)
        """
        query = parse_query(query_str)
        
        assert query.order_by[0][1] is False  # descending
    
    def test_parse_prefix(self):
        """Test parsing PREFIX declarations."""
        query_str = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name
        WHERE { ?s foaf:name ?name }
        """
        query = parse_query(query_str)
        
        assert "foaf" in query.prefixes
        assert query.prefixes["foaf"] == "http://xmlns.com/foaf/0.1/"
    
    def test_parse_ask_query(self):
        """Test parsing ASK query."""
        query_str = """
        ASK WHERE { ?s ?p ?o }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, AskQuery)
    
    def test_parse_quoted_triple(self):
        """Test parsing RDF-Star quoted triple pattern."""
        query_str = """
        SELECT ?who ?when
        WHERE {
            << <http://example.org/s> <http://example.org/p> <http://example.org/o> >> <http://example.org/assertedBy> ?who
        }
        """
        query = parse_query(query_str)
        
        pattern = query.where.patterns[0]
        assert isinstance(pattern.subject, QuotedTriplePattern)
    
    def test_parse_multiple_patterns(self):
        """Test parsing query with multiple triple patterns."""
        query_str = """
        SELECT ?name ?age
        WHERE {
            ?person <http://xmlns.com/foaf/0.1/name> ?name .
            ?person <http://example.org/age> ?age .
        }
        """
        query = parse_query(query_str)
        
        assert len(query.where.patterns) == 2


class TestSPARQLExecutor:
    """Tests for SPARQL-Star query execution."""
    
    @pytest.fixture
    def store_with_data(self):
        """Create a store with test data."""
        store = TripleStore()
        
        # Add some people
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple(
            "http://example.org/person/1",
            "http://xmlns.com/foaf/0.1/name",
            "Alice",
            prov
        )
        store.add_triple(
            "http://example.org/person/1",
            "http://example.org/age",
            30,
            prov
        )
        store.add_triple(
            "http://example.org/person/2",
            "http://xmlns.com/foaf/0.1/name",
            "Bob",
            prov
        )
        store.add_triple(
            "http://example.org/person/2",
            "http://example.org/age",
            25,
            prov
        )
        store.add_triple(
            "http://example.org/person/3",
            "http://xmlns.com/foaf/0.1/name",
            "Charlie",
            prov
        )
        store.add_triple(
            "http://example.org/person/3",
            "http://example.org/age",
            35,
            prov
        )
        
        return store
    
    def test_execute_simple_select(self, store_with_data):
        """Test executing a simple SELECT query."""
        result = execute_sparql(store_with_data, """
            SELECT ?s ?name
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        assert len(result) == 3
        assert "s" in result.columns
        assert "name" in result.columns
    
    def test_execute_with_concrete_subject(self, store_with_data):
        """Test query with concrete subject."""
        result = execute_sparql(store_with_data, """
            SELECT ?p ?o
            WHERE { <http://example.org/person/1> ?p ?o }
        """)
        
        assert len(result) == 2  # name and age
    
    def test_execute_with_concrete_predicate(self, store_with_data):
        """Test query filtering by predicate."""
        result = execute_sparql(store_with_data, """
            SELECT ?s ?name
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        assert len(result) == 3
        names = result["name"].to_list()
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names
    
    def test_execute_with_concrete_object(self, store_with_data):
        """Test query filtering by object value."""
        result = execute_sparql(store_with_data, """
            SELECT ?s
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> "Alice" }
        """)
        
        assert len(result) == 1
        assert result["s"][0] == "http://example.org/person/1"
    
    def test_execute_join(self, store_with_data):
        """Test query with join on shared variable."""
        result = execute_sparql(store_with_data, """
            SELECT ?s ?name ?age
            WHERE {
                ?s <http://xmlns.com/foaf/0.1/name> ?name .
                ?s <http://example.org/age> ?age .
            }
        """)
        
        assert len(result) == 3
        assert "name" in result.columns
        assert "age" in result.columns
    
    def test_execute_limit(self, store_with_data):
        """Test LIMIT clause."""
        result = execute_sparql(store_with_data, """
            SELECT ?s ?name
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
            LIMIT 2
        """)
        
        assert len(result) == 2
    
    def test_execute_distinct(self, store_with_data):
        """Test DISTINCT modifier."""
        # Add duplicate-ish data
        prov = ProvenanceContext(source="test2", confidence=0.9)
        store_with_data.add_triple(
            "http://example.org/person/1",
            "http://xmlns.com/foaf/0.1/name",
            "Alice",  # Same name from different source
            prov
        )
        
        result = execute_sparql(store_with_data, """
            SELECT DISTINCT ?name
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        assert len(result) == 3  # Still only 3 unique names
    
    def test_execute_ask_true(self, store_with_data):
        """Test ASK query returning true."""
        result = execute_sparql(store_with_data, """
            ASK WHERE { <http://example.org/person/1> <http://xmlns.com/foaf/0.1/name> "Alice" }
        """)
        
        assert result is True
    
    def test_execute_ask_false(self, store_with_data):
        """Test ASK query returning false."""
        result = execute_sparql(store_with_data, """
            ASK WHERE { <http://example.org/person/1> <http://xmlns.com/foaf/0.1/name> "Unknown" }
        """)
        
        assert result is False
    
    def test_execute_with_filter_gt(self, store_with_data):
        """Test FILTER with greater than."""
        result = execute_sparql(store_with_data, """
            SELECT ?s ?age
            WHERE {
                ?s <http://example.org/age> ?age
                FILTER(?age > 28)
            }
        """)
        
        # Should return Alice (30) and Charlie (35), not Bob (25)
        assert len(result) == 2
    
    def test_execute_select_star(self, store_with_data):
        """Test SELECT * query."""
        result = execute_sparql(store_with_data, """
            SELECT *
            WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }
        """)
        
        assert len(result) == 3
        assert "s" in result.columns
        assert "name" in result.columns


class TestProvenanceFilters:
    """Tests for RDF-StarBase provenance filter extensions."""
    
    @pytest.fixture
    def store_with_provenance(self):
        """Create a store with data from multiple sources."""
        store = TripleStore()
        
        # Data from CRM
        crm_prov = ProvenanceContext(source="CRM", confidence=0.8)
        store.add_triple(
            "http://example.org/customer/1",
            "http://example.org/age",
            34,
            crm_prov
        )
        
        # Data from DataLake
        lake_prov = ProvenanceContext(source="DataLake", confidence=0.95)
        store.add_triple(
            "http://example.org/customer/1",
            "http://example.org/age",
            36,
            lake_prov
        )
        
        # Low confidence data
        ml_prov = ProvenanceContext(source="ML_Model", confidence=0.4)
        store.add_triple(
            "http://example.org/customer/1",
            "http://example.org/churnRisk",
            0.7,
            ml_prov
        )
        
        return store
    
    def test_competing_claims_visible(self, store_with_provenance):
        """Test that competing claims are returned by standard query."""
        result = execute_sparql(store_with_provenance, """
            SELECT ?age
            WHERE {
                <http://example.org/customer/1> <http://example.org/age> ?age
            }
        """)
        
        # Both values should be returned
        assert len(result) == 2
        ages = result["age"].to_list()
        assert "34" in ages
        assert "36" in ages


class TestProvenanceAnnotations:
    """Tests for RDF-Star provenance predicate recognition."""
    
    def test_prov_wasAttributedTo_sets_source(self):
        """Test that prov:wasAttributedTo sets the source field."""
        store = TripleStore()
        
        # Insert with RDF-Star provenance annotation
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            
            INSERT DATA {
                ex:alice ex:name "Alice" .
                << ex:alice ex:name "Alice" >> prov:wasAttributedTo "Wikipedia" .
            }
        """)
        
        assert result["count"] == 1  # Only 1 actual triple, annotation is metadata
        
        # Query and check source
        triples = store.get_triples(subject="http://example.org/alice")
        assert len(triples) == 1
        assert triples["source"][0] == "Wikipedia"
    
    def test_prov_value_sets_confidence(self):
        """Test that prov:value sets the confidence field."""
        store = TripleStore()
        
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            
            INSERT DATA {
                ex:bob ex:age "30" .
                << ex:bob ex:age "30" >> prov:value 0.85 .
            }
        """)
        
        assert result["count"] == 1
        
        triples = store.get_triples(subject="http://example.org/bob")
        assert len(triples) == 1
        assert triples["confidence"][0] == 0.85
    
    def test_combined_source_and_confidence(self):
        """Test that both source and confidence can be set."""
        store = TripleStore()
        
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            
            INSERT DATA {
                ex:movie ex:title "Inception" .
                << ex:movie ex:title "Inception" >> prov:wasAttributedTo "IMDb" .
                << ex:movie ex:title "Inception" >> prov:value 0.99 .
            }
        """)
        
        assert result["count"] == 1
        
        triples = store.get_triples(subject="http://example.org/movie")
        assert len(triples) == 1
        assert triples["source"][0] == "IMDb"
        assert triples["confidence"][0] == 0.99
    
    def test_annotation_only_creates_triple(self):
        """Test that annotation-only patterns still create the base triple."""
        store = TripleStore()
        
        # Only annotations, no explicit base triple
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            
            INSERT DATA {
                << ex:person ex:knows ex:friend >> prov:wasAttributedTo "SocialGraph" .
                << ex:person ex:knows ex:friend >> prov:value 0.75 .
            }
        """)
        
        assert result["count"] == 1  # Base triple should be created
        
        triples = store.get_triples(
            subject="http://example.org/person",
            predicate="http://example.org/knows"
        )
        assert len(triples) == 1
        assert triples["source"][0] == "SocialGraph"
        assert triples["confidence"][0] == 0.75
    
    def test_dublin_core_source(self):
        """Test that Dublin Core source predicate is recognized."""
        store = TripleStore()
        
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            
            INSERT DATA {
                ex:data ex:value "test" .
                << ex:data ex:value "test" >> dcterms:source "OpenData" .
            }
        """)
        
        assert result["count"] == 1
        
        triples = store.get_triples(subject="http://example.org/data")
        assert triples["source"][0] == "OpenData"
    
    def test_pav_createdBy(self):
        """Test that PAV createdBy is recognized as source."""
        store = TripleStore()
        
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX pav: <http://purl.org/pav/>
            
            INSERT DATA {
                ex:doc ex:content "Hello" .
                << ex:doc ex:content "Hello" >> pav:createdBy "AuthorSystem" .
            }
        """)
        
        assert result["count"] == 1
        
        triples = store.get_triples(subject="http://example.org/doc")
        assert triples["source"][0] == "AuthorSystem"
    
    def test_dqv_quality_measurement(self):
        """Test that DQV quality measurement sets confidence."""
        store = TripleStore()
        
        result = execute_sparql(store, """
            PREFIX ex: <http://example.org/>
            PREFIX dqv: <http://www.w3.org/ns/dqv#>
            
            INSERT DATA {
                ex:sensor ex:reading "42.5" .
                << ex:sensor ex:reading "42.5" >> dqv:hasQualityMeasurement 0.92 .
            }
        """)
        
        assert result["count"] == 1
        
        triples = store.get_triples(subject="http://example.org/sensor")
        assert triples["confidence"][0] == 0.92
