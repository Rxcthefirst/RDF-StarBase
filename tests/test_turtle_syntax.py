"""
Tests for Turtle-style syntax support in INSERT DATA.

Tests that the parser correctly handles:
- Semicolons for property lists (same subject, different predicates)
- Commas for object lists (same subject+predicate, multiple objects)
- Prefixed names
- The 'a' keyword for rdf:type
"""

import pytest
from rdf_starbase.sparql.parser import parse_query
from rdf_starbase.sparql.ast import InsertDataQuery, TriplePattern, IRI, Literal


class TestTurtleSyntaxParsing:
    """Test Turtle-style syntax in INSERT DATA."""
    
    def test_simple_triple(self):
        """Test basic triple with full IRIs."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 1
        assert query.triples[0].subject.value == "http://example.org/alice"
        assert query.triples[0].predicate.value == "http://xmlns.com/foaf/0.1/name"
        assert query.triples[0].object.value == "Alice"
    
    def test_semicolon_property_list(self):
        """Test semicolon for property lists (same subject, different predicates)."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> 
                    <http://xmlns.com/foaf/0.1/name> "Alice" ;
                    <http://xmlns.com/foaf/0.1/age> "30" ;
                    <http://xmlns.com/foaf/0.1/email> "alice@example.org" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 3
        
        # All should have same subject
        subjects = [t.subject.value for t in query.triples]
        assert all(s == "http://example.org/alice" for s in subjects)
        
        # Check predicates
        predicates = [t.predicate.value for t in query.triples]
        assert "http://xmlns.com/foaf/0.1/name" in predicates
        assert "http://xmlns.com/foaf/0.1/age" in predicates
        assert "http://xmlns.com/foaf/0.1/email" in predicates
    
    def test_comma_object_list(self):
        """Test comma for object lists (same subject+predicate, multiple objects)."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> <http://xmlns.com/foaf/0.1/interest> 
                    "Music" , "Art" , "Science" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 3
        
        # All should have same subject and predicate
        for t in query.triples:
            assert t.subject.value == "http://example.org/alice"
            assert t.predicate.value == "http://xmlns.com/foaf/0.1/interest"
        
        # Different objects
        objects = [t.object.value for t in query.triples]
        assert "Music" in objects
        assert "Art" in objects
        assert "Science" in objects
    
    def test_combined_semicolon_and_comma(self):
        """Test combined semicolon and comma syntax."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> 
                    <http://xmlns.com/foaf/0.1/name> "Alice" ;
                    <http://xmlns.com/foaf/0.1/interest> "Music" , "Art" ;
                    <http://xmlns.com/foaf/0.1/age> "30" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 4
        
        # Check we have correct distribution
        objects = [t.object.value for t in query.triples]
        assert "Alice" in objects
        assert "Music" in objects
        assert "Art" in objects
        assert "30" in objects
    
    def test_prefixed_names(self):
        """Test prefixed names."""
        query = parse_query("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>
            
            INSERT DATA {
                ex:alice foaf:name "Alice" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 1
        assert query.triples[0].subject.value == "ex:alice"
        assert query.triples[0].predicate.value == "foaf:name"
    
    def test_a_keyword_for_rdf_type(self):
        """Test 'a' keyword as shorthand for rdf:type."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> a <http://xmlns.com/foaf/0.1/Person> .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 1
        assert query.triples[0].predicate.value == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    
    def test_multiple_subjects(self):
        """Test multiple subject blocks."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
                <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 2
        
        subjects = [t.subject.value for t in query.triples]
        assert "http://example.org/alice" in subjects
        assert "http://example.org/bob" in subjects
    
    def test_complex_turtle_block(self):
        """Test complex Turtle-style data block."""
        query = parse_query("""
            PREFIX schema: <http://schema.org/>
            PREFIX tc: <http://techcorp.com/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            
            INSERT DATA {
                tc:customer/C001 
                    rdf:type tc:EnterpriseCustomer ;
                    schema:name "Alice Johnson" ;
                    schema:email "alice@megacorp.com" ;
                    tc:tier "Enterprise" .
                
                tc:product/CloudSuite
                    rdf:type tc:Product ;
                    schema:name "CloudSuite Pro" ;
                    tc:price "299.99" .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        # 4 triples for customer + 3 for product = 7
        assert len(query.triples) == 7
    
    def test_trailing_semicolon(self):
        """Test trailing semicolon (optional in Turtle)."""
        query = parse_query("""
            INSERT DATA {
                <http://example.org/alice> 
                    <http://xmlns.com/foaf/0.1/name> "Alice" ;
                    <http://xmlns.com/foaf/0.1/age> "30" ;
                .
            }
        """)
        assert isinstance(query, InsertDataQuery)
        assert len(query.triples) == 2


class TestTurtleSyntaxExecution:
    """Test execution of Turtle-style INSERT DATA."""
    
    def test_insert_turtle_data_executes(self):
        """Test that Turtle-style INSERT DATA executes correctly."""
        from rdf_starbase.store import TripleStore
        from rdf_starbase.sparql.executor import execute_sparql
        from rdf_starbase.models import ProvenanceContext
        
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=0.9)
        
        # Insert using Turtle syntax
        result = execute_sparql(store, """
            INSERT DATA {
                <http://example.org/alice> 
                    <http://xmlns.com/foaf/0.1/name> "Alice" ;
                    <http://xmlns.com/foaf/0.1/age> "30" ;
                    <http://xmlns.com/foaf/0.1/email> "alice@example.org" .
            }
        """, prov)
        
        assert result["count"] == 3
        
        # Verify data is in the store
        df = store.get_triples(subject="http://example.org/alice")
        assert len(df) == 3
    
    def test_insert_multiple_subjects_executes(self):
        """Test INSERT DATA with multiple subject blocks."""
        from rdf_starbase.store import TripleStore
        from rdf_starbase.sparql.executor import execute_sparql
        from rdf_starbase.models import ProvenanceContext
        
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=0.9)
        
        result = execute_sparql(store, """
            INSERT DATA {
                <http://example.org/alice> 
                    <http://xmlns.com/foaf/0.1/name> "Alice" ;
                    <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .
                
                <http://example.org/bob>
                    <http://xmlns.com/foaf/0.1/name> "Bob" .
            }
        """, prov)
        
        assert result["count"] == 3
        
        # Verify both subjects
        alice = store.get_triples(subject="http://example.org/alice")
        bob = store.get_triples(subject="http://example.org/bob")
        assert len(alice) == 2
        assert len(bob) == 1
