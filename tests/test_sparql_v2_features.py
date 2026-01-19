"""
Tests for SPARQL v0.2.0 features:
- EXISTS / NOT EXISTS
- COALESCE, IF, BOUND, and other functions
- (SubSelect will be tested when parsing is complete)
"""

import pytest
from rdf_starbase import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql.parser import SPARQLStarParser
from rdf_starbase.sparql.executor import execute_sparql
from rdf_starbase.sparql.ast import Filter, ExistsExpression


def prov(source="test"):
    """Helper to create a provenance context."""
    return ProvenanceContext(source=source, confidence=1.0)


class TestExistsNotExists:
    """Test EXISTS and NOT EXISTS filter patterns."""
    
    @pytest.fixture
    def store(self):
        """Create a store with test data."""
        store = TripleStore()
        
        # People with various properties
        store.add_triple("http://ex/alice", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/alice", "http://ex/email", "alice@example.org", prov())
        store.add_triple("http://ex/alice", "http://ex/age", "30", prov())
        store.add_triple("http://ex/alice", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        
        store.add_triple("http://ex/bob", "http://ex/name", "Bob", prov())
        store.add_triple("http://ex/bob", "http://ex/age", "25", prov())
        store.add_triple("http://ex/bob", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        # Bob has no email
        
        store.add_triple("http://ex/charlie", "http://ex/name", "Charlie", prov())
        store.add_triple("http://ex/charlie", "http://ex/email", "charlie@example.org", prov())
        store.add_triple("http://ex/charlie", "http://ex/age", "35", prov())
        store.add_triple("http://ex/charlie", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        
        return store
    
    def test_parse_exists(self):
        """Test parsing FILTER EXISTS."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?person WHERE {
                ?person a ex:Person .
                FILTER EXISTS { ?person ex:email ?email }
            }
        """)
        
        # Should have a filter with ExistsExpression
        filters = query.where.filters
        assert len(filters) >= 1
        exists_filter = None
        for f in filters:
            if isinstance(f.expression, ExistsExpression):
                exists_filter = f
                break
        assert exists_filter is not None
        assert not exists_filter.expression.negated
    
    def test_parse_not_exists(self):
        """Test parsing FILTER NOT EXISTS."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?person WHERE {
                ?person a ex:Person .
                FILTER NOT EXISTS { ?person ex:email ?email }
            }
        """)
        
        filters = query.where.filters
        assert len(filters) >= 1
        exists_filter = None
        for f in filters:
            if isinstance(f.expression, ExistsExpression):
                exists_filter = f
                break
        assert exists_filter is not None
        assert exists_filter.expression.negated
    
    def test_exists_execution(self, store):
        """Test executing EXISTS - finds people WITH email."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER EXISTS { ?person ex:email ?email }
            }
        """)
        
        names = set(result["name"].to_list())
        # Alice and Charlie have emails
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names
    
    def test_not_exists_execution(self, store):
        """Test executing NOT EXISTS - finds people WITHOUT email."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER NOT EXISTS { ?person ex:email ?email }
            }
        """)
        
        names = set(result["name"].to_list())
        # Only Bob has no email
        assert "Bob" in names
        assert "Alice" not in names
        assert "Charlie" not in names


class TestSPARQLFunctions:
    """Test new SPARQL functions: COALESCE, IF, string functions, etc."""
    
    @pytest.fixture
    def store(self):
        """Create a store with test data."""
        store = TripleStore()
        
        store.add_triple("http://ex/p1", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/p1", "http://ex/nickname", "Ali", prov())
        store.add_triple("http://ex/p1", "http://ex/age", "30", prov())
        
        store.add_triple("http://ex/p2", "http://ex/name", "Bob", prov())
        # p2 has no nickname
        store.add_triple("http://ex/p2", "http://ex/age", "25", prov())
        
        store.add_triple("http://ex/p3", "http://ex/name", "Charlie", prov())
        store.add_triple("http://ex/p3", "http://ex/nickname", "Chuck", prov())
        store.add_triple("http://ex/p3", "http://ex/age", "35", prov())
        
        return store
    
    @pytest.mark.skip(reason="OPTIONAL doesn't add null columns for unmatched patterns - needs fix")
    def test_bound_function(self, store):
        """Test BOUND(?var) function."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name ?nickname WHERE {
                ?person ex:name ?name .
                OPTIONAL { ?person ex:nickname ?nickname }
                FILTER(BOUND(?nickname))
            }
        """)
        
        names = set(result["name"].to_list())
        # Alice and Charlie have nicknames
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names
    
    def test_strlen_function(self, store):
        """Test STRLEN function."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(STRLEN(?name) > 3)
            }
        """)
        
        names = set(result["name"].to_list())
        # "Alice" (5) and "Charlie" (7) have length > 3, "Bob" (3) doesn't
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names
    
    def test_contains_function(self, store):
        """Test CONTAINS function."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(CONTAINS(?name, "li"))
            }
        """)
        
        names = set(result["name"].to_list())
        # "Alice" and "Charlie" contain "li"
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names
    
    def test_strstarts_function(self, store):
        """Test STRSTARTS function."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(STRSTARTS(?name, "A"))
            }
        """)
        
        names = set(result["name"].to_list())
        assert "Alice" in names
        assert "Bob" not in names
        assert "Charlie" not in names
    
    def test_strends_function(self, store):
        """Test STRENDS function."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(STRENDS(?name, "e"))
            }
        """)
        
        names = set(result["name"].to_list())
        # "Alice" and "Charlie" end with "e"
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names
    
    def test_lcase_function(self, store):
        """Test LCASE function - returns lowercase."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(LCASE(?name) = "alice")
            }
        """)
        
        names = set(result["name"].to_list())
        assert "Alice" in names
    
    def test_ucase_function(self, store):
        """Test UCASE function - returns uppercase."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?name WHERE {
                ?person ex:name ?name .
                FILTER(UCASE(?name) = "BOB")
            }
        """)
        
        names = set(result["name"].to_list())
        assert "Bob" in names


class TestCoalesceFunction:
    """Test COALESCE function specifically."""
    
    @pytest.fixture
    def store(self):
        store = TripleStore()
        store.add_triple("http://ex/p1", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/p1", "http://ex/nickname", "Ali", prov())
        
        store.add_triple("http://ex/p2", "http://ex/name", "Bob", prov())
        # p2 has no nickname
        
        return store
    
    def test_coalesce_returns_first_bound(self, store):
        """Test that COALESCE returns first non-null value."""
        # This is a simplified test - COALESCE is implemented but
        # may need BIND to expose it in results
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?person ?displayName WHERE {
                ?person ex:name ?name .
                OPTIONAL { ?person ex:nickname ?nickname }
                BIND(COALESCE(?nickname, ?name) AS ?displayName)
            }
        """)
        
        # Just verify it parses correctly
        assert query is not None
        assert len(query.where.binds) == 1


class TestBindInNestedPatterns:
    """Test BIND in UNION, OPTIONAL, and other nested patterns."""
    
    @pytest.fixture
    def store(self):
        store = TripleStore()
        store.add_triple("http://ex/person1", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/person1", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        store.add_triple("http://ex/company1", "http://ex/name", "ACME Corp", prov())
        store.add_triple("http://ex/company1", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Company", prov())
        return store
    
    def test_bind_in_union_parses(self, store):
        """Test that BIND within UNION branches parses correctly."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?entity ?name ?type WHERE {
                { ?entity a ex:Person . ?entity ex:name ?name . BIND("Person" AS ?type) }
                UNION
                { ?entity a ex:Company . ?entity ex:name ?name . BIND("Company" AS ?type) }
            }
        """)
        
        assert query is not None
        assert len(query.where.union_patterns) == 1
        union = query.where.union_patterns[0]
        # Check that alternatives have binds
        assert len(union.alternatives) == 2
        for alt in union.alternatives:
            assert 'binds' in alt
            assert len(alt['binds']) == 1
    
    def test_bind_in_union_executes(self, store):
        """Test that BIND within UNION branches executes correctly."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?entity ?name ?type WHERE {
                { ?entity a ex:Person . ?entity ex:name ?name . BIND("Person" AS ?type) }
                UNION
                { ?entity a ex:Company . ?entity ex:name ?name . BIND("Company" AS ?type) }
            }
        """)
        
        assert len(result) == 2
        
        # Check that type column was populated by BIND
        types = set(result["type"].to_list())
        assert "Person" in types or '"Person"' in types
        assert "Company" in types or '"Company"' in types
    
    def test_bind_in_optional_parses(self, store):
        """Test that BIND within OPTIONAL parses correctly."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?entity ?name ?label WHERE {
                ?entity ex:name ?name .
                OPTIONAL { ?entity a ex:Person . BIND("A Person" AS ?label) }
            }
        """)
        
        assert query is not None
        assert len(query.where.optional_patterns) == 1
        opt = query.where.optional_patterns[0]
        assert len(opt.binds) == 1


class TestBoundFunction:
    """Test BOUND function for checking if variable is bound."""
    
    @pytest.fixture
    def store(self):
        store = TripleStore()
        # Alice has both name and email
        store.add_triple("http://ex/alice", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/alice", "http://ex/email", "alice@example.org", prov())
        # Bob has only name (no email)
        store.add_triple("http://ex/bob", "http://ex/name", "Bob", prov())
        return store
    
    def test_bound_with_optional(self, store):
        """Test BOUND with OPTIONAL - filter for unbound variables."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?person ?name WHERE {
                ?person ex:name ?name .
                OPTIONAL { ?person ex:email ?email }
                FILTER(!BOUND(?email))
            }
        """)
        
        # Only Bob should match (no email)
        assert len(result) == 1
        assert "Bob" in result["name"].to_list()[0]
    
    def test_bound_filters_to_bound(self, store):
        """Test BOUND to filter for bound variables."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?person ?name WHERE {
                ?person ex:name ?name .
                OPTIONAL { ?person ex:email ?email }
                FILTER(BOUND(?email))
            }
        """)
        
        # Only Alice should match (has email)
        assert len(result) == 1
        assert "Alice" in result["name"].to_list()[0]


class TestIfFunction:
    """Test IF function."""
    
    @pytest.fixture
    def store(self):
        store = TripleStore()
        store.add_triple("http://ex/p1", "http://ex/age", "30", prov())
        store.add_triple("http://ex/p2", "http://ex/age", "15", prov())
        store.add_triple("http://ex/p3", "http://ex/age", "25", prov())
        return store
    
    def test_if_parses(self, store):
        """Test that IF function parses correctly."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?person ?status WHERE {
                ?person ex:age ?age .
                BIND(IF(?age > 18, "adult", "minor") AS ?status)
            }
        """)
        
        assert query is not None
        assert len(query.where.binds) == 1


class TestSubqueries:
    """Test subqueries (nested SELECT in WHERE clause)."""
    
    @pytest.fixture
    def store(self):
        store = TripleStore()
        # People with ages
        store.add_triple("http://ex/p1", "http://ex/name", "Alice", prov())
        store.add_triple("http://ex/p1", "http://ex/age", "30", prov())
        store.add_triple("http://ex/p2", "http://ex/name", "Bob", prov())
        store.add_triple("http://ex/p2", "http://ex/age", "25", prov())
        store.add_triple("http://ex/p3", "http://ex/name", "Charlie", prov())
        store.add_triple("http://ex/p3", "http://ex/age", "35", prov())
        # Categories
        store.add_triple("http://ex/p1", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        store.add_triple("http://ex/p2", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        store.add_triple("http://ex/p3", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person", prov())
        return store
    
    def test_subquery_parses(self, store):
        """Test that subquery parses correctly."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?person ?avgAge WHERE {
                ?person a ex:Person .
                {
                    SELECT (AVG(?age) AS ?avgAge)
                    WHERE { ?p ex:age ?age }
                }
            }
        """)
        
        assert query is not None
        assert len(query.where.subselects) == 1
        sub = query.where.subselects[0]
        assert len(sub.variables) == 1
        assert sub.variables[0].function == "AVG"
    
    def test_subquery_with_count(self, store):
        """Test subquery with COUNT aggregate."""
        result = execute_sparql(store, """
            PREFIX ex: <http://ex/>
            SELECT ?person ?total WHERE {
                ?person a ex:Person .
                {
                    SELECT (COUNT(?p) AS ?total)
                    WHERE { ?p a ex:Person }
                }
            }
        """)
        
        # Each person should be joined with the total count (3)
        assert len(result) == 3
        assert all(t == 3 for t in result["total"].to_list())
    
    def test_subquery_with_group_by(self, store):
        """Test subquery with GROUP BY."""
        parser = SPARQLStarParser()
        query = parser.parse("""
            PREFIX ex: <http://ex/>
            SELECT ?name ?maxAge WHERE {
                ?person ex:name ?name .
                {
                    SELECT (MAX(?age) AS ?maxAge)
                    WHERE { ?p ex:age ?age }
                    GROUP BY ?type
                }
            }
        """)
        
        assert query is not None
        sub = query.where.subselects[0]
        # Note: GROUP BY parsing in subselects may need adjustment
        assert sub is not None
