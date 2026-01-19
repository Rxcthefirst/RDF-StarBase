"""
Tests for SPARQL UPDATE operations (INSERT DATA, DELETE DATA, DELETE WHERE).
"""

import pytest

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.sparql import parse_query, SPARQLExecutor
from rdf_starbase.sparql.ast import InsertDataQuery, DeleteDataQuery, DeleteWhereQuery
from rdf_starbase.sparql.executor import execute_sparql


class TestDeleteDataParsing:
    """Tests for DELETE DATA query parsing."""
    
    def test_parse_simple_delete_data(self):
        """Test parsing a simple DELETE DATA query."""
        query_str = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, DeleteDataQuery)
        assert len(query.triples) == 1
        
        triple = query.triples[0]
        assert triple.subject.value == "http://example.org/alice"
        assert triple.predicate.value == "http://xmlns.com/foaf/0.1/name"
        assert triple.object.value == "Alice"
    
    def test_parse_delete_data_multiple_triples(self):
        """Test DELETE DATA with multiple triples."""
        query_str = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/age> "30" .
            <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
        }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, DeleteDataQuery)
        assert len(query.triples) == 3
    
    def test_parse_delete_data_with_prefix(self):
        """Test DELETE DATA with PREFIX declaration."""
        query_str = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/>
        DELETE DATA {
            ex:alice foaf:name "Alice" .
        }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, DeleteDataQuery)
        assert len(query.triples) == 1
        assert "foaf" in query.prefixes
        assert "ex" in query.prefixes


class TestDeleteWhereParsing:
    """Tests for DELETE WHERE query parsing."""
    
    def test_parse_simple_delete_where(self):
        """Test parsing a simple DELETE WHERE query."""
        query_str = """
        DELETE WHERE {
            <http://example.org/alice> ?p ?o .
        }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, DeleteWhereQuery)
        assert len(query.where.patterns) == 1
    
    def test_parse_delete_where_with_filter(self):
        """Test DELETE WHERE with FILTER clause."""
        query_str = """
        DELETE WHERE {
            ?s <http://example.org/age> ?age .
            FILTER(?age < 18)
        }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, DeleteWhereQuery)
        assert len(query.where.patterns) == 1
        assert len(query.where.filters) == 1


class TestInsertDataExecution:
    """Tests for INSERT DATA execution."""
    
    @pytest.fixture
    def store(self):
        """Create an empty store for testing."""
        return TripleStore()
    
    @pytest.fixture
    def executor(self, store):
        """Create executor with store."""
        return SPARQLExecutor(store)
    
    def test_insert_data_adds_triple(self, store, executor):
        """Test that INSERT DATA actually adds a triple."""
        query_str = """
        INSERT DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        
        # Initially empty
        assert len(store._df) == 0
        
        # Execute insert
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 1
        assert result["operation"] == "INSERT DATA"
        
        # Verify triple was added
        assert len(store._df) == 1
        assert store._df["subject"][0] == "http://example.org/alice"
        assert store._df["object"][0] == "Alice"
    
    def test_insert_data_multiple_triples(self, store, executor):
        """Test INSERT DATA with multiple triples."""
        query_str = """
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        INSERT DATA {
            ex:alice foaf:name "Alice" .
            ex:alice foaf:age "30" .
            ex:bob foaf:name "Bob" .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 3
        assert len(store._df) == 3


class TestDeleteDataExecution:
    """Tests for DELETE DATA execution."""
    
    @pytest.fixture
    def store(self):
        """Create store with test data."""
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", "Alice", prov)
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/age", "30", prov)
        store.add_triple("http://example.org/bob", "http://xmlns.com/foaf/0.1/name", "Bob", prov)
        
        return store
    
    @pytest.fixture
    def executor(self, store):
        """Create executor with store."""
        return SPARQLExecutor(store)
    
    def test_delete_data_removes_triple(self, store, executor):
        """Test that DELETE DATA marks a triple as deleted."""
        # Verify initial state
        assert len(store._df) == 3
        assert store._df.filter(~store._df["deprecated"]).height == 3
        
        query_str = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 1
        assert result["operation"] == "DELETE DATA"
        
        # Verify triple was marked as deprecated
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 2
        
        # The Alice name triple should be deprecated
        alice_name = store._df.filter(
            (store._df["subject"] == "http://example.org/alice") &
            (store._df["object"] == "Alice")
        )
        assert alice_name["deprecated"][0] is True
    
    def test_delete_data_multiple_triples(self, store, executor):
        """Test DELETE DATA with multiple triples."""
        query_str = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
            <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 2
        
        # Only Alice's age should remain non-deprecated
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 1
        assert non_deprecated["object"][0] == "30"
    
    def test_delete_data_nonexistent_triple(self, store, executor):
        """Test DELETE DATA for a triple that doesn't exist."""
        query_str = """
        DELETE DATA {
            <http://example.org/charlie> <http://xmlns.com/foaf/0.1/name> "Charlie" .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        # Should return count of 0 since triple doesn't exist
        assert result["count"] == 0
        
        # No triples should be deprecated
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 3
    
    def test_delete_data_with_prefix(self, store, executor):
        """Test DELETE DATA with PREFIX declarations."""
        query_str = """
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        DELETE DATA {
            ex:alice foaf:name "Alice" .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 1


class TestDeleteWhereExecution:
    """Tests for DELETE WHERE execution."""
    
    @pytest.fixture
    def store(self):
        """Create store with test data."""
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", "Alice", prov)
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/age", "30", prov)
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/knows", "http://example.org/bob", prov)
        store.add_triple("http://example.org/bob", "http://xmlns.com/foaf/0.1/name", "Bob", prov)
        store.add_triple("http://example.org/bob", "http://xmlns.com/foaf/0.1/age", "25", prov)
        
        return store
    
    @pytest.fixture
    def executor(self, store):
        """Create executor with store."""
        return SPARQLExecutor(store)
    
    def test_delete_where_single_pattern(self, store, executor):
        """Test DELETE WHERE with a single pattern."""
        query_str = """
        DELETE WHERE {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 1
        assert result["operation"] == "DELETE WHERE"
        
        # Alice's name should be deprecated
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 4
    
    def test_delete_where_all_properties(self, store, executor):
        """Test DELETE WHERE removing all properties of a subject."""
        query_str = """
        DELETE WHERE {
            <http://example.org/alice> ?p ?o .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        # Alice has 3 triples (name, age, knows)
        assert result["count"] == 3
        
        # Only Bob's triples should remain
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 2
    
    def test_delete_where_no_matches(self, store, executor):
        """Test DELETE WHERE with no matching triples."""
        query_str = """
        DELETE WHERE {
            <http://example.org/charlie> ?p ?o .
        }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["count"] == 0
        
        # All triples should remain non-deprecated
        non_deprecated = store._df.filter(~store._df["deprecated"])
        assert non_deprecated.height == 5


class TestDeletedTriplesNotQueried:
    """Test that deleted triples don't appear in query results."""
    
    @pytest.fixture
    def store(self):
        """Create store with test data."""
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", "Alice", prov)
        store.add_triple("http://example.org/bob", "http://xmlns.com/foaf/0.1/name", "Bob", prov)
        
        return store
    
    def test_deleted_triple_not_in_select(self, store):
        """Test that deleted triples don't appear in SELECT results."""
        # Delete Alice's name
        delete_query = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        execute_sparql(store, delete_query)
        
        # Query for names
        select_query = """
        SELECT ?s ?name WHERE {
            ?s <http://xmlns.com/foaf/0.1/name> ?name .
        }
        """
        result = execute_sparql(store, select_query)
        
        # Should only return Bob
        assert len(result) == 1
        assert result["name"][0] == "Bob"
    
    def test_deleted_triple_not_in_ask(self, store):
        """Test that deleted triples don't affect ASK results."""
        # Delete Alice's name
        delete_query = """
        DELETE DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        execute_sparql(store, delete_query)
        
        # ASK for Alice's name - should return False
        ask_query = """
        ASK WHERE {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
        """
        result = execute_sparql(store, ask_query)
        
        assert result is False
        
        # ASK for Bob's name - should return True
        ask_query_bob = """
        ASK WHERE {
            <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
        }
        """
        result_bob = execute_sparql(store, ask_query_bob)
        
        assert result_bob is True


class TestModifyQueryParsing:
    """Tests for DELETE/INSERT WHERE (MODIFY) query parsing."""
    
    def test_parse_delete_insert_where(self):
        """Test parsing DELETE { } INSERT { } WHERE { }."""
        from rdf_starbase.sparql.ast import ModifyQuery
        
        query_str = """
        PREFIX ex: <http://example.org/>
        DELETE { ?s ex:oldProp ?o }
        INSERT { ?s ex:newProp ?o }
        WHERE { ?s ex:oldProp ?o }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, ModifyQuery)
        assert len(query.delete_patterns) == 1
        assert len(query.insert_patterns) == 1
        assert len(query.where.patterns) == 1
    
    def test_parse_delete_only_where(self):
        """Test parsing DELETE { } WHERE { } without INSERT."""
        from rdf_starbase.sparql.ast import ModifyQuery
        
        query_str = """
        DELETE { ?s <http://example.org/temp> ?o }
        WHERE { ?s <http://example.org/temp> ?o }
        """
        query = parse_query(query_str)
        
        assert isinstance(query, ModifyQuery)
        assert len(query.delete_patterns) == 1
        assert len(query.insert_patterns) == 0


class TestModifyQueryExecution:
    """Tests for DELETE/INSERT WHERE (MODIFY) execution."""
    
    @pytest.fixture
    def store(self):
        """Create store with test data."""
        store = TripleStore()
        prov = ProvenanceContext(source="test", confidence=1.0)
        
        store.add_triple("http://example.org/alice", "http://example.org/status", "active", prov)
        store.add_triple("http://example.org/bob", "http://example.org/status", "active", prov)
        store.add_triple("http://example.org/charlie", "http://example.org/status", "inactive", prov)
        
        return store
    
    def test_modify_delete_and_insert(self, store):
        """Test DELETE/INSERT WHERE modifies matching triples."""
        query_str = """
        PREFIX ex: <http://example.org/>
        DELETE { ?s ex:status "active" }
        INSERT { ?s ex:status "archived" }
        WHERE { ?s ex:status "active" }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["operation"] == "MODIFY"
        assert result["deleted"] == 2  # Alice and Bob
        assert result["inserted"] == 2  # New archived status for both
        
        # Verify the changes
        # Active triples should be deprecated
        active = store._df.filter(
            (store._df["object"] == "active") & (~store._df["deprecated"])
        )
        assert active.height == 0
        
        # Archived triples should exist
        archived = store._df.filter(
            (store._df["object"] == "archived") & (~store._df["deprecated"])
        )
        assert archived.height == 2
    
    def test_modify_delete_only(self, store):
        """Test DELETE { } WHERE { } without INSERT."""
        query_str = """
        DELETE { ?s <http://example.org/status> "inactive" }
        WHERE { ?s <http://example.org/status> "inactive" }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["deleted"] == 1  # Charlie
        assert result["inserted"] == 0
        
        # Inactive should be deprecated
        inactive = store._df.filter(
            (store._df["object"] == "inactive") & (~store._df["deprecated"])
        )
        assert inactive.height == 0
    
    def test_modify_no_matches(self, store):
        """Test MODIFY with no matching triples."""
        query_str = """
        DELETE { ?s <http://example.org/status> "unknown" }
        INSERT { ?s <http://example.org/status> "found" }
        WHERE { ?s <http://example.org/status> "unknown" }
        """
        
        result = execute_sparql(store, query_str)
        
        assert result["deleted"] == 0
        assert result["inserted"] == 0
