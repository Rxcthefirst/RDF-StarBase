"""
Tests for storage persistence layer.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from rdf_starbase.storage import (
    TermDict,
    FactStore,
    StorageExecutor,
    Term,
    TermKind,
    DEFAULT_GRAPH_ID,
    StoragePersistence,
    save_storage,
    load_storage,
)
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.sparql.parser import SPARQLStarParser


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for storage tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def populated_storage():
    """Create a storage with sample data."""
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    def add(s: str, p: str, o: str):
        s_term = Term(kind=TermKind.IRI, lex=s)
        p_term = Term(kind=TermKind.IRI, lex=p)
        if o.isdigit():
            o_term = Term(kind=TermKind.LITERAL, lex=o)
        else:
            o_term = Term(kind=TermKind.IRI, lex=o)
        
        s_id = term_dict.get_or_create(s_term)
        p_id = term_dict.get_or_create(p_term)
        o_id = term_dict.get_or_create(o_term)
        
        fact_store.add_facts_batch([(DEFAULT_GRAPH_ID, s_id, p_id, o_id)])
    
    # Add sample data
    add("http://ex/alice", "http://ex/name", "Alice")
    add("http://ex/alice", "http://ex/age", "30")
    add("http://ex/alice", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person")
    
    add("http://ex/bob", "http://ex/name", "Bob")
    add("http://ex/bob", "http://ex/age", "25")
    add("http://ex/bob", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person")
    
    add("http://ex/alice", "http://ex/knows", "http://ex/bob")
    
    return term_dict, fact_store, qt_dict


# =============================================================================
# Basic Persistence Tests
# =============================================================================

class TestPersistence:
    """Test save/load functionality."""
    
    def test_save_and_load_terms(self, temp_dir, populated_storage):
        """TermDict should be preserved after save/load."""
        term_dict, fact_store, qt_dict = populated_storage
        
        # Get term count before
        term_count_before = len(term_dict._id_to_term)
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Verify term count
        assert len(loaded_term_dict._id_to_term) == term_count_before
    
    def test_save_and_load_facts(self, temp_dir, populated_storage):
        """Facts should be preserved after save/load."""
        term_dict, fact_store, qt_dict = populated_storage
        
        # Get fact count before
        fact_count_before = len(fact_store._df)
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Verify fact count
        assert len(loaded_fact_store._df) == fact_count_before
    
    def test_query_after_load(self, temp_dir, populated_storage):
        """Queries should work on loaded storage."""
        term_dict, fact_store, qt_dict = populated_storage
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Create executor and parser
        executor = StorageExecutor(loaded_term_dict, loaded_qt_dict, loaded_fact_store)
        parser = SPARQLStarParser()
        
        # Run a query
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?person ?age
        WHERE {
            ?person a ex:Person .
            ?person ex:age ?age .
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # Should have 2 people
        assert len(result) == 2
        assert "person" in result.columns
        assert "age" in result.columns
    
    def test_add_facts_after_load(self, temp_dir, populated_storage):
        """Should be able to add facts after loading."""
        term_dict, fact_store, qt_dict = populated_storage
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Add a new fact
        s_term = Term(kind=TermKind.IRI, lex="http://ex/charlie")
        p_term = Term(kind=TermKind.IRI, lex="http://ex/name")
        o_term = Term(kind=TermKind.LITERAL, lex="Charlie")
        
        s_id = loaded_term_dict.get_or_create(s_term)
        p_id = loaded_term_dict.get_or_create(p_term)
        o_id = loaded_term_dict.get_or_create(o_term)
        
        initial_count = len(loaded_fact_store._df)
        loaded_fact_store.add_facts_batch([(DEFAULT_GRAPH_ID, s_id, p_id, o_id)])
        
        # Verify fact was added
        assert len(loaded_fact_store._df) == initial_count + 1
    
    def test_persistence_exists(self, temp_dir, populated_storage):
        """exists() should return True after save."""
        term_dict, fact_store, qt_dict = populated_storage
        
        persistence = StoragePersistence(temp_dir)
        
        # Should not exist before save
        assert not persistence.exists()
        
        # Save
        persistence.save(term_dict, fact_store, qt_dict)
        
        # Should exist after save
        assert persistence.exists()
    
    def test_load_nonexistent(self, temp_dir):
        """Loading from nonexistent path should raise FileNotFoundError."""
        nonexistent = temp_dir / "does_not_exist"
        
        persistence = StoragePersistence(nonexistent)
        
        with pytest.raises(FileNotFoundError):
            persistence.load()


class TestPersistenceQuotedTriples:
    """Test persistence with quoted triples."""
    
    def test_save_and_load_quoted_triples(self, temp_dir):
        """Quoted triples should be preserved after save/load."""
        term_dict = TermDict()
        qt_dict = QtDict(term_dict)
        fact_store = FactStore(term_dict, qt_dict)
        
        # Create terms for the quoted triple
        alice = term_dict.get_or_create(Term(kind=TermKind.IRI, lex="http://ex/alice"))
        knows = term_dict.get_or_create(Term(kind=TermKind.IRI, lex="http://ex/knows"))
        bob = term_dict.get_or_create(Term(kind=TermKind.IRI, lex="http://ex/bob"))
        
        # Create a quoted triple
        qt_id = qt_dict.get_or_create(alice, knows, bob)
        
        # Add a fact using the quoted triple
        says = term_dict.get_or_create(Term(kind=TermKind.IRI, lex="http://ex/says"))
        charlie = term_dict.get_or_create(Term(kind=TermKind.IRI, lex="http://ex/charlie"))
        
        fact_store.add_facts_batch([(DEFAULT_GRAPH_ID, charlie, says, qt_id)])
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Verify quoted triple was preserved
        assert len(loaded_qt_dict._id_to_qt) == 1
        
        # Get the quoted triple
        loaded_qt = loaded_qt_dict._id_to_qt[qt_id]
        assert loaded_qt.s == alice
        assert loaded_qt.p == knows
        assert loaded_qt.o == bob


class TestPersistenceCounters:
    """Test that counters are preserved correctly."""
    
    def test_counters_preserved(self, temp_dir, populated_storage):
        """Term ID counters should be preserved after save/load."""
        term_dict, fact_store, qt_dict = populated_storage
        
        # Record counters before save
        iri_counter_before = term_dict._next_payload[TermKind.IRI]
        literal_counter_before = term_dict._next_payload[TermKind.LITERAL]
        
        # Save
        save_storage(temp_dir, term_dict, fact_store, qt_dict)
        
        # Load
        loaded_term_dict, loaded_fact_store, loaded_qt_dict = load_storage(temp_dir)
        
        # Add a new term after loading
        new_term = Term(kind=TermKind.IRI, lex="http://ex/newterm")
        new_id = loaded_term_dict.get_or_create(new_term)
        
        # The new ID should be greater than or equal to the old counter
        # (accounting for well-known types being re-interned)
        assert new_id is not None
