"""
Tests for Named Graph Management functionality.

Tests CREATE, DROP, CLEAR, LOAD, COPY, MOVE, ADD SPARQL operations
and the underlying store methods.
"""

import pytest
import tempfile
import os
from pathlib import Path

from rdf_starbase.store import TripleStore
from rdf_starbase.sparql.parser import parse_query
from rdf_starbase.sparql.executor import SPARQLExecutor, execute_sparql
from rdf_starbase.sparql.ast import (
    CreateGraphQuery, DropGraphQuery, ClearGraphQuery,
    LoadQuery, CopyGraphQuery, MoveGraphQuery, AddGraphQuery
)
from rdf_starbase.models import ProvenanceContext


@pytest.fixture
def store():
    """Create a fresh triple store for each test."""
    return TripleStore()


@pytest.fixture
def populated_store():
    """Create a store with some test data."""
    store = TripleStore()
    prov = ProvenanceContext(source="test", confidence=0.9)
    
    # Add triples to default graph
    store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", "Alice", prov)
    store.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/age", "30", prov)
    
    # Add triples to named graph
    store.add_triple(
        "http://example.org/bob", 
        "http://xmlns.com/foaf/0.1/name", 
        "Bob", 
        prov, 
        graph="http://example.org/people"
    )
    store.add_triple(
        "http://example.org/bob", 
        "http://xmlns.com/foaf/0.1/age", 
        "25", 
        prov, 
        graph="http://example.org/people"
    )
    
    return store


class TestStoreGraphManagement:
    """Test the TripleStore graph management methods directly."""
    
    def test_list_graphs_empty(self, store):
        """Test listing graphs on empty store."""
        graphs = store.list_graphs()
        assert isinstance(graphs, list)
    
    def test_list_graphs_with_data(self, populated_store):
        """Test listing graphs with data."""
        graphs = populated_store.list_graphs()
        assert "http://example.org/people" in graphs
    
    def test_create_graph(self, store):
        """Test creating a named graph."""
        result = store.create_graph("http://example.org/newgraph")
        # create_graph returns True if graph didn't exist (was "created")
        assert result is True
    
    def test_create_graph_duplicate(self, store):
        """Test creating a duplicate graph still returns True (graphs are implicit)."""
        store.create_graph("http://example.org/newgraph")
        # Second call still returns True (no error, graph existed implicitly)
        result = store.create_graph("http://example.org/newgraph")
        assert result is True  # Graph already exists so result is True (no triples)
    
    def test_drop_graph(self, populated_store):
        """Test dropping a named graph."""
        result = populated_store.drop_graph("http://example.org/people")
        # drop_graph returns count of triples removed
        assert result == 2  # Bob's name and age
        
        graphs = populated_store.list_graphs()
        assert "http://example.org/people" not in graphs
    
    def test_drop_graph_nonexistent(self, store):
        """Test dropping non-existent graph returns 0."""
        result = store.drop_graph("http://example.org/nonexistent")
        assert result == 0
    
    def test_drop_graph_silent(self, store):
        """Test dropping non-existent graph with SILENT."""
        result = store.drop_graph("http://example.org/nonexistent", silent=True)
        assert result == 0
    
    def test_clear_graph(self, populated_store):
        """Test clearing a named graph."""
        count = populated_store.clear_graph("http://example.org/people")
        assert count == 2  # Bob's name and age
        
        # Triples are removed but graph can still be queried (just empty)
    
    def test_clear_default_graph(self, populated_store):
        """Test clearing the default graph."""
        count = populated_store.clear_graph(None)
        assert count == 2  # Alice's name and age
    
    def test_copy_graph(self, populated_store):
        """Test copying a named graph."""
        count = populated_store.copy_graph(
            "http://example.org/people",
            "http://example.org/people_backup"
        )
        assert count == 2
        
        # Original should still exist
        graphs = populated_store.list_graphs()
        assert "http://example.org/people" in graphs
        assert "http://example.org/people_backup" in graphs
    
    def test_move_graph(self, populated_store):
        """Test moving a named graph."""
        count = populated_store.move_graph(
            "http://example.org/people",
            "http://example.org/people_moved"
        )
        assert count == 2
        
        # Original should be cleared
        graphs = populated_store.list_graphs()
        assert "http://example.org/people_moved" in graphs
    
    def test_add_graph(self, populated_store):
        """Test adding triples from one graph to another."""
        # Create a target with existing data
        prov = ProvenanceContext(source="test", confidence=0.9)
        populated_store.add_triple(
            "http://example.org/charlie",
            "http://xmlns.com/foaf/0.1/name",
            "Charlie",
            prov,
            graph="http://example.org/target"
        )
        
        # Add people graph to target
        count = populated_store.add_graph(
            "http://example.org/people",
            "http://example.org/target"
        )
        assert count == 2
        
        # Both original and target should have data
        graphs = populated_store.list_graphs()
        assert "http://example.org/people" in graphs
        assert "http://example.org/target" in graphs


class TestGraphManagementParsing:
    """Test parsing of graph management SPARQL operations."""
    
    def test_parse_create_graph(self):
        """Test parsing CREATE GRAPH."""
        query = parse_query("CREATE GRAPH <http://example.org/newgraph>")
        assert isinstance(query, CreateGraphQuery)
        assert query.graph_uri.value == "http://example.org/newgraph"
        assert query.silent is False
    
    def test_parse_create_silent_graph(self):
        """Test parsing CREATE SILENT GRAPH."""
        query = parse_query("CREATE SILENT GRAPH <http://example.org/newgraph>")
        assert isinstance(query, CreateGraphQuery)
        assert query.silent is True
    
    def test_parse_drop_graph(self):
        """Test parsing DROP GRAPH."""
        query = parse_query("DROP GRAPH <http://example.org/oldgraph>")
        assert isinstance(query, DropGraphQuery)
        assert query.graph_uri.value == "http://example.org/oldgraph"
        assert query.target == "graph"
    
    def test_parse_drop_default(self):
        """Test parsing DROP DEFAULT."""
        query = parse_query("DROP DEFAULT")
        assert isinstance(query, DropGraphQuery)
        assert query.target == "default"
    
    def test_parse_drop_named(self):
        """Test parsing DROP NAMED."""
        query = parse_query("DROP NAMED")
        assert isinstance(query, DropGraphQuery)
        assert query.target == "named"
    
    def test_parse_drop_all(self):
        """Test parsing DROP ALL."""
        query = parse_query("DROP ALL")
        assert isinstance(query, DropGraphQuery)
        assert query.target == "all"
    
    def test_parse_drop_silent(self):
        """Test parsing DROP SILENT GRAPH."""
        query = parse_query("DROP SILENT GRAPH <http://example.org/graph>")
        assert isinstance(query, DropGraphQuery)
        assert query.silent is True
    
    def test_parse_clear_graph(self):
        """Test parsing CLEAR GRAPH."""
        query = parse_query("CLEAR GRAPH <http://example.org/graph>")
        assert isinstance(query, ClearGraphQuery)
        assert query.graph_uri.value == "http://example.org/graph"
    
    def test_parse_clear_default(self):
        """Test parsing CLEAR DEFAULT."""
        query = parse_query("CLEAR DEFAULT")
        assert isinstance(query, ClearGraphQuery)
        assert query.target == "default"
    
    def test_parse_clear_all(self):
        """Test parsing CLEAR ALL."""
        query = parse_query("CLEAR ALL")
        assert isinstance(query, ClearGraphQuery)
        assert query.target == "all"
    
    def test_parse_load(self):
        """Test parsing LOAD query."""
        query = parse_query("LOAD <file:///data/test.ttl>")
        assert isinstance(query, LoadQuery)
        assert query.source_uri.value == "file:///data/test.ttl"
        assert query.graph_uri is None
    
    def test_parse_load_into_graph(self):
        """Test parsing LOAD INTO GRAPH."""
        query = parse_query("LOAD <file:///data/test.ttl> INTO GRAPH <http://example.org/data>")
        assert isinstance(query, LoadQuery)
        assert query.source_uri.value == "file:///data/test.ttl"
        assert query.graph_uri.value == "http://example.org/data"
    
    def test_parse_load_silent(self):
        """Test parsing LOAD SILENT."""
        query = parse_query("LOAD SILENT <file:///data/test.ttl>")
        assert isinstance(query, LoadQuery)
        assert query.silent is True
    
    def test_parse_copy(self):
        """Test parsing COPY."""
        query = parse_query("COPY GRAPH <http://example.org/a> TO GRAPH <http://example.org/b>")
        assert isinstance(query, CopyGraphQuery)
        assert query.source_graph.value == "http://example.org/a"
        assert query.dest_graph.value == "http://example.org/b"
    
    def test_parse_copy_default(self):
        """Test parsing COPY DEFAULT TO."""
        query = parse_query("COPY DEFAULT TO GRAPH <http://example.org/backup>")
        assert isinstance(query, CopyGraphQuery)
        assert query.source_is_default is True
    
    def test_parse_move(self):
        """Test parsing MOVE."""
        query = parse_query("MOVE GRAPH <http://example.org/a> TO GRAPH <http://example.org/b>")
        assert isinstance(query, MoveGraphQuery)
        assert query.source_graph.value == "http://example.org/a"
        assert query.dest_graph.value == "http://example.org/b"
    
    def test_parse_add(self):
        """Test parsing ADD."""
        query = parse_query("ADD GRAPH <http://example.org/a> TO GRAPH <http://example.org/b>")
        assert isinstance(query, AddGraphQuery)
        assert query.source_graph.value == "http://example.org/a"
        assert query.dest_graph.value == "http://example.org/b"


class TestGraphManagementExecution:
    """Test execution of graph management SPARQL operations."""
    
    def test_execute_create_graph(self, store):
        """Test executing CREATE GRAPH."""
        result = execute_sparql(store, "CREATE GRAPH <http://example.org/newgraph>")
        assert result["success"] is True
        assert result["operation"] == "CREATE GRAPH"
        # Note: Empty graphs don't appear in list_graphs() until they have data
    
    def test_execute_drop_graph(self, populated_store):
        """Test executing DROP GRAPH."""
        result = execute_sparql(
            populated_store, 
            "DROP GRAPH <http://example.org/people>"
        )
        assert result["success"] is True
        
        graphs = populated_store.list_graphs()
        assert "http://example.org/people" not in graphs
    
    def test_execute_drop_silent_nonexistent(self, store):
        """Test executing DROP SILENT on non-existent graph."""
        result = execute_sparql(
            store, 
            "DROP SILENT GRAPH <http://example.org/nonexistent>"
        )
        # SILENT means no error is raised, operation succeeds with 0 triples dropped
        assert result["success"] is True
    
    def test_execute_clear_graph(self, populated_store):
        """Test executing CLEAR GRAPH."""
        result = execute_sparql(
            populated_store, 
            "CLEAR GRAPH <http://example.org/people>"
        )
        assert result["success"] is True
        assert result["triples_cleared"] == 2
    
    def test_execute_clear_default(self, populated_store):
        """Test executing CLEAR DEFAULT."""
        result = execute_sparql(populated_store, "CLEAR DEFAULT")
        assert result["success"] is True
        assert result["triples_cleared"] == 2
    
    def test_execute_copy(self, populated_store):
        """Test executing COPY."""
        result = execute_sparql(
            populated_store, 
            "COPY GRAPH <http://example.org/people> TO GRAPH <http://example.org/backup>"
        )
        assert result["success"] is True
        assert result["triples_copied"] == 2
    
    def test_execute_move(self, populated_store):
        """Test executing MOVE."""
        result = execute_sparql(
            populated_store, 
            "MOVE GRAPH <http://example.org/people> TO GRAPH <http://example.org/moved>"
        )
        assert result["success"] is True
        assert result["triples_moved"] == 2
    
    def test_execute_add(self, populated_store):
        """Test executing ADD."""
        result = execute_sparql(
            populated_store, 
            "ADD GRAPH <http://example.org/people> TO DEFAULT"
        )
        assert result["success"] is True
        assert result["triples_added"] == 2


class TestLoadFromFile:
    """Test LOAD functionality with actual files."""
    
    def test_load_turtle_file(self, store):
        """Test loading a Turtle file."""
        # Create a temp Turtle file
        turtle_data = """
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex: <http://example.org/> .

ex:alice foaf:name "Alice" ;
         foaf:age "30" .
"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.ttl', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(turtle_data)
            temp_path = f.name
        
        try:
            # Convert Windows path to file URI
            file_uri = Path(temp_path).as_uri()
            
            result = execute_sparql(store, f"LOAD <{file_uri}>")
            assert result["success"] is True
            assert result["triples_loaded"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_load_into_named_graph(self, store):
        """Test loading into a named graph."""
        turtle_data = """
@prefix ex: <http://example.org/> .

ex:test ex:value "test" .
"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.ttl', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(turtle_data)
            temp_path = f.name
        
        try:
            file_uri = Path(temp_path).as_uri()
            
            result = execute_sparql(
                store, 
                f"LOAD <{file_uri}> INTO GRAPH <http://example.org/loaded>"
            )
            assert result["success"] is True
            
            graphs = store.list_graphs()
            assert "http://example.org/loaded" in graphs
        finally:
            os.unlink(temp_path)
    
    def test_load_silent_nonexistent(self, store):
        """Test LOAD SILENT with non-existent file."""
        result = execute_sparql(
            store, 
            "LOAD SILENT <file:///nonexistent/path/data.ttl>"
        )
        # SILENT means the operation succeeds but loads 0 triples
        assert result["success"] is True
        assert result["triples_loaded"] == 0


class TestGraphManagementIntegration:
    """Integration tests for graph management workflow."""
    
    def test_full_workflow(self, store):
        """Test a complete graph management workflow."""
        prov = ProvenanceContext(source="test", confidence=0.95)
        
        # 1. Create a named graph
        result = execute_sparql(store, "CREATE GRAPH <http://example.org/mydata>")
        assert result["success"]
        
        # 2. Add data to it
        store.add_triple(
            "http://example.org/item1",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Item One",
            prov,
            graph="http://example.org/mydata"
        )
        store.add_triple(
            "http://example.org/item2",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Item Two",
            prov,
            graph="http://example.org/mydata"
        )
        
        # 3. Copy to backup
        result = execute_sparql(
            store, 
            "COPY GRAPH <http://example.org/mydata> TO GRAPH <http://example.org/backup>"
        )
        assert result["triples_copied"] == 2
        
        # 4. Clear original
        result = execute_sparql(store, "CLEAR GRAPH <http://example.org/mydata>")
        assert result["triples_cleared"] == 2
        
        # 5. Restore from backup
        result = execute_sparql(
            store, 
            "MOVE GRAPH <http://example.org/backup> TO GRAPH <http://example.org/mydata>"
        )
        assert result["triples_moved"] == 2
        
        # 6. Verify final state
        graphs = store.list_graphs()
        assert "http://example.org/mydata" in graphs
    
    def test_multiple_graph_operations(self, store):
        """Test operations across multiple graphs."""
        prov = ProvenanceContext(source="test", confidence=0.9)
        
        # Create graphs with data
        for i in range(3):
            store.add_triple(
                f"http://example.org/entity{i}",
                "http://www.w3.org/2000/01/rdf-schema#label",
                f"Entity {i}",
                prov,
                graph=f"http://example.org/graph{i}"
            )
        
        # Add all to default graph
        for i in range(3):
            execute_sparql(store, f"ADD GRAPH <http://example.org/graph{i}> TO DEFAULT")
        
        # Verify graphs exist
        graphs = store.list_graphs()
        assert len([g for g in graphs if g]) >= 3  # At least 3 named graphs
