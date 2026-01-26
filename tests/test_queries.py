"""
Tests for Saved Queries.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from rdf_starbase.storage.queries import (
    SavedQueryManager,
    SavedQuery,
    QueryCollection,
    QueryParameter,
    QueryExecution,
    QueryType,
    detect_query_type,
    extract_parameters,
    save_query,
    load_query,
)


# Sample queries for testing
SAMPLE_SELECT = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?email
WHERE {
    ?person foaf:name ?name .
    ?person foaf:email ?email .
}
"""

SAMPLE_CONSTRUCT = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
CONSTRUCT {
    ?person foaf:label ?name .
}
WHERE {
    ?person foaf:name ?name .
}
"""

SAMPLE_ASK = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
ASK {
    ?person foaf:name "Alice" .
}
"""

SAMPLE_INSERT = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
INSERT {
    ?person foaf:age 30 .
}
WHERE {
    ?person foaf:name "Alice" .
}
"""

SAMPLE_TEMPLATE = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {
    ?person foaf:name "{{name}}" .
    ?person foaf:email ?email .
}
"""


class TestQueryType:
    """Tests for QueryType enum."""
    
    def test_query_type_values(self):
        """Test QueryType values."""
        assert QueryType.SELECT.value == "select"
        assert QueryType.CONSTRUCT.value == "construct"
        assert QueryType.ASK.value == "ask"
        assert QueryType.INSERT.value == "insert"


class TestDetectQueryType:
    """Tests for detect_query_type function."""
    
    def test_detect_select(self):
        """Test detecting SELECT query."""
        assert detect_query_type(SAMPLE_SELECT) == QueryType.SELECT
    
    def test_detect_construct(self):
        """Test detecting CONSTRUCT query."""
        assert detect_query_type(SAMPLE_CONSTRUCT) == QueryType.CONSTRUCT
    
    def test_detect_ask(self):
        """Test detecting ASK query."""
        assert detect_query_type(SAMPLE_ASK) == QueryType.ASK
    
    def test_detect_insert(self):
        """Test detecting INSERT query."""
        assert detect_query_type(SAMPLE_INSERT) == QueryType.INSERT
    
    def test_detect_with_no_prefix(self):
        """Test detecting query without prefixes."""
        simple = "SELECT ?s WHERE { ?s ?p ?o }"
        assert detect_query_type(simple) == QueryType.SELECT
    
    def test_detect_lowercase(self):
        """Test detecting query with lowercase keywords."""
        simple = "select ?s where { ?s ?p ?o }"
        # Case should be handled
        assert detect_query_type(simple) == QueryType.SELECT


class TestExtractParameters:
    """Tests for extract_parameters function."""
    
    def test_extract_simple_parameter(self):
        """Test extracting a simple parameter."""
        query = "SELECT ?s WHERE { ?s foaf:name \"{{name}}\" }"
        params = extract_parameters(query)
        
        assert len(params) == 1
        assert params[0].name == "name"
    
    def test_extract_multiple_parameters(self):
        """Test extracting multiple parameters."""
        query = "SELECT ?s WHERE { ?s foaf:name \"{{name}}\" . ?s foaf:age {{age}} }"
        params = extract_parameters(query)
        
        assert len(params) == 2
        names = [p.name for p in params]
        assert "name" in names
        assert "age" in names
    
    def test_extract_with_description(self):
        """Test extracting parameter with description."""
        query = "SELECT ?s WHERE { ?s foaf:name \"{{name:Person's name}}\" }"
        params = extract_parameters(query)
        
        assert len(params) == 1
        assert params[0].name == "name"
        assert params[0].description == "Person's name"
    
    def test_extract_no_duplicates(self):
        """Test that same parameter used twice is only extracted once."""
        query = "SELECT ?s WHERE { ?s foaf:name \"{{name}}\" . ?s rdfs:label \"{{name}}\" }"
        params = extract_parameters(query)
        
        assert len(params) == 1


class TestQueryParameter:
    """Tests for QueryParameter dataclass."""
    
    def test_create_parameter(self):
        """Test creating a parameter."""
        param = QueryParameter(
            name="limit",
            description="Maximum results",
            default_value="100",
            required=False,
            value_type="integer"
        )
        assert param.name == "limit"
        assert param.default_value == "100"
        assert not param.required
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        param = QueryParameter(
            name="uri",
            description="Resource URI",
            required=True,
            value_type="uri"
        )
        
        d = param.to_dict()
        restored = QueryParameter.from_dict(d)
        
        assert restored.name == param.name
        assert restored.description == param.description
        assert restored.required == param.required
        assert restored.value_type == param.value_type


class TestSavedQuery:
    """Tests for SavedQuery dataclass."""
    
    def test_create_query(self):
        """Test creating a saved query."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Find people",
            sparql=SAMPLE_SELECT,
            query_type=QueryType.SELECT,
            description="Finds all people with emails",
            tags=["foaf", "people"]
        )
        
        assert query.query_id == "query-abc123"
        assert query.name == "Find people"
        assert query.query_type == QueryType.SELECT
        assert "foaf" in query.tags
    
    def test_query_hash(self):
        """Test query hash calculation."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Test",
            sparql=SAMPLE_SELECT,
            query_type=QueryType.SELECT
        )
        
        # Hash should be consistent
        hash1 = query.query_hash
        hash2 = query.query_hash
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_avg_execution_time(self):
        """Test average execution time calculation."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Test",
            sparql=SAMPLE_SELECT,
            query_type=QueryType.SELECT,
            execution_count=10,
            total_execution_time_ms=1000.0
        )
        
        assert query.avg_execution_time_ms == 100.0
    
    def test_avg_execution_time_zero(self):
        """Test average execution time with no executions."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Test",
            sparql=SAMPLE_SELECT,
            query_type=QueryType.SELECT
        )
        
        assert query.avg_execution_time_ms == 0.0
    
    def test_fill_template(self):
        """Test filling template parameters."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Find by name",
            sparql=SAMPLE_TEMPLATE,
            query_type=QueryType.SELECT,
            is_template=True,
            parameters=[QueryParameter(name="name")]
        )
        
        filled = query.fill_template({"name": "Alice"})
        assert '"Alice"' in filled
        assert "{{name}}" not in filled
    
    def test_fill_template_missing_required(self):
        """Test error when required parameter is missing."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Find by name",
            sparql=SAMPLE_TEMPLATE,
            query_type=QueryType.SELECT,
            is_template=True,
            parameters=[QueryParameter(name="name", required=True)]
        )
        
        with pytest.raises(ValueError, match="Required parameter"):
            query.fill_template({})
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        query = SavedQuery(
            query_id="query-abc123",
            name="Find people",
            sparql=SAMPLE_SELECT,
            query_type=QueryType.SELECT,
            description="Test query",
            tags=["test"],
            is_favorite=True,
            execution_count=5
        )
        
        d = query.to_dict()
        restored = SavedQuery.from_dict(d)
        
        assert restored.query_id == query.query_id
        assert restored.name == query.name
        assert restored.query_type == query.query_type
        assert restored.is_favorite == query.is_favorite
        assert restored.execution_count == query.execution_count


class TestQueryExecution:
    """Tests for QueryExecution dataclass."""
    
    def test_create_execution(self):
        """Test creating an execution record."""
        execution = QueryExecution(
            query_id="query-abc123",
            executed_at=datetime.now(),
            duration_ms=50.5,
            result_count=100,
            success=True
        )
        
        assert execution.query_id == "query-abc123"
        assert execution.duration_ms == 50.5
        assert execution.success
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        now = datetime.now()
        execution = QueryExecution(
            query_id="query-abc123",
            executed_at=now,
            duration_ms=100.0,
            result_count=50,
            parameters={"name": "Alice"}
        )
        
        d = execution.to_dict()
        restored = QueryExecution.from_dict(d)
        
        assert restored.query_id == execution.query_id
        assert restored.duration_ms == execution.duration_ms
        assert restored.parameters == {"name": "Alice"}


class TestQueryCollection:
    """Tests for QueryCollection dataclass."""
    
    def test_create_collection(self):
        """Test creating a collection."""
        queries = [
            SavedQuery(
                query_id="q1",
                name="Query 1",
                sparql="SELECT ?s WHERE { ?s ?p ?o }",
                query_type=QueryType.SELECT
            ),
            SavedQuery(
                query_id="q2",
                name="Query 2",
                sparql="ASK { ?s ?p ?o }",
                query_type=QueryType.ASK
            )
        ]
        
        collection = QueryCollection(
            name="My Queries",
            description="A test collection",
            queries=queries,
            author="Test User"
        )
        
        assert collection.name == "My Queries"
        assert len(collection.queries) == 2
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        collection = QueryCollection(
            name="Test Collection",
            description="For testing",
            queries=[
                SavedQuery(
                    query_id="q1",
                    name="Test",
                    sparql="SELECT ?s WHERE { ?s ?p ?o }",
                    query_type=QueryType.SELECT
                )
            ],
            version="2.0"
        )
        
        d = collection.to_dict()
        restored = QueryCollection.from_dict(d)
        
        assert restored.name == collection.name
        assert restored.version == "2.0"
        assert len(restored.queries) == 1


class TestSavedQueryManager:
    """Tests for SavedQueryManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a query manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SavedQueryManager(Path(tmpdir))
    
    def test_save_query(self, manager):
        """Test saving a query."""
        query = manager.save(
            name="Find people",
            sparql=SAMPLE_SELECT,
            description="Find all people",
            tags=["foaf"]
        )
        
        assert query.query_id.startswith("query-")
        assert query.name == "Find people"
        assert query.query_type == QueryType.SELECT
    
    def test_save_detects_type(self, manager):
        """Test that save detects query type."""
        query = manager.save(
            name="Check existence",
            sparql=SAMPLE_ASK
        )
        
        assert query.query_type == QueryType.ASK
    
    def test_save_template(self, manager):
        """Test saving a template query."""
        query = manager.save(
            name="Find by name",
            sparql=SAMPLE_TEMPLATE,
            is_template=True
        )
        
        assert query.is_template
        assert len(query.parameters) == 1
        assert query.parameters[0].name == "name"
    
    def test_get_query(self, manager):
        """Test retrieving a query."""
        saved = manager.save(name="Test", sparql=SAMPLE_SELECT)
        
        retrieved = manager.get(saved.query_id)
        
        assert retrieved is not None
        assert retrieved.name == "Test"
    
    def test_get_nonexistent(self, manager):
        """Test retrieving nonexistent query."""
        result = manager.get("nonexistent-id")
        assert result is None
    
    def test_update_query(self, manager):
        """Test updating a query."""
        query = manager.save(name="Original", sparql=SAMPLE_SELECT)
        original_version = query.version
        
        updated = manager.update(
            query.query_id,
            name="Updated Name",
            description="New description"
        )
        
        assert updated.name == "Updated Name"
        assert updated.description == "New description"
        assert updated.version == original_version + 1
    
    def test_update_sparql(self, manager):
        """Test updating query SPARQL."""
        query = manager.save(name="Test", sparql=SAMPLE_SELECT)
        
        updated = manager.update(
            query.query_id,
            sparql=SAMPLE_ASK
        )
        
        assert updated.query_type == QueryType.ASK
    
    def test_update_nonexistent(self, manager):
        """Test updating nonexistent query."""
        with pytest.raises(KeyError):
            manager.update("nonexistent-id", name="New Name")
    
    def test_delete_query(self, manager):
        """Test deleting a query."""
        query = manager.save(name="To Delete", sparql=SAMPLE_SELECT)
        
        result = manager.delete(query.query_id)
        
        assert result is True
        assert manager.get(query.query_id) is None
    
    def test_delete_nonexistent(self, manager):
        """Test deleting nonexistent query."""
        result = manager.delete("nonexistent-id")
        assert result is False
    
    def test_list_queries(self, manager):
        """Test listing queries."""
        manager.save(name="Query 1", sparql=SAMPLE_SELECT)
        manager.save(name="Query 2", sparql=SAMPLE_ASK)
        manager.save(name="Query 3", sparql=SAMPLE_CONSTRUCT)
        
        queries = manager.list()
        
        assert len(queries) == 3
    
    def test_list_filter_by_type(self, manager):
        """Test filtering by query type."""
        manager.save(name="Select 1", sparql=SAMPLE_SELECT)
        manager.save(name="Ask 1", sparql=SAMPLE_ASK)
        manager.save(name="Select 2", sparql=SAMPLE_SELECT)
        
        select_queries = manager.list(query_type=QueryType.SELECT)
        
        assert len(select_queries) == 2
    
    def test_list_filter_by_tags(self, manager):
        """Test filtering by tags."""
        manager.save(name="Query 1", sparql=SAMPLE_SELECT, tags=["foaf"])
        manager.save(name="Query 2", sparql=SAMPLE_ASK, tags=["rdf"])
        manager.save(name="Query 3", sparql=SAMPLE_SELECT, tags=["foaf", "people"])
        
        foaf_queries = manager.list(tags=["foaf"])
        
        assert len(foaf_queries) == 2
    
    def test_list_favorites(self, manager):
        """Test listing favorites only."""
        q1 = manager.save(name="Query 1", sparql=SAMPLE_SELECT)
        manager.save(name="Query 2", sparql=SAMPLE_ASK)
        manager.update(q1.query_id, is_favorite=True)
        
        favorites = manager.list(favorites_only=True)
        
        assert len(favorites) == 1
        assert favorites[0].query_id == q1.query_id
    
    def test_list_with_limit(self, manager):
        """Test listing with limit."""
        for i in range(5):
            manager.save(name=f"Query {i}", sparql=SAMPLE_SELECT)
        
        queries = manager.list(limit=3)
        
        assert len(queries) == 3
    
    def test_search_by_name(self, manager):
        """Test searching by name."""
        manager.save(name="Find users by email", sparql=SAMPLE_SELECT)
        manager.save(name="Get user age", sparql=SAMPLE_SELECT)
        manager.save(name="Check existence", sparql=SAMPLE_ASK)
        
        results = manager.search("user")
        
        assert len(results) == 2
    
    def test_search_by_sparql(self, manager):
        """Test searching within query text."""
        manager.save(name="Q1", sparql="SELECT ?s WHERE { ?s foaf:name ?name }")
        manager.save(name="Q2", sparql="SELECT ?s WHERE { ?s rdfs:label ?label }")
        
        results = manager.search("foaf")
        
        assert len(results) == 1


class TestQueryManagerPersistence:
    """Tests for query persistence."""
    
    def test_queries_persist(self):
        """Test that queries persist across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with first manager
            manager1 = SavedQueryManager(Path(tmpdir))
            query = manager1.save(name="Persistent", sparql=SAMPLE_SELECT)
            query_id = query.query_id
            
            # Load with second manager
            manager2 = SavedQueryManager(Path(tmpdir))
            loaded = manager2.get(query_id)
            
            assert loaded is not None
            assert loaded.name == "Persistent"


class TestExportImport:
    """Tests for export/import functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a query manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SavedQueryManager(Path(tmpdir))
    
    def test_export_collection(self, manager):
        """Test exporting a collection."""
        q1 = manager.save(name="Q1", sparql=SAMPLE_SELECT, tags=["test"])
        q2 = manager.save(name="Q2", sparql=SAMPLE_ASK, tags=["test"])
        
        collection = manager.export_collection(
            name="Test Collection",
            tags=["test"]
        )
        
        assert len(collection.queries) == 2
        assert collection.name == "Test Collection"
    
    def test_export_by_ids(self, manager):
        """Test exporting specific queries by ID."""
        q1 = manager.save(name="Q1", sparql=SAMPLE_SELECT)
        q2 = manager.save(name="Q2", sparql=SAMPLE_ASK)
        q3 = manager.save(name="Q3", sparql=SAMPLE_CONSTRUCT)
        
        collection = manager.export_collection(
            name="Selected",
            query_ids=[q1.query_id, q3.query_id]
        )
        
        assert len(collection.queries) == 2
    
    def test_export_to_file(self, manager):
        """Test exporting to file."""
        manager.save(name="Q1", sparql=SAMPLE_SELECT)
        manager.save(name="Q2", sparql=SAMPLE_ASK)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            count = manager.export_to_file(f.name, name="Export Test")
            assert count == 2
            
            # Verify file exists and contains data
            import json
            with open(f.name) as rf:
                data = json.load(rf)
            assert data["name"] == "Export Test"
            assert len(data["queries"]) == 2
    
    def test_import_collection(self, manager):
        """Test importing a collection."""
        # Create collection
        collection = QueryCollection(
            name="Import Test",
            description="Testing import",
            queries=[
                SavedQuery(
                    query_id="old-id-1",
                    name="Imported Q1",
                    sparql=SAMPLE_SELECT,
                    query_type=QueryType.SELECT
                )
            ]
        )
        
        id_mapping = manager.import_collection(collection)
        
        assert len(id_mapping) == 1
        new_id = id_mapping["old-id-1"]
        
        imported = manager.get(new_id)
        assert imported is not None
        assert imported.name == "Imported Q1"
    
    def test_import_from_file(self, manager):
        """Test importing from file."""
        # Export first
        manager.save(name="Original", sparql=SAMPLE_SELECT)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manager.export_to_file(f.name)
            
            # Import to new manager
            with tempfile.TemporaryDirectory() as tmpdir2:
                manager2 = SavedQueryManager(Path(tmpdir2))
                id_mapping = manager2.import_from_file(f.name)
                
                assert len(id_mapping) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_save_query(self):
        """Test save_query convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            query = save_query(
                tmpdir,
                name="Quick Save",
                sparql=SAMPLE_SELECT
            )
            
            assert query.name == "Quick Save"
    
    def test_load_query(self):
        """Test load_query convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_query(
                tmpdir,
                name="Load Test",
                sparql=SAMPLE_SELECT
            )
            
            loaded = load_query(tmpdir, saved.query_id)
            
            assert loaded is not None
            assert loaded.name == "Load Test"
