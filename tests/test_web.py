"""Tests for the RDF-StarBase Web API."""

import pytest
from fastapi.testclient import TestClient

from rdf_starbase import TripleStore, ProvenanceContext, AssertionRegistry, SourceType
from rdf_starbase.web import create_app


@pytest.fixture
def store_with_data():
    """Create a store with test data."""
    store = TripleStore()
    
    imdb = ProvenanceContext(source="IMDB", confidence=0.95)
    wiki = ProvenanceContext(source="Wikipedia", confidence=0.8)
    
    # Movies
    store.add_triple("http://ex.org/movie/1", "http://ex.org/title", "Inception", imdb)
    store.add_triple("http://ex.org/movie/1", "http://ex.org/year", "2010", imdb)
    store.add_triple("http://ex.org/movie/1", "http://ex.org/director", "http://ex.org/person/nolan", imdb)
    
    store.add_triple("http://ex.org/movie/2", "http://ex.org/title", "Interstellar", imdb)
    store.add_triple("http://ex.org/movie/2", "http://ex.org/year", "2014", wiki)
    
    # Competing claims
    store.add_triple("http://ex.org/movie/1", "http://ex.org/rating", "8.8", imdb)
    store.add_triple("http://ex.org/movie/1", "http://ex.org/rating", "8.5", wiki)
    
    return store


@pytest.fixture
def registry_with_sources():
    """Create a registry with test sources."""
    registry = AssertionRegistry()
    
    registry.register_source(
        name="IMDB",
        source_type=SourceType.API,
        uri="https://api.imdb.com",
        owner="media-team",
        tags=["movies", "ratings"]
    )
    
    registry.register_source(
        name="Wikipedia",
        source_type=SourceType.DATASET,
        owner="data-team",
        tags=["general", "encyclopedia"]
    )
    
    return registry


@pytest.fixture
def client(store_with_data, registry_with_sources):
    """Create test client with pre-populated data."""
    app = create_app(store=store_with_data, registry=registry_with_sources)
    return TestClient(app)


@pytest.fixture
def empty_client():
    """Create test client with empty store."""
    app = create_app()
    return TestClient(app)


class TestInfoEndpoints:
    """Test info and health endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RDF-StarBase"
        assert "version" in data
    
    def test_health(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "store" in data
        assert "registry" in data
        assert data["store"]["total_assertions"] == 7
        assert data["registry"]["total_sources"] == 2


class TestTripleEndpoints:
    """Test triple management endpoints."""
    
    def test_get_all_triples(self, client):
        """Test getting all triples."""
        response = client.get("/triples")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 7
        assert len(data["triples"]) == 7
    
    def test_get_triples_by_subject(self, client):
        """Test filtering triples by subject."""
        response = client.get("/triples?subject=http://ex.org/movie/1")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5  # title, year, director, 2x rating
    
    def test_get_triples_by_source(self, client):
        """Test filtering triples by source."""
        response = client.get("/triples?source=IMDB")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
    
    def test_get_triples_by_confidence(self, client):
        """Test filtering by minimum confidence."""
        response = client.get("/triples?min_confidence=0.9")
        assert response.status_code == 200
        data = response.json()
        # Only IMDB triples have confidence >= 0.9
        assert all(t["confidence"] >= 0.9 for t in data["triples"])
    
    def test_add_triple(self, empty_client):
        """Test adding a new triple."""
        response = empty_client.post("/triples", json={
            "subject": "http://example.org/test",
            "predicate": "http://example.org/name",
            "object": "Test Entity",
            "provenance": {
                "source": "test",
                "confidence": 0.9
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "assertion_id" in data
        
        # Verify it was added
        response = empty_client.get("/triples")
        assert response.json()["count"] == 1
    
    def test_get_competing_claims(self, client):
        """Test getting competing claims."""
        # Use the URI directly in the path (FastAPI :path converter handles it)
        response = client.get("/triples/http://ex.org/movie/1/claims", params={"predicate": "http://ex.org/rating"})
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 2
        assert data["has_conflicts"] is True
        assert data["unique_values"] == 2
    
    def test_get_competing_claims_no_conflict(self, client):
        """Test competing claims when there's no conflict."""
        # Use the URI directly in the path (FastAPI :path converter handles it)
        response = client.get("/triples/http://ex.org/movie/1/claims", params={"predicate": "http://ex.org/title"})
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 1
        assert data["has_conflicts"] is False


class TestSPARQLEndpoints:
    """Test SPARQL query endpoints."""
    
    def test_select_query(self, client):
        """Test SELECT query."""
        response = client.post("/sparql", json={
            "query": "SELECT ?s ?title WHERE { ?s <http://ex.org/title> ?title }"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["type"] == "select"
        assert data["count"] == 2
        assert "s" in data["columns"]
        assert "title" in data["columns"]
    
    def test_ask_query_true(self, client):
        """Test ASK query returning true."""
        response = client.post("/sparql", json={
            "query": 'ASK WHERE { ?s <http://ex.org/title> "Inception" }'
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["type"] == "ask"
        assert data["result"] is True
    
    def test_ask_query_false(self, client):
        """Test ASK query returning false."""
        response = client.post("/sparql", json={
            "query": 'ASK WHERE { ?s <http://ex.org/title> "NonExistent" }'
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["type"] == "ask"
        assert data["result"] is False
    
    def test_invalid_query(self, client):
        """Test invalid SPARQL query."""
        response = client.post("/sparql", json={
            "query": "NOT VALID SPARQL"
        })
        assert response.status_code == 400
    
    def test_parse_query(self, client):
        """Test parsing SPARQL without executing."""
        response = client.post("/sparql/parse", json={
            "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["type"] == "SelectQuery"
        assert data["pattern_count"] == 1


class TestRegistryEndpoints:
    """Test registry management endpoints."""
    
    def test_get_sources(self, client):
        """Test listing sources."""
        response = client.get("/sources")
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 2
        names = [s["name"] for s in data["sources"]]
        assert "IMDB" in names
        assert "Wikipedia" in names
    
    def test_get_sources_by_type(self, client):
        """Test filtering sources by type."""
        response = client.get("/sources?source_type=api")
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 1
        assert data["sources"][0]["name"] == "IMDB"
    
    def test_get_sources_by_tag(self, client):
        """Test filtering sources by tag."""
        response = client.get("/sources?tag=movies")
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 1
        assert data["sources"][0]["name"] == "IMDB"
    
    def test_register_source(self, empty_client):
        """Test registering a new source."""
        response = empty_client.post("/sources", json={
            "name": "Test API",
            "source_type": "api",
            "uri": "https://test.api.com",
            "owner": "test-team",
            "tags": ["test"]
        })
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Test API"
        assert "id" in data
    
    def test_register_source_invalid_type(self, empty_client):
        """Test registering with invalid source type."""
        response = empty_client.post("/sources", json={
            "name": "Test",
            "source_type": "invalid_type"
        })
        assert response.status_code == 400
    
    def test_get_source_by_id(self, client):
        """Test getting source by ID."""
        # First get all sources to get an ID
        response = client.get("/sources")
        source_id = response.json()["sources"][0]["id"]
        
        # Get by ID
        response = client.get(f"/sources/{source_id}")
        assert response.status_code == 200
        assert response.json()["id"] == source_id
    
    def test_get_source_not_found(self, client):
        """Test getting non-existent source."""
        response = client.get("/sources/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestVisualizationEndpoints:
    """Test graph visualization endpoints."""
    
    def test_get_nodes(self, client):
        """Test getting graph nodes."""
        response = client.get("/graph/nodes")
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] > 0
        assert all("id" in n and "label" in n for n in data["nodes"])
    
    def test_get_edges(self, client):
        """Test getting graph edges."""
        response = client.get("/graph/edges")
        assert response.status_code == 200
        data = response.json()
        
        # Only edges where target is a URI (1 edge: director -> nolan)
        assert data["count"] == 1
        assert all(
            "source" in e and "target" in e and "predicate" in e 
            for e in data["edges"]
        )
    
    def test_get_subgraph(self, client):
        """Test getting subgraph around a node."""
        # Use the URI directly in the path (FastAPI :path converter handles it)
        response = client.get("/graph/subgraph/http://ex.org/movie/1")
        assert response.status_code == 200
        data = response.json()

        assert data["center"] == "http://ex.org/movie/1"
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0
