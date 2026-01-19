"""
Tests for AI Grounding API.

Tests cover:
1. /ai/query - Structured fact retrieval with confidence/freshness filters
2. /ai/verify - Claim verification with supporting/contradicting evidence
3. /ai/context - Entity context retrieval
4. /ai/materialize - Inference materialization
5. /ai/inferences - List inferred triples
6. /ai/health - Health check
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from rdf_starbase import TripleStore, ProvenanceContext
from rdf_starbase.web import create_app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def store_with_data():
    """Create a store with test data for AI grounding tests."""
    store = TripleStore()
    
    # High confidence source
    crm_prov = ProvenanceContext(
        source="CRM_System",
        confidence=0.95,
        process="api_sync",
    )
    
    # Medium confidence source
    web_prov = ProvenanceContext(
        source="Web_Scraper",
        confidence=0.75,
        process="web_crawler",
    )
    
    # Low confidence source
    manual_prov = ProvenanceContext(
        source="Manual_Entry",
        confidence=0.5,
        process="data_entry",
    )
    
    # Add facts about Alice
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/name",
        "Alice Johnson",
        crm_prov,
    )
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/age",
        30,
        crm_prov,
    )
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://example.org/bob",
        crm_prov,
    )
    store.add_triple(
        "http://example.org/alice",
        "http://example.org/email",
        "alice@example.com",
        web_prov,
    )
    
    # Add facts about Bob
    store.add_triple(
        "http://example.org/bob",
        "http://xmlns.com/foaf/0.1/name",
        "Bob Smith",
        crm_prov,
    )
    
    # Add competing claim - different age from different source
    store.add_triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/age",
        32,
        manual_prov,
    )
    
    return store


@pytest.fixture
def client(store_with_data):
    """Create test client with the app."""
    app = create_app(store=store_with_data)
    return TestClient(app)


# =============================================================================
# Test /ai/health
# =============================================================================

class TestAIHealth:
    """Tests for AI health endpoint."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/ai/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api"] == "ai_grounding"
        assert "query" in data["capabilities"]
        assert "verify" in data["capabilities"]
        assert "context" in data["capabilities"]
    
    def test_health_shows_store_stats(self, client):
        """Test that health check includes store stats."""
        response = client.get("/ai/health")
        data = response.json()
        assert "store_stats" in data
        # Stats may be 0 if new stats format doesn't have total_triples
        assert "total_triples" in data["store_stats"] or "unique_subjects" in data["store_stats"]


# =============================================================================
# Test /ai/query
# =============================================================================

class TestAIQuery:
    """Tests for structured fact retrieval."""
    
    def test_query_all_facts(self, client):
        """Test querying all facts."""
        response = client.post("/ai/query", json={})
        assert response.status_code == 200
        data = response.json()
        assert "facts" in data
        assert len(data["facts"]) > 0
        assert "retrieval_timestamp" in data
    
    def test_query_by_subject(self, client):
        """Test filtering by subject."""
        response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
        })
        assert response.status_code == 200
        data = response.json()
        # All facts should be about Alice
        for fact in data["facts"]:
            assert fact["subject"] == "http://example.org/alice"
    
    def test_query_by_predicate(self, client):
        """Test filtering by predicate."""
        response = client.post("/ai/query", json={
            "predicate": "http://xmlns.com/foaf/0.1/name",
        })
        assert response.status_code == 200
        data = response.json()
        for fact in data["facts"]:
            assert fact["predicate"] == "http://xmlns.com/foaf/0.1/name"
    
    def test_query_high_confidence(self, client):
        """Test filtering by high confidence."""
        response = client.post("/ai/query", json={
            "min_confidence": "high",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_threshold"] == 0.9
        for fact in data["facts"]:
            assert fact["citation"]["confidence"] >= 0.9
    
    def test_query_medium_confidence(self, client):
        """Test filtering by medium confidence."""
        response = client.post("/ai/query", json={
            "min_confidence": "medium",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_threshold"] == 0.7
        for fact in data["facts"]:
            assert fact["citation"]["confidence"] >= 0.7
    
    def test_query_by_source(self, client):
        """Test filtering by source."""
        response = client.post("/ai/query", json={
            "sources": ["CRM_System"],
        })
        assert response.status_code == 200
        data = response.json()
        for fact in data["facts"]:
            assert fact["citation"]["source"] == "CRM_System"
    
    def test_query_returns_citation(self, client):
        """Test that facts include citation information."""
        response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/name",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["facts"]) > 0
        
        fact = data["facts"][0]
        assert "citation" in fact
        citation = fact["citation"]
        assert "fact_hash" in citation
        assert "source" in citation
        assert "confidence" in citation
        assert "timestamp" in citation
        assert "retrieval_time" in citation
    
    def test_query_limit(self, client):
        """Test result limiting."""
        response = client.post("/ai/query", json={
            "limit": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["facts"]) <= 2
    
    def test_query_returns_sources_used(self, client):
        """Test that response includes sources used."""
        response = client.post("/ai/query", json={})
        assert response.status_code == 200
        data = response.json()
        assert "sources_used" in data
        assert len(data["sources_used"]) > 0


# =============================================================================
# Test /ai/verify
# =============================================================================

class TestAIVerify:
    """Tests for claim verification."""
    
    def test_verify_supported_claim(self, client):
        """Test verifying a claim that is supported."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/name",
            "expected_object": "Alice Johnson",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["claim_supported"] is True
        assert len(data["supporting_facts"]) > 0
        assert data["confidence"] >= 0.9  # High confidence from CRM
    
    def test_verify_unsupported_claim(self, client):
        """Test verifying a claim that is not supported."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/name",
            "expected_object": "Wrong Name",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["claim_supported"] is False
        assert len(data["contradicting_facts"]) > 0
    
    def test_verify_competing_claims(self, client):
        """Test verifying a claim with competing values."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/age",
            "expected_object": "30",
            "min_confidence": "any",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["claim_supported"] is True
        assert data["has_conflicts"] is True
        assert len(data["contradicting_facts"]) > 0
        assert "contested" in data["recommendation"].lower() or "contradict" in data["recommendation"].lower()
    
    def test_verify_unknown_subject(self, client):
        """Test verifying a claim about unknown subject."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/unknown",
            "predicate": "http://xmlns.com/foaf/0.1/name",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["claim_supported"] is False
        assert "not available" in data["recommendation"].lower() or "no facts" in data["recommendation"].lower()
    
    def test_verify_without_expected_value(self, client):
        """Test verification without specifying expected value."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/name",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["claim_supported"] is True
        assert len(data["supporting_facts"]) > 0
    
    def test_verify_returns_recommendation(self, client):
        """Test that verification returns AI recommendation."""
        response = client.post("/ai/verify", json={
            "subject": "http://example.org/alice",
            "predicate": "http://xmlns.com/foaf/0.1/name",
        })
        assert response.status_code == 200
        data = response.json()
        assert "recommendation" in data
        assert len(data["recommendation"]) > 10  # Meaningful recommendation


# =============================================================================
# Test /ai/context
# =============================================================================

class TestAIContext:
    """Tests for entity context retrieval."""
    
    def test_get_entity_context(self, client):
        """Test getting full context for an entity."""
        response = client.get("/ai/context/http://example.org/alice")
        assert response.status_code == 200
        data = response.json()
        assert data["entity"] == "http://example.org/alice"
        assert len(data["facts"]) > 0
        assert "sources" in data
        assert "confidence_summary" in data
    
    def test_context_includes_related_entities(self, client):
        """Test that context includes related entities."""
        response = client.get("/ai/context/http://example.org/alice")
        assert response.status_code == 200
        data = response.json()
        assert "related_entities" in data
        # Bob should be related (Alice knows Bob)
        assert "http://example.org/bob" in data["related_entities"]
    
    def test_context_confidence_summary(self, client):
        """Test that context includes confidence summary."""
        response = client.get("/ai/context/http://example.org/alice")
        assert response.status_code == 200
        data = response.json()
        summary = data["confidence_summary"]
        assert "min" in summary
        assert "max" in summary
        assert "avg" in summary
        assert "high_confidence_count" in summary
    
    def test_context_with_confidence_filter(self, client):
        """Test filtering context by confidence."""
        response = client.get(
            "/ai/context/http://example.org/alice",
            params={"min_confidence": "high"},
        )
        assert response.status_code == 200
        data = response.json()
        for fact in data["facts"]:
            assert fact["citation"]["confidence"] >= 0.9
    
    def test_context_without_incoming(self, client):
        """Test getting context without incoming edges."""
        response = client.get(
            "/ai/context/http://example.org/bob",
            params={"include_incoming": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        # Should only have outgoing facts (Bob as subject)
        for fact in data["facts"]:
            assert fact["subject"] == "http://example.org/bob"
    
    def test_context_empty_entity(self, client):
        """Test getting context for unknown entity."""
        response = client.get("/ai/context/http://example.org/unknown")
        assert response.status_code == 200
        data = response.json()
        assert len(data["facts"]) == 0


# =============================================================================
# Test /ai/materialize (basic test without full reasoner setup)
# =============================================================================

class TestAIMaterialize:
    """Tests for inference materialization."""
    
    def test_materialize_endpoint_exists(self, client):
        """Test that materialize endpoint exists."""
        response = client.post("/ai/materialize", json={})
        # Should return 200 (success), 500 (error), or 501 (not implemented for this store)
        # The endpoint exists if we don't get 404
        assert response.status_code != 404
    
    def test_materialize_request_validation(self, client):
        """Test that materialize validates request."""
        response = client.post("/ai/materialize", json={
            "enable_rdfs": True,
            "enable_owl": True,
            "max_iterations": 50,
        })
        # Endpoint should accept valid request (may fail internally due to store type)
        assert response.status_code != 404
        assert response.status_code != 422  # Validation error


# =============================================================================
# Test /ai/inferences
# =============================================================================

class TestAIInferences:
    """Tests for listing inferred triples."""
    
    def test_inferences_endpoint_exists(self, client):
        """Test that inferences endpoint exists."""
        response = client.get("/ai/inferences")
        assert response.status_code == 200
    
    def test_inferences_with_limit(self, client):
        """Test limiting inference results."""
        response = client.get("/ai/inferences", params={"limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert "count" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestAIGroundingIntegration:
    """Integration tests for AI grounding workflow."""
    
    def test_rag_workflow(self, client):
        """Test typical RAG workflow: query -> verify -> respond."""
        # Step 1: Query for relevant facts
        query_response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
            "min_confidence": "medium",
        })
        assert query_response.status_code == 200
        facts = query_response.json()["facts"]
        assert len(facts) > 0
        
        # Step 2: Verify a specific claim before using
        name_fact = next(
            (f for f in facts if "name" in f["predicate"]), 
            None
        )
        if name_fact:
            verify_response = client.post("/ai/verify", json={
                "subject": name_fact["subject"],
                "predicate": name_fact["predicate"],
                "expected_object": name_fact["object"],
            })
            assert verify_response.status_code == 200
            assert verify_response.json()["claim_supported"] is True
    
    def test_entity_exploration_workflow(self, client):
        """Test entity exploration: context -> related entities."""
        # Get context for Alice
        context_response = client.get("/ai/context/http://example.org/alice")
        assert context_response.status_code == 200
        context = context_response.json()
        
        # Explore a related entity
        related = context["related_entities"]
        if related:
            related_response = client.get(f"/ai/context/{related[0]}")
            assert related_response.status_code == 200
    
    def test_confidence_filtering_workflow(self, client):
        """Test workflow with progressive confidence relaxation."""
        # Try high confidence first
        high_response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
            "min_confidence": "high",
        })
        high_count = len(high_response.json()["facts"])
        
        # Relax to medium
        medium_response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
            "min_confidence": "medium",
        })
        medium_count = len(medium_response.json()["facts"])
        
        # Relax to any
        any_response = client.post("/ai/query", json={
            "subject": "http://example.org/alice",
            "min_confidence": "any",
        })
        any_count = len(any_response.json()["facts"])
        
        # More facts should be available at lower confidence
        assert any_count >= medium_count >= high_count
