"""Tests for the federation module."""

import pytest
from datetime import datetime

from rdf_starbase.storage.federation import (
    EndpointStatus,
    SyncDirection,
    SyncStrategy,
    QueryPushdownType,
    RemoteEndpoint,
    ServiceCall,
    FederatedResult,
    SyncState,
    SyncOperation,
    EndpointRegistry,
    ServiceClauseParser,
    FederatedQueryExecutor,
    CrossInstanceSync,
    DistributedQueryPlanner,
    create_federation_executor,
    create_sync_manager,
)


# =============================================================================
# RemoteEndpoint Tests
# =============================================================================

class TestRemoteEndpoint:
    """Tests for RemoteEndpoint dataclass."""
    
    def test_create_endpoint(self):
        """Test creating a remote endpoint."""
        endpoint = RemoteEndpoint(url="http://example.org/sparql")
        assert endpoint.url == "http://example.org/sparql"
        assert endpoint.name == "example.org"
        assert endpoint.timeout_seconds == 30.0
        assert endpoint.status == EndpointStatus.UNKNOWN
    
    def test_custom_name(self):
        """Test endpoint with custom name."""
        endpoint = RemoteEndpoint(
            url="http://example.org/sparql",
            name="My Endpoint"
        )
        assert endpoint.name == "My Endpoint"
    
    def test_success_rate(self):
        """Test success rate calculation."""
        endpoint = RemoteEndpoint(url="http://test.org/sparql")
        endpoint.success_count = 8
        endpoint.error_count = 2
        assert endpoint.success_rate == 0.8
    
    def test_success_rate_zero(self):
        """Test success rate with no requests."""
        endpoint = RemoteEndpoint(url="http://test.org/sparql")
        assert endpoint.success_rate == 0.0
    
    def test_to_dict(self):
        """Test serialization."""
        endpoint = RemoteEndpoint(
            url="http://test.org/sparql",
            name="Test",
            timeout_seconds=60.0,
        )
        endpoint.status = EndpointStatus.HEALTHY
        endpoint.success_count = 10
        
        data = endpoint.to_dict()
        assert data["url"] == "http://test.org/sparql"
        assert data["name"] == "Test"
        assert data["status"] == "HEALTHY"
        assert data["success_rate"] == 1.0


# =============================================================================
# EndpointRegistry Tests
# =============================================================================

class TestEndpointRegistry:
    """Tests for EndpointRegistry."""
    
    def test_register_endpoint(self):
        """Test registering an endpoint."""
        registry = EndpointRegistry()
        ep = registry.register("http://example.org/sparql", name="Example")
        
        assert ep.url == "http://example.org/sparql"
        assert len(registry.list_all()) == 1
    
    def test_register_with_alias(self):
        """Test registering with alias."""
        registry = EndpointRegistry()
        registry.register(
            "http://dbpedia.org/sparql",
            name="DBpedia",
            alias="dbpedia"
        )
        
        ep = registry.get("dbpedia")
        assert ep is not None
        assert ep.url == "http://dbpedia.org/sparql"
    
    def test_resolve_url(self):
        """Test URL resolution."""
        registry = EndpointRegistry()
        registry.register("http://example.org/sparql", alias="ex")
        
        assert registry.resolve_url("ex") == "http://example.org/sparql"
        assert registry.resolve_url("http://other.org") == "http://other.org"
    
    def test_unregister(self):
        """Test unregistering an endpoint."""
        registry = EndpointRegistry()
        registry.register("http://example.org/sparql", alias="ex")
        
        assert registry.unregister("ex")
        assert registry.get("ex") is None
        assert len(registry.list_all()) == 0
    
    def test_list_healthy(self):
        """Test listing healthy endpoints."""
        registry = EndpointRegistry()
        ep1 = registry.register("http://healthy.org/sparql")
        ep2 = registry.register("http://unhealthy.org/sparql")
        
        ep1.status = EndpointStatus.HEALTHY
        ep2.status = EndpointStatus.UNREACHABLE
        
        healthy = registry.list_healthy()
        assert len(healthy) == 1
        assert healthy[0].url == "http://healthy.org/sparql"
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        registry = EndpointRegistry()
        registry.register("http://example.org/sparql", alias="ex")
        
        data = registry.to_dict()
        restored = EndpointRegistry.from_dict(data)
        
        assert len(restored.list_all()) == 1
        assert restored.resolve_url("ex") == "http://example.org/sparql"


# =============================================================================
# ServiceClauseParser Tests
# =============================================================================

class TestServiceClauseParser:
    """Tests for SERVICE clause parsing."""
    
    def test_parse_simple_service(self):
        """Test parsing a simple SERVICE clause."""
        query = """
        SELECT * WHERE {
            ?s ?p ?o .
            SERVICE <http://example.org/sparql> {
                ?s rdfs:label ?label .
            }
        }
        """
        
        services = ServiceClauseParser.parse(query)
        assert len(services) == 1
        assert services[0].endpoint_url == "http://example.org/sparql"
        assert "label" in services[0].variables_used
    
    def test_parse_silent_service(self):
        """Test parsing SERVICE SILENT."""
        query = """
        SELECT * WHERE {
            SERVICE SILENT <http://example.org/sparql> {
                ?s ?p ?o .
            }
        }
        """
        
        services = ServiceClauseParser.parse(query)
        assert len(services) == 1
        assert services[0].is_silent
    
    def test_parse_multiple_services(self):
        """Test parsing multiple SERVICE clauses."""
        query = """
        SELECT * WHERE {
            SERVICE <http://a.org/sparql> { ?s ?p ?o }
            SERVICE <http://b.org/sparql> { ?x ?y ?z }
        }
        """
        
        services = ServiceClauseParser.parse(query)
        assert len(services) == 2
    
    def test_has_service_clause(self):
        """Test checking for SERVICE clause."""
        with_service = "SELECT * WHERE { SERVICE <http://x.org> { ?s ?p ?o } }"
        without_service = "SELECT * WHERE { ?s ?p ?o }"
        
        assert ServiceClauseParser.has_service_clause(with_service)
        assert not ServiceClauseParser.has_service_clause(without_service)
    
    def test_remove_service_clauses(self):
        """Test removing SERVICE clauses."""
        query = """
        SELECT * WHERE {
            ?s ?p ?o .
            SERVICE <http://x.org/sparql> { ?a ?b ?c }
        }
        """
        
        cleaned = ServiceClauseParser.remove_service_clauses(query)
        assert "SERVICE" not in cleaned
        assert "?s ?p ?o" in cleaned


# =============================================================================
# FederatedQueryExecutor Tests
# =============================================================================

class TestFederatedQueryExecutor:
    """Tests for FederatedQueryExecutor."""
    
    def test_create_executor(self):
        """Test creating executor."""
        executor = FederatedQueryExecutor()
        assert executor.cache_enabled
        assert executor.max_concurrent_requests == 5
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        executor = FederatedQueryExecutor()
        key1 = executor._cache_key("http://a.org", "SELECT * WHERE { ?s ?p ?o }")
        key2 = executor._cache_key("http://b.org", "SELECT * WHERE { ?s ?p ?o }")
        key3 = executor._cache_key("http://a.org", "SELECT * WHERE { ?s ?p ?o }")
        
        assert key1 != key2  # Different endpoints
        assert key1 == key3  # Same endpoint and query
    
    def test_clear_cache(self):
        """Test cache clearing."""
        executor = FederatedQueryExecutor()
        executor._cache["test"] = ("result", 0)
        
        executor.clear_cache()
        assert len(executor._cache) == 0
    
    def test_local_query_passthrough(self):
        """Test that queries without SERVICE pass through to local executor."""
        executor = FederatedQueryExecutor()
        
        local_results = [{"s": "http://example.org/s"}]
        results = executor.execute_federated_query(
            "SELECT * WHERE { ?s ?p ?o }",
            lambda q: local_results
        )
        
        assert results == local_results
    
    def test_create_federation_executor(self):
        """Test convenience function."""
        executor = create_federation_executor(
            endpoints=[{"url": "http://example.org/sparql"}],
            cache_enabled=False,
        )
        
        assert not executor.cache_enabled
        assert len(executor.registry.list_all()) == 1


# =============================================================================
# CrossInstanceSync Tests
# =============================================================================

class TestCrossInstanceSync:
    """Tests for cross-instance synchronization."""
    
    def test_create_sync_manager(self):
        """Test creating sync manager."""
        sync = CrossInstanceSync("instance-1")
        assert sync.local_instance_id == "instance-1"
        assert sync.direction == SyncDirection.BIDIRECTIONAL
    
    def test_register_remote(self):
        """Test registering a remote instance."""
        sync = CrossInstanceSync("local")
        state = sync.register_remote("remote-1")
        
        assert state.remote_instance_id == "remote-1"
        assert state.pending_changes == 0
    
    def test_record_local_change(self):
        """Test recording local changes."""
        sync = CrossInstanceSync("local")
        sync.register_remote("remote")
        
        sync.record_local_change("insert", "s", "p", "o")
        
        ops = sync.get_pending_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "insert"
        
        state = sync.get_sync_state("remote")
        assert state.pending_changes == 1
    
    def test_clear_pending(self):
        """Test clearing pending operations."""
        sync = CrossInstanceSync("local")
        sync.register_remote("remote")
        sync.record_local_change("insert", "s", "p", "o")
        
        sync.clear_pending_operations()
        
        assert len(sync.get_pending_operations()) == 0
        assert sync.get_sync_state("remote").pending_changes == 0
    
    def test_detect_conflicts(self):
        """Test conflict detection."""
        sync = CrossInstanceSync("local")
        
        local_ops = [SyncOperation("insert", "s", "p", "o")]
        remote_ops = [SyncOperation("delete", "s", "p", "o")]
        
        conflicts = sync.detect_conflicts(local_ops, remote_ops)
        assert len(conflicts) == 1
    
    def test_resolve_conflict_local_wins(self):
        """Test local-wins conflict resolution."""
        sync = CrossInstanceSync("local", strategy=SyncStrategy.LOCAL_WINS)
        
        local_op = SyncOperation("insert", "s", "p", "o")
        remote_op = SyncOperation("delete", "s", "p", "o")
        
        winner = sync.resolve_conflict(local_op, remote_op)
        assert winner == local_op
    
    def test_resolve_conflict_remote_wins(self):
        """Test remote-wins conflict resolution."""
        sync = CrossInstanceSync("local", strategy=SyncStrategy.REMOTE_WINS)
        
        local_op = SyncOperation("insert", "s", "p", "o")
        remote_op = SyncOperation("delete", "s", "p", "o")
        
        winner = sync.resolve_conflict(local_op, remote_op)
        assert winner == remote_op
    
    def test_resolve_conflict_most_recent(self):
        """Test most-recent conflict resolution."""
        sync = CrossInstanceSync("local", strategy=SyncStrategy.MOST_RECENT)
        
        local_op = SyncOperation(
            "insert", "s", "p", "o",
            timestamp=datetime(2026, 1, 1)
        )
        remote_op = SyncOperation(
            "delete", "s", "p", "o",
            timestamp=datetime(2026, 1, 2)
        )
        
        winner = sync.resolve_conflict(local_op, remote_op)
        assert winner == remote_op
    
    def test_sync_with_remote(self):
        """Test full sync operation."""
        sync = CrossInstanceSync("local")
        sync.register_remote("remote")
        
        remote_ops = [
            SyncOperation("insert", "s1", "p", "o"),
            SyncOperation("insert", "s2", "p", "o"),
        ]
        
        applied = []
        def apply_op(op):
            applied.append(op)
            return True
        
        stats = sync.sync_with_remote("remote", remote_ops, apply_op)
        
        assert stats["applied"] == 2
        assert len(applied) == 2
    
    def test_create_sync_manager_func(self):
        """Test convenience function."""
        sync = create_sync_manager(
            "instance-1",
            direction="push",
            strategy="local_wins",
        )
        
        assert sync.direction == SyncDirection.PUSH
        assert sync.strategy == SyncStrategy.LOCAL_WINS


# =============================================================================
# DistributedQueryPlanner Tests
# =============================================================================

class TestDistributedQueryPlanner:
    """Tests for distributed query planning."""
    
    def test_create_planner(self):
        """Test creating query planner."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        assert planner.registry == registry
    
    def test_analyze_query(self):
        """Test query analysis."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        
        query = """
        SELECT * WHERE {
            ?s ?p ?o .
            SERVICE <http://example.org/sparql> {
                ?s rdfs:label ?label .
                FILTER(?label = "test")
            }
        }
        """
        
        analysis = planner.analyze_query(query)
        assert analysis["service_calls"] == 1
        assert "http://example.org/sparql" in analysis["endpoints"]
        assert len(analysis["pushdown_opportunities"]) > 0
    
    def test_analyze_query_no_services(self):
        """Test analyzing query without SERVICE clauses."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        
        analysis = planner.analyze_query("SELECT * WHERE { ?s ?p ?o }")
        assert analysis["service_calls"] == 0
    
    def test_create_execution_plan(self):
        """Test execution plan creation."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        
        query = """
        SELECT * WHERE {
            ?s ?p ?o .
            SERVICE <http://example.org/sparql> { ?s rdfs:label ?l }
        }
        """
        
        plan = planner.create_execution_plan(query)
        
        assert plan["service_calls"] == 1
        assert len(plan["execution_steps"]) == 3  # remote, local, join
        
        # Check phases
        phases = [s["phase"] for s in plan["execution_steps"]]
        assert "remote_execution" in phases
        assert "local_execution" in phases
        assert "join" in phases
    
    def test_optimize_query(self):
        """Test query optimization."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        
        query = "SELECT * WHERE { ?s ?p ?o }"
        optimized = planner.optimize_query(query)
        
        # For now, optimization returns original query
        assert optimized == query
    
    def test_estimate_cardinality(self):
        """Test cardinality estimation."""
        registry = EndpointRegistry()
        planner = DistributedQueryPlanner(registry)
        
        simple = "?s ?p ?o"
        complex = "?s ?p ?o . ?o ?q ?r . FILTER(?r > 10)"
        
        simple_card = planner._estimate_cardinality("http://x.org", simple)
        complex_card = planner._estimate_cardinality("http://x.org", complex)
        
        # Complex queries should have lower estimated cardinality
        assert simple_card > complex_card


# =============================================================================
# Integration Tests
# =============================================================================

class TestFederationIntegration:
    """Integration tests for federation module."""
    
    def test_full_workflow(self):
        """Test complete federation workflow."""
        # Set up registry
        registry = EndpointRegistry()
        registry.register("http://dbpedia.org/sparql", alias="dbpedia")
        registry.register("http://wikidata.org/sparql", alias="wikidata")
        
        # Create executor
        executor = FederatedQueryExecutor(registry=registry, cache_enabled=True)
        
        # Create planner
        planner = DistributedQueryPlanner(registry)
        
        # Analyze a federated query
        query = """
        SELECT * WHERE {
            ?person a :Person .
            SERVICE <http://dbpedia.org/sparql> {
                ?person rdfs:label ?label .
            }
        }
        """
        
        analysis = planner.analyze_query(query)
        assert analysis["service_calls"] == 1
        
        plan = planner.create_execution_plan(query)
        assert len(plan["execution_steps"]) == 3
    
    def test_sync_round_trip(self):
        """Test sync data round trip."""
        sync1 = CrossInstanceSync("instance-1")
        sync2 = CrossInstanceSync("instance-2")
        
        sync1.register_remote("instance-2")
        sync2.register_remote("instance-1")
        
        # Instance 1 records changes
        sync1.record_local_change("insert", "s1", "p", "o1")
        sync1.record_local_change("insert", "s2", "p", "o2")
        
        # Get operations from instance 1
        ops = sync1.get_pending_operations()
        
        # Sync to instance 2
        applied = []
        stats = sync2.sync_with_remote("instance-1", ops, lambda o: applied.append(o) or True)
        
        assert stats["applied"] == 2
        assert len(applied) == 2
