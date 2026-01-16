"""Tests for the Assertion Registry."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from rdf_starbase import (
    AssertionRegistry,
    RegisteredSource,
    SourceType,
    SourceStatus,
    TripleStore,
)


class TestAssertionRegistry:
    """Test suite for the Assertion Registry."""
    
    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = AssertionRegistry()
        stats = registry.get_stats()
        
        assert stats["total_sources"] == 0
        assert stats["total_sync_runs"] == 0
    
    def test_register_source(self):
        """Test registering a new data source."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Salesforce CRM",
            source_type=SourceType.API,
            uri="https://api.salesforce.com/v52",
            description="Customer relationship data",
            owner="sales-team",
            sync_frequency="hourly",
        )
        
        assert source.name == "Salesforce CRM"
        assert source.source_type == SourceType.API
        assert source.status == SourceStatus.ACTIVE
        assert source.owner == "sales-team"
        
        # Should be retrievable
        retrieved = registry.get_source(source.id)
        assert retrieved is not None
        assert retrieved.name == "Salesforce CRM"
    
    def test_register_multiple_source_types(self):
        """Test registering different types of sources."""
        registry = AssertionRegistry()
        
        # Register various types
        dataset = registry.register_source(
            name="Customer Master",
            source_type=SourceType.DATASET,
            tags=["master-data", "customers"]
        )
        
        api = registry.register_source(
            name="Weather API",
            source_type=SourceType.API,
            uri="https://api.weather.gov",
            tags=["external", "weather"]
        )
        
        mapping = registry.register_source(
            name="Customer Transform",
            source_type=SourceType.MAPPING,
            description="Maps CRM to master schema"
        )
        
        process = registry.register_source(
            name="ML Enrichment",
            source_type=SourceType.PROCESS,
            owner="data-science"
        )
        
        stats = registry.get_stats()
        assert stats["total_sources"] == 4
        assert stats["sources_by_type"]["dataset"] == 1
        assert stats["sources_by_type"]["api"] == 1
        assert stats["sources_by_type"]["mapping"] == 1
        assert stats["sources_by_type"]["process"] == 1
    
    def test_get_source_by_name(self):
        """Test retrieving source by name."""
        registry = AssertionRegistry()
        
        registry.register_source(
            name="Primary CRM",
            source_type=SourceType.API,
        )
        
        source = registry.get_source_by_name("Primary CRM")
        assert source is not None
        assert source.source_type == SourceType.API
        
        # Non-existent source
        assert registry.get_source_by_name("Unknown") is None
    
    def test_filter_sources_by_type(self):
        """Test filtering sources by type."""
        registry = AssertionRegistry()
        
        registry.register_source(name="API 1", source_type=SourceType.API)
        registry.register_source(name="API 2", source_type=SourceType.API)
        registry.register_source(name="Dataset 1", source_type=SourceType.DATASET)
        
        apis = registry.get_sources(source_type=SourceType.API)
        assert len(apis) == 2
        
        datasets = registry.get_sources(source_type=SourceType.DATASET)
        assert len(datasets) == 1
    
    def test_filter_sources_by_owner(self):
        """Test filtering sources by owner."""
        registry = AssertionRegistry()
        
        registry.register_source(
            name="Sales Data",
            source_type=SourceType.DATASET,
            owner="sales-team"
        )
        registry.register_source(
            name="Marketing Data",
            source_type=SourceType.DATASET,
            owner="marketing-team"
        )
        
        sales_sources = registry.get_sources(owner="sales-team")
        assert len(sales_sources) == 1
        assert sales_sources[0].name == "Sales Data"
    
    def test_filter_sources_by_tag(self):
        """Test filtering sources by tag."""
        registry = AssertionRegistry()
        
        registry.register_source(
            name="Customer API",
            source_type=SourceType.API,
            tags=["customers", "external"]
        )
        registry.register_source(
            name="Internal API",
            source_type=SourceType.API,
            tags=["internal"]
        )
        
        external = registry.get_sources(tag="external")
        assert len(external) == 1
        assert external[0].name == "Customer API"
    
    def test_deprecate_source(self):
        """Test deprecating a source."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Legacy System",
            source_type=SourceType.API,
        )
        
        assert source.status == SourceStatus.ACTIVE
        
        registry.deprecate_source(source.id)
        
        updated = registry.get_source(source.id)
        assert updated.status == SourceStatus.DEPRECATED
    
    def test_start_sync(self):
        """Test starting a sync run."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Data Source",
            source_type=SourceType.API,
        )
        
        run = registry.start_sync(source.id, metadata={"batch_id": "123"})
        
        assert run.source_id == source.id
        assert run.status == "running"
        
        # Source should be in syncing status
        updated_source = registry.get_source(source.id)
        assert updated_source.status == SourceStatus.SYNCING
    
    def test_complete_sync_success(self):
        """Test completing a successful sync run."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Data Source",
            source_type=SourceType.API,
        )
        
        run = registry.start_sync(source.id)
        
        registry.complete_sync(
            run.id,
            assertions_created=100,
            assertions_updated=50,
            status="success"
        )
        
        # Check sync history
        history = registry.get_sync_history(source.id)
        assert len(history) == 1
        assert history["status"][0] == "success"
        assert history["assertions_created"][0] == 100
        
        # Source should be active again
        updated_source = registry.get_source(source.id)
        assert updated_source.status == SourceStatus.ACTIVE
        assert updated_source.last_sync is not None
    
    def test_complete_sync_failure(self):
        """Test completing a failed sync run."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Flaky API",
            source_type=SourceType.API,
        )
        
        run = registry.start_sync(source.id)
        
        registry.complete_sync(
            run.id,
            status="failed",
            errors=["Connection timeout", "Rate limited"]
        )
        
        # Source should be in error status
        updated_source = registry.get_source(source.id)
        assert updated_source.status == SourceStatus.ERROR
    
    def test_get_last_sync(self):
        """Test getting the most recent sync run."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Data Source",
            source_type=SourceType.API,
        )
        
        # No syncs yet
        assert registry.get_last_sync(source.id) is None
        
        # Do two syncs
        run1 = registry.start_sync(source.id)
        registry.complete_sync(run1.id, assertions_created=50)
        
        run2 = registry.start_sync(source.id)
        registry.complete_sync(run2.id, assertions_created=75)
        
        last = registry.get_last_sync(source.id)
        assert last is not None
        assert last.id == run2.id
        assert last.assertions_created == 75
    
    def test_sync_stats(self):
        """Test sync statistics calculation."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="Data Source",
            source_type=SourceType.API,
        )
        
        # Successful sync
        run1 = registry.start_sync(source.id)
        registry.complete_sync(run1.id, status="success")
        
        # Failed sync
        run2 = registry.start_sync(source.id)
        registry.complete_sync(run2.id, status="failed")
        
        # Another success
        run3 = registry.start_sync(source.id)
        registry.complete_sync(run3.id, status="success")
        
        stats = registry.get_stats()
        assert stats["total_sync_runs"] == 3
        assert stats["successful_sync_runs"] == 2
        assert stats["sync_success_rate"] == 2/3
    
    def test_create_provenance_context(self):
        """Test creating provenance context from registered source."""
        registry = AssertionRegistry()
        
        source = registry.register_source(
            name="CRM System",
            source_type=SourceType.API,
            uri="https://api.crm.com"
        )
        
        prov = registry.create_provenance_context(
            source.id,
            confidence=0.95,
            process="daily_sync"
        )
        
        assert prov.source == "CRM System"
        assert prov.confidence == 0.95
        assert prov.process == "daily_sync"
        assert prov.metadata["source_id"] == str(source.id)
    
    def test_provenance_context_unknown_source(self):
        """Test that creating provenance for unknown source fails."""
        registry = AssertionRegistry()
        
        with pytest.raises(ValueError, match="not found"):
            registry.create_provenance_context(uuid4())


class TestRegistryIntegration:
    """Integration tests with TripleStore."""
    
    def test_registry_with_triple_store(self):
        """Test using registry with triple store for full provenance."""
        registry = AssertionRegistry()
        store = TripleStore()
        
        # Register our CRM as a source
        crm = registry.register_source(
            name="Salesforce",
            source_type=SourceType.API,
            owner="sales-team",
        )
        
        # Start a sync
        run = registry.start_sync(crm.id)
        
        # Create provenance from registry
        prov = registry.create_provenance_context(crm.id, confidence=0.9)
        
        # Add triples with tracked provenance
        store.add_triple(
            "http://example.org/customer/1",
            "http://xmlns.com/foaf/0.1/name",
            "Alice",
            prov
        )
        store.add_triple(
            "http://example.org/customer/2",
            "http://xmlns.com/foaf/0.1/name",
            "Bob",
            prov
        )
        
        # Complete the sync
        registry.complete_sync(run.id, assertions_created=2)
        
        # Verify
        last_sync = registry.get_last_sync(crm.id)
        assert last_sync.assertions_created == 2
        
        # Query store - assertions have CRM provenance
        results = store.get_triples(source="Salesforce")
        assert len(results) == 2


class TestRegistryPersistence:
    """Test persistence of the registry."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading registry."""
        registry = AssertionRegistry()
        
        # Add sources
        source = registry.register_source(
            name="Test API",
            source_type=SourceType.API,
            tags=["test"]
        )
        
        # Do a sync
        run = registry.start_sync(source.id)
        registry.complete_sync(run.id, assertions_created=10)
        
        # Save
        path = str(tmp_path / "registry")
        registry.save(path)
        
        # Load
        loaded = AssertionRegistry.load(path)
        
        # Verify
        assert loaded.get_stats()["total_sources"] == 1
        assert loaded.get_stats()["total_sync_runs"] == 1
        
        loaded_source = loaded.get_source_by_name("Test API")
        assert loaded_source is not None
        assert loaded_source.tags == ["test"]
