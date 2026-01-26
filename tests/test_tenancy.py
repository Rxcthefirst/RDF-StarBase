"""Tests for the multi-tenancy module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from rdf_starbase.storage.tenancy import (
    TenantStatus,
    QuotaType,
    IsolationLevel,
    ResourceQuota,
    TenantConfig,
    QuotaExceededError,
    TenantNotFoundError,
    TenantSuspendedError,
    TenantManager,
    TenantContext,
    UsageTracker,
    create_tenant_manager,
    get_current_tenant,
)


# =============================================================================
# ResourceQuota Tests
# =============================================================================

class TestResourceQuota:
    """Tests for ResourceQuota."""
    
    def test_create_quota(self):
        """Test creating a quota."""
        quota = ResourceQuota(
            quota_type=QuotaType.REPOSITORIES,
            limit=10,
        )
        assert quota.limit == 10
        assert quota.current_usage == 0
        assert quota.remaining == 10
    
    def test_usage_percent(self):
        """Test usage percentage calculation."""
        quota = ResourceQuota(QuotaType.TRIPLES, limit=100)
        quota.current_usage = 25
        assert quota.usage_percent == 25.0
    
    def test_usage_percent_zero_limit(self):
        """Test usage percent with zero limit."""
        quota = ResourceQuota(QuotaType.TRIPLES, limit=0)
        assert quota.usage_percent == 100.0
    
    def test_is_exceeded(self):
        """Test exceeded check."""
        quota = ResourceQuota(QuotaType.QUERIES_PER_MINUTE, limit=5)
        assert not quota.is_exceeded
        
        quota.current_usage = 5
        assert quota.is_exceeded
    
    def test_consume_within_limit(self):
        """Test consuming within limit."""
        quota = ResourceQuota(QuotaType.REPOSITORIES, limit=10)
        
        assert quota.consume(3)
        assert quota.current_usage == 3
        assert quota.remaining == 7
    
    def test_consume_exceeds_limit(self):
        """Test consuming exceeds limit."""
        quota = ResourceQuota(QuotaType.REPOSITORIES, limit=5)
        quota.current_usage = 4
        
        assert not quota.consume(2)  # Would exceed
        assert quota.current_usage == 4  # Unchanged
    
    def test_release(self):
        """Test releasing quota."""
        quota = ResourceQuota(QuotaType.CONCURRENT_QUERIES, limit=10)
        quota.current_usage = 5
        
        quota.release(2)
        assert quota.current_usage == 3
    
    def test_release_floor_zero(self):
        """Test release doesn't go below zero."""
        quota = ResourceQuota(QuotaType.CONCURRENT_QUERIES, limit=10)
        quota.current_usage = 2
        
        quota.release(5)
        assert quota.current_usage == 0
    
    def test_reset_interval(self):
        """Test rate limit reset."""
        quota = ResourceQuota(
            QuotaType.QUERIES_PER_MINUTE,
            limit=100,
            reset_interval_seconds=60,
        )
        quota.current_usage = 50
        quota.last_reset = datetime.now() - timedelta(seconds=120)
        
        # Should reset
        assert quota.check_and_reset()
        assert quota.current_usage == 0
    
    def test_to_dict(self):
        """Test serialization."""
        quota = ResourceQuota(QuotaType.STORAGE_BYTES, limit=1000)
        quota.current_usage = 250
        
        data = quota.to_dict()
        assert data["quota_type"] == "STORAGE_BYTES"
        assert data["limit"] == 1000
        assert data["remaining"] == 750
        assert data["usage_percent"] == 25.0


# =============================================================================
# TenantConfig Tests
# =============================================================================

class TestTenantConfig:
    """Tests for TenantConfig."""
    
    def test_create_tenant(self):
        """Test creating a tenant."""
        tenant = TenantConfig(
            tenant_id="t1",
            name="acme",
            display_name="Acme Corp",
        )
        assert tenant.tenant_id == "t1"
        assert tenant.name == "acme"
        assert tenant.status == TenantStatus.ACTIVE
    
    def test_default_display_name(self):
        """Test default display name."""
        tenant = TenantConfig(tenant_id="t1", name="acme")
        assert tenant.display_name == "acme"
    
    def test_add_quota(self):
        """Test adding a quota."""
        tenant = TenantConfig(tenant_id="t1", name="acme")
        tenant.add_quota(QuotaType.REPOSITORIES, 10)
        
        assert QuotaType.REPOSITORIES in tenant.quotas
        assert tenant.quotas[QuotaType.REPOSITORIES].limit == 10
    
    def test_check_quota_success(self):
        """Test successful quota check."""
        tenant = TenantConfig(tenant_id="t1", name="acme")
        tenant.add_quota(QuotaType.REPOSITORIES, 5)
        
        assert tenant.check_quota(QuotaType.REPOSITORIES)
        assert tenant.quotas[QuotaType.REPOSITORIES].current_usage == 1
    
    def test_check_quota_no_quota_defined(self):
        """Test quota check when no quota defined."""
        tenant = TenantConfig(tenant_id="t1", name="acme")
        # No quota defined = unlimited
        assert tenant.check_quota(QuotaType.REPOSITORIES)
    
    def test_is_feature_enabled(self):
        """Test feature flag checks."""
        tenant = TenantConfig(tenant_id="t1", name="acme")
        tenant.features_enabled.add("feature_a")
        tenant.features_disabled.add("feature_b")
        
        assert tenant.is_feature_enabled("feature_a")
        assert not tenant.is_feature_enabled("feature_b")
        assert tenant.is_feature_enabled("feature_c")  # Default enabled
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        tenant = TenantConfig(
            tenant_id="t1",
            name="acme",
            display_name="Acme Corp",
            isolation_level=IsolationLevel.DEDICATED,
        )
        tenant.add_quota(QuotaType.REPOSITORIES, 10)
        tenant.features_enabled.add("feature_x")
        tenant.repositories.add("repo1")
        
        data = tenant.to_dict()
        restored = TenantConfig.from_dict(data)
        
        assert restored.tenant_id == "t1"
        assert restored.name == "acme"
        assert restored.isolation_level == IsolationLevel.DEDICATED
        assert QuotaType.REPOSITORIES in restored.quotas
        assert "feature_x" in restored.features_enabled
        assert "repo1" in restored.repositories


# =============================================================================
# TenantManager Tests
# =============================================================================

class TestTenantManager:
    """Tests for TenantManager."""
    
    def test_create_tenant(self):
        """Test creating a tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme", display_name="Acme Corp")
        
        assert tenant.name == "acme"
        assert tenant.display_name == "Acme Corp"
        assert tenant.tenant_id is not None
    
    def test_create_tenant_duplicate_name(self):
        """Test creating tenant with duplicate name."""
        manager = TenantManager()
        manager.create_tenant("acme")
        
        with pytest.raises(ValueError, match="already exists"):
            manager.create_tenant("acme")
    
    def test_get_tenant(self):
        """Test getting a tenant by ID."""
        manager = TenantManager()
        created = manager.create_tenant("acme")
        
        fetched = manager.get_tenant(created.tenant_id)
        assert fetched.name == "acme"
    
    def test_get_tenant_not_found(self):
        """Test getting non-existent tenant."""
        manager = TenantManager()
        
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant("nonexistent")
    
    def test_get_tenant_by_name(self):
        """Test getting tenant by name."""
        manager = TenantManager()
        created = manager.create_tenant("acme")
        
        fetched = manager.get_tenant_by_name("acme")
        assert fetched.tenant_id == created.tenant_id
    
    def test_list_tenants(self):
        """Test listing tenants."""
        manager = TenantManager()
        manager.create_tenant("acme")
        manager.create_tenant("globex")
        
        tenants = manager.list_tenants()
        assert len(tenants) == 2
    
    def test_list_tenants_by_status(self):
        """Test listing by status."""
        manager = TenantManager()
        t1 = manager.create_tenant("acme")
        t2 = manager.create_tenant("globex")
        manager.suspend_tenant(t2.tenant_id)
        
        active = manager.list_tenants(status=TenantStatus.ACTIVE)
        suspended = manager.list_tenants(status=TenantStatus.SUSPENDED)
        
        assert len(active) == 1
        assert len(suspended) == 1
    
    def test_update_tenant(self):
        """Test updating tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        updated = manager.update_tenant(
            tenant.tenant_id,
            display_name="Acme Corporation",
            settings={"theme": "dark"},
        )
        
        assert updated.display_name == "Acme Corporation"
        assert updated.settings["theme"] == "dark"
    
    def test_suspend_tenant(self):
        """Test suspending tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        suspended = manager.suspend_tenant(tenant.tenant_id, reason="Non-payment")
        
        assert suspended.status == TenantStatus.SUSPENDED
        assert suspended.settings["suspension_reason"] == "Non-payment"
    
    def test_activate_tenant(self):
        """Test activating suspended tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.suspend_tenant(tenant.tenant_id)
        
        activated = manager.activate_tenant(tenant.tenant_id)
        assert activated.status == TenantStatus.ACTIVE
    
    def test_delete_tenant(self):
        """Test deleting tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        assert manager.delete_tenant(tenant.tenant_id)
        
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant(tenant.tenant_id)
    
    def test_delete_tenant_with_repos(self):
        """Test deleting tenant with repositories."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.add_repository(tenant.tenant_id, "repo1")
        
        with pytest.raises(ValueError, match="has .* repositories"):
            manager.delete_tenant(tenant.tenant_id)
        
        # Force delete works
        assert manager.delete_tenant(tenant.tenant_id, force=True)
    
    def test_check_quota_success(self):
        """Test quota check succeeds."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        assert manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
    
    def test_check_quota_exceeded(self):
        """Test quota exceeded."""
        manager = TenantManager(default_quotas={QuotaType.REPOSITORIES: 1})
        tenant = manager.create_tenant("acme")
        
        # First should succeed
        manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
        
        # Second should fail
        with pytest.raises(QuotaExceededError):
            manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
    
    def test_check_quota_suspended_tenant(self):
        """Test quota check on suspended tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.suspend_tenant(tenant.tenant_id)
        
        with pytest.raises(TenantSuspendedError):
            manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
    
    def test_add_repository(self):
        """Test adding repository to tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        manager.add_repository(tenant.tenant_id, "repo1")
        
        updated = manager.get_tenant(tenant.tenant_id)
        assert "repo1" in updated.repositories
    
    def test_remove_repository(self):
        """Test removing repository from tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.add_repository(tenant.tenant_id, "repo1")
        
        manager.remove_repository(tenant.tenant_id, "repo1")
        
        updated = manager.get_tenant(tenant.tenant_id)
        assert "repo1" not in updated.repositories
    
    def test_get_tenant_for_repository(self):
        """Test finding tenant for repository."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.add_repository(tenant.tenant_id, "repo1")
        
        found = manager.get_tenant_for_repository("repo1")
        assert found is not None
        assert found.tenant_id == tenant.tenant_id
    
    def test_enable_disable_feature(self):
        """Test enabling/disabling features."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        manager.enable_feature(tenant.tenant_id, "feature_x")
        manager.disable_feature(tenant.tenant_id, "feature_y")
        
        updated = manager.get_tenant(tenant.tenant_id)
        assert updated.is_feature_enabled("feature_x")
        assert not updated.is_feature_enabled("feature_y")
    
    def test_persistence(self):
        """Test tenant persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            
            # Create and save
            manager1 = TenantManager(storage_path=path)
            tenant = manager1.create_tenant("acme")
            manager1.add_repository(tenant.tenant_id, "repo1")
            
            # Load in new manager
            manager2 = TenantManager(storage_path=path)
            
            loaded = manager2.get_tenant_by_name("acme")
            assert loaded.tenant_id == tenant.tenant_id
            assert "repo1" in loaded.repositories


# =============================================================================
# TenantContext Tests
# =============================================================================

class TestTenantContext:
    """Tests for TenantContext."""
    
    def test_context_manager(self):
        """Test context manager sets current tenant."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        with TenantContext(manager, tenant.tenant_id):
            assert TenantContext.get_current_tenant_id() == tenant.tenant_id
        
        assert TenantContext.get_current_tenant_id() is None
    
    def test_nested_contexts(self):
        """Test nested tenant contexts."""
        manager = TenantManager()
        t1 = manager.create_tenant("acme")
        t2 = manager.create_tenant("globex")
        
        with TenantContext(manager, t1.tenant_id):
            assert TenantContext.get_current_tenant_id() == t1.tenant_id
            
            with TenantContext(manager, t2.tenant_id):
                assert TenantContext.get_current_tenant_id() == t2.tenant_id
            
            assert TenantContext.get_current_tenant_id() == t1.tenant_id
    
    def test_context_check_quota(self):
        """Test checking quota in context."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        with TenantContext(manager, tenant.tenant_id) as ctx:
            assert ctx.check_quota(QuotaType.REPOSITORIES)
    
    def test_context_feature_check(self):
        """Test feature check in context."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        manager.enable_feature(tenant.tenant_id, "feature_x")
        
        with TenantContext(manager, tenant.tenant_id) as ctx:
            assert ctx.is_feature_enabled("feature_x")
    
    def test_get_current_tenant(self):
        """Test convenience function."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        
        assert get_current_tenant() is None
        
        with TenantContext(manager, tenant.tenant_id):
            assert get_current_tenant() == tenant.tenant_id


# =============================================================================
# UsageTracker Tests
# =============================================================================

class TestUsageTracker:
    """Tests for UsageTracker."""
    
    def test_record_usage(self):
        """Test recording usage."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        tracker = UsageTracker(manager)
        
        tracker.record_usage(tenant.tenant_id, "queries", 10)
        tracker.record_usage(tenant.tenant_id, "queries", 5)
        
        summary = tracker.get_usage_summary(tenant.tenant_id)
        assert summary["queries"] == 15
    
    def test_usage_by_time_range(self):
        """Test usage filtering by time range."""
        manager = TenantManager()
        tenant = manager.create_tenant("acme")
        tracker = UsageTracker(manager)
        
        tracker.record_usage(tenant.tenant_id, "queries", 10)
        
        future = datetime.now() + timedelta(hours=1)
        summary = tracker.get_usage_summary(
            tenant.tenant_id,
            start_time=future,
        )
        
        assert len(summary) == 0
    
    def test_all_tenants_usage(self):
        """Test getting all tenants' usage."""
        manager = TenantManager()
        t1 = manager.create_tenant("acme")
        t2 = manager.create_tenant("globex")
        tracker = UsageTracker(manager)
        
        tracker.record_usage(t1.tenant_id, "queries", 10)
        tracker.record_usage(t2.tenant_id, "queries", 20)
        
        all_usage = tracker.get_all_tenants_usage()
        
        assert t1.tenant_id in all_usage
        assert t2.tenant_id in all_usage
        assert all_usage[t1.tenant_id]["queries"] == 10
        assert all_usage[t2.tenant_id]["queries"] == 20
    
    def test_clear_old_entries(self):
        """Test clearing old entries."""
        manager = TenantManager()
        tracker = UsageTracker(manager)
        
        # Add old entry manually
        tracker._usage_log.append({
            "timestamp": (datetime.now() - timedelta(days=100)).isoformat(),
            "tenant_id": "old",
            "resource_type": "queries",
            "amount": 10,
            "metadata": {},
        })
        
        tracker.record_usage("new", "queries", 5)
        
        tracker.clear_old_entries(older_than_days=90)
        
        assert len(tracker._usage_log) == 1
        assert tracker._usage_log[0]["tenant_id"] == "new"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_tenant_manager(self):
        """Test create_tenant_manager function."""
        manager = create_tenant_manager(
            default_quotas={"repositories": 5}
        )
        
        tenant = manager.create_tenant("test")
        assert tenant.quotas[QuotaType.REPOSITORIES].limit == 5
    
    def test_create_tenant_manager_with_path(self):
        """Test create_tenant_manager with storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_tenant_manager(storage_path=tmpdir)
            manager.create_tenant("test")
            
            # File should exist
            assert (Path(tmpdir) / "tenants.json").exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestTenancyIntegration:
    """Integration tests for multi-tenancy."""
    
    def test_full_tenant_lifecycle(self):
        """Test complete tenant lifecycle."""
        manager = TenantManager()
        
        # Create
        tenant = manager.create_tenant(
            "acme",
            display_name="Acme Corporation",
            isolation_level=IsolationLevel.DEDICATED,
        )
        
        # Configure quotas
        manager.update_quota(tenant.tenant_id, QuotaType.REPOSITORIES, 5)
        manager.update_quota(tenant.tenant_id, QuotaType.TRIPLES, 1_000_000)
        
        # Add repositories
        manager.add_repository(tenant.tenant_id, "main")
        manager.add_repository(tenant.tenant_id, "analytics")
        
        # Use in context
        with TenantContext(manager, tenant.tenant_id) as ctx:
            assert ctx.tenant.name == "acme"
            assert len(ctx.tenant.repositories) == 2
        
        # Suspend
        manager.suspend_tenant(tenant.tenant_id, reason="Maintenance")
        assert manager.get_tenant(tenant.tenant_id).status == TenantStatus.SUSPENDED
        
        # Activate
        manager.activate_tenant(tenant.tenant_id)
        assert manager.get_tenant(tenant.tenant_id).status == TenantStatus.ACTIVE
        
        # Delete
        manager.delete_tenant(tenant.tenant_id, force=True)
        
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant(tenant.tenant_id)
    
    def test_quota_enforcement(self):
        """Test quota enforcement across operations."""
        manager = TenantManager(default_quotas={
            QuotaType.REPOSITORIES: 2,
            QuotaType.QUERIES_PER_MINUTE: 5,
        })
        
        tenant = manager.create_tenant("test")
        
        # Repository quota
        manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
        manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
        
        with pytest.raises(QuotaExceededError):
            manager.check_quota(tenant.tenant_id, QuotaType.REPOSITORIES)
        
        # Rate limit quota
        for _ in range(5):
            manager.check_quota(tenant.tenant_id, QuotaType.QUERIES_PER_MINUTE)
        
        with pytest.raises(QuotaExceededError):
            manager.check_quota(tenant.tenant_id, QuotaType.QUERIES_PER_MINUTE)
