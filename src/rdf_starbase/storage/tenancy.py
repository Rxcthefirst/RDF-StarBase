"""
Multi-tenancy module for RDF-StarBase.

Provides isolated namespaces with resource quotas, tenant management,
and cross-tenant isolation for enterprise deployments.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class TenantStatus(Enum):
    """Tenant lifecycle status."""
    ACTIVE = auto()
    SUSPENDED = auto()
    PENDING = auto()
    DELETED = auto()


class QuotaType(Enum):
    """Types of resource quotas."""
    REPOSITORIES = auto()      # Max number of repositories
    TRIPLES = auto()           # Max total triples across all repos
    STORAGE_BYTES = auto()     # Max storage in bytes
    QUERIES_PER_MINUTE = auto()  # Query rate limit
    QUERIES_PER_DAY = auto()   # Daily query limit
    INGEST_TRIPLES_PER_MINUTE = auto()  # Ingestion rate limit
    CONCURRENT_QUERIES = auto()  # Max concurrent queries
    MAX_QUERY_TIME_SECONDS = auto()  # Max query execution time


class IsolationLevel(Enum):
    """Tenant isolation levels."""
    SHARED = auto()      # Shared resources, logical isolation only
    DEDICATED = auto()   # Dedicated resources per tenant
    HYBRID = auto()      # Some shared, some dedicated


@dataclass
class ResourceQuota:
    """Resource quota configuration for a tenant."""
    quota_type: QuotaType
    limit: int
    current_usage: int = 0
    reset_interval_seconds: Optional[int] = None  # For rate limits
    last_reset: Optional[datetime] = None
    
    @property
    def remaining(self) -> int:
        """Calculate remaining quota."""
        return max(0, self.limit - self.current_usage)
    
    @property
    def usage_percent(self) -> float:
        """Calculate usage percentage."""
        if self.limit == 0:
            return 100.0
        return (self.current_usage / self.limit) * 100
    
    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current_usage >= self.limit
    
    def check_and_reset(self) -> bool:
        """Check if rate limit should reset, return True if reset occurred."""
        if self.reset_interval_seconds is None:
            return False
        
        now = datetime.now()
        if self.last_reset is None:
            self.last_reset = now
            return False
        
        elapsed = (now - self.last_reset).total_seconds()
        if elapsed >= self.reset_interval_seconds:
            self.current_usage = 0
            self.last_reset = now
            return True
        return False
    
    def consume(self, amount: int = 1) -> bool:
        """
        Consume quota. Returns True if successful, False if exceeded.
        """
        self.check_and_reset()
        
        if self.current_usage + amount > self.limit:
            return False
        
        self.current_usage += amount
        return True
    
    def release(self, amount: int = 1):
        """Release consumed quota."""
        self.current_usage = max(0, self.current_usage - amount)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize quota."""
        return {
            "quota_type": self.quota_type.name,
            "limit": self.limit,
            "current_usage": self.current_usage,
            "remaining": self.remaining,
            "usage_percent": self.usage_percent,
            "reset_interval_seconds": self.reset_interval_seconds,
            "last_reset": self.last_reset.isoformat() if self.last_reset else None,
        }


@dataclass
class TenantConfig:
    """Configuration for a tenant namespace."""
    tenant_id: str
    name: str
    display_name: Optional[str] = None
    status: TenantStatus = TenantStatus.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    # Resource limits
    quotas: Dict[QuotaType, ResourceQuota] = field(default_factory=dict)
    
    # Feature flags
    features_enabled: Set[str] = field(default_factory=set)
    features_disabled: Set[str] = field(default_factory=set)
    
    # Custom settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Repository membership
    repositories: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name
    
    def add_quota(
        self,
        quota_type: QuotaType,
        limit: int,
        reset_interval_seconds: Optional[int] = None,
    ):
        """Add or update a resource quota."""
        self.quotas[quota_type] = ResourceQuota(
            quota_type=quota_type,
            limit=limit,
            reset_interval_seconds=reset_interval_seconds,
        )
        self.updated_at = datetime.now()
    
    def check_quota(self, quota_type: QuotaType, amount: int = 1) -> bool:
        """Check if quota allows the requested amount."""
        if quota_type not in self.quotas:
            return True  # No quota = unlimited
        return self.quotas[quota_type].consume(amount)
    
    def get_quota_usage(self, quota_type: QuotaType) -> Optional[Dict[str, Any]]:
        """Get quota usage information."""
        if quota_type not in self.quotas:
            return None
        return self.quotas[quota_type].to_dict()
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled for this tenant."""
        if feature in self.features_disabled:
            return False
        if feature in self.features_enabled:
            return True
        return True  # Default to enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tenant config."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status.name,
            "isolation_level": self.isolation_level.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "quotas": {qt.name: q.to_dict() for qt, q in self.quotas.items()},
            "features_enabled": list(self.features_enabled),
            "features_disabled": list(self.features_disabled),
            "settings": self.settings,
            "repositories": list(self.repositories),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantConfig":
        """Deserialize tenant config."""
        config = cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            display_name=data.get("display_name"),
            status=TenantStatus[data.get("status", "ACTIVE")],
            isolation_level=IsolationLevel[data.get("isolation_level", "SHARED")],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            created_by=data.get("created_by"),
            features_enabled=set(data.get("features_enabled", [])),
            features_disabled=set(data.get("features_disabled", [])),
            settings=data.get("settings", {}),
            repositories=set(data.get("repositories", [])),
        )
        
        # Restore quotas
        for qt_name, q_data in data.get("quotas", {}).items():
            qt = QuotaType[qt_name]
            config.quotas[qt] = ResourceQuota(
                quota_type=qt,
                limit=q_data["limit"],
                current_usage=q_data.get("current_usage", 0),
                reset_interval_seconds=q_data.get("reset_interval_seconds"),
            )
        
        return config


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their quota."""
    
    def __init__(self, tenant_id: str, quota_type: QuotaType, limit: int, current: int):
        self.tenant_id = tenant_id
        self.quota_type = quota_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"Tenant {tenant_id} exceeded {quota_type.name} quota: "
            f"{current}/{limit}"
        )


class TenantNotFoundError(Exception):
    """Raised when a tenant is not found."""
    pass


class TenantSuspendedError(Exception):
    """Raised when operating on a suspended tenant."""
    pass


class TenantManager:
    """
    Manages multi-tenant namespaces and resource isolation.
    
    Provides:
    - Tenant CRUD operations
    - Quota management and enforcement
    - Tenant-scoped repository access
    - Usage tracking and reporting
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        default_quotas: Optional[Dict[QuotaType, int]] = None,
    ):
        self.storage_path = storage_path
        self._tenants: Dict[str, TenantConfig] = {}
        self._lock = threading.RLock()
        
        # Default quotas for new tenants
        self._default_quotas = default_quotas or {
            QuotaType.REPOSITORIES: 10,
            QuotaType.TRIPLES: 10_000_000,
            QuotaType.STORAGE_BYTES: 10 * 1024 * 1024 * 1024,  # 10 GB
            QuotaType.QUERIES_PER_MINUTE: 100,
            QuotaType.CONCURRENT_QUERIES: 5,
            QuotaType.MAX_QUERY_TIME_SECONDS: 300,
        }
        
        # Load existing tenants
        if storage_path:
            self._load_tenants()
    
    def _load_tenants(self):
        """Load tenants from storage."""
        if not self.storage_path:
            return
        
        tenants_file = self.storage_path / "tenants.json"
        if tenants_file.exists():
            with open(tenants_file, "r") as f:
                data = json.load(f)
                for tenant_data in data.get("tenants", []):
                    tenant = TenantConfig.from_dict(tenant_data)
                    self._tenants[tenant.tenant_id] = tenant
    
    def _save_tenants(self):
        """Save tenants to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        tenants_file = self.storage_path / "tenants.json"
        
        data = {
            "tenants": [t.to_dict() for t in self._tenants.values()],
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(tenants_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_tenant(
        self,
        name: str,
        display_name: Optional[str] = None,
        isolation_level: IsolationLevel = IsolationLevel.SHARED,
        quotas: Optional[Dict[QuotaType, int]] = None,
        created_by: Optional[str] = None,
    ) -> TenantConfig:
        """
        Create a new tenant.
        
        Args:
            name: Unique tenant name (slug-like)
            display_name: Human-readable name
            isolation_level: Resource isolation level
            quotas: Custom quotas (uses defaults if not specified)
            created_by: User who created the tenant
            
        Returns:
            Created TenantConfig
        """
        with self._lock:
            # Check for duplicate name
            for t in self._tenants.values():
                if t.name == name:
                    raise ValueError(f"Tenant with name '{name}' already exists")
            
            tenant_id = str(uuid.uuid4())
            
            tenant = TenantConfig(
                tenant_id=tenant_id,
                name=name,
                display_name=display_name or name,
                status=TenantStatus.ACTIVE,
                isolation_level=isolation_level,
                created_by=created_by,
            )
            
            # Apply quotas
            effective_quotas = {**self._default_quotas, **(quotas or {})}
            for qt, limit in effective_quotas.items():
                reset_interval = None
                if qt == QuotaType.QUERIES_PER_MINUTE:
                    reset_interval = 60
                elif qt == QuotaType.QUERIES_PER_DAY:
                    reset_interval = 86400
                elif qt == QuotaType.INGEST_TRIPLES_PER_MINUTE:
                    reset_interval = 60
                
                tenant.add_quota(qt, limit, reset_interval)
            
            self._tenants[tenant_id] = tenant
            self._save_tenants()
            
            return tenant
    
    def get_tenant(self, tenant_id: str) -> TenantConfig:
        """Get tenant by ID."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")
            return self._tenants[tenant_id]
    
    def get_tenant_by_name(self, name: str) -> TenantConfig:
        """Get tenant by name."""
        with self._lock:
            for tenant in self._tenants.values():
                if tenant.name == name:
                    return tenant
            raise TenantNotFoundError(f"Tenant with name '{name}' not found")
    
    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        isolation_level: Optional[IsolationLevel] = None,
    ) -> List[TenantConfig]:
        """List tenants with optional filtering."""
        with self._lock:
            tenants = list(self._tenants.values())
            
            if status is not None:
                tenants = [t for t in tenants if t.status == status]
            
            if isolation_level is not None:
                tenants = [t for t in tenants if t.isolation_level == isolation_level]
            
            return tenants
    
    def update_tenant(
        self,
        tenant_id: str,
        display_name: Optional[str] = None,
        status: Optional[TenantStatus] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> TenantConfig:
        """Update tenant configuration."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            
            if display_name is not None:
                tenant.display_name = display_name
            
            if status is not None:
                tenant.status = status
            
            if settings is not None:
                tenant.settings.update(settings)
            
            tenant.updated_at = datetime.now()
            self._save_tenants()
            
            return tenant
    
    def suspend_tenant(self, tenant_id: str, reason: Optional[str] = None) -> TenantConfig:
        """Suspend a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.status = TenantStatus.SUSPENDED
            tenant.settings["suspension_reason"] = reason
            tenant.settings["suspended_at"] = datetime.now().isoformat()
            tenant.updated_at = datetime.now()
            self._save_tenants()
            return tenant
    
    def activate_tenant(self, tenant_id: str) -> TenantConfig:
        """Activate a suspended tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.status = TenantStatus.ACTIVE
            tenant.settings.pop("suspension_reason", None)
            tenant.settings.pop("suspended_at", None)
            tenant.updated_at = datetime.now()
            self._save_tenants()
            return tenant
    
    def delete_tenant(self, tenant_id: str, force: bool = False) -> bool:
        """
        Delete a tenant.
        
        Args:
            tenant_id: Tenant to delete
            force: If True, delete even if tenant has repositories
            
        Returns:
            True if deleted
        """
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            
            if tenant.repositories and not force:
                raise ValueError(
                    f"Tenant has {len(tenant.repositories)} repositories. "
                    "Use force=True to delete anyway."
                )
            
            tenant.status = TenantStatus.DELETED
            del self._tenants[tenant_id]
            self._save_tenants()
            
            return True
    
    def update_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        limit: int,
        reset_interval_seconds: Optional[int] = None,
    ):
        """Update a specific quota for a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.add_quota(quota_type, limit, reset_interval_seconds)
            self._save_tenants()
    
    def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> bool:
        """
        Check and consume quota.
        
        Returns True if allowed, raises QuotaExceededError if not.
        """
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            
            if tenant.status == TenantStatus.SUSPENDED:
                raise TenantSuspendedError(f"Tenant {tenant_id} is suspended")
            
            if quota_type not in tenant.quotas:
                return True  # No quota defined = unlimited
            
            quota = tenant.quotas[quota_type]
            if not quota.consume(amount):
                raise QuotaExceededError(
                    tenant_id, quota_type, quota.limit, quota.current_usage
                )
            
            return True
    
    def release_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ):
        """Release previously consumed quota."""
        with self._lock:
            if tenant_id not in self._tenants:
                return
            
            tenant = self._tenants[tenant_id]
            if quota_type in tenant.quotas:
                tenant.quotas[quota_type].release(amount)
    
    def get_quota_usage(self, tenant_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all quota usage for a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            return {
                qt.name: q.to_dict()
                for qt, q in tenant.quotas.items()
            }
    
    def add_repository(self, tenant_id: str, repo_name: str):
        """Associate a repository with a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            
            # Check repository quota
            self.check_quota(tenant_id, QuotaType.REPOSITORIES)
            
            tenant.repositories.add(repo_name)
            tenant.updated_at = datetime.now()
            self._save_tenants()
    
    def remove_repository(self, tenant_id: str, repo_name: str):
        """Remove a repository from a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.repositories.discard(repo_name)
            
            # Release the repository quota
            self.release_quota(tenant_id, QuotaType.REPOSITORIES)
            
            tenant.updated_at = datetime.now()
            self._save_tenants()
    
    def get_tenant_for_repository(self, repo_name: str) -> Optional[TenantConfig]:
        """Find which tenant owns a repository."""
        with self._lock:
            for tenant in self._tenants.values():
                if repo_name in tenant.repositories:
                    return tenant
            return None
    
    def enable_feature(self, tenant_id: str, feature: str):
        """Enable a feature for a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.features_enabled.add(feature)
            tenant.features_disabled.discard(feature)
            tenant.updated_at = datetime.now()
            self._save_tenants()
    
    def disable_feature(self, tenant_id: str, feature: str):
        """Disable a feature for a tenant."""
        with self._lock:
            tenant = self.get_tenant(tenant_id)
            tenant.features_disabled.add(feature)
            tenant.features_enabled.discard(feature)
            tenant.updated_at = datetime.now()
            self._save_tenants()


class TenantContext:
    """
    Thread-local context for tenant-scoped operations.
    
    Use as context manager:
        with TenantContext(manager, tenant_id):
            # All operations scoped to tenant
    """
    
    _local = threading.local()
    
    def __init__(self, manager: TenantManager, tenant_id: str):
        self.manager = manager
        self.tenant_id = tenant_id
        self._previous_context: Optional[str] = None
    
    def __enter__(self) -> "TenantContext":
        self._previous_context = getattr(self._local, "tenant_id", None)
        self._local.tenant_id = self.tenant_id
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._previous_context:
            self._local.tenant_id = self._previous_context
        else:
            delattr(self._local, "tenant_id")
        return False
    
    @classmethod
    def get_current_tenant_id(cls) -> Optional[str]:
        """Get the current tenant ID from context."""
        return getattr(cls._local, "tenant_id", None)
    
    @property
    def tenant(self) -> TenantConfig:
        """Get the tenant configuration."""
        return self.manager.get_tenant(self.tenant_id)
    
    def check_quota(self, quota_type: QuotaType, amount: int = 1) -> bool:
        """Check quota in current tenant context."""
        return self.manager.check_quota(self.tenant_id, quota_type, amount)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled for current tenant."""
        return self.tenant.is_feature_enabled(feature)


class UsageTracker:
    """
    Tracks resource usage across tenants for billing and monitoring.
    """
    
    def __init__(self, manager: TenantManager):
        self.manager = manager
        self._usage_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def record_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record resource usage."""
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "amount": amount,
                "metadata": metadata or {},
            }
            self._usage_log.append(entry)
    
    def get_usage_summary(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Get usage summary for a tenant."""
        with self._lock:
            summary: Dict[str, int] = {}
            
            for entry in self._usage_log:
                if entry["tenant_id"] != tenant_id:
                    continue
                
                entry_time = datetime.fromisoformat(entry["timestamp"])
                
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                
                resource = entry["resource_type"]
                if resource not in summary:
                    summary[resource] = 0
                summary[resource] += entry["amount"]
            
            return summary
    
    def get_all_tenants_usage(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Get usage summary for all tenants."""
        tenant_ids = set()
        with self._lock:
            for entry in self._usage_log:
                tenant_ids.add(entry["tenant_id"])
        
        return {
            tid: self.get_usage_summary(tid, start_time, end_time)
            for tid in tenant_ids
        }
    
    def clear_old_entries(self, older_than_days: int = 90):
        """Clear usage log entries older than specified days."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        
        with self._lock:
            self._usage_log = [
                e for e in self._usage_log
                if datetime.fromisoformat(e["timestamp"]) >= cutoff
            ]


# Convenience functions

def create_tenant_manager(
    storage_path: Optional[str] = None,
    default_quotas: Optional[Dict[str, int]] = None,
) -> TenantManager:
    """Create a tenant manager with optional configuration."""
    path = Path(storage_path) if storage_path else None
    
    quotas = None
    if default_quotas:
        quotas = {
            QuotaType[k.upper()]: v
            for k, v in default_quotas.items()
        }
    
    return TenantManager(storage_path=path, default_quotas=quotas)


def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID from context."""
    return TenantContext.get_current_tenant_id()
