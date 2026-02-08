"""
Authentication and Authorization module for RDF-StarBase API.

Provides API key management, role-based access control, scoped tokens,
and rate limiting for enterprise security.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Role(Enum):
    """Access roles for repository permissions."""
    
    READER = "reader"       # Can query, cannot modify
    WRITER = "writer"       # Can query and modify data
    ADMIN = "admin"         # Full access including config
    
    def can_read(self) -> bool:
        """Check if role allows read operations."""
        return True  # All roles can read
    
    def can_write(self) -> bool:
        """Check if role allows write operations."""
        return self in (Role.WRITER, Role.ADMIN)
    
    def can_admin(self) -> bool:
        """Check if role allows admin operations."""
        return self == Role.ADMIN


class Operation(Enum):
    """Operations that can be scoped in tokens."""
    
    # Read operations
    QUERY = "query"
    DESCRIBE = "describe"
    EXPORT = "export"
    
    # Write operations
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    LOAD = "load"
    
    # Admin operations
    CREATE_REPO = "create_repo"
    DELETE_REPO = "delete_repo"
    BACKUP = "backup"
    RESTORE = "restore"
    CONFIG = "config"
    MANAGE_KEYS = "manage_keys"


# Operation categories
READ_OPERATIONS = {Operation.QUERY, Operation.DESCRIBE, Operation.EXPORT}
WRITE_OPERATIONS = {Operation.INSERT, Operation.DELETE, Operation.UPDATE, Operation.LOAD}
ADMIN_OPERATIONS = {Operation.CREATE_REPO, Operation.DELETE_REPO, 
                    Operation.BACKUP, Operation.RESTORE, Operation.CONFIG, 
                    Operation.MANAGE_KEYS}


@dataclass
class APIKey:
    """An API key for authentication."""
    
    key_id: str
    key_hash: str
    name: str
    role: Role
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    last_used: datetime | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Rate limiting
    rate_limit_queries: int | None = None    # Queries per minute
    rate_limit_ingestion: int | None = None  # Triples per minute
    
    # Scoping
    allowed_repos: set[str] | None = None    # None = all repos
    allowed_operations: set[Operation] | None = None  # None = role default
    
    def is_valid(self) -> bool:
        """Check if key is valid (enabled and not expired)."""
        if not self.enabled:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def can_access_repo(self, repo_name: str) -> bool:
        """Check if key can access a specific repository."""
        if self.allowed_repos is None:
            return True
        return repo_name in self.allowed_repos
    
    def can_perform(self, operation: Operation) -> bool:
        """Check if key can perform a specific operation."""
        if not self.is_valid():
            return False
            
        # Check explicit operation restrictions
        if self.allowed_operations is not None:
            return operation in self.allowed_operations
        
        # Fall back to role-based permissions
        if operation in READ_OPERATIONS:
            return self.role.can_read()
        elif operation in WRITE_OPERATIONS:
            return self.role.can_write()
        elif operation in ADMIN_OPERATIONS:
            return self.role.can_admin()
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes key_hash for safety)."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "rate_limit_queries": self.rate_limit_queries,
            "rate_limit_ingestion": self.rate_limit_ingestion,
            "allowed_repos": list(self.allowed_repos) if self.allowed_repos else None,
            "allowed_operations": [op.value for op in self.allowed_operations] if self.allowed_operations else None,
        }
    
    def _to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary including hash for storage."""
        d = self.to_dict()
        d["key_hash"] = self.key_hash
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "APIKey":
        """Create from dictionary."""
        allowed_ops = None
        if data.get("allowed_operations"):
            allowed_ops = {Operation(op) for op in data["allowed_operations"]}
        
        return cls(
            key_id=data["key_id"],
            key_hash=data.get("key_hash", ""),
            name=data["name"],
            role=Role(data["role"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
            rate_limit_queries=data.get("rate_limit_queries"),
            rate_limit_ingestion=data.get("rate_limit_ingestion"),
            allowed_repos=set(data["allowed_repos"]) if data.get("allowed_repos") else None,
            allowed_operations=allowed_ops,
        )


@dataclass
class ScopedToken:
    """A short-lived token with specific permissions."""
    
    token_id: str
    token_hash: str
    source_key_id: str               # The API key that created this token
    operations: set[Operation]
    repos: set[str] | None           # None = inherit from key
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    max_uses: int | None = None      # None = unlimited
    use_count: int = 0
    
    def is_valid(self) -> bool:
        """Check if token is still valid."""
        if datetime.now() > self.expires_at:
            return False
        if self.max_uses is not None and self.use_count >= self.max_uses:
            return False
        return True
    
    def can_perform(self, operation: Operation) -> bool:
        """Check if token allows operation."""
        return self.is_valid() and operation in self.operations
    
    def can_access_repo(self, repo_name: str) -> bool:
        """Check if token can access repository."""
        if self.repos is None:
            return True
        return repo_name in self.repos
    
    def use(self) -> None:
        """Record a use of this token."""
        self.use_count += 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "token_hash": self.token_hash,
            "source_key_id": self.source_key_id,
            "operations": [op.value for op in self.operations],
            "repos": list(self.repos) if self.repos else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "max_uses": self.max_uses,
            "use_count": self.use_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopedToken":
        """Create from dictionary."""
        return cls(
            token_id=data["token_id"],
            token_hash=data["token_hash"],
            source_key_id=data["source_key_id"],
            operations={Operation(op) for op in data["operations"]},
            repos=set(data["repos"]) if data.get("repos") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            max_uses=data.get("max_uses"),
            use_count=data.get("use_count", 0),
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Window duration in seconds
    window_seconds: float = 60.0
    
    # Global limits (apply to all keys)
    global_queries_per_window: int | None = None
    global_ingestion_per_window: int | None = None
    
    # Default per-key limits (used when key has no specific limit)
    default_queries_per_window: int | None = 1000
    default_ingestion_per_window: int | None = 100000
    
    # Burst allowance (percentage over limit for short bursts)
    burst_allowance: float = 0.1  # 10% burst allowed
    
    # Whether to track rate limits per repository
    per_repo_tracking: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_seconds": self.window_seconds,
            "global_queries_per_window": self.global_queries_per_window,
            "global_ingestion_per_window": self.global_ingestion_per_window,
            "default_queries_per_window": self.default_queries_per_window,
            "default_ingestion_per_window": self.default_ingestion_per_window,
            "burst_allowance": self.burst_allowance,
            "per_repo_tracking": self.per_repo_tracking,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RateLimitConfig":
        """Create from dictionary."""
        return cls(
            window_seconds=data.get("window_seconds", 60.0),
            global_queries_per_window=data.get("global_queries_per_window"),
            global_ingestion_per_window=data.get("global_ingestion_per_window"),
            default_queries_per_window=data.get("default_queries_per_window", 1000),
            default_ingestion_per_window=data.get("default_ingestion_per_window", 100000),
            burst_allowance=data.get("burst_allowance", 0.1),
            per_repo_tracking=data.get("per_repo_tracking", False),
        )


@dataclass
class RateLimitState:
    """Tracks rate limit state for a key."""
    
    key_id: str
    window_start: float = field(default_factory=time.time)
    query_count: int = 0
    triple_count: int = 0
    
    def reset_if_needed(self, window_seconds: float = 60.0) -> None:
        """Reset counters if window has passed."""
        now = time.time()
        if now - self.window_start >= window_seconds:
            self.window_start = now
            self.query_count = 0
            self.triple_count = 0
    
    def check_query_limit(self, limit: int | None, burst_allowance: float = 0.0) -> bool:
        """Check if query is allowed under rate limit."""
        if limit is None:
            return True
        self.reset_if_needed()
        effective_limit = int(limit * (1 + burst_allowance))
        return self.query_count < effective_limit
    
    def check_ingestion_limit(self, triple_count: int, limit: int | None, burst_allowance: float = 0.0) -> bool:
        """Check if ingestion is allowed under rate limit."""
        if limit is None:
            return True
        self.reset_if_needed()
        effective_limit = int(limit * (1 + burst_allowance))
        return self.triple_count + triple_count <= effective_limit
    
    def record_query(self) -> None:
        """Record a query."""
        self.reset_if_needed()
        self.query_count += 1
    
    def record_ingestion(self, count: int) -> None:
        """Record triple ingestion."""
        self.reset_if_needed()
        self.triple_count += count
    
    def get_remaining(self, query_limit: int | None, ingestion_limit: int | None) -> dict[str, int | None]:
        """Get remaining quota for this window."""
        self.reset_if_needed()
        return {
            "queries_remaining": (query_limit - self.query_count) if query_limit else None,
            "ingestion_remaining": (ingestion_limit - self.triple_count) if ingestion_limit else None,
            "window_resets_in_seconds": int(60.0 - (time.time() - self.window_start)),
        }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, key_id: str, limit_type: str, limit: int, retry_after: int = 60):
        self.key_id = key_id
        self.limit_type = limit_type
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {key_id}: {limit_type} limit of {limit}/min")


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, operation: Operation | None = None, repo: str | None = None):
        self.operation = operation
        self.repo = repo
        super().__init__(message)


class APIKeyManager:
    """Manages API keys for authentication."""
    
    KEY_PREFIX_LENGTH = 8
    KEY_LENGTH = 32
    
    def __init__(self, storage_path: Path | None = None):
        """Initialize key manager.
        
        Args:
            storage_path: Path to persist keys (None = in-memory only)
        """
        self.storage_path = storage_path
        self._keys: dict[str, APIKey] = {}
        self._tokens: dict[str, ScopedToken] = {}
        self._rate_limits: dict[str, RateLimitState] = {}
        
        if storage_path and storage_path.exists():
            self._load()
    
    def generate_key(
        self,
        name: str,
        role: Role = Role.READER,
        expires_in: timedelta | None = None,
        rate_limit_queries: int | None = None,
        rate_limit_ingestion: int | None = None,
        allowed_repos: set[str] | None = None,
        allowed_operations: set[Operation] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, APIKey]:
        """Generate a new API key.
        
        Returns:
            Tuple of (raw_key, APIKey). The raw_key is only returned once!
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(self.KEY_LENGTH)
        key_id = raw_key[:self.KEY_PREFIX_LENGTH]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            expires_at=expires_at,
            rate_limit_queries=rate_limit_queries,
            rate_limit_ingestion=rate_limit_ingestion,
            allowed_repos=allowed_repos,
            allowed_operations=allowed_operations,
            metadata=metadata or {},
        )
        
        self._keys[key_id] = api_key
        self._save()
        
        return raw_key, api_key
    
    def validate_key(self, raw_key: str) -> APIKey | None:
        """Validate an API key and return the key object if valid."""
        if len(raw_key) < self.KEY_PREFIX_LENGTH:
            return None
        
        key_id = raw_key[:self.KEY_PREFIX_LENGTH]
        api_key = self._keys.get(key_id)
        
        if api_key is None:
            return None
        
        # Verify hash
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if not hmac.compare_digest(key_hash, api_key.key_hash):
            return None
        
        if not api_key.is_valid():
            return None
        
        # Update last used
        api_key.last_used = datetime.now()
        self._save()
        
        return api_key
    
    def get_key(self, key_id: str) -> APIKey | None:
        """Get an API key by ID."""
        return self._keys.get(key_id)
    
    def list_keys(self) -> list[APIKey]:
        """List all API keys."""
        return list(self._keys.values())
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id not in self._keys:
            return False
        
        self._keys[key_id].enabled = False
        self._save()
        return True
    
    def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key."""
        if key_id not in self._keys:
            return False
        
        del self._keys[key_id]
        self._save()
        return True
    
    def update_key(
        self,
        key_id: str,
        name: str | None = None,
        role: Role | None = None,
        enabled: bool | None = None,
        rate_limit_queries: int | None = None,
        rate_limit_ingestion: int | None = None,
        allowed_repos: set[str] | None = None,
        allowed_operations: set[Operation] | None = None,
    ) -> APIKey | None:
        """Update an API key's settings."""
        api_key = self._keys.get(key_id)
        if api_key is None:
            return None
        
        if name is not None:
            api_key.name = name
        if role is not None:
            api_key.role = role
        if enabled is not None:
            api_key.enabled = enabled
        if rate_limit_queries is not None:
            api_key.rate_limit_queries = rate_limit_queries
        if rate_limit_ingestion is not None:
            api_key.rate_limit_ingestion = rate_limit_ingestion
        if allowed_repos is not None:
            api_key.allowed_repos = allowed_repos
        if allowed_operations is not None:
            api_key.allowed_operations = allowed_operations
        
        self._save()
        return api_key
    
    def create_scoped_token(
        self,
        api_key: APIKey,
        operations: set[Operation],
        repos: set[str] | None = None,
        expires_in: timedelta = timedelta(hours=1),
        max_uses: int | None = None,
    ) -> tuple[str, ScopedToken]:
        """Create a scoped token from an API key.
        
        The token can only have permissions the key has.
        """
        # Filter to operations the key can perform
        valid_ops = {op for op in operations if api_key.can_perform(op)}
        
        if not valid_ops:
            raise AuthorizationError("No valid operations for scoped token")
        
        # Filter to repos the key can access
        if repos and api_key.allowed_repos:
            valid_repos = repos & api_key.allowed_repos
        elif repos:
            valid_repos = repos
        else:
            valid_repos = api_key.allowed_repos
        
        raw_token = secrets.token_urlsafe(self.KEY_LENGTH)
        token_id = raw_token[:self.KEY_PREFIX_LENGTH]
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        
        token = ScopedToken(
            token_id=token_id,
            token_hash=token_hash,
            source_key_id=api_key.key_id,
            operations=valid_ops,
            repos=valid_repos,
            expires_at=datetime.now() + expires_in,
            max_uses=max_uses,
        )
        
        self._tokens[token_id] = token
        return raw_token, token
    
    def validate_token(self, raw_token: str) -> ScopedToken | None:
        """Validate a scoped token."""
        if len(raw_token) < self.KEY_PREFIX_LENGTH:
            return None
        
        token_id = raw_token[:self.KEY_PREFIX_LENGTH]
        token = self._tokens.get(token_id)
        
        if token is None:
            return None
        
        # Verify hash
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        if not hmac.compare_digest(token_hash, token.token_hash):
            return None
        
        if not token.is_valid():
            return None
        
        token.use()
        return token
    
    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens. Returns count removed."""
        expired = [tid for tid, t in self._tokens.items() if not t.is_valid()]
        for tid in expired:
            del self._tokens[tid]
        return len(expired)
    
    def check_rate_limit(
        self,
        key_id: str,
        limit_queries: int | None,
        limit_ingestion: int | None = None,
        triple_count: int = 0,
    ) -> None:
        """Check and update rate limits. Raises RateLimitExceeded if exceeded."""
        if key_id not in self._rate_limits:
            self._rate_limits[key_id] = RateLimitState(key_id)
        
        state = self._rate_limits[key_id]
        
        if limit_queries is not None:
            if not state.check_query_limit(limit_queries):
                raise RateLimitExceeded(key_id, "query", limit_queries)
            state.record_query()
        
        if limit_ingestion is not None and triple_count > 0:
            if not state.check_ingestion_limit(triple_count, limit_ingestion):
                raise RateLimitExceeded(key_id, "ingestion", limit_ingestion)
            state.record_ingestion(triple_count)
    
    def get_rate_limit_state(self, key_id: str) -> RateLimitState | None:
        """Get current rate limit state for a key."""
        return self._rate_limits.get(key_id)
    
    def _save(self) -> None:
        """Persist keys to storage."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "keys": {kid: k._to_storage_dict() for kid, k in self._keys.items()},
            "tokens": {tid: t.to_dict() for tid, t in self._tokens.items()},
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load keys from storage."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        self._keys = {kid: APIKey.from_dict(kd) for kid, kd in data.get("keys", {}).items()}
        self._tokens = {tid: ScopedToken.from_dict(td) for tid, td in data.get("tokens", {}).items()}


class AuthContext:
    """Authorization context for a request."""
    
    def __init__(
        self,
        key: APIKey | None = None,
        token: ScopedToken | None = None,
        manager: APIKeyManager | None = None,
    ):
        """Initialize auth context.
        
        Args:
            key: The API key (if authenticated with key)
            token: The scoped token (if authenticated with token)
            manager: The key manager for rate limiting
        """
        self.key = key
        self.token = token
        self.manager = manager
    
    @property
    def is_authenticated(self) -> bool:
        """Check if context is authenticated."""
        return self.key is not None or self.token is not None
    
    @property
    def key_id(self) -> str | None:
        """Get the key ID."""
        if self.key:
            return self.key.key_id
        if self.token:
            return self.token.source_key_id
        return None
    
    @property
    def role(self) -> Role | None:
        """Get the role."""
        if self.key:
            return self.key.role
        return None
    
    def can_perform(self, operation: Operation, repo: str | None = None) -> bool:
        """Check if operation is allowed."""
        if self.token:
            if not self.token.can_perform(operation):
                return False
            if repo and not self.token.can_access_repo(repo):
                return False
            return True
        
        if self.key:
            if not self.key.can_perform(operation):
                return False
            if repo and not self.key.can_access_repo(repo):
                return False
            return True
        
        return False
    
    def require(self, operation: Operation, repo: str | None = None) -> None:
        """Require permission for operation. Raises AuthorizationError if denied."""
        if not self.is_authenticated:
            raise AuthorizationError("Authentication required", operation, repo)
        
        if not self.can_perform(operation, repo):
            raise AuthorizationError(
                f"Permission denied for {operation.value}" + (f" on {repo}" if repo else ""),
                operation,
                repo,
            )
        
        # Check rate limits
        if self.manager and self.key:
            self.manager.check_rate_limit(
                self.key.key_id,
                self.key.rate_limit_queries,
            )
    
    def require_ingestion(self, triple_count: int, repo: str | None = None) -> None:
        """Require permission for data ingestion with rate limiting."""
        self.require(Operation.INSERT, repo)
        
        if self.manager and self.key and self.key.rate_limit_ingestion:
            self.manager.check_rate_limit(
                self.key.key_id,
                None,
                self.key.rate_limit_ingestion,
                triple_count,
            )


class PermissionPolicy:
    """Defines permission policies for repositories."""
    
    def __init__(self):
        """Initialize permission policy."""
        self._repo_policies: dict[str, dict[str, Role]] = {}  # repo -> {key_id -> role}
        self._default_role: Role = Role.READER
    
    def set_repo_policy(self, repo: str, key_id: str, role: Role) -> None:
        """Set role for a key on a specific repository."""
        if repo not in self._repo_policies:
            self._repo_policies[repo] = {}
        self._repo_policies[repo][key_id] = role
    
    def get_repo_role(self, repo: str, key_id: str) -> Role | None:
        """Get role for a key on a repository."""
        return self._repo_policies.get(repo, {}).get(key_id)
    
    def remove_repo_policy(self, repo: str, key_id: str) -> bool:
        """Remove role for a key on a repository."""
        if repo in self._repo_policies and key_id in self._repo_policies[repo]:
            del self._repo_policies[repo][key_id]
            return True
        return False
    
    def list_repo_policies(self, repo: str) -> dict[str, Role]:
        """List all policies for a repository."""
        return self._repo_policies.get(repo, {}).copy()


# Convenience functions

def create_key_manager(storage_path: Path | None = None) -> APIKeyManager:
    """Create a new API key manager."""
    return APIKeyManager(storage_path)


def require_auth(
    raw_key_or_token: str,
    manager: APIKeyManager,
    operation: Operation,
    repo: str | None = None,
) -> AuthContext:
    """Validate credentials and require permission.
    
    Returns AuthContext if authorized, raises exception otherwise.
    """
    # Try as API key first
    key = manager.validate_key(raw_key_or_token)
    if key:
        ctx = AuthContext(key=key, manager=manager)
        ctx.require(operation, repo)
        return ctx
    
    # Try as scoped token
    token = manager.validate_token(raw_key_or_token)
    if token:
        ctx = AuthContext(token=token, manager=manager)
        ctx.require(operation, repo)
        return ctx
    
    raise AuthorizationError("Invalid credentials")
