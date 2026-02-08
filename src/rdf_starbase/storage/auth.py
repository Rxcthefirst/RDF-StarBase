"""
Authentication and Authorization module for RDF-StarBase.

DEPRECATED: This module has been moved to api.auth.
This file provides backward compatibility - import from api.auth instead.
"""

import warnings

warnings.warn(
    "rdf_starbase.storage.auth is deprecated. Import from api.auth instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from api.auth import (
    Role,
    Operation,
    READ_OPERATIONS,
    WRITE_OPERATIONS,
    ADMIN_OPERATIONS,
    APIKey,
    ScopedToken,
    RateLimitState,
    RateLimitExceeded,
    AuthorizationError,
    APIKeyManager,
    AuthContext,
    PermissionPolicy,
    create_key_manager,
    require_auth,
)

__all__ = [
    "Role",
    "Operation",
    "READ_OPERATIONS",
    "WRITE_OPERATIONS",
    "ADMIN_OPERATIONS",
    "APIKey",
    "ScopedToken",
    "RateLimitState",
    "RateLimitExceeded",
    "AuthorizationError",
    "APIKeyManager",
    "AuthContext",
    "PermissionPolicy",
    "create_key_manager",
    "require_auth",
]
