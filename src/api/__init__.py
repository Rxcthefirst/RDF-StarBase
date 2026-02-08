"""
RDF-StarBase API Layer

FastAPI-based REST API for the RDF-Star knowledge graph engine.
Separates API concerns from the core engine (rdf_starbase).
"""

__version__ = "0.3.0"

from api.auth import (
    APIKey,
    APIKeyManager,
    AuthContext,
    AuthorizationError,
    Operation,
    PermissionPolicy,
    RateLimitExceeded,
    RateLimitState,
    Role,
    ScopedToken,
    create_key_manager,
    require_auth,
)

# OIDC exports (optional - requires PyJWT)
# Import directly: from api.oidc import OIDCManager, OIDCProviderConfig

# Note: api.web imports are NOT included here to avoid circular imports
# (api.web imports from rdf_starbase, which imports storage.auth shim)
# Import directly from api.web when needed: from api.web import create_app

__all__ = [
    # Auth
    "APIKey",
    "APIKeyManager",
    "AuthContext",
    "AuthorizationError",
    "Operation",
    "PermissionPolicy",
    "RateLimitExceeded",
    "RateLimitState",
    "Role",
    "ScopedToken",
    "create_key_manager",
    "require_auth",
]
