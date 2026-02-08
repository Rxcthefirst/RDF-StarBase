"""
OAuth2/OIDC Authentication Provider for RDF-StarBase API.

Supports:
- OIDC discovery (well-known configuration)
- JWT token validation with JWKS
- Multiple identity provider configurations
- Claims-to-role mapping
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

from api.auth import Role, Operation, AuthContext, AuthorizationError


# Optional dependencies - graceful degradation
try:
    import jwt
    from jwt import PyJWKClient
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    jwt = None  # type: ignore
    PyJWKClient = None  # type: ignore

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore


class OIDCError(Exception):
    """Base exception for OIDC errors."""
    pass


class TokenValidationError(OIDCError):
    """Raised when token validation fails."""
    pass


class ProviderConfigError(OIDCError):
    """Raised when provider configuration is invalid."""
    pass


@dataclass
class ClaimsMapping:
    """Maps JWT claims to RDF-StarBase roles and permissions."""
    
    # Claim name containing roles (e.g., "roles", "groups", "realm_access.roles")
    role_claim: str = "roles"
    
    # Map external role names to internal roles
    role_map: dict[str, Role] = field(default_factory=lambda: {
        "admin": Role.ADMIN,
        "writer": Role.WRITER,
        "reader": Role.READER,
        "rdf-starbase-admin": Role.ADMIN,
        "rdf-starbase-writer": Role.WRITER,
        "rdf-starbase-reader": Role.READER,
    })
    
    # Claim name for repository access (optional)
    repos_claim: str | None = "rdf_repos"
    
    # Claim name for user identifier
    subject_claim: str = "sub"
    
    # Claim name for username/email (for audit)
    username_claim: str = "preferred_username"
    
    # Default role if no role claim matches
    default_role: Role = Role.READER
    
    def get_nested_claim(self, claims: dict[str, Any], path: str) -> Any:
        """Get a nested claim value using dot notation."""
        parts = path.split(".")
        value = claims
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def extract_role(self, claims: dict[str, Any]) -> Role:
        """Extract the highest role from claims."""
        role_value = self.get_nested_claim(claims, self.role_claim)
        
        if role_value is None:
            return self.default_role
        
        # Handle both single value and list of roles
        roles = role_value if isinstance(role_value, list) else [role_value]
        
        # Find highest matching role
        best_role = self.default_role
        for role_name in roles:
            if role_name in self.role_map:
                mapped = self.role_map[role_name]
                # ADMIN > WRITER > READER
                if mapped == Role.ADMIN:
                    return Role.ADMIN
                elif mapped == Role.WRITER and best_role != Role.ADMIN:
                    best_role = Role.WRITER
        
        return best_role
    
    def extract_repos(self, claims: dict[str, Any]) -> set[str] | None:
        """Extract allowed repositories from claims."""
        if not self.repos_claim:
            return None
        
        repos = self.get_nested_claim(claims, self.repos_claim)
        if repos is None:
            return None
        
        if isinstance(repos, list):
            return set(repos)
        elif isinstance(repos, str):
            return {repos}
        return None
    
    def extract_subject(self, claims: dict[str, Any]) -> str | None:
        """Extract subject identifier."""
        return self.get_nested_claim(claims, self.subject_claim)
    
    def extract_username(self, claims: dict[str, Any]) -> str | None:
        """Extract username for display/audit."""
        return self.get_nested_claim(claims, self.username_claim)


@dataclass
class OIDCProviderConfig:
    """Configuration for an OIDC identity provider."""
    
    # Provider identifier
    provider_id: str
    
    # OIDC issuer URL (e.g., https://auth.example.com/realms/myapp)
    issuer: str
    
    # Expected audience (client_id)
    audience: str
    
    # Claims mapping configuration
    claims_mapping: ClaimsMapping = field(default_factory=ClaimsMapping)
    
    # Optional: JWKS URI override (normally discovered)
    jwks_uri: str | None = None
    
    # Token validation options
    verify_exp: bool = True
    verify_aud: bool = True
    verify_iss: bool = True
    
    # Leeway for time-based validation (seconds)
    leeway: int = 30
    
    # Cache JWKS for this duration (seconds)
    jwks_cache_seconds: int = 3600
    
    # Provider enabled status
    enabled: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "provider_id": self.provider_id,
            "issuer": self.issuer,
            "audience": self.audience,
            "claims_mapping": {
                "role_claim": self.claims_mapping.role_claim,
                "role_map": {k: v.value for k, v in self.claims_mapping.role_map.items()},
                "repos_claim": self.claims_mapping.repos_claim,
                "subject_claim": self.claims_mapping.subject_claim,
                "username_claim": self.claims_mapping.username_claim,
                "default_role": self.claims_mapping.default_role.value,
            },
            "jwks_uri": self.jwks_uri,
            "verify_exp": self.verify_exp,
            "verify_aud": self.verify_aud,
            "verify_iss": self.verify_iss,
            "leeway": self.leeway,
            "jwks_cache_seconds": self.jwks_cache_seconds,
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OIDCProviderConfig":
        """Create from dictionary."""
        cm_data = data.get("claims_mapping", {})
        claims_mapping = ClaimsMapping(
            role_claim=cm_data.get("role_claim", "roles"),
            role_map={k: Role(v) for k, v in cm_data.get("role_map", {}).items()},
            repos_claim=cm_data.get("repos_claim"),
            subject_claim=cm_data.get("subject_claim", "sub"),
            username_claim=cm_data.get("username_claim", "preferred_username"),
            default_role=Role(cm_data.get("default_role", "reader")),
        )
        
        return cls(
            provider_id=data["provider_id"],
            issuer=data["issuer"],
            audience=data["audience"],
            claims_mapping=claims_mapping,
            jwks_uri=data.get("jwks_uri"),
            verify_exp=data.get("verify_exp", True),
            verify_aud=data.get("verify_aud", True),
            verify_iss=data.get("verify_iss", True),
            leeway=data.get("leeway", 30),
            jwks_cache_seconds=data.get("jwks_cache_seconds", 3600),
            enabled=data.get("enabled", True),
        )


@dataclass
class ValidatedToken:
    """Result of successful token validation."""
    
    provider_id: str
    subject: str
    username: str | None
    role: Role
    allowed_repos: set[str] | None
    claims: dict[str, Any]
    expires_at: datetime | None
    
    def to_auth_context(self) -> "OIDCAuthContext":
        """Convert to an AuthContext for request handling."""
        return OIDCAuthContext(
            validated_token=self,
        )


class OIDCAuthContext(AuthContext):
    """Auth context for OIDC-authenticated requests."""
    
    def __init__(self, validated_token: ValidatedToken):
        """Initialize OIDC auth context."""
        super().__init__(key=None, token=None, manager=None)
        self.validated_token = validated_token
    
    @property
    def is_authenticated(self) -> bool:
        return True
    
    @property
    def key_id(self) -> str | None:
        return f"oidc:{self.validated_token.provider_id}:{self.validated_token.subject}"
    
    @property
    def role(self) -> Role | None:
        return self.validated_token.role
    
    @property
    def username(self) -> str | None:
        return self.validated_token.username
    
    def can_perform(self, operation: Operation, repo: str | None = None) -> bool:
        """Check if operation is allowed based on OIDC claims."""
        role = self.validated_token.role
        
        # Check role permissions
        from api.auth import READ_OPERATIONS, WRITE_OPERATIONS, ADMIN_OPERATIONS
        
        if operation in READ_OPERATIONS:
            allowed = role.can_read()
        elif operation in WRITE_OPERATIONS:
            allowed = role.can_write()
        elif operation in ADMIN_OPERATIONS:
            allowed = role.can_admin()
        else:
            allowed = False
        
        if not allowed:
            return False
        
        # Check repository restrictions
        if repo and self.validated_token.allowed_repos is not None:
            if repo not in self.validated_token.allowed_repos:
                return False
        
        return True
    
    def require(self, operation: Operation, repo: str | None = None) -> None:
        """Require permission for operation."""
        if not self.can_perform(operation, repo):
            raise AuthorizationError(
                f"Permission denied for {operation.value}" + (f" on {repo}" if repo else ""),
                operation,
                repo,
            )


class OIDCManager:
    """Manages OIDC provider configurations and token validation."""
    
    def __init__(self, storage_path: Path | None = None):
        """Initialize OIDC manager.
        
        Args:
            storage_path: Path to persist provider configurations
        """
        if not HAS_JWT:
            raise ImportError(
                "PyJWT is required for OIDC support. "
                "Install with: pip install 'rdf-starbase[auth]'"
            )
        
        self.storage_path = storage_path
        self._providers: dict[str, OIDCProviderConfig] = {}
        self._jwks_clients: dict[str, Any] = {}  # provider_id -> PyJWKClient
        self._discovery_cache: dict[str, tuple[dict, float]] = {}  # issuer -> (config, timestamp)
        
        if storage_path and storage_path.exists():
            self._load()
    
    def add_provider(self, config: OIDCProviderConfig) -> None:
        """Add or update an OIDC provider configuration."""
        self._providers[config.provider_id] = config
        # Clear cached JWKS client
        self._jwks_clients.pop(config.provider_id, None)
        self._save()
    
    def remove_provider(self, provider_id: str) -> bool:
        """Remove a provider configuration."""
        if provider_id not in self._providers:
            return False
        del self._providers[provider_id]
        self._jwks_clients.pop(provider_id, None)
        self._save()
        return True
    
    def get_provider(self, provider_id: str) -> OIDCProviderConfig | None:
        """Get a provider configuration."""
        return self._providers.get(provider_id)
    
    def list_providers(self) -> list[OIDCProviderConfig]:
        """List all provider configurations."""
        return list(self._providers.values())
    
    def discover_oidc_config(self, issuer: str) -> dict[str, Any]:
        """Fetch OIDC discovery document from well-known endpoint."""
        if not HAS_HTTPX:
            raise ProviderConfigError(
                "httpx is required for OIDC discovery. "
                "Install with: pip install httpx"
            )
        
        # Check cache
        if issuer in self._discovery_cache:
            config, timestamp = self._discovery_cache[issuer]
            if time.time() - timestamp < 3600:  # 1 hour cache
                return config
        
        well_known_url = urljoin(issuer.rstrip("/") + "/", ".well-known/openid-configuration")
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(well_known_url)
                response.raise_for_status()
                config = response.json()
        except Exception as e:
            raise ProviderConfigError(f"Failed to fetch OIDC config from {well_known_url}: {e}")
        
        self._discovery_cache[issuer] = (config, time.time())
        return config
    
    def _get_jwks_client(self, provider: OIDCProviderConfig) -> Any:
        """Get or create JWKS client for provider."""
        if provider.provider_id in self._jwks_clients:
            return self._jwks_clients[provider.provider_id]
        
        jwks_uri = provider.jwks_uri
        if not jwks_uri:
            # Discover from OIDC config
            oidc_config = self.discover_oidc_config(provider.issuer)
            jwks_uri = oidc_config.get("jwks_uri")
            if not jwks_uri:
                raise ProviderConfigError(f"No JWKS URI found for issuer {provider.issuer}")
        
        client = PyJWKClient(
            jwks_uri,
            cache_keys=True,
            lifespan=provider.jwks_cache_seconds,
        )
        self._jwks_clients[provider.provider_id] = client
        return client
    
    def validate_token(self, token: str) -> ValidatedToken:
        """Validate a JWT token against configured providers.
        
        Tries each enabled provider in order until one succeeds.
        """
        errors = []
        
        for provider in self._providers.values():
            if not provider.enabled:
                continue
            
            try:
                return self._validate_with_provider(token, provider)
            except TokenValidationError as e:
                errors.append(f"{provider.provider_id}: {e}")
                continue
            except Exception as e:
                errors.append(f"{provider.provider_id}: unexpected error: {e}")
                continue
        
        if not self._providers:
            raise TokenValidationError("No OIDC providers configured")
        
        raise TokenValidationError(
            f"Token validation failed for all providers: {'; '.join(errors)}"
        )
    
    def _validate_with_provider(
        self, 
        token: str, 
        provider: OIDCProviderConfig
    ) -> ValidatedToken:
        """Validate token with a specific provider."""
        try:
            jwks_client = self._get_jwks_client(provider)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            # Build validation options
            options = {
                "verify_exp": provider.verify_exp,
                "verify_aud": provider.verify_aud,
                "verify_iss": provider.verify_iss,
            }
            
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=provider.audience if provider.verify_aud else None,
                issuer=provider.issuer if provider.verify_iss else None,
                leeway=provider.leeway,
                options=options,
            )
        except jwt.ExpiredSignatureError:
            raise TokenValidationError("Token has expired")
        except jwt.InvalidAudienceError:
            raise TokenValidationError("Invalid audience")
        except jwt.InvalidIssuerError:
            raise TokenValidationError("Invalid issuer")
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {e}")
        except Exception as e:
            raise TokenValidationError(f"Token validation failed: {e}")
        
        # Extract role and permissions from claims
        mapping = provider.claims_mapping
        
        role = mapping.extract_role(claims)
        repos = mapping.extract_repos(claims)
        subject = mapping.extract_subject(claims) or "unknown"
        username = mapping.extract_username(claims)
        
        # Get expiration
        exp = claims.get("exp")
        expires_at = datetime.fromtimestamp(exp) if exp else None
        
        return ValidatedToken(
            provider_id=provider.provider_id,
            subject=subject,
            username=username,
            role=role,
            allowed_repos=repos,
            claims=claims,
            expires_at=expires_at,
        )
    
    def validate_token_for_provider(
        self, 
        token: str, 
        provider_id: str
    ) -> ValidatedToken:
        """Validate token with a specific provider by ID."""
        provider = self._providers.get(provider_id)
        if not provider:
            raise TokenValidationError(f"Unknown provider: {provider_id}")
        if not provider.enabled:
            raise TokenValidationError(f"Provider {provider_id} is disabled")
        
        return self._validate_with_provider(token, provider)
    
    def _save(self) -> None:
        """Persist provider configurations."""
        if self.storage_path is None:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "providers": {pid: p.to_dict() for pid, p in self._providers.items()},
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load provider configurations from storage."""
        if self.storage_path is None or not self.storage_path.exists():
            return
        
        data = json.loads(self.storage_path.read_text())
        self._providers = {
            pid: OIDCProviderConfig.from_dict(pd) 
            for pid, pd in data.get("providers", {}).items()
        }


def create_oidc_manager(storage_path: Path | None = None) -> OIDCManager:
    """Create a new OIDC manager."""
    return OIDCManager(storage_path)


# Pre-configured provider templates

def keycloak_provider(
    provider_id: str,
    issuer: str,
    audience: str,
    realm_roles: bool = True,
) -> OIDCProviderConfig:
    """Create a Keycloak OIDC provider configuration.
    
    Args:
        provider_id: Unique identifier for this provider
        issuer: Keycloak realm URL (e.g., https://auth.example.com/realms/myapp)
        audience: Client ID
        realm_roles: If True, use realm_access.roles; if False, use resource_access
    """
    role_claim = "realm_access.roles" if realm_roles else f"resource_access.{audience}.roles"
    
    return OIDCProviderConfig(
        provider_id=provider_id,
        issuer=issuer,
        audience=audience,
        claims_mapping=ClaimsMapping(
            role_claim=role_claim,
            role_map={
                "admin": Role.ADMIN,
                "rdf-admin": Role.ADMIN,
                "writer": Role.WRITER,
                "rdf-writer": Role.WRITER,
                "reader": Role.READER,
                "rdf-reader": Role.READER,
            },
            username_claim="preferred_username",
        ),
    )


def azure_ad_provider(
    provider_id: str,
    tenant_id: str,
    client_id: str,
) -> OIDCProviderConfig:
    """Create an Azure AD OIDC provider configuration.
    
    Args:
        provider_id: Unique identifier for this provider
        tenant_id: Azure AD tenant ID
        client_id: Application (client) ID
    """
    return OIDCProviderConfig(
        provider_id=provider_id,
        issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
        audience=client_id,
        claims_mapping=ClaimsMapping(
            role_claim="roles",  # App roles in Azure AD
            role_map={
                "Admin": Role.ADMIN,
                "RDF.Admin": Role.ADMIN,
                "Writer": Role.WRITER,
                "RDF.Writer": Role.WRITER,
                "Reader": Role.READER,
                "RDF.Reader": Role.READER,
            },
            username_claim="preferred_username",
            subject_claim="oid",  # Object ID is more stable than sub
        ),
    )


def okta_provider(
    provider_id: str,
    domain: str,
    client_id: str,
) -> OIDCProviderConfig:
    """Create an Okta OIDC provider configuration.
    
    Args:
        provider_id: Unique identifier for this provider
        domain: Okta domain (e.g., dev-123456.okta.com)
        client_id: Client ID
    """
    return OIDCProviderConfig(
        provider_id=provider_id,
        issuer=f"https://{domain}",
        audience=client_id,
        claims_mapping=ClaimsMapping(
            role_claim="groups",  # Okta groups as roles
            role_map={
                "RDF-Admins": Role.ADMIN,
                "RDF-Writers": Role.WRITER,
                "RDF-Readers": Role.READER,
                "Everyone": Role.READER,
            },
            username_claim="email",
        ),
    )


def auth0_provider(
    provider_id: str,
    domain: str,
    audience: str,
) -> OIDCProviderConfig:
    """Create an Auth0 OIDC provider configuration.
    
    Args:
        provider_id: Unique identifier for this provider
        domain: Auth0 domain (e.g., myapp.auth0.com)
        audience: API identifier
    """
    return OIDCProviderConfig(
        provider_id=provider_id,
        issuer=f"https://{domain}/",
        audience=audience,
        claims_mapping=ClaimsMapping(
            role_claim="https://rdf-starbase.io/roles",  # Custom claim namespace
            role_map={
                "admin": Role.ADMIN,
                "writer": Role.WRITER,
                "reader": Role.READER,
            },
            username_claim="email",
        ),
    )
