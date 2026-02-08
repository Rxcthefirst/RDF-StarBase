"""
Tests for OAuth2/OIDC authentication.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from api.auth import Role, Operation
from api.oidc import (
    ClaimsMapping,
    OIDCProviderConfig,
    ValidatedToken,
    OIDCAuthContext,
    OIDCManager,
    OIDCError,
    TokenValidationError,
    ProviderConfigError,
    keycloak_provider,
    azure_ad_provider,
    okta_provider,
    auth0_provider,
    HAS_JWT,
)


class TestClaimsMapping:
    """Tests for claims mapping functionality."""
    
    def test_default_mapping(self):
        """Test default claims mapping configuration."""
        mapping = ClaimsMapping()
        assert mapping.role_claim == "roles"
        assert mapping.default_role == Role.READER
    
    def test_get_nested_claim_simple(self):
        """Test getting a simple claim."""
        mapping = ClaimsMapping()
        claims = {"sub": "user123", "roles": ["admin"]}
        assert mapping.get_nested_claim(claims, "sub") == "user123"
    
    def test_get_nested_claim_nested(self):
        """Test getting a nested claim with dot notation."""
        mapping = ClaimsMapping()
        claims = {
            "realm_access": {
                "roles": ["admin", "user"]
            }
        }
        assert mapping.get_nested_claim(claims, "realm_access.roles") == ["admin", "user"]
    
    def test_get_nested_claim_missing(self):
        """Test getting a missing claim returns None."""
        mapping = ClaimsMapping()
        claims = {"sub": "user123"}
        assert mapping.get_nested_claim(claims, "missing") is None
        assert mapping.get_nested_claim(claims, "deep.missing.path") is None
    
    def test_extract_role_admin(self):
        """Test extracting admin role."""
        mapping = ClaimsMapping(role_map={"admin": Role.ADMIN})
        claims = {"roles": ["admin"]}
        assert mapping.extract_role(claims) == Role.ADMIN
    
    def test_extract_role_highest_wins(self):
        """Test that highest role is selected when multiple match."""
        mapping = ClaimsMapping(role_map={
            "admin": Role.ADMIN,
            "writer": Role.WRITER,
            "reader": Role.READER,
        })
        claims = {"roles": ["reader", "writer", "admin"]}
        assert mapping.extract_role(claims) == Role.ADMIN
    
    def test_extract_role_writer(self):
        """Test writer role extraction."""
        mapping = ClaimsMapping(role_map={
            "writer": Role.WRITER,
            "reader": Role.READER,
        })
        claims = {"roles": ["writer", "reader"]}
        assert mapping.extract_role(claims) == Role.WRITER
    
    def test_extract_role_default(self):
        """Test default role when no match."""
        mapping = ClaimsMapping(default_role=Role.READER)
        claims = {"roles": ["unknown-role"]}
        assert mapping.extract_role(claims) == Role.READER
    
    def test_extract_role_missing_claim(self):
        """Test default role when claim missing."""
        mapping = ClaimsMapping(default_role=Role.READER)
        claims = {}
        assert mapping.extract_role(claims) == Role.READER
    
    def test_extract_role_single_value(self):
        """Test role extraction with single string value (not list)."""
        mapping = ClaimsMapping(role_map={"admin": Role.ADMIN})
        claims = {"roles": "admin"}
        assert mapping.extract_role(claims) == Role.ADMIN
    
    def test_extract_repos(self):
        """Test repository extraction."""
        mapping = ClaimsMapping(repos_claim="rdf_repos")
        claims = {"rdf_repos": ["repo1", "repo2"]}
        assert mapping.extract_repos(claims) == {"repo1", "repo2"}
    
    def test_extract_repos_single(self):
        """Test repository extraction with single value."""
        mapping = ClaimsMapping(repos_claim="rdf_repos")
        claims = {"rdf_repos": "single-repo"}
        assert mapping.extract_repos(claims) == {"single-repo"}
    
    def test_extract_repos_none_configured(self):
        """Test repos extraction when not configured."""
        mapping = ClaimsMapping(repos_claim=None)
        claims = {"rdf_repos": ["repo1"]}
        assert mapping.extract_repos(claims) is None
    
    def test_extract_subject(self):
        """Test subject extraction."""
        mapping = ClaimsMapping()
        claims = {"sub": "user-uuid-123"}
        assert mapping.extract_subject(claims) == "user-uuid-123"
    
    def test_extract_username(self):
        """Test username extraction."""
        mapping = ClaimsMapping(username_claim="email")
        claims = {"email": "user@example.com"}
        assert mapping.extract_username(claims) == "user@example.com"


class TestOIDCProviderConfig:
    """Tests for OIDC provider configuration."""
    
    def test_create_provider(self):
        """Test basic provider creation."""
        config = OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
        )
        assert config.provider_id == "test"
        assert config.issuer == "https://auth.example.com"
        assert config.audience == "my-app"
        assert config.enabled is True
    
    def test_to_dict(self):
        """Test serialization to dict."""
        config = OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
        )
        data = config.to_dict()
        assert data["provider_id"] == "test"
        assert data["issuer"] == "https://auth.example.com"
        assert "claims_mapping" in data
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "provider_id": "test",
            "issuer": "https://auth.example.com",
            "audience": "my-app",
            "claims_mapping": {
                "role_claim": "groups",
                "role_map": {"admin": "admin"},
                "default_role": "reader",
            },
        }
        config = OIDCProviderConfig.from_dict(data)
        assert config.provider_id == "test"
        assert config.claims_mapping.role_claim == "groups"
    
    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
            claims_mapping=ClaimsMapping(
                role_claim="custom_roles",
                role_map={"super-admin": Role.ADMIN},
            ),
        )
        data = original.to_dict()
        restored = OIDCProviderConfig.from_dict(data)
        assert restored.provider_id == original.provider_id
        assert restored.issuer == original.issuer
        assert restored.claims_mapping.role_claim == original.claims_mapping.role_claim


class TestValidatedToken:
    """Tests for ValidatedToken."""
    
    def test_create_token(self):
        """Test creating a validated token."""
        token = ValidatedToken(
            provider_id="keycloak",
            subject="user123",
            username="alice",
            role=Role.WRITER,
            allowed_repos={"repo1", "repo2"},
            claims={"sub": "user123"},
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert token.subject == "user123"
        assert token.role == Role.WRITER
    
    def test_to_auth_context(self):
        """Test converting to auth context."""
        token = ValidatedToken(
            provider_id="keycloak",
            subject="user123",
            username="alice",
            role=Role.ADMIN,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = token.to_auth_context()
        assert isinstance(ctx, OIDCAuthContext)
        assert ctx.is_authenticated
        assert ctx.role == Role.ADMIN


class TestOIDCAuthContext:
    """Tests for OIDC auth context."""
    
    def test_is_authenticated(self):
        """Test authentication status."""
        token = ValidatedToken(
            provider_id="keycloak",
            subject="user123",
            username=None,
            role=Role.READER,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.is_authenticated is True
    
    def test_key_id_format(self):
        """Test key ID format for OIDC."""
        token = ValidatedToken(
            provider_id="keycloak",
            subject="user123",
            username=None,
            role=Role.READER,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.key_id == "oidc:keycloak:user123"
    
    def test_can_perform_read(self):
        """Test read permission check."""
        token = ValidatedToken(
            provider_id="test",
            subject="user",
            username=None,
            role=Role.READER,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.can_perform(Operation.QUERY) is True
        assert ctx.can_perform(Operation.INSERT) is False
    
    def test_can_perform_write(self):
        """Test write permission check."""
        token = ValidatedToken(
            provider_id="test",
            subject="user",
            username=None,
            role=Role.WRITER,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.can_perform(Operation.QUERY) is True
        assert ctx.can_perform(Operation.INSERT) is True
        assert ctx.can_perform(Operation.CREATE_REPO) is False
    
    def test_can_perform_admin(self):
        """Test admin permission check."""
        token = ValidatedToken(
            provider_id="test",
            subject="user",
            username=None,
            role=Role.ADMIN,
            allowed_repos=None,
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.can_perform(Operation.QUERY) is True
        assert ctx.can_perform(Operation.INSERT) is True
        assert ctx.can_perform(Operation.CREATE_REPO) is True
    
    def test_can_perform_with_repo_restriction(self):
        """Test permission with repository restrictions."""
        token = ValidatedToken(
            provider_id="test",
            subject="user",
            username=None,
            role=Role.WRITER,
            allowed_repos={"allowed-repo"},
            claims={},
            expires_at=None,
        )
        ctx = OIDCAuthContext(token)
        assert ctx.can_perform(Operation.INSERT, "allowed-repo") is True
        assert ctx.can_perform(Operation.INSERT, "other-repo") is False


class TestProviderTemplates:
    """Tests for pre-configured provider templates."""
    
    def test_keycloak_provider(self):
        """Test Keycloak provider template."""
        config = keycloak_provider(
            provider_id="my-keycloak",
            issuer="https://auth.example.com/realms/myapp",
            audience="my-client",
        )
        assert config.provider_id == "my-keycloak"
        assert "realm_access.roles" in config.claims_mapping.role_claim
    
    def test_keycloak_provider_client_roles(self):
        """Test Keycloak provider with client roles."""
        config = keycloak_provider(
            provider_id="my-keycloak",
            issuer="https://auth.example.com/realms/myapp",
            audience="my-client",
            realm_roles=False,
        )
        assert "resource_access.my-client.roles" in config.claims_mapping.role_claim
    
    def test_azure_ad_provider(self):
        """Test Azure AD provider template."""
        config = azure_ad_provider(
            provider_id="azure",
            tenant_id="tenant-uuid",
            client_id="client-uuid",
        )
        assert config.provider_id == "azure"
        assert "tenant-uuid" in config.issuer
        assert config.claims_mapping.subject_claim == "oid"
    
    def test_okta_provider(self):
        """Test Okta provider template."""
        config = okta_provider(
            provider_id="okta",
            domain="dev-123456.okta.com",
            client_id="my-client",
        )
        assert config.provider_id == "okta"
        assert "dev-123456.okta.com" in config.issuer
        assert config.claims_mapping.role_claim == "groups"
    
    def test_auth0_provider(self):
        """Test Auth0 provider template."""
        config = auth0_provider(
            provider_id="auth0",
            domain="myapp.auth0.com",
            audience="https://api.example.com",
        )
        assert config.provider_id == "auth0"
        assert "myapp.auth0.com" in config.issuer


@pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
class TestOIDCManager:
    """Tests for OIDC manager (requires PyJWT)."""
    
    def test_add_provider(self, tmp_path):
        """Test adding a provider."""
        manager = OIDCManager(storage_path=tmp_path / "oidc.json")
        config = OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
        )
        manager.add_provider(config)
        assert manager.get_provider("test") is not None
    
    def test_remove_provider(self, tmp_path):
        """Test removing a provider."""
        manager = OIDCManager(storage_path=tmp_path / "oidc.json")
        config = OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
        )
        manager.add_provider(config)
        assert manager.remove_provider("test") is True
        assert manager.get_provider("test") is None
    
    def test_list_providers(self, tmp_path):
        """Test listing providers."""
        manager = OIDCManager(storage_path=tmp_path / "oidc.json")
        manager.add_provider(OIDCProviderConfig(
            provider_id="p1",
            issuer="https://auth1.example.com",
            audience="app1",
        ))
        manager.add_provider(OIDCProviderConfig(
            provider_id="p2",
            issuer="https://auth2.example.com",
            audience="app2",
        ))
        providers = manager.list_providers()
        assert len(providers) == 2
    
    def test_persistence(self, tmp_path):
        """Test provider persistence."""
        storage = tmp_path / "oidc.json"
        
        # Create and add provider
        manager1 = OIDCManager(storage_path=storage)
        manager1.add_provider(OIDCProviderConfig(
            provider_id="test",
            issuer="https://auth.example.com",
            audience="my-app",
        ))
        
        # Create new manager and verify provider loaded
        manager2 = OIDCManager(storage_path=storage)
        provider = manager2.get_provider("test")
        assert provider is not None
        assert provider.issuer == "https://auth.example.com"
    
    def test_validate_token_no_providers(self, tmp_path):
        """Test validation fails with no providers."""
        manager = OIDCManager(storage_path=tmp_path / "oidc.json")
        with pytest.raises(TokenValidationError, match="No OIDC providers"):
            manager.validate_token("fake.jwt.token")
    
    def test_validate_token_unknown_provider(self, tmp_path):
        """Test validation fails for unknown provider."""
        manager = OIDCManager(storage_path=tmp_path / "oidc.json")
        with pytest.raises(TokenValidationError, match="Unknown provider"):
            manager.validate_token_for_provider("fake.jwt.token", "nonexistent")
