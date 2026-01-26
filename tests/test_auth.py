"""Tests for the auth module - API keys, roles, tokens, rate limiting."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from rdf_starbase.storage.auth import (
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


class TestRole:
    """Tests for Role enum."""
    
    def test_roles(self):
        assert Role.READER.value == "reader"
        assert Role.WRITER.value == "writer"
        assert Role.ADMIN.value == "admin"
    
    def test_can_read(self):
        assert Role.READER.can_read()
        assert Role.WRITER.can_read()
        assert Role.ADMIN.can_read()
    
    def test_can_write(self):
        assert not Role.READER.can_write()
        assert Role.WRITER.can_write()
        assert Role.ADMIN.can_write()
    
    def test_can_admin(self):
        assert not Role.READER.can_admin()
        assert not Role.WRITER.can_admin()
        assert Role.ADMIN.can_admin()


class TestOperation:
    """Tests for Operation enum."""
    
    def test_operation_categories(self):
        assert Operation.QUERY in READ_OPERATIONS
        assert Operation.INSERT in WRITE_OPERATIONS
        assert Operation.BACKUP in ADMIN_OPERATIONS
    
    def test_all_operations_categorized(self):
        all_ops = READ_OPERATIONS | WRITE_OPERATIONS | ADMIN_OPERATIONS
        assert len(all_ops) == len(Operation)


class TestAPIKey:
    """Tests for APIKey dataclass."""
    
    def test_creation(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        assert key.key_id == "test1234"
        assert key.role == Role.READER
        assert key.enabled
    
    def test_is_valid(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        assert key.is_valid()
        
        key.enabled = False
        assert not key.is_valid()
    
    def test_expired_key(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert not key.is_valid()
    
    def test_can_access_repo(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
            allowed_repos={"repo1", "repo2"},
        )
        assert key.can_access_repo("repo1")
        assert not key.can_access_repo("repo3")
    
    def test_can_access_all_repos(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        assert key.can_access_repo("any_repo")
    
    def test_can_perform_reader(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        assert key.can_perform(Operation.QUERY)
        assert not key.can_perform(Operation.INSERT)
        assert not key.can_perform(Operation.BACKUP)
    
    def test_can_perform_writer(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.WRITER,
        )
        assert key.can_perform(Operation.QUERY)
        assert key.can_perform(Operation.INSERT)
        assert not key.can_perform(Operation.BACKUP)
    
    def test_can_perform_admin(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.ADMIN,
        )
        assert key.can_perform(Operation.QUERY)
        assert key.can_perform(Operation.INSERT)
        assert key.can_perform(Operation.BACKUP)
    
    def test_explicit_operations(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.ADMIN,
            allowed_operations={Operation.QUERY},
        )
        assert key.can_perform(Operation.QUERY)
        assert not key.can_perform(Operation.INSERT)
    
    def test_to_dict(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        d = key.to_dict()
        assert d["key_id"] == "test1234"
        assert d["role"] == "reader"
        assert "key_hash" not in d  # Should not expose hash
    
    def test_from_dict(self):
        data = {
            "key_id": "test1234",
            "key_hash": "abc123",
            "name": "Test Key",
            "role": "writer",
            "created_at": datetime.now().isoformat(),
        }
        key = APIKey.from_dict(data)
        assert key.key_id == "test1234"
        assert key.role == Role.WRITER


class TestScopedToken:
    """Tests for ScopedToken dataclass."""
    
    def test_creation(self):
        token = ScopedToken(
            token_id="tok12345",
            token_hash="hash123",
            source_key_id="key123",
            operations={Operation.QUERY},
            repos=None,
        )
        assert token.token_id == "tok12345"
        assert token.is_valid()
    
    def test_expired_token(self):
        token = ScopedToken(
            token_id="tok12345",
            token_hash="hash123",
            source_key_id="key123",
            operations={Operation.QUERY},
            repos=None,
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert not token.is_valid()
    
    def test_max_uses(self):
        token = ScopedToken(
            token_id="tok12345",
            token_hash="hash123",
            source_key_id="key123",
            operations={Operation.QUERY},
            repos=None,
            max_uses=2,
        )
        assert token.is_valid()
        token.use()
        assert token.is_valid()
        token.use()
        assert not token.is_valid()
    
    def test_can_perform(self):
        token = ScopedToken(
            token_id="tok12345",
            token_hash="hash123",
            source_key_id="key123",
            operations={Operation.QUERY, Operation.DESCRIBE},
            repos=None,
        )
        assert token.can_perform(Operation.QUERY)
        assert not token.can_perform(Operation.INSERT)


class TestRateLimitState:
    """Tests for RateLimitState."""
    
    def test_check_query_limit(self):
        state = RateLimitState(key_id="key1")
        assert state.check_query_limit(10)
        
        for _ in range(9):
            state.record_query()
        
        assert state.check_query_limit(10)
        state.record_query()
        assert not state.check_query_limit(10)
    
    def test_check_ingestion_limit(self):
        state = RateLimitState(key_id="key1")
        assert state.check_ingestion_limit(100, 1000)
        state.record_ingestion(900)
        assert state.check_ingestion_limit(100, 1000)
        assert not state.check_ingestion_limit(200, 1000)
    
    def test_no_limit(self):
        state = RateLimitState(key_id="key1")
        assert state.check_query_limit(None)
        assert state.check_ingestion_limit(1000000, None)


class TestAPIKeyManager:
    """Tests for APIKeyManager."""
    
    def test_generate_key(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            role=Role.WRITER,
        )
        
        assert len(raw_key) > 20
        assert api_key.name == "Test Key"
        assert api_key.role == Role.WRITER
    
    def test_validate_key(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            role=Role.READER,
        )
        
        validated = manager.validate_key(raw_key)
        assert validated is not None
        assert validated.key_id == api_key.key_id
    
    def test_validate_invalid_key(self):
        manager = APIKeyManager()
        assert manager.validate_key("invalid_key") is None
    
    def test_revoke_key(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test Key",
            role=Role.READER,
        )
        
        assert manager.validate_key(raw_key) is not None
        
        manager.revoke_key(api_key.key_id)
        
        assert manager.validate_key(raw_key) is None
    
    def test_delete_key(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(name="Test", role=Role.READER)
        
        assert manager.delete_key(api_key.key_id)
        assert manager.get_key(api_key.key_id) is None
    
    def test_list_keys(self):
        manager = APIKeyManager()
        manager.generate_key(name="Key 1", role=Role.READER)
        manager.generate_key(name="Key 2", role=Role.WRITER)
        
        keys = manager.list_keys()
        assert len(keys) == 2
    
    def test_update_key(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(name="Test", role=Role.READER)
        
        updated = manager.update_key(api_key.key_id, role=Role.ADMIN)
        assert updated.role == Role.ADMIN
    
    def test_create_scoped_token(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test",
            role=Role.ADMIN,
        )
        
        raw_token, token = manager.create_scoped_token(
            api_key,
            operations={Operation.QUERY},
            expires_in=timedelta(minutes=30),
        )
        
        assert token.source_key_id == api_key.key_id
        assert Operation.QUERY in token.operations
    
    def test_validate_scoped_token(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(name="Test", role=Role.ADMIN)
        raw_token, token = manager.create_scoped_token(
            api_key,
            operations={Operation.QUERY},
        )
        
        validated = manager.validate_token(raw_token)
        assert validated is not None
        assert validated.token_id == token.token_id
    
    def test_rate_limit_check(self):
        manager = APIKeyManager()
        
        # Use up 9 queries
        for _ in range(9):
            manager.check_rate_limit("key1", limit_queries=10)
        
        # 10th should still pass
        manager.check_rate_limit("key1", limit_queries=10)
        
        # 11th should fail
        with pytest.raises(RateLimitExceeded):
            manager.check_rate_limit("key1", limit_queries=10)
    
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "keys.json"
            
            manager1 = APIKeyManager(storage_path=path)
            raw_key, api_key = manager1.generate_key(
                name="Persistent Key",
                role=Role.WRITER,
            )
            
            # Load in new manager
            manager2 = APIKeyManager(storage_path=path)
            loaded = manager2.get_key(api_key.key_id)
            
            assert loaded is not None
            assert loaded.name == "Persistent Key"
    
    def test_cleanup_expired_tokens(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(name="Test", role=Role.ADMIN)
        
        # Create expired token
        raw_token, token = manager.create_scoped_token(
            api_key,
            operations={Operation.QUERY},
            expires_in=timedelta(seconds=-1),  # Already expired
        )
        
        count = manager.cleanup_expired_tokens()
        assert count == 1


class TestAuthContext:
    """Tests for AuthContext."""
    
    def test_with_key(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.WRITER,
        )
        ctx = AuthContext(key=key)
        
        assert ctx.is_authenticated
        assert ctx.role == Role.WRITER
        assert ctx.can_perform(Operation.INSERT)
    
    def test_with_token(self):
        token = ScopedToken(
            token_id="tok12345",
            token_hash="hash123",
            source_key_id="key123",
            operations={Operation.QUERY},
            repos=None,
        )
        ctx = AuthContext(token=token)
        
        assert ctx.is_authenticated
        assert ctx.can_perform(Operation.QUERY)
        assert not ctx.can_perform(Operation.INSERT)
    
    def test_require_success(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.WRITER,
        )
        ctx = AuthContext(key=key)
        
        ctx.require(Operation.INSERT)  # Should not raise
    
    def test_require_failure(self):
        key = APIKey(
            key_id="test1234",
            key_hash="abc123",
            name="Test Key",
            role=Role.READER,
        )
        ctx = AuthContext(key=key)
        
        with pytest.raises(AuthorizationError):
            ctx.require(Operation.INSERT)
    
    def test_require_not_authenticated(self):
        ctx = AuthContext()
        
        with pytest.raises(AuthorizationError):
            ctx.require(Operation.QUERY)


class TestPermissionPolicy:
    """Tests for PermissionPolicy."""
    
    def test_set_repo_policy(self):
        policy = PermissionPolicy()
        policy.set_repo_policy("repo1", "key1", Role.WRITER)
        
        assert policy.get_repo_role("repo1", "key1") == Role.WRITER
    
    def test_list_repo_policies(self):
        policy = PermissionPolicy()
        policy.set_repo_policy("repo1", "key1", Role.READER)
        policy.set_repo_policy("repo1", "key2", Role.WRITER)
        
        policies = policy.list_repo_policies("repo1")
        assert len(policies) == 2
    
    def test_remove_repo_policy(self):
        policy = PermissionPolicy()
        policy.set_repo_policy("repo1", "key1", Role.WRITER)
        
        assert policy.remove_repo_policy("repo1", "key1")
        assert policy.get_repo_role("repo1", "key1") is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_key_manager(self):
        manager = create_key_manager()
        assert isinstance(manager, APIKeyManager)
    
    def test_require_auth_success(self):
        manager = APIKeyManager()
        raw_key, api_key = manager.generate_key(
            name="Test",
            role=Role.READER,
        )
        
        ctx = require_auth(raw_key, manager, Operation.QUERY)
        assert ctx.is_authenticated
    
    def test_require_auth_failure(self):
        manager = APIKeyManager()
        
        with pytest.raises(AuthorizationError):
            require_auth("invalid_key", manager, Operation.QUERY)
