"""Tests for repository configuration and versioning."""
import pytest
import json
from pathlib import Path
from datetime import datetime

from rdf_starbase.storage.repo_config import (
    ReasoningLevel,
    MemoryConfig,
    QueryConfig,
    ReasoningConfig,
    StorageConfig,
    RepositoryConfig,
    MigrationStep,
    SchemaMigrator,
    ConfigValidator,
    ConfigManager,
    SchemaVersionError,
    ConfigValidationError,
    CURRENT_SCHEMA_VERSION,
    parse_version,
    compare_versions,
    get_schema_version,
    create_default_config,
)


# ========== ReasoningLevel Tests ==========

class TestReasoningLevel:
    def test_levels(self):
        assert ReasoningLevel.NONE.value == "none"
        assert ReasoningLevel.RDFS.value == "rdfs"
        assert ReasoningLevel.RDFS_PLUS.value == "rdfs_plus"
        assert ReasoningLevel.OWL_RL.value == "owl_rl"


# ========== MemoryConfig Tests ==========

class TestMemoryConfig:
    def test_defaults(self):
        config = MemoryConfig()
        assert config.max_memory_mb == 1024
        assert config.term_dict_cache_mb == 256
    
    def test_to_dict(self):
        config = MemoryConfig(max_memory_mb=2048)
        d = config.to_dict()
        assert d["max_memory_mb"] == 2048
    
    def test_from_dict(self):
        config = MemoryConfig.from_dict({"max_memory_mb": 512})
        assert config.max_memory_mb == 512


# ========== QueryConfig Tests ==========

class TestQueryConfig:
    def test_defaults(self):
        config = QueryConfig()
        assert config.default_timeout_seconds == 30.0
        assert config.slow_query_threshold_seconds == 1.0
    
    def test_to_dict(self):
        config = QueryConfig(max_results=50000)
        assert config.to_dict()["max_results"] == 50000
    
    def test_from_dict(self):
        config = QueryConfig.from_dict({"enable_query_cache": False})
        assert config.enable_query_cache is False


# ========== ReasoningConfig Tests ==========

class TestReasoningConfig:
    def test_defaults(self):
        config = ReasoningConfig()
        assert config.level == ReasoningLevel.RDFS
        assert config.incremental_reasoning is True
    
    def test_custom_rules(self):
        config = ReasoningConfig(custom_rules=["rule1", "rule2"])
        assert len(config.custom_rules) == 2
    
    def test_from_dict_invalid_level(self):
        config = ReasoningConfig.from_dict({"level": "invalid"})
        assert config.level == ReasoningLevel.RDFS  # Default


# ========== StorageConfig Tests ==========

class TestStorageConfig:
    def test_defaults(self):
        config = StorageConfig()
        assert config.enable_wal is True
        assert config.wal_sync_mode == "normal"
    
    def test_partition_threshold(self):
        config = StorageConfig(partition_threshold=50000)
        assert config.partition_threshold == 50000


# ========== RepositoryConfig Tests ==========

class TestRepositoryConfig:
    def test_creation(self):
        config = RepositoryConfig(name="test-repo")
        assert config.name == "test-repo"
        assert config.uuid is not None
        assert config.schema_version == CURRENT_SCHEMA_VERSION
    
    def test_to_dict(self):
        config = RepositoryConfig(name="test", description="A test repo")
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "A test repo"
        assert "memory" in d
        assert "query" in d
    
    def test_from_dict(self):
        data = {
            "uuid": "test-uuid",
            "name": "my-repo",
            "schema_version": "0.2.0",
            "memory": {"max_memory_mb": 2048},
            "tags": ["production"]
        }
        config = RepositoryConfig.from_dict(data)
        assert config.uuid == "test-uuid"
        assert config.name == "my-repo"
        assert config.memory.max_memory_mb == 2048
        assert "production" in config.tags
    
    def test_save_and_load(self, tmp_path):
        config = RepositoryConfig(
            name="persist-test",
            description="Testing persistence"
        )
        config.save(tmp_path)
        
        loaded = RepositoryConfig.load(tmp_path)
        assert loaded.name == "persist-test"
        assert loaded.description == "Testing persistence"
    
    def test_custom_metadata(self):
        config = RepositoryConfig()
        config.custom["env"] = "production"
        config.custom["owner"] = "team-a"
        
        d = config.to_dict()
        assert d["custom"]["env"] == "production"


# ========== Version Parsing Tests ==========

class TestVersionParsing:
    def test_parse_version(self):
        assert parse_version("0.3.0") == (0, 3, 0)
        assert parse_version("1.2.3") == (1, 2, 3)
        assert parse_version("invalid") == (0, 0, 0)
    
    def test_compare_versions(self):
        assert compare_versions("0.2.0", "0.3.0") == -1
        assert compare_versions("0.3.0", "0.3.0") == 0
        assert compare_versions("1.0.0", "0.3.0") == 1


# ========== SchemaMigrator Tests ==========

class TestSchemaMigrator:
    def test_needs_migration(self):
        migrator = SchemaMigrator()
        old_config = RepositoryConfig(schema_version="0.1.0")
        assert migrator.needs_migration(old_config)
        
        current_config = RepositoryConfig()
        assert not migrator.needs_migration(current_config)
    
    def test_get_migration_path(self):
        migrator = SchemaMigrator()
        path = migrator.get_migration_path("0.1.0", "0.3.0")
        assert len(path) == 2
        assert path[0].from_version == "0.1.0"
        assert path[1].to_version == "0.3.0"
    
    def test_migrate_0_2_to_0_3(self, tmp_path):
        migrator = SchemaMigrator()
        config = RepositoryConfig(schema_version="0.2.0")
        
        result = migrator.migrate(tmp_path, config, "0.3.0")
        assert result.schema_version == "0.3.0"
        assert result.uuid is not None
    
    def test_no_migration_needed(self, tmp_path):
        migrator = SchemaMigrator()
        config = RepositoryConfig()
        
        result = migrator.migrate(tmp_path, config)
        assert result.schema_version == CURRENT_SCHEMA_VERSION
    
    def test_register_custom_migration(self, tmp_path):
        migrator = SchemaMigrator()
        
        def custom_migrate(path, config):
            config.custom["migrated"] = True
            return config
        
        migrator.register_migration(MigrationStep(
            from_version="0.3.0",
            to_version="0.4.0",
            description="Test migration",
            migrate=custom_migrate
        ))
        
        config = RepositoryConfig(schema_version="0.3.0")
        result = migrator.migrate(tmp_path, config, "0.4.0")
        assert result.custom.get("migrated") is True


# ========== ConfigValidator Tests ==========

class TestConfigValidator:
    def test_valid_config(self):
        config = RepositoryConfig()
        errors = ConfigValidator.validate(config)
        assert len(errors) == 0
    
    def test_invalid_memory(self):
        config = RepositoryConfig()
        config.memory.max_memory_mb = 32  # Too small
        errors = ConfigValidator.validate(config)
        assert any("64 MB" in e for e in errors)
    
    def test_invalid_timeout(self):
        config = RepositoryConfig()
        config.query.default_timeout_seconds = -1
        errors = ConfigValidator.validate(config)
        assert any("positive" in e for e in errors)
    
    def test_timeout_order(self):
        config = RepositoryConfig()
        config.query.default_timeout_seconds = 100
        config.query.max_timeout_seconds = 50
        errors = ConfigValidator.validate(config)
        assert any("max_timeout" in e for e in errors)
    
    def test_invalid_wal_mode(self):
        config = RepositoryConfig()
        config.storage.wal_sync_mode = "invalid"
        errors = ConfigValidator.validate(config)
        assert any("wal_sync_mode" in e for e in errors)
    
    def test_validate_or_raise(self):
        config = RepositoryConfig()
        config.memory.max_memory_mb = 10
        
        with pytest.raises(ConfigValidationError):
            ConfigValidator.validate_or_raise(config)


# ========== ConfigManager Tests ==========

class TestConfigManager:
    def test_create_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        config = manager.create_config("test-repo", description="Test")
        
        assert config.name == "test-repo"
        assert config.description == "Test"
        assert (tmp_path / "test-repo" / "config.json").exists()
    
    def test_get_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.create_config("my-repo")
        
        config = manager.get_config("my-repo")
        assert config.name == "my-repo"
    
    def test_get_nonexistent(self, tmp_path):
        manager = ConfigManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.get_config("nonexistent")
    
    def test_update_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.create_config("update-test")
        
        updated = manager.update_config("update-test", {
            "description": "Updated description",
            "memory": {"max_memory_mb": 2048}
        })
        
        assert updated.description == "Updated description"
        assert updated.memory.max_memory_mb == 2048
    
    def test_list_configs(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.create_config("repo1")
        manager.create_config("repo2")
        
        configs = manager.list_configs()
        assert "repo1" in configs
        assert "repo2" in configs
    
    def test_delete_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.create_config("to-delete")
        
        manager.delete_config("to-delete")
        assert "to-delete" not in manager.list_configs()


# ========== Convenience Functions Tests ==========

class TestConvenienceFunctions:
    def test_get_schema_version(self):
        assert get_schema_version() == CURRENT_SCHEMA_VERSION
    
    def test_create_default_config(self):
        config = create_default_config("test")
        assert config.name == "test"
        assert config.schema_version == CURRENT_SCHEMA_VERSION
