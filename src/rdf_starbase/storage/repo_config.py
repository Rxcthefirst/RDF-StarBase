"""
Repository Configuration and Versioning for RDF-StarBase.

Provides:
- Schema versioning with migration paths
- Per-repository configuration
- Repository metadata management
- Configuration validation
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)

# Current schema version
CURRENT_SCHEMA_VERSION = "0.3.0"


class ReasoningLevel(Enum):
    """Reasoning inference levels."""
    NONE = "none"           # No inference
    RDFS = "rdfs"           # RDFS rules only
    RDFS_PLUS = "rdfs_plus" # RDFS + OWL subset
    OWL_RL = "owl_rl"       # OWL 2 RL profile


@dataclass
class MemoryConfig:
    """Memory configuration for a repository."""
    max_memory_mb: int = 1024
    term_dict_cache_mb: int = 256
    query_cache_mb: int = 128
    compaction_threshold_mb: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "term_dict_cache_mb": self.term_dict_cache_mb,
            "query_cache_mb": self.query_cache_mb,
            "compaction_threshold_mb": self.compaction_threshold_mb
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        return cls(
            max_memory_mb=data.get("max_memory_mb", 1024),
            term_dict_cache_mb=data.get("term_dict_cache_mb", 256),
            query_cache_mb=data.get("query_cache_mb", 128),
            compaction_threshold_mb=data.get("compaction_threshold_mb", 512)
        )


@dataclass
class QueryConfig:
    """Query configuration for a repository."""
    default_timeout_seconds: float = 30.0
    max_timeout_seconds: float = 300.0
    max_results: int = 100000
    enable_query_cache: bool = True
    slow_query_threshold_seconds: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_timeout_seconds": self.max_timeout_seconds,
            "max_results": self.max_results,
            "enable_query_cache": self.enable_query_cache,
            "slow_query_threshold_seconds": self.slow_query_threshold_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryConfig":
        return cls(
            default_timeout_seconds=data.get("default_timeout_seconds", 30.0),
            max_timeout_seconds=data.get("max_timeout_seconds", 300.0),
            max_results=data.get("max_results", 100000),
            enable_query_cache=data.get("enable_query_cache", True),
            slow_query_threshold_seconds=data.get("slow_query_threshold_seconds", 1.0)
        )


@dataclass
class ReasoningConfig:
    """Reasoning configuration for a repository."""
    level: ReasoningLevel = ReasoningLevel.RDFS
    materialize_on_load: bool = False
    incremental_reasoning: bool = True
    custom_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "materialize_on_load": self.materialize_on_load,
            "incremental_reasoning": self.incremental_reasoning,
            "custom_rules": self.custom_rules
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningConfig":
        level_str = data.get("level", "rdfs")
        try:
            level = ReasoningLevel(level_str)
        except ValueError:
            level = ReasoningLevel.RDFS
        
        return cls(
            level=level,
            materialize_on_load=data.get("materialize_on_load", False),
            incremental_reasoning=data.get("incremental_reasoning", True),
            custom_rules=data.get("custom_rules", [])
        )


@dataclass
class StorageConfig:
    """Storage configuration for a repository."""
    enable_wal: bool = True
    wal_sync_mode: str = "normal"  # "off", "normal", "full"
    compaction_enabled: bool = True
    compaction_interval_minutes: int = 60
    enable_partitioning: bool = True
    partition_threshold: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_wal": self.enable_wal,
            "wal_sync_mode": self.wal_sync_mode,
            "compaction_enabled": self.compaction_enabled,
            "compaction_interval_minutes": self.compaction_interval_minutes,
            "enable_partitioning": self.enable_partitioning,
            "partition_threshold": self.partition_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        return cls(
            enable_wal=data.get("enable_wal", True),
            wal_sync_mode=data.get("wal_sync_mode", "normal"),
            compaction_enabled=data.get("compaction_enabled", True),
            compaction_interval_minutes=data.get("compaction_interval_minutes", 60),
            enable_partitioning=data.get("enable_partitioning", True),
            partition_threshold=data.get("partition_threshold", 10000)
        )


@dataclass
class RepositoryConfig:
    """
    Complete configuration for a repository.
    
    Stored in repo metadata and used to configure store behavior.
    """
    # Identity
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Schema version
    schema_version: str = CURRENT_SCHEMA_VERSION
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Sub-configs
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Custom metadata
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "memory": self.memory.to_dict(),
            "query": self.query.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "storage": self.storage.to_dict(),
            "tags": self.tags,
            "custom": self.custom
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryConfig":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()
        
        return cls(
            uuid=data.get("uuid", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
            created_at=created_at,
            updated_at=updated_at,
            memory=MemoryConfig.from_dict(data.get("memory", {})),
            query=QueryConfig.from_dict(data.get("query", {})),
            reasoning=ReasoningConfig.from_dict(data.get("reasoning", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
            tags=data.get("tags", []),
            custom=data.get("custom", {})
        )
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        self.updated_at = datetime.now()
        config_file = path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RepositoryConfig":
        """Load configuration from file."""
        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        return cls()


@dataclass
class MigrationStep:
    """A single migration step."""
    from_version: str
    to_version: str
    description: str
    migrate: Callable[[Path, RepositoryConfig], RepositoryConfig]


class SchemaVersionError(Exception):
    """Schema version error."""
    pass


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse semver string to tuple."""
    parts = version.split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (0, 0, 0)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    t1, t2 = parse_version(v1), parse_version(v2)
    if t1 < t2:
        return -1
    elif t1 > t2:
        return 1
    return 0


class SchemaMigrator:
    """
    Handles schema migrations between versions.
    
    Migrations are registered as steps from one version to another.
    The migrator finds a path from current version to target version
    and applies migrations in order.
    """
    
    def __init__(self):
        self._migrations: Dict[str, MigrationStep] = {}
        self._register_builtin_migrations()
    
    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Migration from 0.2.0 to 0.3.0
        self.register_migration(MigrationStep(
            from_version="0.2.0",
            to_version="0.3.0",
            description="Add per-repo configuration and observability",
            migrate=self._migrate_0_2_to_0_3
        ))
        
        # Migration from 0.1.0 to 0.2.0
        self.register_migration(MigrationStep(
            from_version="0.1.0",
            to_version="0.2.0",
            description="Add WAL and transaction support",
            migrate=self._migrate_0_1_to_0_2
        ))
    
    def _migrate_0_2_to_0_3(
        self,
        path: Path,
        config: RepositoryConfig
    ) -> RepositoryConfig:
        """Migrate from 0.2.0 to 0.3.0."""
        logger.info(f"Migrating {path} from 0.2.0 to 0.3.0")
        
        # Add new configuration sections with defaults
        if not hasattr(config, 'memory') or config.memory is None:
            config.memory = MemoryConfig()
        if not hasattr(config, 'query') or config.query is None:
            config.query = QueryConfig()
        if not hasattr(config, 'reasoning') or config.reasoning is None:
            config.reasoning = ReasoningConfig()
        if not hasattr(config, 'storage') or config.storage is None:
            config.storage = StorageConfig()
        
        # Ensure UUID exists
        if not config.uuid:
            config.uuid = str(uuid.uuid4())
        
        config.schema_version = "0.3.0"
        return config
    
    def _migrate_0_1_to_0_2(
        self,
        path: Path,
        config: RepositoryConfig
    ) -> RepositoryConfig:
        """Migrate from 0.1.0 to 0.2.0."""
        logger.info(f"Migrating {path} from 0.1.0 to 0.2.0")
        
        # 0.2.0 added WAL - create WAL directory if needed
        wal_dir = path / "wal"
        if not wal_dir.exists():
            wal_dir.mkdir(exist_ok=True)
        
        config.schema_version = "0.2.0"
        return config
    
    def register_migration(self, step: MigrationStep) -> None:
        """Register a migration step."""
        key = f"{step.from_version}->{step.to_version}"
        self._migrations[key] = step
    
    def get_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> List[MigrationStep]:
        """Find migration path between versions."""
        if compare_versions(from_version, to_version) >= 0:
            return []  # Already at or ahead of target
        
        # BFS to find shortest path
        visited = {from_version}
        queue = [(from_version, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            for key, step in self._migrations.items():
                if step.from_version == current and step.to_version not in visited:
                    new_path = path + [step]
                    
                    if step.to_version == to_version:
                        return new_path
                    
                    if compare_versions(step.to_version, to_version) <= 0:
                        visited.add(step.to_version)
                        queue.append((step.to_version, new_path))
        
        return []
    
    def migrate(
        self,
        path: Path,
        config: RepositoryConfig,
        target_version: str = CURRENT_SCHEMA_VERSION
    ) -> RepositoryConfig:
        """
        Migrate configuration to target version.
        
        Returns updated configuration.
        Raises SchemaVersionError if migration fails.
        """
        current = config.schema_version or "0.1.0"
        
        if compare_versions(current, target_version) >= 0:
            logger.debug(f"No migration needed: {current} >= {target_version}")
            return config
        
        steps = self.get_migration_path(current, target_version)
        
        if not steps:
            # Direct upgrade - just update version
            logger.info(f"Direct upgrade from {current} to {target_version}")
            config.schema_version = target_version
            return config
        
        # Apply migrations
        for step in steps:
            logger.info(f"Applying migration: {step.description}")
            try:
                config = step.migrate(path, config)
            except Exception as e:
                raise SchemaVersionError(
                    f"Migration failed at {step.from_version} -> {step.to_version}: {e}"
                )
        
        config.schema_version = target_version
        return config
    
    def needs_migration(
        self,
        config: RepositoryConfig,
        target_version: str = CURRENT_SCHEMA_VERSION
    ) -> bool:
        """Check if migration is needed."""
        current = config.schema_version or "0.1.0"
        return compare_versions(current, target_version) < 0


class ConfigValidator:
    """Validates repository configuration."""
    
    @staticmethod
    def validate(config: RepositoryConfig) -> List[str]:
        """
        Validate configuration.
        
        Returns list of error messages (empty if valid).
        """
        errors = []
        
        # Memory config validation
        if config.memory.max_memory_mb < 64:
            errors.append("max_memory_mb must be at least 64 MB")
        
        if config.memory.term_dict_cache_mb > config.memory.max_memory_mb:
            errors.append("term_dict_cache_mb cannot exceed max_memory_mb")
        
        # Query config validation
        if config.query.default_timeout_seconds <= 0:
            errors.append("default_timeout_seconds must be positive")
        
        if config.query.max_timeout_seconds < config.query.default_timeout_seconds:
            errors.append("max_timeout_seconds cannot be less than default_timeout_seconds")
        
        if config.query.max_results < 1:
            errors.append("max_results must be at least 1")
        
        # Storage config validation
        if config.storage.wal_sync_mode not in ("off", "normal", "full"):
            errors.append(f"Invalid wal_sync_mode: {config.storage.wal_sync_mode}")
        
        if config.storage.compaction_interval_minutes < 1:
            errors.append("compaction_interval_minutes must be at least 1")
        
        return errors
    
    @staticmethod
    def validate_or_raise(config: RepositoryConfig) -> None:
        """Validate configuration, raising on errors."""
        errors = ConfigValidator.validate(config)
        if errors:
            raise ConfigValidationError("; ".join(errors))


class ConfigManager:
    """
    Manages repository configurations.
    
    Provides:
    - Load/save configuration
    - Migration on load
    - Validation
    - Default configuration
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.migrator = SchemaMigrator()
        self._configs: Dict[str, RepositoryConfig] = {}
    
    def get_config(self, repo_name: str) -> RepositoryConfig:
        """Get configuration for a repository."""
        if repo_name in self._configs:
            return self._configs[repo_name]
        
        repo_path = self.base_path / repo_name
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_name}")
        
        config = RepositoryConfig.load(repo_path)
        
        # Migrate if needed
        if self.migrator.needs_migration(config):
            config = self.migrator.migrate(repo_path, config)
            config.save(repo_path)
        
        self._configs[repo_name] = config
        return config
    
    def set_config(
        self,
        repo_name: str,
        config: RepositoryConfig,
        validate: bool = True
    ) -> None:
        """Set configuration for a repository."""
        if validate:
            ConfigValidator.validate_or_raise(config)
        
        repo_path = self.base_path / repo_name
        if not repo_path.exists():
            repo_path.mkdir(parents=True, exist_ok=True)
        
        config.name = repo_name
        config.save(repo_path)
        self._configs[repo_name] = config
    
    def update_config(
        self,
        repo_name: str,
        updates: Dict[str, Any]
    ) -> RepositoryConfig:
        """Update specific configuration fields."""
        config = self.get_config(repo_name)
        
        # Apply updates
        if "memory" in updates:
            for k, v in updates["memory"].items():
                setattr(config.memory, k, v)
        
        if "query" in updates:
            for k, v in updates["query"].items():
                setattr(config.query, k, v)
        
        if "reasoning" in updates:
            for k, v in updates["reasoning"].items():
                if k == "level":
                    config.reasoning.level = ReasoningLevel(v)
                else:
                    setattr(config.reasoning, k, v)
        
        if "storage" in updates:
            for k, v in updates["storage"].items():
                setattr(config.storage, k, v)
        
        if "description" in updates:
            config.description = updates["description"]
        
        if "tags" in updates:
            config.tags = updates["tags"]
        
        if "custom" in updates:
            config.custom.update(updates["custom"])
        
        # Validate and save
        ConfigValidator.validate_or_raise(config)
        repo_path = self.base_path / repo_name
        config.save(repo_path)
        
        return config
    
    def create_config(
        self,
        repo_name: str,
        description: str = "",
        **kwargs
    ) -> RepositoryConfig:
        """Create a new repository configuration."""
        config = RepositoryConfig(
            uuid=str(uuid.uuid4()),
            name=repo_name,
            description=description,
            schema_version=CURRENT_SCHEMA_VERSION
        )
        
        # Apply any custom settings
        if "memory" in kwargs:
            config.memory = MemoryConfig.from_dict(kwargs["memory"])
        if "query" in kwargs:
            config.query = QueryConfig.from_dict(kwargs["query"])
        if "reasoning" in kwargs:
            config.reasoning = ReasoningConfig.from_dict(kwargs["reasoning"])
        if "storage" in kwargs:
            config.storage = StorageConfig.from_dict(kwargs["storage"])
        if "tags" in kwargs:
            config.tags = kwargs["tags"]
        
        self.set_config(repo_name, config)
        return config
    
    def list_configs(self) -> List[str]:
        """List all repository configurations."""
        configs = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    configs.append(item.name)
        return configs
    
    def delete_config(self, repo_name: str) -> None:
        """Delete a repository configuration."""
        if repo_name in self._configs:
            del self._configs[repo_name]
        
        config_file = self.base_path / repo_name / "config.json"
        if config_file.exists():
            config_file.unlink()


# Convenience functions
def get_schema_version() -> str:
    """Get current schema version."""
    return CURRENT_SCHEMA_VERSION


def create_default_config(name: str = "") -> RepositoryConfig:
    """Create a default repository configuration."""
    return RepositoryConfig(
        uuid=str(uuid.uuid4()),
        name=name,
        schema_version=CURRENT_SCHEMA_VERSION
    )
