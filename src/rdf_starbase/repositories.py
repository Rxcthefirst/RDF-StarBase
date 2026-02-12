"""
Repository Manager for RDF-StarBase.

Manages multiple named TripleStore instances (repositories/projects).
Similar to how GraphDB or Neo4j manage multiple databases.

Features:
- Create/delete named repositories
- Persist repositories to disk
- Switch between repositories
- List all repositories with metadata
- Backup/restore/clone repositories
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import shutil
import uuid

from rdf_starbase.store import TripleStore
from rdf_starbase.storage.backup import (
    BackupManager,
    BackupManifest,
    RestoreOptions,
)
from rdf_starbase.storage.repo_config import ReasoningLevel, ReasoningConfig


@dataclass
class RepositoryInfo:
    """Metadata about a repository."""
    name: str
    created_at: datetime
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Unique ID (stable across renames)
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Schema version for migrations
    schema_version: int = 1
    
    # Reasoning configuration (like GraphDB)
    reasoning_level: str = "none"  # none, rdfs, rdfs_plus, owl_rl
    materialize_on_load: bool = False
    
    # Stats (populated on demand)
    triple_count: int = 0
    subject_count: int = 0
    predicate_count: int = 0
    inferred_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "tags": self.tags,
            "uuid": self.uuid,
            "schema_version": self.schema_version,
            "reasoning_level": self.reasoning_level,
            "materialize_on_load": self.materialize_on_load,
            "triple_count": self.triple_count,
            "subject_count": self.subject_count,
            "predicate_count": self.predicate_count,
            "inferred_count": self.inferred_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RepositoryInfo":
        return cls(
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            uuid=data.get("uuid", str(uuid.uuid4())),
            schema_version=data.get("schema_version", 1),
            reasoning_level=data.get("reasoning_level", "none"),
            materialize_on_load=data.get("materialize_on_load", False),
            triple_count=data.get("triple_count", 0),
            subject_count=data.get("subject_count", 0),
            predicate_count=data.get("predicate_count", 0),
            inferred_count=data.get("inferred_count", 0),
        )


class RepositoryManager:
    """
    Manages multiple named TripleStore repositories.
    
    Provides:
    - CRUD operations for repositories
    - Persistence to a workspace directory
    - In-memory caching of active repositories
    
    Usage:
        manager = RepositoryManager("./data/repositories")
        
        # Create a new repository
        manager.create("my-project", description="Test project")
        
        # Get the store for a repository
        store = manager.get_store("my-project")
        store.add_triple(...)
        
        # List all repositories
        repos = manager.list_repositories()
        
        # Persist changes
        manager.save("my-project")
    """
    
    def __init__(self, workspace_path: str | Path):
        """
        Initialize the repository manager.
        
        Args:
            workspace_path: Directory to store all repositories
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of loaded repositories
        self._stores: Dict[str, TripleStore] = {}
        self._info: Dict[str, RepositoryInfo] = {}
        
        # Load existing repository metadata
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata for all repositories in the workspace."""
        for repo_dir in self.workspace_path.iterdir():
            if repo_dir.is_dir():
                meta_file = repo_dir / "repository.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            data = json.load(f)
                        self._info[repo_dir.name] = RepositoryInfo.from_dict(data)
                    except Exception as e:
                        print(f"Warning: Failed to load metadata for {repo_dir.name}: {e}")
    
    def _save_metadata(self, name: str) -> None:
        """Save metadata for a repository."""
        repo_dir = self.workspace_path / name
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        info = self._info.get(name)
        if info:
            # Update stats if store is loaded
            if name in self._stores:
                stats = self._stores[name].stats()
                info.triple_count = stats.get("active_assertions", 0)
                info.subject_count = stats.get("unique_subjects", 0)
                info.predicate_count = stats.get("unique_predicates", 0)
            
            with open(repo_dir / "repository.json", "w") as f:
                json.dump(info.to_dict(), f, indent=2)
    
    def create(
        self, 
        name: str, 
        description: str = "",
        tags: Optional[list[str]] = None,
        reasoning_level: str = "none",
        materialize_on_load: bool = False,
    ) -> RepositoryInfo:
        """
        Create a new repository.
        
        Args:
            name: Unique repository name (alphanumeric + hyphens)
            description: Human-readable description
            tags: Optional tags for categorization
            reasoning_level: Inference level - "none", "rdfs", "rdfs_plus", "owl_rl"
            materialize_on_load: If True, run inference after data load
            
        Returns:
            RepositoryInfo for the new repository
            
        Raises:
            ValueError: If name is invalid or already exists
        """
        # Validate name
        if not name:
            raise ValueError("Repository name cannot be empty")
        if not all(c.isalnum() or c in '-_' for c in name):
            raise ValueError("Repository name can only contain alphanumeric characters, hyphens, and underscores")
        if name in self._info:
            raise ValueError(f"Repository '{name}' already exists")
        
        # Validate reasoning level
        valid_levels = {"none", "rdfs", "rdfs_plus", "owl_rl"}
        if reasoning_level not in valid_levels:
            raise ValueError(f"Invalid reasoning_level '{reasoning_level}'. Valid options: {valid_levels}")
        
        # Create repository
        info = RepositoryInfo(
            name=name,
            created_at=datetime.now(timezone.utc),
            description=description,
            tags=tags or [],
            reasoning_level=reasoning_level,
            materialize_on_load=materialize_on_load,
        )
        
        self._info[name] = info
        self._stores[name] = TripleStore()
        
        # Persist metadata
        self._save_metadata(name)
        
        return info
    
    def delete(self, name: str, force: bool = False) -> bool:
        """
        Delete a repository.
        
        Args:
            name: Repository name
            force: If True, delete even if repository has data
            
        Returns:
            True if deleted
            
        Raises:
            ValueError: If repository doesn't exist
            ValueError: If repository has data and force=False
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        # Check if repository has data
        store = self.get_store(name)
        stats = store.stats()
        if stats.get("active_assertions", 0) > 0 and not force:
            raise ValueError(
                f"Repository '{name}' contains {stats['active_assertions']} assertions. "
                "Use force=True to delete anyway."
            )
        
        # Remove from memory
        self._stores.pop(name, None)
        self._info.pop(name, None)
        
        # Remove from disk
        repo_dir = self.workspace_path / name
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        
        return True
    
    def get_store(self, name: str) -> TripleStore:
        """
        Get the TripleStore for a repository.
        
        Loads from disk if not already in memory.
        
        Args:
            name: Repository name
            
        Returns:
            TripleStore instance
            
        Raises:
            ValueError: If repository doesn't exist
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        if name not in self._stores:
            # Load from disk
            repo_dir = self.workspace_path / name
            store_file = repo_dir / "store.parquet"
            
            if store_file.exists():
                self._stores[name] = TripleStore.load(store_file)
            else:
                self._stores[name] = TripleStore()
        
        return self._stores[name]
    
    def get_info(self, name: str) -> RepositoryInfo:
        """
        Get metadata for a repository.
        
        Args:
            name: Repository name
            
        Returns:
            RepositoryInfo
            
        Raises:
            ValueError: If repository doesn't exist
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        info = self._info[name]
        
        # Update stats if store is loaded
        if name in self._stores:
            stats = self._stores[name].stats()
            info.triple_count = stats.get("active_assertions", 0)
            info.subject_count = stats.get("unique_subjects", 0)
            info.predicate_count = stats.get("unique_predicates", 0)
        
        return info
    
    def materialize_inferences(self, name: str, level: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference and materialize inferred triples.
        
        Similar to GraphDB's materialization - runs the configured reasoner
        and stores inferred triples with INFERRED flag.
        
        Args:
            name: Repository name
            level: Override reasoning level (uses repo config if None)
            
        Returns:
            Dict with inference statistics
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        from rdf_starbase.storage.reasoner import RDFSReasoner
        
        info = self._info[name]
        store = self.get_store(name)
        
        # Determine reasoning level
        reasoning_level = level or info.reasoning_level
        if reasoning_level == "none":
            return {"status": "skipped", "reason": "reasoning_level is 'none'"}
        
        # Configure reasoner based on level
        enable_owl = reasoning_level in ("rdfs_plus", "owl_rl")
        
        reasoner = RDFSReasoner(
            term_dict=store._term_dict,
            fact_store=store._fact_store,
            enable_owl=enable_owl,
        )
        
        # Run inference
        stats = reasoner.reason()
        
        # Invalidate store cache
        store._invalidate_cache()
        
        # Update info with inferred count
        info.inferred_count = stats.triples_inferred
        self._save_metadata(name)
        
        return {
            "status": "success",
            "reasoning_level": reasoning_level,
            "iterations": stats.iterations,
            "triples_inferred": stats.triples_inferred,
            "rdfs_inferences": {
                "domain": stats.rdfs2_inferences,
                "range": stats.rdfs3_inferences,
                "subPropertyOf_transitivity": stats.rdfs5_inferences,
                "property_inheritance": stats.rdfs7_inferences,
                "type_inheritance": stats.rdfs9_inferences,
                "subClassOf_transitivity": stats.rdfs11_inferences,
            },
            "owl_inferences": {
                "sameAs": stats.owl_same_as_inferences,
                "equivalentClass": stats.owl_equivalent_class_inferences,
                "equivalentProperty": stats.owl_equivalent_property_inferences,
                "inverseOf": stats.owl_inverse_of_inferences,
                "transitive": stats.owl_transitive_inferences,
                "symmetric": stats.owl_symmetric_inferences,
                "functional": stats.owl_functional_inferences,
                "inverseFunctional": stats.owl_inverse_functional_inferences,
                "hasValue": stats.owl_has_value_inferences,
            } if enable_owl else {},
        }
    
    def update_reasoning_config(
        self, 
        name: str, 
        reasoning_level: Optional[str] = None,
        materialize_on_load: Optional[bool] = None,
    ) -> RepositoryInfo:
        """
        Update reasoning configuration for a repository.
        
        Args:
            name: Repository name
            reasoning_level: New reasoning level (none/rdfs/rdfs_plus/owl_rl)
            materialize_on_load: Whether to auto-materialize on data load
            
        Returns:
            Updated RepositoryInfo
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        info = self._info[name]
        
        if reasoning_level is not None:
            valid_levels = {"none", "rdfs", "rdfs_plus", "owl_rl"}
            if reasoning_level not in valid_levels:
                raise ValueError(f"Invalid reasoning_level. Valid: {valid_levels}")
            info.reasoning_level = reasoning_level
        
        if materialize_on_load is not None:
            info.materialize_on_load = materialize_on_load
        
        self._save_metadata(name)
        return info
    
    def list_repositories(self) -> list[RepositoryInfo]:
        """
        List all repositories with their metadata.
        
        Returns:
            List of RepositoryInfo objects
        """
        result = []
        for name, info in sorted(self._info.items()):
            # Update stats only if store is already loaded (use cached stats)
            if name in self._stores:
                stats = self._stores[name].stats()
                info.triple_count = stats.get("active_assertions", 0)
                info.subject_count = stats.get("unique_subjects", 0)
                info.predicate_count = stats.get("unique_predicates", 0)
            # Otherwise use persisted metadata counts (already on info)
            result.append(info)
        return result
    
    def save(self, name: str) -> None:
        """
        Persist a repository to disk.
        
        Args:
            name: Repository name
            
        Raises:
            ValueError: If repository doesn't exist or isn't loaded
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        if name not in self._stores:
            # Nothing to save - not loaded
            return
        
        repo_dir = self.workspace_path / name
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Save store
        store_file = repo_dir / "store.parquet"
        self._stores[name].save(store_file)
        
        # Update metadata
        self._save_metadata(name)
    
    def save_all(self) -> None:
        """Persist all loaded repositories to disk."""
        for name in self._stores:
            self.save(name)
    
    def exists(self, name: str) -> bool:
        """Check if a repository exists."""
        return name in self._info
    
    def unload(self, name: str) -> None:
        """
        Unload a repository from memory (after saving).
        
        Useful for freeing memory when many repositories exist.
        
        Args:
            name: Repository name
        """
        if name in self._stores:
            self.save(name)
            del self._stores[name]
    
    def rename(self, old_name: str, new_name: str) -> RepositoryInfo:
        """
        Rename a repository.
        
        Args:
            old_name: Current repository name
            new_name: New repository name
            
        Returns:
            Updated RepositoryInfo
        """
        if old_name not in self._info:
            raise ValueError(f"Repository '{old_name}' does not exist")
        if new_name in self._info:
            raise ValueError(f"Repository '{new_name}' already exists")
        if not all(c.isalnum() or c in '-_' for c in new_name):
            raise ValueError("Repository name can only contain alphanumeric characters, hyphens, and underscores")
        
        # Save first
        if old_name in self._stores:
            self.save(old_name)
        
        # Move directory
        old_dir = self.workspace_path / old_name
        new_dir = self.workspace_path / new_name
        if old_dir.exists():
            old_dir.rename(new_dir)
        
        # Update in-memory state
        info = self._info.pop(old_name)
        info.name = new_name
        self._info[new_name] = info
        
        if old_name in self._stores:
            self._stores[new_name] = self._stores.pop(old_name)
        
        # Update metadata file
        self._save_metadata(new_name)
        
        return info
    
    def update_info(
        self, 
        name: str, 
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> RepositoryInfo:
        """
        Update repository metadata.
        
        Args:
            name: Repository name
            description: New description (if provided)
            tags: New tags (if provided)
            
        Returns:
            Updated RepositoryInfo
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        info = self._info[name]
        if description is not None:
            info.description = description
        if tags is not None:
            info.tags = tags
        
        self._save_metadata(name)
        return info

    def clone(
        self,
        source_name: str,
        target_name: str,
        description: Optional[str] = None,
    ) -> RepositoryInfo:
        """
        Clone a repository to a new name.
        
        Creates a complete copy of the source repository including
        all data, but with a new name and UUID.
        
        Args:
            source_name: Name of repository to clone
            target_name: Name for the new cloned repository
            description: Optional description (defaults to "Clone of {source}")
            
        Returns:
            RepositoryInfo for the cloned repository
            
        Raises:
            ValueError: If source doesn't exist or target already exists
        """
        if source_name not in self._info:
            raise ValueError(f"Source repository '{source_name}' does not exist")
        if target_name in self._info:
            raise ValueError(f"Target repository '{target_name}' already exists")
        if not all(c.isalnum() or c in '-_' for c in target_name):
            raise ValueError("Repository name can only contain alphanumeric characters, hyphens, and underscores")
        
        # Save source first to ensure we have latest data
        if source_name in self._stores:
            self.save(source_name)
        
        # Copy the directory
        source_dir = self.workspace_path / source_name
        target_dir = self.workspace_path / target_name
        
        if source_dir.exists():
            shutil.copytree(source_dir, target_dir)
        else:
            target_dir.mkdir(parents=True)
        
        # Create new metadata with new UUID
        source_info = self._info[source_name]
        target_info = RepositoryInfo(
            name=target_name,
            created_at=datetime.now(timezone.utc),
            description=description or f"Clone of {source_name}",
            tags=source_info.tags.copy(),
            uuid=str(uuid.uuid4()),  # New UUID for clone
            schema_version=source_info.schema_version,
            triple_count=source_info.triple_count,
            subject_count=source_info.subject_count,
            predicate_count=source_info.predicate_count,
        )
        
        self._info[target_name] = target_info
        self._save_metadata(target_name)
        
        return target_info
    
    def snapshot(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> BackupManifest:
        """
        Create a backup snapshot of a repository.
        
        Args:
            name: Repository name
            description: Backup description
            tags: Optional tags for the backup
            
        Returns:
            BackupManifest describing the snapshot
            
        Raises:
            ValueError: If repository doesn't exist
        """
        if name not in self._info:
            raise ValueError(f"Repository '{name}' does not exist")
        
        # Save first
        if name in self._stores:
            self.save(name)
        
        # Get backup directory
        backup_dir = self.workspace_path / "_backups"
        manager = BackupManager(backup_dir)
        
        info = self._info[name]
        repo_dir = self.workspace_path / name
        
        return manager.snapshot(
            source_path=repo_dir,
            repo_name=name,
            description=description,
            tags=tags,
            repo_uuid=info.uuid,
        )
    
    def restore(
        self,
        snapshot_id: str,
        target_name: str,
        overwrite: bool = False,
    ) -> RepositoryInfo:
        """
        Restore a repository from a backup snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to restore
            target_name: Name for the restored repository
            overwrite: If True, overwrite existing repository
            
        Returns:
            RepositoryInfo for the restored repository
            
        Raises:
            ValueError: If snapshot doesn't exist or target conflicts
        """
        if target_name in self._info and not overwrite:
            raise ValueError(
                f"Repository '{target_name}' already exists. "
                "Use overwrite=True to replace."
            )
        
        if not all(c.isalnum() or c in '-_' for c in target_name):
            raise ValueError("Repository name can only contain alphanumeric characters, hyphens, and underscores")
        
        # Get backup directory
        backup_dir = self.workspace_path / "_backups"
        manager = BackupManager(backup_dir)
        
        # Verify snapshot exists
        manifest = manager.get_backup(snapshot_id)
        if not manifest:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
        
        target_dir = self.workspace_path / target_name
        
        # If overwriting, delete existing first
        if target_name in self._info and overwrite:
            self.delete(target_name, force=True)
        
        # Restore files
        manager.restore(
            snapshot_id=snapshot_id,
            target_path=target_dir,
            options=RestoreOptions(
                target_name=target_name,
                overwrite=overwrite,
            ),
        )
        
        # Create new metadata (restored repo gets new identity)
        info = RepositoryInfo(
            name=target_name,
            created_at=datetime.now(timezone.utc),
            description=f"Restored from {manifest.repository_name} ({snapshot_id})",
            tags=manifest.tags.copy() if manifest.tags else [],
            uuid=str(uuid.uuid4()),  # New UUID for restored repo
            triple_count=manifest.fact_count,
        )
        
        self._info[target_name] = info
        self._save_metadata(target_name)
        
        return info
    
    def list_backups(
        self,
        repo_name: Optional[str] = None,
    ) -> List[BackupManifest]:
        """
        List available backups.
        
        Args:
            repo_name: Filter by repository name (optional)
            
        Returns:
            List of backup manifests
        """
        backup_dir = self.workspace_path / "_backups"
        if not backup_dir.exists():
            return []
        
        manager = BackupManager(backup_dir)
        return manager.list_backups(repo_name=repo_name)
    
    def delete_backup(self, snapshot_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            snapshot_id: ID of snapshot to delete
            
        Returns:
            True if deleted, False if not found
        """
        backup_dir = self.workspace_path / "_backups"
        if not backup_dir.exists():
            return False
        
        manager = BackupManager(backup_dir)
        return manager.delete_backup(snapshot_id)
