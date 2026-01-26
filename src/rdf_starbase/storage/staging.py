"""
Import Staging Workflow for RDF-StarBase.

Provides a safe workflow for importing data with:
- Preview: Parse and inspect data before committing
- Validation: Check against constraints/SHACL shapes  
- Dry-run: Execute import without committing
- Undo: Rollback recent imports

This prevents data corruption and allows review before commit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from rdf_starbase.store import TripleStore

logger = logging.getLogger(__name__)


class StagingState(Enum):
    """State of a staging session."""
    PENDING = "pending"      # Data parsed, awaiting review
    VALIDATED = "validated"  # Passed validation
    COMMITTED = "committed"  # Import completed
    ROLLED_BACK = "rolled_back"  # Import undone
    FAILED = "failed"        # Import or validation failed
    EXPIRED = "expired"      # Session timed out


@dataclass
class ValidationResult:
    """Result of validating staged data."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats
        }


@dataclass 
class StagedTriple:
    """A triple staged for import."""
    subject: str
    predicate: str
    object: str
    graph: str | None = None
    provenance: dict[str, Any] | None = None
    
    def to_tuple(self) -> tuple:
        return (self.subject, self.predicate, self.object, self.graph)


@dataclass
class StagingManifest:
    """Manifest for a staging session."""
    session_id: str
    repository: str
    source: str  # File path or "api" or "stream"
    format: str  # turtle, ntriples, jsonld, etc
    state: StagingState
    created_at: datetime
    updated_at: datetime
    triple_count: int = 0
    graph_count: int = 0
    subject_count: int = 0
    predicate_count: int = 0
    validation_result: ValidationResult | None = None
    commit_snapshot_id: str | None = None  # For undo
    error_message: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "repository": self.repository,
            "source": self.source,
            "format": self.format,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "triple_count": self.triple_count,
            "graph_count": self.graph_count,
            "subject_count": self.subject_count,
            "predicate_count": self.predicate_count,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "commit_snapshot_id": self.commit_snapshot_id,
            "error_message": self.error_message,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> StagingManifest:
        validation = None
        if data.get("validation_result"):
            vr = data["validation_result"]
            validation = ValidationResult(
                valid=vr["valid"],
                errors=vr.get("errors", []),
                warnings=vr.get("warnings", []),
                stats=vr.get("stats", {})
            )
        
        return cls(
            session_id=data["session_id"],
            repository=data["repository"],
            source=data["source"],
            format=data["format"],
            state=StagingState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            triple_count=data.get("triple_count", 0),
            graph_count=data.get("graph_count", 0),
            subject_count=data.get("subject_count", 0),
            predicate_count=data.get("predicate_count", 0),
            validation_result=validation,
            commit_snapshot_id=data.get("commit_snapshot_id"),
            error_message=data.get("error_message"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {})
        )


@dataclass
class ImportPreview:
    """Preview of staged import data."""
    session_id: str
    triple_count: int
    graph_count: int
    subject_count: int
    predicate_count: int
    sample_triples: list[StagedTriple]
    graphs: list[str]
    top_subjects: list[tuple[str, int]]  # (subject, count)
    top_predicates: list[tuple[str, int]]  # (predicate, count)
    namespaces: dict[str, str]
    format_detected: str
    size_bytes: int
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "triple_count": self.triple_count,
            "graph_count": self.graph_count,
            "subject_count": self.subject_count,
            "predicate_count": self.predicate_count,
            "sample_triples": [
                {"s": t.subject, "p": t.predicate, "o": t.object, "g": t.graph}
                for t in self.sample_triples
            ],
            "graphs": self.graphs,
            "top_subjects": [{"subject": s, "count": c} for s, c in self.top_subjects],
            "top_predicates": [{"predicate": p, "count": c} for p, c in self.top_predicates],
            "namespaces": self.namespaces,
            "format_detected": self.format_detected,
            "size_bytes": self.size_bytes
        }


class StagingSession:
    """
    An import staging session that holds data for preview/validation.
    
    Workflow:
    1. stage_file() or stage_data() - Parse and stage data
    2. preview() - Get statistics and sample triples
    3. validate() - Check constraints (optional)
    4. commit() - Apply to store (creates undo snapshot)
    5. rollback() - Undo committed import (if within window)
    """
    
    def __init__(
        self,
        session_id: str,
        repository: str,
        staging_dir: Path,
        source: str = "api",
        format: str = "turtle",
        ttl_seconds: int = 3600  # 1 hour default
    ):
        self.session_id = session_id
        self.repository = repository
        self.staging_dir = staging_dir
        self.source = source
        self.format = format
        
        # Storage for staged triples
        self._triples: list[StagedTriple] = []
        self._subjects: set[str] = set()
        self._predicates: set[str] = set()
        self._graphs: set[str] = set()
        self._namespaces: dict[str, str] = {}
        self._size_bytes = 0
        
        now = datetime.now()
        self.manifest = StagingManifest(
            session_id=session_id,
            repository=repository,
            source=source,
            format=format,
            state=StagingState.PENDING,
            created_at=now,
            updated_at=now,
            expires_at=datetime.fromtimestamp(now.timestamp() + ttl_seconds)
        )
        
        # Create session directory
        self.session_path = staging_dir / session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        graph: str | None = None,
        provenance: dict | None = None
    ) -> None:
        """Add a triple to the staging area."""
        triple = StagedTriple(subject, predicate, obj, graph, provenance)
        self._triples.append(triple)
        self._subjects.add(subject)
        self._predicates.add(predicate)
        if graph:
            self._graphs.add(graph)
        
        # Update stats
        self.manifest.triple_count = len(self._triples)
        self.manifest.subject_count = len(self._subjects)
        self.manifest.predicate_count = len(self._predicates)
        self.manifest.graph_count = len(self._graphs)
        self.manifest.updated_at = datetime.now()
    
    def add_namespace(self, prefix: str, uri: str) -> None:
        """Register a namespace prefix."""
        self._namespaces[prefix] = uri
    
    def set_size(self, size_bytes: int) -> None:
        """Set the source data size."""
        self._size_bytes = size_bytes
    
    def preview(self, sample_size: int = 10) -> ImportPreview:
        """Generate a preview of the staged data."""
        # Get sample triples
        sample = self._triples[:sample_size]
        
        # Count subjects and predicates
        subject_counts: dict[str, int] = {}
        predicate_counts: dict[str, int] = {}
        
        for triple in self._triples:
            subject_counts[triple.subject] = subject_counts.get(triple.subject, 0) + 1
            predicate_counts[triple.predicate] = predicate_counts.get(triple.predicate, 0) + 1
        
        # Get top 10
        top_subjects = sorted(subject_counts.items(), key=lambda x: -x[1])[:10]
        top_predicates = sorted(predicate_counts.items(), key=lambda x: -x[1])[:10]
        
        return ImportPreview(
            session_id=self.session_id,
            triple_count=len(self._triples),
            graph_count=len(self._graphs),
            subject_count=len(self._subjects),
            predicate_count=len(self._predicates),
            sample_triples=sample,
            graphs=list(self._graphs),
            top_subjects=top_subjects,
            top_predicates=top_predicates,
            namespaces=self._namespaces.copy(),
            format_detected=self.format,
            size_bytes=self._size_bytes
        )
    
    def validate(
        self,
        validators: list[Callable[[list[StagedTriple]], ValidationResult]] | None = None
    ) -> ValidationResult:
        """
        Validate staged data against constraints.
        
        Default validation checks:
        - No empty subjects/predicates/objects
        - Valid IRI format (basic check)
        - No duplicate triples
        
        Custom validators can be passed for SHACL, etc.
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        # Basic validation
        seen_triples: set[tuple] = set()
        duplicate_count = 0
        
        for i, triple in enumerate(self._triples):
            # Check for empty values
            if not triple.subject:
                errors.append(f"Triple {i}: Empty subject")
            if not triple.predicate:
                errors.append(f"Triple {i}: Empty predicate")
            if not triple.object:
                errors.append(f"Triple {i}: Empty object")
            
            # Check for basic IRI validity
            for name, value in [("subject", triple.subject), ("predicate", triple.predicate)]:
                if value and not value.startswith(("http://", "https://", "_:", "urn:")):
                    if not value.startswith("<") and "://" not in value:
                        warnings.append(f"Triple {i}: {name} '{value[:50]}' may not be a valid IRI")
            
            # Check duplicates
            key = triple.to_tuple()
            if key in seen_triples:
                duplicate_count += 1
            else:
                seen_triples.add(key)
        
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate triples")
        
        # Run custom validators
        if validators:
            for validator in validators:
                try:
                    result = validator(self._triples)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                except Exception as e:
                    errors.append(f"Validator error: {e}")
        
        # Build result
        valid = len(errors) == 0
        result = ValidationResult(
            valid=valid,
            errors=errors[:100],  # Limit errors
            warnings=warnings[:100],  # Limit warnings
            stats={
                "total_triples": len(self._triples),
                "unique_triples": len(seen_triples),
                "duplicate_count": duplicate_count,
                "unique_subjects": len(self._subjects),
                "unique_predicates": len(self._predicates),
                "graphs": len(self._graphs)
            }
        )
        
        self.manifest.validation_result = result
        self.manifest.state = StagingState.VALIDATED if valid else StagingState.FAILED
        self.manifest.updated_at = datetime.now()
        self._save_manifest()
        
        return result
    
    def dry_run(self, store: TripleStore) -> dict[str, Any]:
        """
        Simulate the import without committing.
        
        Returns statistics about what would happen.
        """
        # Count how many triples already exist
        existing_count = 0
        new_count = 0
        
        for triple in self._triples:
            # Check if triple exists using get_triples
            results = store.get_triples(
                subject=triple.subject,
                predicate=triple.predicate,
                obj=triple.object,
                graph=triple.graph
            )
            if len(results) > 0:
                existing_count += 1
            else:
                new_count += 1
        
        return {
            "session_id": self.session_id,
            "dry_run": True,
            "total_triples": len(self._triples),
            "new_triples": new_count,
            "existing_triples": existing_count,
            "graphs_affected": list(self._graphs) if self._graphs else ["default"],
            "estimated_time_ms": len(self._triples) * 0.1  # Rough estimate
        }
    
    def commit(
        self,
        store: TripleStore,
        backup_manager: Any | None = None,
        skip_validation: bool = False
    ) -> dict[str, Any]:
        """
        Commit staged data to the store.
        
        If backup_manager is provided, creates a snapshot for undo.
        """
        from rdf_starbase.models import ProvenanceContext
        
        if not skip_validation and self.manifest.state == StagingState.PENDING:
            self.validate()
        
        if self.manifest.validation_result and not self.manifest.validation_result.valid:
            raise ValueError(
                f"Cannot commit: validation failed with {len(self.manifest.validation_result.errors)} errors"
            )
        
        # Create pre-commit snapshot for undo
        if backup_manager:
            try:
                snapshot = backup_manager.snapshot(
                    store,
                    description=f"Pre-import snapshot for session {self.session_id}"
                )
                self.manifest.commit_snapshot_id = snapshot.snapshot_id
            except Exception as e:
                logger.warning(f"Failed to create pre-commit snapshot: {e}")
        
        # Commit triples
        start_time = time.time()
        added_count = 0
        
        try:
            for triple in self._triples:
                prov = ProvenanceContext(
                    source=f"staging:{self.session_id}",
                    confidence=1.0
                )
                if triple.provenance:
                    if "source" in triple.provenance:
                        prov = ProvenanceContext(
                            source=triple.provenance["source"],
                            confidence=triple.provenance.get("confidence", 1.0)
                        )
                
                store.add_triple(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    prov,
                    graph=triple.graph
                )
                added_count += 1
            
            self.manifest.state = StagingState.COMMITTED
            self.manifest.updated_at = datetime.now()
            self._save_manifest()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "session_id": self.session_id,
                "committed": True,
                "triples_added": added_count,
                "elapsed_ms": elapsed_ms,
                "snapshot_id": self.manifest.commit_snapshot_id,
                "can_undo": self.manifest.commit_snapshot_id is not None
            }
            
        except Exception as e:
            self.manifest.state = StagingState.FAILED
            self.manifest.error_message = str(e)
            self.manifest.updated_at = datetime.now()
            self._save_manifest()
            raise
    
    def rollback(self, store: TripleStore, backup_manager: Any) -> dict[str, Any]:
        """
        Rollback a committed import using the pre-commit snapshot.
        """
        if self.manifest.state != StagingState.COMMITTED:
            raise ValueError(f"Cannot rollback: session is in state {self.manifest.state.value}")
        
        if not self.manifest.commit_snapshot_id:
            raise ValueError("Cannot rollback: no pre-commit snapshot available")
        
        # Restore from snapshot
        backup_manager.restore(self.manifest.commit_snapshot_id, store)
        
        self.manifest.state = StagingState.ROLLED_BACK
        self.manifest.updated_at = datetime.now()
        self._save_manifest()
        
        return {
            "session_id": self.session_id,
            "rolled_back": True,
            "restored_snapshot": self.manifest.commit_snapshot_id
        }
    
    def get_triples(self) -> Iterator[StagedTriple]:
        """Iterate over staged triples."""
        return iter(self._triples)
    
    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        manifest_path = self.session_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
    
    def _save_triples(self) -> None:
        """Save staged triples to disk."""
        triples_path = self.session_path / "triples.jsonl"
        with open(triples_path, "w") as f:
            for triple in self._triples:
                line = json.dumps({
                    "s": triple.subject,
                    "p": triple.predicate,
                    "o": triple.object,
                    "g": triple.graph,
                    "prov": triple.provenance
                })
                f.write(line + "\n")
    
    def cleanup(self) -> None:
        """Remove session data from disk."""
        if self.session_path.exists():
            shutil.rmtree(self.session_path)


class StagingManager:
    """
    Manages staging sessions for safe data import.
    
    Usage:
        manager = StagingManager(workspace_path / "_staging")
        
        # Stage a file
        session = manager.stage_file(
            repository="my-repo",
            file_path="data.ttl",
            format="turtle"
        )
        
        # Preview what will be imported
        preview = session.preview()
        print(f"Will import {preview.triple_count} triples")
        
        # Validate
        result = session.validate()
        if not result.valid:
            print(f"Errors: {result.errors}")
            return
        
        # Commit
        store = repo_manager.get_store("my-repo")
        session.commit(store, backup_manager=backup_mgr)
    """
    
    def __init__(self, staging_dir: Path | str, session_ttl: int = 3600):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.session_ttl = session_ttl
        self._sessions: dict[str, StagingSession] = {}
    
    def stage_file(
        self,
        repository: str,
        file_path: str | Path,
        format: str | None = None
    ) -> StagingSession:
        """
        Stage data from a file for import.
        
        Auto-detects format from extension if not specified.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format from extension
        if format is None:
            ext_map = {
                ".ttl": "turtle",
                ".turtle": "turtle",
                ".nt": "ntriples",
                ".ntriples": "ntriples",
                ".nq": "nquads",
                ".nquads": "nquads",
                ".jsonld": "jsonld",
                ".json": "jsonld",
                ".rdf": "rdfxml",
                ".xml": "rdfxml",
                ".trig": "trig",
                ".trix": "trix",
                ".n3": "n3"
            }
            format = ext_map.get(file_path.suffix.lower(), "turtle")
        
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.stage_data(
            repository=repository,
            data=content,
            format=format,
            source=str(file_path)
        )
    
    def stage_data(
        self,
        repository: str,
        data: str,
        format: str = "turtle",
        source: str = "api"
    ) -> StagingSession:
        """
        Stage RDF data for import.
        
        Parses the data and stores it in a staging session.
        """
        from rdf_starbase.formats import turtle, ntriples, jsonld
        
        session_id = f"staging-{uuid.uuid4().hex[:12]}"
        session = StagingSession(
            session_id=session_id,
            repository=repository,
            staging_dir=self.staging_dir,
            source=source,
            format=format,
            ttl_seconds=self.session_ttl
        )
        session.set_size(len(data.encode("utf-8")))
        
        # Parse data based on format
        try:
            if format == "turtle":
                self._parse_turtle(data, session)
            elif format == "ntriples":
                self._parse_ntriples(data, session)
            elif format == "jsonld":
                self._parse_jsonld(data, session)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._sessions[session_id] = session
            session._save_manifest()
            session._save_triples()
            
            logger.info(
                f"Staged {session.manifest.triple_count} triples in session {session_id}"
            )
            
            return session
            
        except Exception as e:
            session.manifest.state = StagingState.FAILED
            session.manifest.error_message = str(e)
            session._save_manifest()
            raise
    
    def _parse_turtle(self, data: str, session: StagingSession) -> None:
        """Parse Turtle data into staging session."""
        from rdf_starbase.formats.turtle import TurtleParser
        
        parser = TurtleParser()
        parser.parse(data)
        
        # Add namespaces
        for prefix, uri in parser.prefixes.items():
            session.add_namespace(prefix, uri)
        
        # Add triples (Triple is a dataclass with subject, predicate, object attrs)
        for triple in parser.triples:
            session.add_triple(triple.subject, triple.predicate, triple.object)
    
    def _parse_ntriples(self, data: str, session: StagingSession) -> None:
        """Parse N-Triples data into staging session."""
        from rdf_starbase.formats.ntriples import NTriplesParser
        
        parser = NTriplesParser()
        doc = parser.parse(data)
        
        # Add triples (Triple is a dataclass with subject, predicate, object attrs)
        for triple in doc.triples:
            session.add_triple(triple.subject, triple.predicate, triple.object)
    
    def _parse_jsonld(self, data: str, session: StagingSession) -> None:
        """Parse JSON-LD data into staging session."""
        from rdf_starbase.formats.jsonld import JSONLDParser
        
        parser = JSONLDParser()
        parser.parse(data)
        
        # Add triples (Triple is a dataclass with subject, predicate, object attrs)
        for triple in parser.triples:
            session.add_triple(triple.subject, triple.predicate, triple.object)
    
    def get_session(self, session_id: str) -> StagingSession | None:
        """Get a staging session by ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Try to load from disk
        session_path = self.staging_dir / session_id
        manifest_path = session_path / "manifest.json"
        
        if manifest_path.exists():
            return self._load_session(session_id)
        
        return None
    
    def _load_session(self, session_id: str) -> StagingSession:
        """Load a session from disk."""
        session_path = self.staging_dir / session_id
        manifest_path = session_path / "manifest.json"
        triples_path = session_path / "triples.jsonl"
        
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        
        manifest = StagingManifest.from_dict(manifest_data)
        
        session = StagingSession(
            session_id=session_id,
            repository=manifest.repository,
            staging_dir=self.staging_dir,
            source=manifest.source,
            format=manifest.format,
            ttl_seconds=self.session_ttl
        )
        session.manifest = manifest
        
        # Load triples
        if triples_path.exists():
            with open(triples_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    session.add_triple(
                        data["s"],
                        data["p"],
                        data["o"],
                        data.get("g"),
                        data.get("prov")
                    )
        
        self._sessions[session_id] = session
        return session
    
    def list_sessions(
        self,
        repository: str | None = None,
        state: StagingState | None = None
    ) -> list[StagingManifest]:
        """List all staging sessions, optionally filtered."""
        sessions = []
        
        for entry in self.staging_dir.iterdir():
            if entry.is_dir():
                manifest_path = entry / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = StagingManifest.from_dict(json.load(f))
                        
                        # Apply filters
                        if repository and manifest.repository != repository:
                            continue
                        if state and manifest.state != state:
                            continue
                        
                        # Check expiration
                        if manifest.expires_at and datetime.now() > manifest.expires_at:
                            if manifest.state == StagingState.PENDING:
                                manifest.state = StagingState.EXPIRED
                        
                        sessions.append(manifest)
                    except Exception as e:
                        logger.warning(f"Failed to load session {entry.name}: {e}")
        
        return sorted(sessions, key=lambda m: m.created_at, reverse=True)
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of sessions removed."""
        removed = 0
        now = datetime.now()
        
        for entry in list(self.staging_dir.iterdir()):
            if entry.is_dir():
                manifest_path = entry / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = StagingManifest.from_dict(json.load(f))
                        
                        if manifest.expires_at and now > manifest.expires_at:
                            if manifest.state in (StagingState.PENDING, StagingState.VALIDATED):
                                shutil.rmtree(entry)
                                removed += 1
                                logger.info(f"Cleaned up expired session {entry.name}")
                    except Exception as e:
                        logger.warning(f"Failed to check session {entry.name}: {e}")
        
        return removed
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a staging session."""
        session_path = self.staging_dir / session_id
        
        if session_path.exists():
            shutil.rmtree(session_path)
            if session_id in self._sessions:
                del self._sessions[session_id]
            return True
        
        return False


# Convenience functions

def stage_import(
    staging_dir: Path | str,
    repository: str,
    data: str,
    format: str = "turtle"
) -> StagingSession:
    """Quick way to stage data for import."""
    manager = StagingManager(staging_dir)
    return manager.stage_data(repository, data, format)


def preview_import(session: StagingSession, sample_size: int = 10) -> ImportPreview:
    """Get a preview of staged data."""
    return session.preview(sample_size)
