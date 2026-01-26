"""
Tests for Import Staging Workflow.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from rdf_starbase.storage.staging import (
    StagingManager,
    StagingSession,
    StagingManifest,
    StagingState,
    StagedTriple,
    ImportPreview,
    ValidationResult,
    stage_import,
    preview_import,
)
from rdf_starbase.storage.backup import BackupManager
from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext


# Sample data for testing
SAMPLE_TURTLE = """
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice foaf:name "Alice" ;
         foaf:knows ex:bob, ex:carol .

ex:bob foaf:name "Bob" ;
       foaf:age "30"^^<http://www.w3.org/2001/XMLSchema#integer> .

ex:carol foaf:name "Carol" .
"""

SAMPLE_NTRIPLES = """
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
"""

INVALID_DATA = """
This is not valid RDF data at all!
{ something: broken }
"""


class TestStagedTriple:
    """Tests for StagedTriple dataclass."""
    
    def test_create_staged_triple(self):
        """Test creating a staged triple."""
        triple = StagedTriple(
            subject="http://example.org/s",
            predicate="http://example.org/p",
            object="http://example.org/o"
        )
        assert triple.subject == "http://example.org/s"
        assert triple.predicate == "http://example.org/p"
        assert triple.object == "http://example.org/o"
        assert triple.graph is None
        assert triple.provenance is None
    
    def test_staged_triple_with_graph(self):
        """Test staged triple with named graph."""
        triple = StagedTriple(
            subject="http://example.org/s",
            predicate="http://example.org/p",
            object="http://example.org/o",
            graph="http://example.org/graph1"
        )
        assert triple.graph == "http://example.org/graph1"
    
    def test_to_tuple(self):
        """Test converting to tuple."""
        triple = StagedTriple("s", "p", "o", "g")
        assert triple.to_tuple() == ("s", "p", "o", "g")


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True)
        assert result.valid
        assert result.errors == []
        assert result.warnings == []
    
    def test_invalid_result(self):
        """Test invalid result with errors."""
        result = ValidationResult(
            valid=False,
            errors=["Missing subject", "Invalid IRI"],
            warnings=["Deprecated predicate"]
        )
        assert not result.valid
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
    
    def test_to_dict(self):
        """Test serialization."""
        result = ValidationResult(
            valid=True,
            stats={"triple_count": 100}
        )
        d = result.to_dict()
        assert d["valid"] is True
        assert d["stats"]["triple_count"] == 100


class TestStagingManifest:
    """Tests for StagingManifest dataclass."""
    
    def test_create_manifest(self):
        """Test creating a manifest."""
        now = datetime.now()
        manifest = StagingManifest(
            session_id="staging-abc123",
            repository="test-repo",
            source="data.ttl",
            format="turtle",
            state=StagingState.PENDING,
            created_at=now,
            updated_at=now,
            triple_count=100,
            graph_count=2
        )
        assert manifest.session_id == "staging-abc123"
        assert manifest.repository == "test-repo"
        assert manifest.state == StagingState.PENDING
        assert manifest.triple_count == 100
    
    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        now = datetime.now()
        manifest = StagingManifest(
            session_id="staging-abc123",
            repository="test-repo",
            source="api",
            format="turtle",
            state=StagingState.VALIDATED,
            created_at=now,
            updated_at=now,
            triple_count=50,
            validation_result=ValidationResult(valid=True)
        )
        
        d = manifest.to_dict()
        restored = StagingManifest.from_dict(d)
        
        assert restored.session_id == manifest.session_id
        assert restored.repository == manifest.repository
        assert restored.state == StagingState.VALIDATED
        assert restored.triple_count == 50
        assert restored.validation_result.valid is True


class TestStagingSession:
    """Tests for StagingSession."""
    
    @pytest.fixture
    def staging_dir(self):
        """Create temporary staging directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_create_session(self, staging_dir):
        """Test creating a staging session."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir,
            source="api",
            format="turtle"
        )
        assert session.session_id == "test-session"
        assert session.repository == "my-repo"
        assert session.manifest.state == StagingState.PENDING
        assert session.session_path.exists()
    
    def test_add_triple(self, staging_dir):
        """Test adding triples to session."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o"
        )
        
        assert session.manifest.triple_count == 1
        assert session.manifest.subject_count == 1
        assert session.manifest.predicate_count == 1
    
    def test_add_triple_with_graph(self, staging_dir):
        """Test adding triples with named graph."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
            graph="http://example.org/graph1"
        )
        
        assert session.manifest.graph_count == 1
    
    def test_preview(self, staging_dir):
        """Test preview generation."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        # Add sample triples
        for i in range(15):
            session.add_triple(
                f"http://example.org/s{i % 3}",
                f"http://example.org/p{i % 2}",
                f"value{i}"
            )
        
        preview = session.preview(sample_size=5)
        
        assert preview.triple_count == 15
        assert len(preview.sample_triples) == 5
        assert preview.subject_count == 3  # s0, s1, s2
        assert preview.predicate_count == 2  # p0, p1
        assert len(preview.top_subjects) <= 10
        assert len(preview.top_predicates) <= 10
    
    def test_validate_valid_data(self, staging_dir):
        """Test validation of valid data."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name",
            "Alice"
        )
        
        result = session.validate()
        
        assert result.valid
        assert len(result.errors) == 0
        assert session.manifest.state == StagingState.VALIDATED
    
    def test_validate_empty_values(self, staging_dir):
        """Test validation catches empty values."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple("", "http://example.org/p", "value")
        
        result = session.validate()
        
        assert not result.valid
        assert any("Empty subject" in e for e in result.errors)
        assert session.manifest.state == StagingState.FAILED
    
    def test_validate_detects_duplicates(self, staging_dir):
        """Test validation detects duplicate triples."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        # Add same triple twice
        for _ in range(2):
            session.add_triple(
                "http://example.org/s",
                "http://example.org/p",
                "value"
            )
        
        result = session.validate()
        
        # Duplicates are warnings, not errors
        assert result.valid
        assert any("duplicate" in w.lower() for w in result.warnings)
    
    def test_custom_validator(self, staging_dir):
        """Test custom validator integration."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "value"
        )
        
        def custom_validator(triples):
            return ValidationResult(
                valid=False,
                errors=["Custom validation failed"]
            )
        
        result = session.validate(validators=[custom_validator])
        
        assert not result.valid
        assert "Custom validation failed" in result.errors
    
    def test_iterate_triples(self, staging_dir):
        """Test iterating over staged triples."""
        session = StagingSession(
            session_id="test-session",
            repository="my-repo",
            staging_dir=staging_dir
        )
        
        session.add_triple("s1", "p1", "o1")
        session.add_triple("s2", "p2", "o2")
        
        triples = list(session.get_triples())
        assert len(triples) == 2


class TestStagingManager:
    """Tests for StagingManager."""
    
    @pytest.fixture
    def staging_manager(self):
        """Create a staging manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StagingManager(Path(tmpdir))
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        store = TripleStore()
        yield store
    
    def test_stage_turtle_data(self, staging_manager):
        """Test staging Turtle data."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_TURTLE,
            format="turtle"
        )
        
        assert session.manifest.triple_count > 0
        assert session.manifest.state == StagingState.PENDING
        assert "ex" in session._namespaces or "foaf" in session._namespaces
    
    def test_stage_ntriples_data(self, staging_manager):
        """Test staging N-Triples data."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        assert session.manifest.triple_count == 3
    
    def test_get_session(self, staging_manager):
        """Test retrieving a session."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_TURTLE,
            format="turtle"
        )
        
        retrieved = staging_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
    
    def test_list_sessions(self, staging_manager):
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            staging_manager.stage_data(
                repository=f"repo-{i}",
                data=SAMPLE_NTRIPLES,
                format="ntriples"
            )
        
        sessions = staging_manager.list_sessions()
        assert len(sessions) == 3
    
    def test_list_sessions_by_repository(self, staging_manager):
        """Test filtering sessions by repository."""
        staging_manager.stage_data(repository="repo-a", data=SAMPLE_NTRIPLES, format="ntriples")
        staging_manager.stage_data(repository="repo-b", data=SAMPLE_NTRIPLES, format="ntriples")
        staging_manager.stage_data(repository="repo-a", data=SAMPLE_NTRIPLES, format="ntriples")
        
        sessions_a = staging_manager.list_sessions(repository="repo-a")
        sessions_b = staging_manager.list_sessions(repository="repo-b")
        
        assert len(sessions_a) == 2
        assert len(sessions_b) == 1
    
    def test_list_sessions_by_state(self, staging_manager):
        """Test filtering sessions by state."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        session.validate()
        
        pending = staging_manager.list_sessions(state=StagingState.PENDING)
        validated = staging_manager.list_sessions(state=StagingState.VALIDATED)
        
        # After validate, state becomes VALIDATED (not PENDING)
        assert len(pending) == 0
        assert len(validated) == 1
    
    def test_delete_session(self, staging_manager):
        """Test deleting a session."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_TURTLE,
            format="turtle"
        )
        
        assert staging_manager.delete_session(session.session_id)
        assert staging_manager.get_session(session.session_id) is None
    
    def test_delete_nonexistent_session(self, staging_manager):
        """Test deleting nonexistent session."""
        assert not staging_manager.delete_session("nonexistent-session")


class TestStagingCommit:
    """Tests for committing staged data."""
    
    @pytest.fixture
    def staging_manager(self):
        """Create staging manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StagingManager(Path(tmpdir))
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary store."""
        store = TripleStore()
        yield store
    
    def test_commit_to_store(self, staging_manager, temp_store):
        """Test committing staged data to store."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        result = session.commit(temp_store, skip_validation=True)
        
        assert result["committed"]
        assert result["triples_added"] == 3
        assert session.manifest.state == StagingState.COMMITTED
        
        # Verify data is in store
        triples = temp_store.get_triples(
            subject="http://example.org/alice"
        )
        assert len(triples) >= 1
    
    def test_commit_validates_first(self, staging_manager, temp_store):
        """Test that commit validates if not already validated."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        result = session.commit(temp_store)  # skip_validation=False by default
        
        assert result["committed"]
        assert session.manifest.validation_result is not None
        assert session.manifest.validation_result.valid
    
    def test_commit_fails_on_invalid_data(self, staging_manager, temp_store):
        """Test commit fails if validation fails."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        # Add invalid triple
        session.add_triple("", "", "")
        
        with pytest.raises(ValueError, match="validation failed"):
            session.commit(temp_store)
    
    def test_dry_run(self, staging_manager, temp_store):
        """Test dry run analysis."""
        session = staging_manager.stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        result = session.dry_run(temp_store)
        
        assert result["dry_run"]
        assert result["total_triples"] == 3
        # All triples are new since store is empty
        assert result["new_triples"] == 3
        assert result["existing_triples"] == 0


class TestStagingWithBackup:
    """Tests for staging with backup integration."""
    
    @pytest.fixture
    def setup(self):
        """Create staging manager and store with backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            staging_mgr = StagingManager(tmpdir / "staging")
            backup_mgr = BackupManager(tmpdir / "backups")
            store = TripleStore()
            
            yield {
                "staging": staging_mgr,
                "backup": backup_mgr,
                "store": store,
                "workspace": tmpdir
            }
    
    def test_commit_without_backup(self, setup):
        """Test that commit works without backup manager."""
        # Add initial data
        setup["store"].add_triple(
            "http://example.org/existing",
            "http://example.org/p",
            "initial",
            ProvenanceContext(source="test")
        )
        
        session = setup["staging"].stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        # Commit without backup manager
        result = session.commit(setup["store"])
        
        assert result["committed"]
        assert result["can_undo"] is False
        assert result["snapshot_id"] is None
    
    def test_commit_state_transition(self, setup):
        """Test that commit changes state correctly."""
        session = setup["staging"].stage_data(
            repository="test-repo",
            data=SAMPLE_NTRIPLES,
            format="ntriples"
        )
        
        assert session.manifest.state == StagingState.PENDING
        
        session.commit(setup["store"])
        
        assert session.manifest.state == StagingState.COMMITTED


class TestStagingFile:
    """Tests for staging files."""
    
    @pytest.fixture
    def staging_manager(self):
        """Create staging manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StagingManager(Path(tmpdir))
    
    def test_stage_turtle_file(self, staging_manager):
        """Test staging a Turtle file."""
        with tempfile.NamedTemporaryFile(suffix=".ttl", mode="w", delete=False) as f:
            f.write(SAMPLE_TURTLE)
            f.flush()
            
            session = staging_manager.stage_file(
                repository="test-repo",
                file_path=f.name
            )
            
            assert session.manifest.triple_count > 0
            assert session.format == "turtle"
    
    def test_stage_file_autodetect_format(self, staging_manager):
        """Test format auto-detection from extension."""
        with tempfile.NamedTemporaryFile(suffix=".nt", mode="w", delete=False) as f:
            f.write(SAMPLE_NTRIPLES)
            f.flush()
            
            session = staging_manager.stage_file(
                repository="test-repo",
                file_path=f.name
            )
            
            assert session.format == "ntriples"
    
    def test_stage_file_not_found(self, staging_manager):
        """Test staging nonexistent file."""
        with pytest.raises(FileNotFoundError):
            staging_manager.stage_file(
                repository="test-repo",
                file_path="/nonexistent/file.ttl"
            )


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_stage_import(self):
        """Test stage_import convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = stage_import(
                staging_dir=tmpdir,
                repository="test-repo",
                data=SAMPLE_NTRIPLES,
                format="ntriples"
            )
            
            assert session.manifest.triple_count == 3
    
    def test_preview_import(self):
        """Test preview_import convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = stage_import(
                staging_dir=tmpdir,
                repository="test-repo",
                data=SAMPLE_NTRIPLES,
                format="ntriples"
            )
            
            preview = preview_import(session, sample_size=2)
            
            assert preview.triple_count == 3
            assert len(preview.sample_triples) == 2


class TestImportPreview:
    """Tests for ImportPreview."""
    
    def test_to_dict(self):
        """Test preview serialization."""
        preview = ImportPreview(
            session_id="test-session",
            triple_count=100,
            graph_count=2,
            subject_count=50,
            predicate_count=10,
            sample_triples=[
                StagedTriple("s", "p", "o")
            ],
            graphs=["g1", "g2"],
            top_subjects=[("s1", 20), ("s2", 15)],
            top_predicates=[("p1", 50)],
            namespaces={"ex": "http://example.org/"},
            format_detected="turtle",
            size_bytes=1024
        )
        
        d = preview.to_dict()
        
        assert d["triple_count"] == 100
        assert d["graph_count"] == 2
        assert len(d["sample_triples"]) == 1
        assert len(d["graphs"]) == 2
        assert d["namespaces"]["ex"] == "http://example.org/"
