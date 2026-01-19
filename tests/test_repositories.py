"""
Tests for the RepositoryManager.
"""

import pytest
import tempfile
from pathlib import Path
import shutil

from rdf_starbase.repositories import RepositoryManager, RepositoryInfo


class TestRepositoryManager:
    """Test RepositoryManager functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        path = tempfile.mkdtemp(prefix="rdf_starbase_test_")
        yield Path(path)
        shutil.rmtree(path, ignore_errors=True)
    
    @pytest.fixture
    def manager(self, temp_workspace):
        """Create a RepositoryManager with a temp workspace."""
        return RepositoryManager(temp_workspace)
    
    def test_create_repository(self, manager):
        """Test creating a new repository."""
        info = manager.create("test-repo", description="Test repository", tags=["test"])
        
        assert info.name == "test-repo"
        assert info.description == "Test repository"
        assert "test" in info.tags
        assert manager.exists("test-repo")
    
    def test_create_duplicate_fails(self, manager):
        """Test that creating a duplicate repository fails."""
        manager.create("test-repo")
        
        with pytest.raises(ValueError, match="already exists"):
            manager.create("test-repo")
    
    def test_create_invalid_name_fails(self, manager):
        """Test that invalid repository names fail."""
        with pytest.raises(ValueError, match="empty"):
            manager.create("")
        
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.create("test repo")  # spaces not allowed
        
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.create("test/repo")  # slashes not allowed
    
    def test_get_store(self, manager):
        """Test getting a store for a repository."""
        manager.create("test-repo")
        
        store = manager.get_store("test-repo")
        assert store is not None
        
        # Same store instance returned on second call
        store2 = manager.get_store("test-repo")
        assert store is store2
    
    def test_get_nonexistent_store_fails(self, manager):
        """Test that getting a non-existent repository fails."""
        with pytest.raises(ValueError, match="does not exist"):
            manager.get_store("nonexistent")
    
    def test_list_repositories(self, manager):
        """Test listing all repositories."""
        assert len(manager.list_repositories()) == 0
        
        manager.create("repo-1", description="First")
        manager.create("repo-2", description="Second")
        manager.create("repo-3", description="Third")
        
        repos = manager.list_repositories()
        assert len(repos) == 3
        names = [r.name for r in repos]
        assert "repo-1" in names
        assert "repo-2" in names
        assert "repo-3" in names
    
    def test_delete_empty_repository(self, manager):
        """Test deleting an empty repository."""
        manager.create("test-repo")
        assert manager.exists("test-repo")
        
        manager.delete("test-repo")
        assert not manager.exists("test-repo")
    
    def test_delete_repository_with_data_requires_force(self, manager):
        """Test that deleting a repository with data requires force."""
        manager.create("test-repo")
        store = manager.get_store("test-repo")
        
        # Add some data
        from rdf_starbase.models import ProvenanceContext
        store.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
            ProvenanceContext(source="test")
        )
        
        with pytest.raises(ValueError, match="contains"):
            manager.delete("test-repo")
        
        # Should work with force
        manager.delete("test-repo", force=True)
        assert not manager.exists("test-repo")
    
    def test_save_and_reload(self, temp_workspace):
        """Test saving a repository and reloading it."""
        # Create and populate
        manager1 = RepositoryManager(temp_workspace)
        manager1.create("test-repo", description="Test")
        store = manager1.get_store("test-repo")
        
        from rdf_starbase.models import ProvenanceContext
        store.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
            ProvenanceContext(source="test")
        )
        manager1.save("test-repo")
        
        # Create new manager pointing to same workspace
        manager2 = RepositoryManager(temp_workspace)
        
        # Should see the repository
        assert manager2.exists("test-repo")
        
        info = manager2.get_info("test-repo")
        assert info.description == "Test"
        
        # Store should have the data
        store2 = manager2.get_store("test-repo")
        assert store2.stats()["total_assertions"] == 1
    
    def test_rename_repository(self, manager):
        """Test renaming a repository."""
        manager.create("old-name", description="Test repo")
        store = manager.get_store("old-name")
        
        from rdf_starbase.models import ProvenanceContext
        store.add_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
            ProvenanceContext(source="test")
        )
        
        # Rename
        info = manager.rename("old-name", "new-name")
        
        assert info.name == "new-name"
        assert info.description == "Test repo"
        
        assert not manager.exists("old-name")
        assert manager.exists("new-name")
        
        # Data should still be there
        store2 = manager.get_store("new-name")
        assert store2.stats()["total_assertions"] == 1
    
    def test_update_info(self, manager):
        """Test updating repository metadata."""
        manager.create("test-repo", description="Original", tags=["tag1"])
        
        info = manager.update_info(
            "test-repo",
            description="Updated",
            tags=["tag1", "tag2"]
        )
        
        assert info.description == "Updated"
        assert info.tags == ["tag1", "tag2"]
    
    def test_stats_update_on_get_info(self, manager):
        """Test that stats are updated when getting info."""
        manager.create("test-repo")
        store = manager.get_store("test-repo")
        
        from rdf_starbase.models import ProvenanceContext
        store.add_triple(
            "http://example.org/s1",
            "http://example.org/p",
            "http://example.org/o1",
            ProvenanceContext(source="test")
        )
        store.add_triple(
            "http://example.org/s2",
            "http://example.org/p",
            "http://example.org/o2",
            ProvenanceContext(source="test")
        )
        
        info = manager.get_info("test-repo")
        assert info.triple_count == 2


class TestRepositoryAPI:
    """Test the repository REST API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        from fastapi.testclient import TestClient
        import tempfile
        import os
        
        # Use temp directory for test repositories
        temp_dir = tempfile.mkdtemp(prefix="rdf_api_test_")
        os.environ["RDFSTARBASE_REPOSITORY_PATH"] = temp_dir
        
        from rdf_starbase.web import create_app
        app = create_app()
        
        yield TestClient(app)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_list_repositories_empty(self, client):
        """Test listing repositories when empty."""
        response = client.get("/repositories")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["repositories"] == []
    
    def test_create_repository(self, client):
        """Test creating a repository via API."""
        response = client.post("/repositories", json={
            "name": "test-repo",
            "description": "Test repository",
            "tags": ["test"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["repository"]["name"] == "test-repo"
    
    def test_create_duplicate_fails(self, client):
        """Test that creating a duplicate repository fails."""
        client.post("/repositories", json={"name": "test-repo"})
        
        response = client.post("/repositories", json={"name": "test-repo"})
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_get_repository(self, client):
        """Test getting repository info."""
        client.post("/repositories", json={
            "name": "test-repo",
            "description": "Test"
        })
        
        response = client.get("/repositories/test-repo")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-repo"
        assert data["description"] == "Test"
    
    def test_get_nonexistent_repository_fails(self, client):
        """Test that getting a non-existent repository fails."""
        response = client.get("/repositories/nonexistent")
        assert response.status_code == 404
    
    def test_delete_repository(self, client):
        """Test deleting a repository."""
        client.post("/repositories", json={"name": "test-repo"})
        
        response = client.delete("/repositories/test-repo")
        assert response.status_code == 200
        
        # Should no longer exist
        response = client.get("/repositories/test-repo")
        assert response.status_code == 404
    
    def test_repository_sparql(self, client):
        """Test executing SPARQL against a repository."""
        # Create repository
        client.post("/repositories", json={"name": "test-repo"})
        
        # Insert data
        response = client.post("/repositories/test-repo/sparql", json={
            "query": """
                INSERT DATA {
                    <http://example.org/s> <http://example.org/p> "value" .
                }
            """
        })
        assert response.status_code == 200
        assert response.json()["type"] == "update"
        
        # Query data
        response = client.post("/repositories/test-repo/sparql", json={
            "query": """
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o
                }
            """
        })
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "select"
        assert data["count"] == 1
    
    def test_update_repository_metadata(self, client):
        """Test updating repository metadata."""
        client.post("/repositories", json={
            "name": "test-repo",
            "description": "Original"
        })
        
        response = client.patch("/repositories/test-repo", json={
            "description": "Updated"
        })
        assert response.status_code == 200
        assert response.json()["description"] == "Updated"
    
    def test_repository_stats(self, client):
        """Test getting repository stats."""
        client.post("/repositories", json={"name": "test-repo"})
        
        response = client.get("/repositories/test-repo/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-repo"
        assert "stats" in data
