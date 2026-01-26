"""
Tests for named graph loading options.

Tests the graph_target parameter for import/upload endpoints:
- 'default': Load into default graph
- 'named:<uri>': Load into specific named graph
- 'auto': Auto-generate graph from filename
- Direct URI: Use as named graph
"""

import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import FastAPI

from rdf_starbase.repository_api import create_repository_router


@pytest.fixture
def test_client():
    """Create a test client with temporary repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        router, manager = create_repository_router(tmpdir)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Create test repository
        client.post('/repositories', json={'name': 'graph-test', 'description': 'Test'})
        
        yield client, manager


class TestGraphTargetImport:
    """Test graph_target parameter on import endpoint."""
    
    def test_import_default_graph(self, test_client):
        """Import into default graph (no graph_target)."""
        client, manager = test_client
        
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:alice ex:name "Alice" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle'
        })
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] is None
        assert data['triples_added'] == 1
    
    def test_import_explicit_default(self, test_client):
        """Import into default graph (explicit graph_target='default')."""
        client, manager = test_client
        
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:bob ex:name "Bob" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle',
            'graph_target': 'default'
        })
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] is None
    
    def test_import_named_graph(self, test_client):
        """Import into specific named graph."""
        client, manager = test_client
        
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:charlie ex:name "Charlie" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle',
            'graph_target': 'named:http://example.org/graphs/people'
        })
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] == 'http://example.org/graphs/people'
        assert 'into graph' in data['message']
    
    def test_import_direct_uri(self, test_client):
        """Import using direct URI as graph."""
        client, manager = test_client
        
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:diana ex:name "Diana" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle',
            'graph_target': 'http://example.org/graphs/direct'
        })
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] == 'http://example.org/graphs/direct'
    
    def test_import_auto_generates_graph(self, test_client):
        """Import with auto generates timestamp-based graph."""
        client, manager = test_client
        
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:eve ex:name "Eve" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle',
            'graph_target': 'auto'
        })
        
        assert r.status_code == 200
        data = r.json()
        # Auto without filename should generate timestamp-based URI
        assert data['graph'] is not None
        assert 'http://rdf-starbase.io/graph/import_' in data['graph']
    
    def test_query_named_graph(self, test_client):
        """Verify data is in the correct named graph via SPARQL."""
        client, manager = test_client
        
        # Import into named graph
        turtle_data = """
        @prefix ex: <http://example.org/> .
        ex:frank ex:name "Frank" .
        ex:frank ex:age "30" .
        """
        
        r = client.post('/repositories/graph-test/import', json={
            'data': turtle_data,
            'format': 'turtle',
            'graph_target': 'named:http://example.org/graphs/frank-data'
        })
        assert r.status_code == 200
        
        # Query the named graph
        r = client.post('/repositories/graph-test/sparql', json={
            'query': '''
                SELECT ?s ?p ?o WHERE { 
                    GRAPH <http://example.org/graphs/frank-data> { 
                        ?s ?p ?o 
                    } 
                }
            '''
        })
        
        assert r.status_code == 200
        data = r.json()
        assert data['count'] == 2  # name and age
    
    def test_multiple_named_graphs(self, test_client):
        """Import into different named graphs and verify isolation."""
        client, manager = test_client
        
        # Import Graph A
        r = client.post('/repositories/graph-test/import', json={
            'data': '@prefix ex: <http://example.org/> . ex:a ex:val "A" .',
            'format': 'turtle',
            'graph_target': 'named:http://example.org/graph-a'
        })
        assert r.status_code == 200
        
        # Import Graph B
        r = client.post('/repositories/graph-test/import', json={
            'data': '@prefix ex: <http://example.org/> . ex:b ex:val "B" .',
            'format': 'turtle',
            'graph_target': 'named:http://example.org/graph-b'
        })
        assert r.status_code == 200
        
        # Query Graph A only
        r = client.post('/repositories/graph-test/sparql', json={
            'query': 'SELECT ?o WHERE { GRAPH <http://example.org/graph-a> { ?s ?p ?o } }'
        })
        data = r.json()
        assert data['count'] == 1
        assert data['results'][0]['o'] == 'A'
        
        # Query Graph B only
        r = client.post('/repositories/graph-test/sparql', json={
            'query': 'SELECT ?o WHERE { GRAPH <http://example.org/graph-b> { ?s ?p ?o } }'
        })
        data = r.json()
        assert data['count'] == 1
        assert data['results'][0]['o'] == 'B'


class TestGraphTargetUpload:
    """Test graph_target parameter on file upload endpoint."""
    
    def test_upload_auto_from_filename(self, test_client):
        """Upload with auto generates graph from filename."""
        client, manager = test_client
        
        turtle_content = b'@prefix ex: <http://example.org/> . ex:test ex:val "upload" .'
        
        r = client.post(
            '/repositories/graph-test/upload',
            files={'file': ('my-data-file.ttl', turtle_content, 'text/turtle')},
            data={'graph_target': 'auto'}
        )
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] == 'http://rdf-starbase.io/graph/my-data-file'
    
    def test_upload_named_graph(self, test_client):
        """Upload into specific named graph."""
        client, manager = test_client
        
        turtle_content = b'@prefix ex: <http://example.org/> . ex:upload ex:val "named" .'
        
        r = client.post(
            '/repositories/graph-test/upload',
            files={'file': ('data.ttl', turtle_content, 'text/turtle')},
            data={'graph_target': 'named:http://example.org/graphs/uploaded'}
        )
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] == 'http://example.org/graphs/uploaded'
    
    def test_upload_default_graph(self, test_client):
        """Upload into default graph."""
        client, manager = test_client
        
        turtle_content = b'@prefix ex: <http://example.org/> . ex:default ex:val "test" .'
        
        r = client.post(
            '/repositories/graph-test/upload',
            files={'file': ('data.ttl', turtle_content, 'text/turtle')},
            data={'graph_target': 'default'}
        )
        
        assert r.status_code == 200
        data = r.json()
        assert data['graph'] is None


class TestGraphTargetResolver:
    """Test the _resolve_graph_target helper function."""
    
    def test_resolve_none(self):
        """None should return None (default graph)."""
        from rdf_starbase.repository_api import create_repository_router
        # We need to access the nested function, test via API instead
        pass  # Covered by API tests above
    
    def test_auto_with_complex_filename(self, test_client):
        """Auto mode handles complex filenames correctly."""
        client, manager = test_client
        
        turtle_content = b'@prefix ex: <http://example.org/> . ex:x ex:y "z" .'
        
        # Test with spaces (should be replaced)
        r = client.post(
            '/repositories/graph-test/upload',
            files={'file': ('my data file.ttl', turtle_content, 'text/turtle')},
            data={'graph_target': 'auto'}
        )
        
        assert r.status_code == 200
        data = r.json()
        assert 'my_data_file' in data['graph']
        assert ' ' not in data['graph']
