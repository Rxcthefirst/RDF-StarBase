"""
Integration test for SQL API and AI Grounding across repositories.
"""

from fastapi.testclient import TestClient
from rdf_starbase.repository_api import create_repository_router
from fastapi import FastAPI
import tempfile
import os

# Create a temporary directory for test repos
with tempfile.TemporaryDirectory() as tmpdir:
    router, manager = create_repository_router(tmpdir)
    
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Create test repositories
    print('=== Creating test repositories ===')
    r = client.post('/repositories', json={'name': 'repo-alpha', 'description': 'First repo'})
    print(f'Create repo-alpha: {r.status_code}')
    
    r = client.post('/repositories', json={'name': 'repo-beta', 'description': 'Second repo'})
    print(f'Create repo-beta: {r.status_code}')
    
    # Load different datasets into each
    print()
    print('=== Loading different datasets ===')
    r = client.post('/repositories/repo-alpha/load-example/movies')
    print(f'Load movies into repo-alpha: {r.status_code}')
    
    r = client.post('/repositories/repo-beta/load-example/techcorp')
    print(f'Load techcorp into repo-beta: {r.status_code}')
    
    # Test SQL API on each repo
    print()
    print('=== Testing SQL API on repo-alpha (movies) ===')
    r = client.get('/repositories/repo-alpha/sql/status')
    data = r.json()
    print(f'SQL Status: available={data.get("available")}, tables={len(data.get("tables", []))}')
    
    r = client.post('/repositories/repo-alpha/sql/query', json={'sql': 'SELECT COUNT(*) as cnt FROM triples'})
    data = r.json()
    print(f'Triple count in repo-alpha: {data["rows"][0][0]}')
    
    r = client.post('/repositories/repo-alpha/sql/query', json={
        'sql': 'SELECT predicate, COUNT(*) as cnt FROM triples GROUP BY predicate ORDER BY cnt DESC LIMIT 5'
    })
    data = r.json()
    print(f'Top predicates in repo-alpha: {data["rows"][:3]}')
    
    print()
    print('=== Testing SQL API on repo-beta (techcorp) ===')
    r = client.post('/repositories/repo-beta/sql/query', json={'sql': 'SELECT COUNT(*) as cnt FROM triples'})
    data = r.json()
    print(f'Triple count in repo-beta: {data["rows"][0][0]}')
    
    r = client.get('/repositories/repo-beta/sql/tables')
    data = r.json()
    print(f'Tables: {[t["name"] for t in data["tables"]]}')
    
    # Test AI Grounding on each repo
    print()
    print('=== Testing AI Grounding on repo-alpha (movies) ===')
    r = client.get('/repositories/repo-alpha/ai/health')
    data = r.json()
    print(f'AI Health: status={data.get("status")}, triples={data.get("store_stats", {}).get("total_triples")}')
    
    r = client.post('/repositories/repo-alpha/ai/query', json={'question': 'movies'})
    data = r.json()
    print(f'AI Query returned {len(data.get("facts", []))} facts')
    if data.get('facts'):
        print(f'Sample fact: {data["facts"][0].get("subject", "?")} - {data["facts"][0].get("predicate", "?")[:30]}')
    
    print()
    print('=== Testing AI Grounding on repo-beta (techcorp) ===')
    r = client.get('/repositories/repo-beta/ai/health')
    data = r.json()
    print(f'AI Health: status={data.get("status")}, triples={data.get("store_stats", {}).get("total_triples")}')
    
    r = client.post('/repositories/repo-beta/ai/query', json={'question': 'customer'})
    data = r.json()
    print(f'AI Query returned {len(data.get("facts", []))} facts')
    if data.get('facts'):
        print(f'Sample fact: {data["facts"][0].get("subject", "?")} - {data["facts"][0].get("predicate", "?")[:30]}')
    
    # Verify isolation - use SPARQL to check different namespaces
    print()
    print('=== Verifying repository isolation (via SPARQL) ===')
    
    # Movies repo should have schema.org predicates
    r = client.post('/repositories/repo-alpha/sparql', json={
        'query': 'SELECT (COUNT(*) as ?cnt) WHERE { ?s <http://schema.org/name> ?o }'
    })
    data = r.json()
    results = data.get('results', [])
    alpha_schema_count = results[0].get('cnt', 0) if results else 0
    print(f'repo-alpha: schema.org/name triples = {alpha_schema_count}')
    
    # TechCorp repo should have techcorp.com predicates
    r = client.post('/repositories/repo-beta/sparql', json={
        'query': 'SELECT (COUNT(*) as ?cnt) WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?s), "techcorp")) }'
    })
    data = r.json()
    results = data.get('results', [])
    beta_techcorp_count = results[0].get('cnt', 0) if results else 0
    print(f'repo-beta: techcorp subjects = {beta_techcorp_count}')
    
    # Cross-check: movies repo should NOT have techcorp subjects
    r = client.post('/repositories/repo-alpha/sparql', json={
        'query': 'SELECT (COUNT(*) as ?cnt) WHERE { ?s ?p ?o FILTER(CONTAINS(STR(?s), "techcorp")) }'
    })
    data = r.json()
    results = data.get('results', [])
    alpha_techcorp_count = results[0].get('cnt', 0) if results else 0
    print(f'repo-alpha: techcorp subjects = {alpha_techcorp_count} (should be 0)')
    
    # Cross-check: techcorp repo should NOT have schema.org predicates
    r = client.post('/repositories/repo-beta/sparql', json={
        'query': 'SELECT (COUNT(*) as ?cnt) WHERE { ?s <http://schema.org/name> ?o }'
    })
    data = r.json()
    results = data.get('results', [])
    beta_schema_count = results[0].get('cnt', 0) if results else 0
    print(f'repo-beta: schema.org/name triples = {beta_schema_count} (should be 0)')
    
    if int(alpha_techcorp_count) == 0 and int(beta_schema_count) == 0:
        print('✓ Repository isolation verified!')
    else:
        print('✗ Repository isolation FAILED')
    
    print()
    print('=== All tests completed ===')
