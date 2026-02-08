# Troubleshooting Guide

> Common issues and their solutions

## Quick Diagnostics

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installation
pip show rdf-starbase

# Check imports
python -c "from rdf_starbase import TripleStore; print('Engine OK')"
python -c "from api.web import create_app; print('API OK')"

# Run health check
curl http://localhost:8000/health
```

---

## Import Errors

### `ModuleNotFoundError: No module named 'rdf_starbase'`

**Cause:** Package not installed or wrong virtual environment.

**Fix:**
```bash
# Activate venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install in editable mode
pip install -e "."
```

### `ModuleNotFoundError: No module named 'api'`

**Cause:** API package not in Python path.

**Fix:**
```bash
# Ensure src/ is in PYTHONPATH
set PYTHONPATH=src  # Windows
export PYTHONPATH=src  # Linux/macOS

# Or install with all extras
pip install -e ".[dev,web,query,sql]"
```

### `ImportError: circular import`

**Cause:** Module A imports B, B imports A.

**Fix:** Check for imports at module level that should be inside functions. The API/engine separation should prevent most circular imports.

---

## Server Issues

### `Address already in use`

**Cause:** Another process on the same port.

**Fix:**
```bash
# Find process
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/macOS

# Kill it
taskkill /PID <pid> /F  # Windows
kill -9 <pid>  # Linux/macOS

# Or use different port
uvicorn api.web:app --port 8001
```

### `Connection refused`

**Cause:** Server not running or wrong address.

**Fix:**
```bash
# Check if server is running
curl http://localhost:8000/health

# If 0.0.0.0 needed (Docker):
uvicorn api.web:app --host 0.0.0.0
```

### `Internal Server Error (500)`

**Cause:** Unhandled exception in code.

**Fix:**
```bash
# Check server logs
uvicorn api.web:app --log-level debug

# Look for stack trace in output
```

---

## Query Issues

### `SPARQL syntax error`

**Cause:** Malformed SPARQL query.

**Fix:**
```sparql
# Common issues:
# - Missing prefixes
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# - Missing angle brackets for URIs
<http://example.org/alice>  # Correct
http://example.org/alice    # Wrong

# - Wrong string quotes
"literal"   # Correct
'literal'   # Also correct
literal     # Wrong (unquoted)
```

### `Query timeout`

**Cause:** Query too complex or dataset too large.

**Fix:**
```sparql
# Add LIMIT
SELECT * WHERE { ?s ?p ?o } LIMIT 1000

# Be more specific
SELECT * WHERE {
  ?s a foaf:Person .  # Filter by type
  ?s foaf:name ?name .
}
```

### `No results returned`

**Cause:** Query doesn't match data.

**Fix:**
```bash
# Check if data exists
curl http://localhost:8000/repositories/test/stats

# Try simpler query first
SELECT * WHERE { ?s ?p ?o } LIMIT 10

# Check for typos in URIs
```

---

## Data Issues

### `Repository not found`

**Cause:** Repository doesn't exist or wrong name.

**Fix:**
```bash
# List repositories
curl http://localhost:8000/repositories

# Create if missing
curl -X POST http://localhost:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{"name": "my-repo"}'
```

### `Import failed`

**Cause:** Invalid RDF syntax or wrong format.

**Fix:**
```bash
# Validate RDF locally first
rapper -i turtle -c myfile.ttl  # Using raptor

# Check Content-Type header matches format
-H "Content-Type: text/turtle"      # For Turtle
-H "Content-Type: application/ld+json"  # For JSON-LD

# Try smaller file first
head -100 large-file.ttl > test.ttl
```

### `Data not persisted`

**Cause:** In-memory store or persistence not enabled.

**Fix:**
```python
# Ensure persistence path is set
from rdf_starbase.storage.persistence import PersistenceManager

pm = PersistenceManager(path="/data/repositories/my-repo")
```

---

## Authentication Issues

### `401 Unauthorized`

**Cause:** Missing or invalid API key.

**Fix:**
```bash
# Include Authorization header
curl -H "Authorization: Bearer rsb_your_key_here" \
  http://localhost:8000/repositories
```

### `403 Forbidden`

**Cause:** Key doesn't have permission for operation.

**Fix:**
```python
# Check key's role and permissions
from api.auth import APIKeyManager

manager = APIKeyManager(storage_path="/data/keys")
key_info = manager.get_key_info("rsb_abc123")
print(key_info.role)  # Should be WRITER or ADMIN for writes
```

### `429 Too Many Requests`

**Cause:** Rate limit exceeded.

**Fix:**
```bash
# Wait and retry
# Or request higher rate limit

# Check current limits
curl http://localhost:8000/admin/keys/rsb_abc123
```

---

## Docker Issues

### `Container won't start`

**Cause:** Various (check logs).

**Fix:**
```bash
# Check logs
docker logs rdfstarbase

# Check if port is available
docker ps  # See what's running

# Try interactive mode
docker run -it --rm rxcthefirst/rdf-starbase:latest /bin/sh
```

### `Data not persisted after restart`

**Cause:** No volume mounted.

**Fix:**
```bash
# Mount volume for persistence
docker run -v rdfstarbase-data:/data/repositories ...
```

### `UI not loading`

**Cause:** UI not built or wrong path.

**Fix:**
```bash
# Check if UI files exist in container
docker exec rdfstarbase ls /app/src/ui/dist

# Rebuild image if needed
docker build --no-cache -t rdfstarbase:local -f deploy/docker/Dockerfile.api .
```

---

## Performance Issues

### Slow queries

**Diagnostics:**
```bash
# Check query plan
curl -X POST http://localhost:8000/repositories/test/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "EXPLAIN SELECT * WHERE { ?s ?p ?o }"

# Check repository stats
curl http://localhost:8000/repositories/test/stats
```

**Fixes:**
- Add more specific patterns
- Use LIMIT
- Consider partitioning large datasets

### High memory usage

**Diagnostics:**
```bash
# Check process memory
docker stats rdfstarbase
```

**Fixes:**
```python
# Set memory budget
from rdf_starbase.storage.memory_budget import MemoryBudget
budget = MemoryBudget(max_mb=512)
```

---

## Getting Help

1. Check this guide
2. Search [GitHub Issues](https://github.com/ontus/rdf-starbase/issues)
3. Check server logs (`--log-level debug`)
4. Open new issue with:
   - Python version
   - RDF-StarBase version
   - Error message/stack trace
   - Minimal reproduction steps
