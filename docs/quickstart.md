# Quickstart Guide

Get RDF-StarBase running in 5 minutes.

## Installation

```bash
pip install rdf-starbase[query,web]
```

## Basic Usage

### 1. Create a Store and Add Data

```python
from rdf_starbase import TripleStore

store = TripleStore()

# Add triples with provenance
store.add_triple(
    subject="http://example.org/Alice",
    predicate="http://xmlns.com/foaf/0.1/name",
    object="Alice Johnson",
    source="CRM_System",
    confidence=0.95
)

# Or parse from file
store.parse("data.ttl")
```

### 2. Query with SPARQL

```python
results = store.query("""
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    
    SELECT ?person ?name
    WHERE {
        ?person foaf:name ?name .
    }
""")

for row in results:
    print(f"{row['person']}: {row['name']}")
```

### 3. Filter by Provenance

```python
# Query with confidence filter (RDF-Star extension)
results = store.query("""
    SELECT ?person ?name ?confidence
    WHERE {
        ?person foaf:name ?name .
        << ?person foaf:name ?name >> prov:confidence ?confidence .
    }
    FILTER(?confidence > 0.8)
""")
```

### 4. Time-Travel Queries

```python
# Query the graph as it was at a specific time
results = store.query("""
    SELECT ?person ?name
    AS OF "2025-12-01T00:00:00Z"
    WHERE {
        ?person foaf:name ?name .
    }
""")
```

### 5. Save and Load

```python
# Save to Parquet (fast, columnar)
store.save("./my_graph")

# Load later
store = TripleStore.load("./my_graph")
```

## Start the REST API

```python
from rdf_starbase.web import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then visit:
- `http://localhost:8000/docs` — OpenAPI documentation
- `http://localhost:8000/ai/query` — AI Grounding API

## rdflib Compatibility

If you have existing rdflib code, just change the import:

```python
# Before
from rdflib import Graph, URIRef, Literal

# After
from rdf_starbase.compat.rdflib import Graph, URIRef, Literal

# Everything else works the same!
g = Graph()
g.parse("data.ttl")
for s, p, o in g.triples((None, RDF.type, FOAF.Person)):
    print(s)
```

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) — Understand the columnar design
- [AI Grounding Guide](AI_GROUNDING_GUIDE.md) — Integrate with LLMs
- [Examples](https://github.com/ontus/rdf-starbase/tree/main/examples) — More code samples
