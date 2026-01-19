# RDF-StarBase Documentation

Welcome to RDF-StarBase — a blazingly fast, provenance-native RDF-Star database.

## Quick Links

- [Quickstart](quickstart.md) — Get up and running in 5 minutes
- [Architecture](ARCHITECTURE.md) — Technical deep-dive
- [AI Grounding Guide](AI_GROUNDING_GUIDE.md) — Using RDF-StarBase with LLMs
- [API Reference](api/index.md) — Full API documentation

## Installation

```bash
pip install rdf-starbase
```

For full features:
```bash
pip install rdf-starbase[query,web]
```

## Why RDF-StarBase?

| Feature | rdflib | RDF-StarBase |
|---------|--------|--------------|
| Performance | Baseline | **10-72x faster** |
| RDF-Star | Experimental | **Native** |
| Provenance | None | **First-class** |
| AI Grounding | None | **Built-in API** |

## Quick Example

```python
from rdf_starbase import TripleStore

# Create store and add data
store = TripleStore()
store.parse("data.ttl")

# Query with SPARQL-Star
results = store.query("""
    SELECT ?person ?name ?confidence
    WHERE {
        ?person foaf:name ?name .
        << ?person foaf:name ?name >> prov:confidence ?confidence .
    }
    FILTER(?confidence > 0.8)
""")

for row in results:
    print(row)
```

## Drop-in rdflib Replacement

```python
# Just change the import!
from rdf_starbase.compat.rdflib import Graph

g = Graph()
g.parse("data.ttl")
# Same API, 10-72x faster
```

## License

MIT License — see [LICENSE](https://github.com/ontus/rdf-starbase/blob/main/LICENSE)
