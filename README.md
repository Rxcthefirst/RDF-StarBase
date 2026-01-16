# RDF-StarBase

> **A blazingly fast RDF★ database with native provenance tracking**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-79%25-green.svg)]()

RDF-StarBase is a native RDF★ platform for storing, querying, and visualizing **assertions about data** — not just data itself. Every triple carries full provenance: **who** said it, **when**, **how confident** they were, and **which process** generated it.

## ✨ Key Features

- **🚀 Blazingly Fast** — Built on [Polars](https://pola.rs/) with Rust-speed DataFrame operations
- **⭐ Native RDF-Star** — First-class support for quoted triples and statement metadata
- **📍 Full Provenance** — Every assertion tracked with source, timestamp, confidence, process
- **⚖️ Competing Claims** — See ALL assertions, not just the "winning" one
- **🔍 SPARQL-Star** — Query with standard SPARQL syntax + provenance extensions
- **📋 Assertion Registry** — Track data sources, APIs, and mappings as first-class entities
- **🌐 REST API** — FastAPI-powered web interface with interactive docs
- **📊 Graph Visualization** — React + D3.js frontend for exploring knowledge graphs
- **💾 Parquet Persistence** — Efficient columnar storage for analytics workloads

## 🎯 Why RDF-StarBase?

Traditional databases store **values**.  
Traditional catalogs store **descriptions**.  
**RDF-StarBase stores assertions about reality.**

When your CRM says `customer.age = 34` and your Data Lake says `customer.age = 36`, most systems silently overwrite. RDF-StarBase **keeps both**, letting you:

- See competing claims side-by-side
- Filter by source, confidence, or recency
- Maintain full audit trails
- Let downstream systems choose which to trust

## 📦 Installation

```bash
pip install rdf-starbase
```

Or install from source:

```bash
git clone https://github.com/ontus/rdf-starbase.git
cd rdf-starbase
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
from rdf_starbase import TripleStore, ProvenanceContext

# Create a store
store = TripleStore()

# Add triples with provenance
prov = ProvenanceContext(
    source="CRM_System",
    confidence=0.85,
    process="api_sync"
)

store.add_triple(
    "http://example.org/customer/123",
    "http://xmlns.com/foaf/0.1/name",
    "Alice Johnson",
    prov
)

# Query with provenance filtering
results = store.get_triples(
    subject="http://example.org/customer/123",
    min_confidence=0.8
)

# Detect competing claims
claims = store.get_competing_claims(
    subject="http://example.org/customer/123",
    predicate="http://example.org/age"
)
```

## 🔍 SPARQL-Star Queries

```python
from rdf_starbase import execute_sparql

# Standard SPARQL
results = execute_sparql(store, """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?name WHERE {
        <http://example.org/customer/123> foaf:name ?name
    }
""")

# With provenance extensions
results = execute_sparql(store, """
    SELECT ?s ?p ?o WHERE {
        ?s ?p ?o .
        FILTER_CONFIDENCE(>= 0.9)
        FILTER_SOURCE("CRM_System")
    }
""")

# ASK queries
exists = execute_sparql(store, """
    ASK WHERE {
        <http://example.org/customer/123> <http://xmlns.com/foaf/0.1/name> ?name
    }
""")  # Returns: True
```

## ⭐ RDF-Star: Quoted Triples

RDF-Star allows you to make statements **about statements**:

```python
# The assertion "Alice knows Bob" is claimed by Wikipedia
store.add_quoted_triple(
    subject="<<http://example.org/alice http://xmlns.com/foaf/0.1/knows http://example.org/bob>>",
    predicate="http://example.org/assertedBy",
    obj="http://dbpedia.org/resource/Wikipedia",
    provenance=prov
)
```

Query with SPARQL-Star:

```sparql
SELECT ?who WHERE {
    << ?person foaf:knows ?other >> ex:assertedBy ?who
}
```

## 📊 Competing Claims Detection

```python
# Multiple systems report different ages
crm_prov = ProvenanceContext(source="CRM", confidence=0.85)
lake_prov = ProvenanceContext(source="DataLake", confidence=0.92)

store.add_triple(customer, "http://example.org/age", 34, crm_prov)
store.add_triple(customer, "http://example.org/age", 36, lake_prov)

# See all competing values
claims = store.get_competing_claims(customer, "http://example.org/age")
print(claims)
# shape: (2, 4)
# ┌────────┬──────────┬────────────┬─────────────────────┐
# │ object │ source   │ confidence │ timestamp           │
# ├────────┼──────────┼────────────┼─────────────────────┤
# │ 36     │ DataLake │ 0.92       │ 2026-01-16 03:00:00 │
# │ 34     │ CRM      │ 0.85       │ 2026-01-16 02:00:00 │
# └────────┴──────────┴────────────┴─────────────────────┘
```

## 📁 Persistence

```python
# Save to Parquet (columnar, fast, compressible)
store.save("knowledge_graph.parquet")

# Load back
loaded_store = TripleStore.load("knowledge_graph.parquet")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RDF-StarBase                                │
├─────────────────────────────────────────────────────────────────────┤
│    React + D3.js Frontend    │     REST API (FastAPI)               │
├──────────────────────────────┼──────────────────────────────────────┤
│  SPARQL-Star Parser  │  Query Executor  │  Assertion Registry       │
├─────────────────────────────────────────────────────────────────────┤
│                    Triple Store (Polars DataFrames)                 │
├─────────────────────────────────────────────────────────────────────┤
│  Parquet I/O  │  Provenance Tracking  │  Competing Claims Detection │
└─────────────────────────────────────────────────────────────────────┘
```

**Core Stack:**
- **Polars** — Rust-powered DataFrames for blazing performance
- **FastAPI** — Modern async REST API framework
- **pyparsing** — SPARQL-Star parser
- **Pydantic** — Data model validation
- **D3.js** — Graph visualization
- **PyArrow** — Parquet persistence

## 📈 Performance

RDF-StarBase leverages Polars' Rust backend for:

- **Vectorized operations** on millions of triples
- **Lazy evaluation** for query optimization
- **Zero-copy reads** from Parquet
- **Parallel execution** across cores

## 🌐 Web API

Start the server:

```bash
# Using uvicorn directly
uvicorn rdf_starbase.web:app --reload

# Or with the module
python -m rdf_starbase.web
```

Then open:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/triples` | GET | Query triples with filters |
| `/triples` | POST | Add new triple with provenance |
| `/triples/{subject}/claims` | GET | Get competing claims |
| `/sparql` | POST | Execute SPARQL-Star query |
| `/sources` | GET/POST | Manage data sources |
| `/graph/nodes` | GET | Visualization data |
| `/graph/edges` | GET | Graph edges |
| `/stats` | GET | Database statistics |

## 📋 Assertion Registry

Track data sources as first-class entities:

```python
from rdf_starbase import AssertionRegistry, SourceType

registry = AssertionRegistry()

# Register a data source
source = registry.register_source(
    name="CRM_Production",
    source_type=SourceType.API,
    uri="https://api.crm.example.com/v2",
    owner="sales-team",
    tags=["production", "customer-data"],
)

# Track sync runs
run = registry.start_sync(source.id)
# ... perform sync ...
registry.complete_sync(run.id, records_processed=1000)

# Get sync history
history = registry.get_sync_history(source.id)
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/rdf_starbase

# Format code
black src/ tests/
ruff check src/ tests/
```

## 📊 Frontend (React + D3)

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000 (proxies API to :8000)

## 📚 Examples

See the `examples/` directory:

- `quickstart.py` — Core features demonstration
- `competing_claims.py` — Handling conflicting data from multiple sources
- `sparql_queries.py` — SPARQL-Star query examples
- `registry_demo.py` — Assertion Registry usage

## 🗺️ Roadmap

### ✅ Completed (MVP)
- [x] Native RDF-Star storage
- [x] Provenance tracking (source, timestamp, confidence, process)
- [x] Competing claims detection
- [x] SPARQL-Star parser (SELECT, ASK, FILTER, ORDER BY, LIMIT)
- [x] SPARQL-Star executor with Polars backend
- [x] Provenance filter extensions
- [x] Parquet persistence
- [x] Assertion Registry (datasets, APIs, mappings)
- [x] REST API with FastAPI
- [x] React + D3 graph visualization

### 🔜 Next
- [ ] Trust scoring and decay
- [ ] Time-travel queries
- [ ] CONSTRUCT and DESCRIBE queries
- [ ] Property path queries

### 🚀 Future
- [ ] Federation across instances
- [ ] AI grounding via trusted assertions
- [ ] Governance workflows
- [ ] Reasoning engine integration

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Polars](https://pola.rs/) — The lightning-fast DataFrame library
- [RDF-Star Working Group](https://w3c.github.io/rdf-star/) — For the specification
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [D3.js](https://d3js.org/) — Data visualization library
- [pyparsing](https://pyparsing-docs.readthedocs.io/) — Parser combinators for Python

---

**RDF-StarBase** — *The place where enterprises store beliefs, not just data.*
