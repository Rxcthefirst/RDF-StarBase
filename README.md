# RDF-StarBase

> **A blazingly fast RDF★ database with native provenance tracking**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-410%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-76%25-green.svg)]()

RDF-StarBase is a native RDF★ platform for storing, querying, and visualizing **assertions about data** — not just data itself. Every triple carries full provenance: **who** said it, **when**, **how confident** they were, and **which process** generated it.

##  Key Features

- **Blazingly Fast** — Built on [Polars](https://pola.rs/) with Rust-speed DataFrame operations
- **Native RDF-Star** — First-class support for quoted triples and statement metadata
- **Full Provenance** — Every assertion tracked with source, timestamp, confidence, process
- **Competing Claims** — See ALL assertions, not just the "winning" one
- **SPARQL-Star** — Query with standard SPARQL syntax + provenance extensions
- **Assertion Registry** — Track data sources, APIs, and mappings as first-class entities
- **REST API** — FastAPI-powered web interface with interactive docs
- **Graph Visualization** — React + D3.js frontend for exploring knowledge graphs
- **Parquet Persistence** — Efficient columnar storage for analytics workloads

## Why RDF-StarBase?

Traditional databases store **values**.  
Traditional catalogs store **descriptions**.  
**RDF-StarBase stores assertions about reality.**

When your CRM says `customer.age = 34` and your Data Lake says `customer.age = 36`, most systems silently overwrite. RDF-StarBase **keeps both**, letting you:

- See competing claims side-by-side
- Filter by source, confidence, or recency
- Maintain full audit trails
- Let downstream systems choose which to trust

## Installation

```bash
pip install rdf-starbase
```

Or install from source:

```bash
git clone https://github.com/ontus/rdf-starbase.git
cd rdf-starbase
pip install -e ".[dev]"
```

## Quick Start

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

### Advanced Query Features

```python
# OPTIONAL - include data when available
results = execute_sparql(store, """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?name ?email WHERE {
        ?person foaf:name ?name .
        OPTIONAL { ?person foaf:mbox ?email }
    }
""")

# UNION - combine multiple patterns
results = execute_sparql(store, """
    SELECT ?entity ?label WHERE {
        { ?entity rdfs:label ?label }
        UNION
        { ?entity foaf:name ?label }
    }
""")

# BIND - computed values
results = execute_sparql(store, """
    SELECT ?product ?price ?taxed WHERE {
        ?product ex:price ?price .
        BIND(?price * 1.1 AS ?taxed)
    }
""")

# Aggregates with GROUP BY
results = execute_sparql(store, """
    SELECT ?source (COUNT(*) AS ?count) (AVG(?confidence) AS ?avg_conf) WHERE {
        ?s ?p ?o .
    }
    GROUP BY ?source
    HAVING (COUNT(*) > 10)
""")

# CONSTRUCT - generate new triples
results = execute_sparql(store, """
    CONSTRUCT {
        ?person foaf:knows ?other .
    }
    WHERE {
        ?person ex:worksAt ?company .
        ?other ex:worksAt ?company .
        FILTER(?person != ?other)
    }
""")

# INSERT DATA - add new triples
execute_sparql(store, """
    INSERT DATA {
        <http://example.org/alice> foaf:name "Alice" .
        <http://example.org/alice> foaf:age 30 .
    }
""")

# DELETE DATA - remove specific triples
execute_sparql(store, """
    DELETE DATA {
        <http://example.org/alice> foaf:age 30 .
    }
""")

# DELETE WHERE - remove matching patterns
execute_sparql(store, """
    DELETE WHERE {
        <http://example.org/alice> foaf:knows ?anyone .
    }
""")

# DELETE/INSERT WHERE - update values atomically
execute_sparql(store, """
    DELETE { ?s ex:status "active" }
    INSERT { ?s ex:status "archived" }
    WHERE { ?s ex:status "active" }
""")

# Property paths - navigate graph relationships
results = execute_sparql(store, """
    SELECT ?ancestor WHERE {
        <http://example.org/alice> foaf:knows+ ?ancestor .  # One or more hops
    }
""")

results = execute_sparql(store, """
    SELECT ?connected WHERE {
        <http://example.org/alice> (foaf:knows|foaf:worksWith)* ?connected .  # Zero or more via knows OR worksWith
    }
""")

results = execute_sparql(store, """
    SELECT ?knower WHERE {
        ?knower ^foaf:knows <http://example.org/bob> .  # Inverse: who knows Bob?
    }
""")

# Time-travel queries - query historical state
results = execute_sparql(store, """
    SELECT ?s ?name WHERE {
        ?s foaf:name ?name .
    }
    AS OF "2025-01-15T00:00:00Z"
""")

# ASK with time-travel
existed = execute_sparql(store, """
    ASK WHERE {
        <http://example.org/alice> foaf:name ?name .
    }
    AS OF "2024-06-01"
""")  # Returns: True if Alice existed on that date
```

## 📊 Named Graph Management

RDF-StarBase supports named graphs (graph containers/clusters) with full SPARQL Graph Store Protocol operations:

```python
from rdf_starbase import execute_sparql

# CREATE GRAPH - create a new named graph
execute_sparql(store, """
    CREATE GRAPH <http://example.org/graphs/customers>
""")

# LOAD - load RDF data from a file into a graph
execute_sparql(store, """
    LOAD <file:///data/customers.ttl> 
    INTO GRAPH <http://example.org/graphs/customers>
""")

# Or load from HTTP
execute_sparql(store, """
    LOAD <https://example.org/data/products.ttl>
    INTO GRAPH <http://example.org/graphs/products>
""")

# COPY - copy all triples from one graph to another
execute_sparql(store, """
    COPY GRAPH <http://example.org/graphs/customers>
    TO GRAPH <http://example.org/graphs/customers_backup>
""")

# MOVE - move triples (copy then clear source)
execute_sparql(store, """
    MOVE GRAPH <http://example.org/graphs/staging>
    TO GRAPH <http://example.org/graphs/production>
""")

# ADD - add triples to another graph (merge)
execute_sparql(store, """
    ADD GRAPH <http://example.org/graphs/updates>
    TO GRAPH <http://example.org/graphs/main>
""")

# CLEAR - remove all triples from a graph (graph still exists)
execute_sparql(store, """
    CLEAR GRAPH <http://example.org/graphs/temp>
""")

# DROP - delete a graph and all its triples
execute_sparql(store, """
    DROP GRAPH <http://example.org/graphs/old_data>
""")

# Special graph targets
execute_sparql(store, "CLEAR DEFAULT")  # Clear default graph
execute_sparql(store, "DROP NAMED")     # Drop all named graphs
execute_sparql(store, "CLEAR ALL")      # Clear everything

# SILENT mode - don't fail if graph doesn't exist
execute_sparql(store, """
    DROP SILENT GRAPH <http://example.org/graphs/maybe_exists>
""")

# List all named graphs
graphs = store.list_graphs()
print(graphs)  # ['http://example.org/graphs/customers', 'http://example.org/graphs/products']
```

### Querying Named Graphs

```python
# FROM clause - restrict query to specific graph
results = execute_sparql(store, """
    SELECT ?customer ?name
    FROM <http://example.org/graphs/customers>
    WHERE {
        ?customer foaf:name ?name
    }
""")

# FROM with multiple graphs (union of datasets)
results = execute_sparql(store, """
    SELECT ?entity ?label
    FROM <http://example.org/graphs/customers>
    FROM <http://example.org/graphs/products>
    WHERE {
        ?entity rdfs:label ?label
    }
""")

# GRAPH pattern - query specific named graph in WHERE clause
results = execute_sparql(store, """
    SELECT ?customer ?name WHERE {
        GRAPH <http://example.org/graphs/customers> {
            ?customer foaf:name ?name
        }
    }
""")

# GRAPH with variable - discover which graph contains data
results = execute_sparql(store, """
    SELECT ?graph ?entity ?name WHERE {
        GRAPH ?graph {
            ?entity foaf:name ?name
        }
    }
""")

# Combined patterns - default graph + specific named graph
results = execute_sparql(store, """
    SELECT ?person ?friend ?friendName WHERE {
        ?person foaf:knows ?friend .
        GRAPH <http://example.org/graphs/profiles> {
            ?friend foaf:name ?friendName
        }
    }
""")

# FROM NAMED - specify available named graphs for GRAPH patterns
results = execute_sparql(store, """
    SELECT ?g ?s ?name
    FROM NAMED <http://example.org/graphs/customers>
    FROM NAMED <http://example.org/graphs/employees>
    WHERE {
        GRAPH ?g { ?s foaf:name ?name }
    }
""")
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

## Competing Claims Detection

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

## Persistence

```python
# Save to Parquet (columnar, fast, compressible)
store.save("knowledge_graph.parquet")

# Load back
loaded_store = TripleStore.load("knowledge_graph.parquet")
```

## Architecture

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

## Performance

RDF-StarBase leverages Polars' Rust backend for:

- **Vectorized operations** on millions of triples
- **Lazy evaluation** for query optimization
- **Zero-copy reads** from Parquet
- **Parallel execution** across cores

## Web API

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

### 🤖 AI Grounding API

A specialized API layer designed for AI/LLM consumption, separate from the UI visualization endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ai/query` | POST | Structured fact retrieval with provenance for RAG |
| `/ai/verify` | POST | Verify if a claim is supported by the knowledge base |
| `/ai/context/{iri}` | GET | Get all facts about an entity with citations |
| `/ai/materialize` | POST | Trigger reasoning and persist inferences |
| `/ai/inferences` | GET | List materialized inferences |
| `/ai/health` | GET | AI API health check |

#### Why a Separate AI API?

| Aspect | UI API (`/graph/*`) | AI Grounding API (`/ai/*`) |
|--------|---------------------|----------------------------|
| **Consumer** | D3.js visualization | LLM tool calls / agents |
| **Response format** | Nodes + edges for rendering | Facts + provenance + citations |
| **Query pattern** | Browsing, neighborhood exploration | Precise fact lookup, verification |
| **Filtering** | Limit by count, visual simplicity | Confidence threshold, freshness |

#### Example: Grounding an AI Response

```python
import httpx

# 1. Query relevant facts for RAG
response = httpx.post("http://localhost:8000/ai/query", json={
    "subject": "http://example.org/customer/123",
    "min_confidence": "high",  # high (>=0.9), medium (>=0.7), low (>=0.5), any
    "max_age_days": 30,        # Only recent facts
})
facts = response.json()["facts"]

# 2. Verify a claim before stating it
verify = httpx.post("http://localhost:8000/ai/verify", json={
    "subject": "http://example.org/customer/123",
    "predicate": "http://xmlns.com/foaf/0.1/age",
    "expected_object": "34",
})
result = verify.json()
if result["claim_supported"]:
    print(f"Claim verified with {result['confidence']:.0%} confidence")
elif result["has_conflicts"]:
    print("Warning: Competing claims exist!")
    print(result["recommendation"])

# 3. Get full entity context
context = httpx.get("http://localhost:8000/ai/context/http://example.org/customer/123")
entity_facts = context.json()["facts"]
related = context.json()["related_entities"]
```

#### Inference Materialization

Materialize RDFS/OWL inferences with provenance tracking:

```python
# Run reasoning engine and persist inferred triples
response = httpx.post("http://localhost:8000/ai/materialize", json={
    "enable_rdfs": True,   # RDFS entailment rules
    "enable_owl": True,    # OWL 2 RL rules
    "max_iterations": 100,
})
print(f"Inferred {response.json()['triples_inferred']} triples")

# Query inferred facts (marked with source='reasoner')
inferences = httpx.get("http://localhost:8000/ai/inferences")
for fact in inferences.json()["inferences"]:
    print(f"Inferred: {fact['subject']} {fact['predicate']} {fact['object']}")
```

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
- [x] SPARQL-Star parser (SELECT, ASK, FILTER, ORDER BY, LIMIT, OFFSET)
- [x] SPARQL-Star executor with Polars backend
- [x] Provenance filter extensions
- [x] Parquet persistence
- [x] Assertion Registry (datasets, APIs, mappings)
- [x] REST API with FastAPI
- [x] React + D3 graph visualization

### ✅ Completed (Advanced Query Features)
- [x] OPTIONAL patterns (left outer joins)
- [x] UNION patterns (combine result sets)
- [x] MINUS patterns (set difference)
- [x] FILTER expressions (comparisons, boolean logic, regex, string functions)
- [x] BIND clauses (variable assignment, expressions, functions)
- [x] VALUES inline data
- [x] Aggregate functions (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE)
- [x] GROUP BY and HAVING
- [x] CONSTRUCT queries (template-based triple generation)
- [x] DESCRIBE queries (resource description)
- [x] SPARQL UPDATE (INSERT DATA, DELETE DATA, DELETE WHERE, DELETE/INSERT WHERE)
- [x] OWL reasoning (rdfs:subClassOf, owl:sameAs, owl:inverseOf, owl:TransitiveProperty)
- [x] Property path queries (`/`, `|`, `^`, `*`, `+`, `?`)
- [x] Time-travel queries (`AS OF "2025-01-15T00:00:00Z"`)
- [x] AI Grounding API (`/ai/query`, `/ai/verify`, `/ai/context`)
- [x] Inference materialization (`/ai/materialize`, `/ai/inferences`)
- [x] Named Graph Management (CREATE, DROP, CLEAR, LOAD, COPY, MOVE, ADD)
- [x] FROM clause dataset specification
- [x] GRAPH pattern queries

### 🔜 Next
- [ ] Trust scoring and decay

### 🚀 Future
- [ ] Federation across instances
- [ ] Governance workflows

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
