# Architecture Overview

> Component responsibilities and system design

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Clients                                      │
│    (Browser UI, CLI, Python SDK, REST clients, LLM/RAG pipelines)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Layer (src/api/)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   web.py    │  │   auth.py   │  │repository_  │  │ai_grounding │   │
│  │  (FastAPI)  │  │ (RBAC/Keys) │  │  api.py     │  │    .py      │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Engine Layer (src/rdf_starbase/)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  store.py   │  │ executor.py │  │ reasoner.py │  │  parser.py  │   │
│  │(TripleStore)│  │  (SPARQL)   │  │ (RDFS/OWL)  │  │ (RDF I/O)   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Storage Layer (src/rdf_starbase/storage/)            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  facts.py   │  │  terms.py   │  │   wal.py    │  │persistence  │   │
│  │ (Quad Store)│  │ (Dictionary)│  │  (WAL/TXN)  │  │    .py      │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │indexing.py  │  │partitioning │  │memory_budget│  │  backup.py  │   │
│  │  (B-tree)   │  │    .py      │  │    .py      │  │ (Snapshots) │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Layer (Polars/Parquet)                      │
│                   ┌─────────────────────────────────┐                   │
│                   │   Parquet files on disk         │                   │
│                   │   /data/repositories/{repo}/    │                   │
│                   │     ├── terms.parquet           │                   │
│                   │     ├── facts.parquet           │                   │
│                   │     ├── wal/                    │                   │
│                   │     └── metadata.json           │                   │
│                   └─────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

### `src/api/` — REST API Layer
Separated from engine for clean boundaries. Owns:
- HTTP routing and middleware
- Authentication and authorization
- Rate limiting
- Request/response serialization

| Module | Responsibility |
|--------|----------------|
| `web.py` | FastAPI app, routes, middleware |
| `auth.py` | API keys, RBAC, scoped tokens |
| `repository_api.py` | Repository CRUD endpoints |
| `ai_grounding.py` | LLM/RAG-focused endpoints |

### `src/rdf_starbase/` — Core Engine
The RDF-Star database engine. No HTTP concerns.

| Module | Responsibility |
|--------|----------------|
| `store.py` | TripleStore: main entry point |
| `executor.py` | SPARQL query execution |
| `parser.py` | SPARQL parsing |
| `reasoner.py` | RDFS/OWL inference |
| `formats.py` | RDF serialization (Turtle, JSON-LD, etc.) |

### `src/rdf_starbase/storage/` — Storage Layer
Low-level storage primitives:

| Module | Responsibility |
|--------|----------------|
| `terms.py` | Dictionary encoding (term ↔ ID) |
| `facts.py` | Quad storage and retrieval |
| `wal.py` | Write-ahead log, transactions |
| `persistence.py` | Parquet I/O |
| `indexing.py` | B-tree sorted indexes |
| `partitioning.py` | Predicate-based partitions |
| `memory_budget.py` | Memory management |
| `backup.py` | Snapshot/restore |

### `src/ui/` — Web Frontend
React + TypeScript SPA:
- Monaco SPARQL editor
- D3.js graph visualization
- Repository management UI

---

## Data Flow

### Query Execution
```
1. HTTP request → api/web.py
2. Auth check → api/auth.py
3. Parse SPARQL → rdf_starbase/parser.py
4. Execute → rdf_starbase/executor.py
5. Storage ops → rdf_starbase/storage/
6. Results → JSON response
```

### Data Ingestion
```
1. HTTP POST → api/repository_api.py
2. Parse RDF → rdf_starbase/formats.py
3. Stage → storage/staging.py
4. Validate → storage/import_jobs.py
5. Commit → storage/facts.py + wal.py
```

---

## Key Design Decisions

1. **Columnar Storage**: Polars DataFrames with Parquet persistence
2. **Dictionary Encoding**: All terms mapped to integer IDs
3. **RDF-Star Native**: Quoted triples are first-class terms
4. **API/Engine Separation**: Clean boundary for testability
5. **Single-Node First**: Optimize single-node before distribution

See [ADR-0001](../adr/0001-initial-architecture.md) for rationale.
