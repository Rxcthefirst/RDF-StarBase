# ADR-0001: Initial Architecture

**Status:** Accepted  
**Date:** 2025-06-15  
**Authors:** Ontus Team

## Context

We need to build an RDF-Star database that:
1. Supports quoted triples (RDF-Star) as first-class citizens
2. Achieves high query performance for analytics workloads
3. Integrates well with Python data science tooling
4. Provides full provenance tracking

Existing options:
- **rdflib**: Pure Python, slow, no native RDF-Star
- **Apache Jena**: Java, heavy, RDF-Star via reification
- **Virtuoso**: C, complex licensing, RDF-Star limited
- **Oxigraph**: Rust, fast, but limited provenance

## Decision

We will build a **columnar RDF-Star database** using:

1. **Polars** as the DataFrame engine (Rust-based, fast)
2. **Parquet** for persistence (columnar, compressed)
3. **Dictionary encoding** for term storage (integers, not strings)
4. **Native RDF-Star** with quoted triples as first-class terms
5. **Python-first API** with FastAPI REST layer

### Key Architecture Choices

```
┌─────────────────┐
│   API Layer     │  ← FastAPI, auth, REST
├─────────────────┤
│  Engine Layer   │  ← SPARQL, reasoning
├─────────────────┤
│ Storage Layer   │  ← Terms, facts, WAL
├─────────────────┤
│     Polars      │  ← DataFrame operations
├─────────────────┤
│    Parquet      │  ← Disk persistence
└─────────────────┘
```

## Rationale

### Why Polars?
| Factor | Polars | Pandas | DuckDB |
|--------|--------|--------|--------|
| Speed | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Memory | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Python API | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Lazy eval | ✅ | ❌ | ✅ |

Polars provides Rust-level performance with a Python API.

### Why Dictionary Encoding?
Storing string terms in every triple is wasteful:
```
# Without encoding: 3 strings per triple
("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", "Alice")

# With encoding: 3 integers per triple
(42, 17, 103)  # terms stored separately
```

Benefits:
- 10-50x storage reduction
- Faster joins (integer comparison)
- Cheaper hash lookups

### Why Native RDF-Star?
Alternative: Reification
```turtle
_:stmt1 rdf:subject :alice .
_:stmt1 rdf:predicate :knows .
_:stmt1 rdf:object :bob .
_:stmt1 :certainty 0.9 .
```

Problems with reification:
- 4 triples per statement
- Complex queries
- No standard syntax

Our approach: Quoted triples are terms:
```turtle
<<:alice :knows :bob>> :certainty 0.9 .
```

Benefits:
- 1 triple for metadata
- Natural SPARQL-Star syntax
- Cleaner storage model

## Consequences

### Positive
- 10-72x faster than rdflib (benchmarked)
- Native RDF-Star without performance penalty
- Python ecosystem integration (numpy, pandas, sklearn)
- Incremental persistence with WAL

### Negative
- Single-node only (distributed is future work)
- No built-in full-text search (use external)
- Memory-bound for very large datasets

### Risks
- Polars API changes (mitigate: pin versions)
- RDF-Star spec changes (mitigate: follow W3C closely)

## Alternatives Considered

1. **DuckDB for everything** — Rejected: RDF-Star handling awkward
2. **SQLite + custom indexing** — Rejected: No columnar benefits
3. **Fork rdflib** — Rejected: Fundamental performance limits
4. **Use Oxigraph** — Rejected: Limited provenance model

## References

- [RDF-Star W3C Draft](https://w3c.github.io/rdf-star/cg-spec/)
- [Polars Documentation](https://pola.rs/)
- [Parquet Format](https://parquet.apache.org/)
