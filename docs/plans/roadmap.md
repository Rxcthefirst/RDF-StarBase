# RDF-StarBase Roadmap

> North-star milestones for the project

## Current Status: v1.1.0 ✅

RDF-StarBase is **production-ready** with full SPARQL-Star support, SHACL validation, ontology packs, and enterprise features.

| Layer | Status |
|-------|--------|
| **Storage Engine** | ✅ WAL, ACID, partitioning, indexes, memory budget |
| **SPARQL-Star** | ✅ Full SPARQL 1.1 + RDF-Star extensions |
| **Trust & Security** | ✅ Auth, audit, trust scoring, RBAC |
| **Enterprise Ops** | ✅ Federation, multi-tenancy, observability |
| **Governance** | ✅ SHACL, ontology packs, schema guidance |

**Test Suite:** 1526 tests passing, 71% coverage

---

## Release History

| Version | Release | Theme |
|---------|---------|-------|
| v0.1.0 | Q1 2026 | Alpha - Core engine |
| v0.2.0 | Q2 2026 | Beta - SPARQL patterns |
| v0.3.0 | Q3 2026 | Product Workflows |
| v0.4.0 | Q4 2026 | Trust & Security |
| v1.0.0 | Q1 2027 | Production - Federation |
| v1.1.0 | Q2 2027 | Governance & Ontologies |

---

## Future Directions

### v1.2.0 — Performance & Scale (Q3 2027)

- **Vectorized query execution** — SIMD-accelerated joins
- **Distributed partitioning** — Horizontal scaling across nodes
- **GPU acceleration** — Optional GPU-based aggregations
- **Incremental materialization** — Cached reasoner results

### v1.3.0 — Developer Experience (Q4 2027)

- **GraphQL interface** — Auto-generated from ontology
- **Jupyter integration** — Native notebook support
- **VS Code extension** — SPARQL editing + query execution
- **CLI improvements** — Bulk operations, scripting

---

## Non-Goals

These are explicitly **not planned**:

| Feature | Reason |
|---------|--------|
| Disk-based triple store | Columnar-first with Polars/Parquet |
| Full-text search built-in | Use external FTS (Elasticsearch) |
| OWL 2 Full reasoning | Computational complexity; useful subsets only |
| Distributed consensus | Single-node focus; external coordination |

---

See [ROADMAP.md](../../ROADMAP.md) in project root for detailed feature lists.
