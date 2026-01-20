# RDF-StarBase Roadmap

**From Alpha to Industry Standard**

---

## Current Status: Alpha Release Ready ✅

RDF-StarBase is **production-ready for early adopters**. The core functionality is complete, tested, and benchmarked.

### What's Done (v0.1.0)

| Category | Features | Tests |
|----------|----------|-------|
| **Core Storage** | Dictionary-encoded columnar storage, RDF-Star native, Polars backend | ✅ |
| **SPARQL Query** | SELECT, ASK, CONSTRUCT, DESCRIBE | ✅ |
| **SPARQL Patterns** | OPTIONAL, UNION, MINUS, FILTER, BIND, VALUES, GRAPH, EXISTS/NOT EXISTS | ✅ |
| **Property Paths** | Sequence `/`, Alternative `\|`, Inverse `^`, Modifiers `*`, `+`, `?` | ✅ |
| **Aggregates** | COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE, GROUP BY, HAVING | ✅ |
| **Functions** | COALESCE, IF, STRLEN, CONTAINS, STRSTARTS, STRENDS, LCASE, UCASE | ✅ |
| **SPARQL Update** | INSERT DATA, DELETE DATA, DELETE WHERE, DELETE/INSERT WHERE | ✅ |
| **Graph Management** | Named graphs, CREATE, DROP, CLEAR, COPY, MOVE, ADD, LOAD | ✅ |
| **Time-Travel** | AS OF clause for temporal queries | ✅ |
| **Formats** | Turtle, N-Triples, RDF/XML, JSON-LD (parse + serialize) | ✅ |
| **Persistence** | Parquet-based save/load | ✅ |
| **Reasoning** | RDFS + OWL (subClassOf, sameAs, inverseOf, transitiveProperty) | ✅ |
| **AI Grounding** | /ai/query, /ai/verify, /ai/context, /ai/materialize | ✅ |
| **REST API** | FastAPI endpoints for all features | ✅ |
| **rdflib Compat** | Drop-in replacement layer | ✅ |
| **Visualization** | React + D3 graph visualization | ✅ |

**Test Suite:** 503 tests, 71% coverage  
**Benchmarks:** 10-72x faster than rdflib

---

## Release Milestones

### v0.1.0 — Alpha (Q1 2026) ✅ SHIPPED

**Goal:** PyPI publication, early adopter feedback

- [x] **PyPI publication** (`pip install rdf-starbase`) — **LIVE on PyPI**
- [x] **CI/CD Pipeline** — GitHub Actions for test, benchmark, publish
- [x] **Documentation site** — MkDocs + ReadTheDocs config
- [x] **Quickstart guide** — `docs/quickstart.md`
- [x] **API reference** — Auto-generated from docstrings
- [x] **LICENSE file** — MIT License
- [x] **Benchmark reproducibility** — CI runs benchmarks on every push
- [x] **Performance baseline** — `benchmarks/bench_query.py` (200K triples/sec insert, 10-20ms queries)

**Marketing:**
- Add to Ontus.io product page
- Blog post: "rdflib for the AI Era"
- Post on r/semanticweb, HackerNews, Twitter/X

---

### v0.2.0 — Beta (Q2 2026) ✅ SHIPPED

**Goal:** Production hardening with modern web interface and Docker deployment

#### Web Interface & Visualization
- [x] **Monaco SPARQL Editor** — Syntax highlighting with @monaco-editor/react
- [x] **Schema Browser** — Classes and properties viewer with click-to-insert
- [x] **Import/Export UI** — File upload (Turtle, RDF/XML, N-Triples, JSON-LD)
- [x] **Graph Visualization** — D3.js force-directed graph with node/edge interactions
- [x] **Provenance Panel** — Node properties with RDF-Star metadata display
- [x] **Catppuccin Theme** — Mocha dark and Latte light theme support

#### Docker Deployment
- [x] **Multi-stage Dockerfile** — Node 20 Alpine → Python 3.12 slim
- [x] **docker-compose.yml** — Volume persistence for repository data
- [x] **Production build** — Vite-optimized frontend with proper asset paths
- [x] **Container orchestration** — Single-container deployment with embedded frontend

#### Performance Enhancements
- [x] **Columnar insertion** — `add_triples_columnar()` bypasses dict conversion
- [x] **Lazy DataFrame materialization** — `_df` property uses Polars lazy evaluation
- [x] **Short-circuit optimization** — Early exit when pattern terms don't exist in store
- [x] **Fast term lookup** — O(1) `get_iri_id()` / `get_literal_id()` using cached dictionaries
- [x] **Query plan caching** — LRU cache for parsed queries (21,000x speedup for repeated queries)
- [x] **Memory-mapped Parquet** — `load_streaming()` with `scan_parquet(memory_map=True)`
- [x] **Parallel pattern execution** — ThreadPoolExecutor for independent pattern groups (opt-in for federated queries)
- [x] **Lazy loading for large graphs** — Streaming collect for out-of-core processing

#### SPARQL Completeness
- [x] **EXISTS / NOT EXISTS** — Filter patterns with subquery existence checks
- [x] **COALESCE, IF functions** — Conditional expressions in SELECT and FILTER
- [x] **String functions** — STRLEN, CONTAINS, STRSTARTS, STRENDS, LCASE, UCASE
- [x] **BOUND function** — Check if optional variable is bound (with OPTIONAL null column support)
- [x] **BIND in nested patterns** — BIND within UNION, OPTIONAL, MINUS, EXISTS, GRAPH
- [x] **Subqueries** — Nested SELECT in WHERE clause with aggregates
- [x] **Property path {n,m}** — Fixed-length path modifiers ({n}, {n,m}, {n,})

#### Persistence Enhancements
- [x] **Streaming load** — Memory-mapped Parquet loading for large datasets
- [ ] DuckDB backend option (SQL interface to Parquet, analytical workloads)
- [ ] Incremental persistence (append-only, avoid full rewrites)

---

### v0.3.0 — Trust & Governance (Q3 2026)

**Goal:** Enterprise features for data governance

#### Trust Scoring
- [ ] Configurable trust policies per source
- [ ] Confidence decay over time
- [ ] Conflict resolution strategies (most-recent, highest-confidence, manual)
- [ ] Trust inheritance for inferred facts

#### Governance Workflows
- [ ] Approval workflows for assertions
- [ ] Audit log export
- [ ] Data lineage visualization
- [ ] Source health monitoring

---

### v1.0.0 — Production (Q4 2026)

**Goal:** Enterprise-grade, federation-ready

#### Federation
- [ ] SERVICE clause (SPARQL 1.1 Federated Query)
- [ ] Cross-instance synchronization
- [ ] Distributed query planning

#### Enterprise Features
- [ ] Authentication & authorization
- [ ] Multi-tenancy
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics endpoint
- [ ] OpenTelemetry tracing

#### Certification
- [ ] W3C SPARQL 1.1 compliance test suite
- [ ] RDF-Star Working Group test suite
- [ ] Security audit

---

## Non-Goals (By Design)

These are **not planned** because they conflict with our architecture:

| Feature | Reason |
|---------|--------|
| **Disk-based triple store** | We're columnar-first; use Parquet/DuckDB for persistence |
| **Full-text search built-in** | Integrate with external FTS (Elasticsearch, Typesense) |
| **OWL 2 Full reasoning** | Computational complexity; we support useful OWL subsets |
| **SHACL validation** | Planned as separate package (`rdf-starbase-shacl`) |

---

## How to Contribute

### Priority Areas
1. **Bug reports** — File issues with reproducible examples
2. **SPARQL edge cases** — Complex queries that fail or produce wrong results
3. **Performance regressions** — Queries that are slower than expected
4. **Documentation** — Tutorials, how-tos, and examples

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all 491+ tests pass
5. Submit a pull request

---

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **0.x.y** — Alpha/Beta, API may change
- **1.0.0** — Stable API, production-ready
- **1.x.y** — Backward-compatible features and fixes
- **2.0.0** — Breaking changes (if ever needed)

---

## Contact

- **Product:** [Ontus.io](https://ontus.io)
- **Issues:** GitHub Issues
- **Email:** team@ontus.dev

---

*RDF-StarBase — The semantic bedrock for AI applications*
