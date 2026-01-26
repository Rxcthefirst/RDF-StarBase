# RDF-StarBase Roadmap

**From Alpha to Industry Standard**

---

## Current Status: v1.0.0 Complete ✅

RDF-StarBase is now **production-ready** with federation, multi-tenancy, and enterprise operations (1380 tests, 74% coverage). Ready for enterprise deployment!

### Architecture Assessment

| Layer | Score | Status |
|-------|-------|---------|
| **Storage Engine** | 9/10 | ✅ WAL, ACID, partitioning, indexes, memory budget |
| **Repository Manager** | 9/10 | ✅ Backup/restore, clone, versioning, config |
| **Product Workflows** | 9/10 | ✅ Staging, observability, graph explorer, query tools |
| **Trust & Security** | 9/10 | ✅ Auth, audit, trust scoring, compliance || **Enterprise Ops** | 9/10 | ✅ Federation, multi-tenancy, Prometheus, tracing |
### What's Done (v0.1.0 + v0.2.0 + v0.3.0)

| Category | Features | Tests |
|----------|----------|-------|
| **Core Storage** | Dictionary-encoded columnar storage, RDF-Star native, Polars backend | ✅ |
| **SPARQL Query** | SELECT, ASK, CONSTRUCT, DESCRIBE | ✅ |
| **SPARQL Patterns** | OPTIONAL, UNION, MINUS, FILTER, BIND, VALUES, GRAPH, EXISTS/NOT EXISTS | ✅ |
| **Property Paths** | Sequence `/`, Alternative `\|`, Inverse `^`, Modifiers `*`, `+`, `?`, `{n,m}` | ✅ |
| **Aggregates** | COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE, GROUP BY, HAVING | ✅ |
| **Functions** | COALESCE, IF, STRLEN, CONTAINS, STRSTARTS, STRENDS, LCASE, UCASE, BOUND | ✅ |
| **SPARQL Update** | INSERT DATA, DELETE DATA, DELETE WHERE, DELETE/INSERT WHERE | ✅ |
| **Graph Management** | Named graphs, CREATE, DROP, CLEAR, COPY, MOVE, ADD, LOAD | ✅ |
| **Time-Travel** | AS OF clause for temporal queries | ✅ |
| **Formats** | Turtle, N-Triples, RDF/XML, JSON-LD, TriG, N-Quads (parse + serialize) | ✅ |
| **Persistence** | Parquet-based, WAL, incremental delta files, streaming load | ✅ |
| **Reasoning** | RDFS + OWL (subClassOf, sameAs, inverseOf, transitiveProperty) | ✅ |
| **AI Grounding** | /ai/query, /ai/verify, /ai/context, /ai/materialize | ✅ |
| **REST API** | FastAPI endpoints for all features | ✅ |
| **rdflib Compat** | Drop-in replacement layer | ✅ |
| **Visualization** | React + D3 graph visualization with Monaco editor | ✅ |
| **Database Features** | WAL, ACID transactions, connection pooling, memory budget | ✅ |
| **Indexing** | B-tree sorted index, predicate partitioning, query timeout/cancel | ✅ |
| **Backup/Restore** | Snapshot API, restore to new repo, clone with UUID versioning | ✅ |
| **Import Workflow** | Staging area, preview, validation, dry-run, job tracking, undo | ✅ |
| **Query Tools** | Saved queries, history, slow query log, export, cursor pagination | ✅ |
| **Graph Explorer** | List graphs, DCAT/PROV metadata, statistics, replace/drop/copy | ✅ |
| **Observability** | Metrics collection, health checks, admin dashboard data | ✅ |
| **Repository Config** | Schema versioning, migrations, per-repo configuration | ✅ |
| **Authentication** | API key management, role-based access, scoped tokens, rate limiting | ✅ |
| **Audit & Compliance** | Audit log, CSV/JSON export, data lineage tracking, source health | ✅ |
| **Trust Scoring** | Configurable trust policies, confidence decay, conflict resolution | ✅ |
| **Federation** | SERVICE clause, cross-instance sync, distributed query planning | ✅ |
| **Multi-tenancy** | Tenant isolation, resource quotas, usage tracking | ✅ |
| **Enterprise Ops** | Health checks, Prometheus metrics, OpenTelemetry tracing | ✅ |
| **Kubernetes** | Helm chart, StatefulSet, PVC, Ingress, ServiceMonitor | ✅ |

**Test Suite:** 1380 tests, 74% coverage  
**Benchmarks:** 10-72x faster than rdflib

---

## Release Milestones

### v0.1.0 — Alpha (Q1 2026) ✅ SHIPPED
### v0.2.0 — Beta (Q2 2026) ✅ SHIPPED
### v0.3.0 — Product Workflows (Q3 2026) ✅ SHIPPED
### v0.4.0 — Trust & Security (Q4 2026) ✅ SHIPPED
### v1.0.0 — Production (Q1 2027) ✅ SHIPPED

*See archived sections below for completed features.*

---

### v1.0.0 — Production (Q1 2027)

**Goal:** Transform from "engine" to "database product" with real-world workflows

#### Repository Lifecycle (Priority: HIGH) ✅
- [x] **Backup/snapshot API** — `repo.snapshot()` creates point-in-time backup with manifest ✅
- [x] **Restore to new repo** — `manager.restore(snapshot_path, new_name)` never overwrites ✅
- [x] **Clone repository** — `manager.clone(source, target)` with copy-on-write option ✅
- [x] **Repository versioning** — Schema version in metadata, migration path for upgrades ✅
- [x] **Per-repo configuration** — Reasoning rules, memory limits, sharding config per repo ✅
- [x] **Repository UUIDs** — Stable IDs separate from mutable display names ✅

#### Import Workflow (Priority: HIGH) ✅
- [x] **Staging area** — Imports go to staging manifest before commit ✅
- [x] **Preview before commit** — Show term/fact/graph counts, warnings, errors ✅
- [x] **Validation layer** — Check IRI syntax, datatype validity, RDF-Star structure ✅
- [x] **Dry-run mode** — `import(file, dry_run=True)` returns stats without persisting ✅
- [x] **Undo last import** — Per-commit rollback via WAL transaction IDs ✅
- [x] **Import job tracking** — Progress, status, error log for large imports ✅

#### Query Workspace (Priority: MEDIUM) ✅
- [x] **Saved queries** — Persist queries to repo metadata with name, description, tags ✅
- [x] **Query history** — Last N queries with timestamps, stored per-session or per-repo ✅
- [ ] **SQL tab in UI** — DuckDB interface alongside SPARQL editor (deferred to v0.4.0)
- [x] **Export results** — CSV, JSON, TSV, JSONL, Parquet download from result grid ✅
- [x] **Pagination with cursors** — Efficient cursor-based paging beyond LIMIT/OFFSET ✅

#### Graph Explorer (Priority: MEDIUM) ✅
- [x] **Graph list view** — Named graphs with counts, last modified, source info ✅
- [x] **Graph metadata panel** — DCAT/PROV/DCT structured view (creator, created, source) ✅
- [x] **Per-graph statistics** — Subject/predicate/object counts, RDF-Star annotation counts ✅
- [x] **"Replace graph" action** — Drop + reload from file in single operation ✅

#### Observability (Priority: MEDIUM) ✅
- [x] **Metrics collection** — Ingest rate, query latency, memory usage, compaction status ✅
- [x] **Stats API endpoint** — `/metrics` with JSON or Prometheus format ✅
- [x] **Admin dashboard** — Web UI showing key metrics over time ✅
- [x] **Slow query log** — Queries exceeding threshold logged with EXPLAIN plan ✅

---

### v0.4.0 — Trust & Security (Q4 2026)

**Goal:** Enterprise-ready authentication and data governance

#### Authentication & Authorization (Priority: HIGH) ✅
- [x] **API key management** — Generate, revoke, list API keys ✅
- [x] **Role-based access** — Reader, Writer, Admin roles per repository ✅
- [x] **Scoped tokens** — Tokens with specific repo/operation permissions ✅
- [x] **Rate limiting** — Per-key query/ingestion rate limits ✅

#### Audit & Compliance (Priority: MEDIUM) ✅
- [x] **Audit log** — Who loaded data, who ran queries, when ✅
- [x] **Audit log export** — CSV/JSON export for compliance ✅
- [x] **Data lineage view** — Visual lineage from source to derived facts ✅
- [x] **Source health monitoring** — Track freshness, error rates per source ✅

#### Trust Scoring (Priority: MEDIUM) ✅
- [x] **Configurable trust policies** — Per-source confidence rules ✅
- [x] **Confidence decay** — Time-based decay for aging assertions ✅
- [x] **Conflict resolution** — Strategies: most-recent, highest-confidence, manual review ✅
- [x] **Trust inheritance** — Inferred facts inherit trust from premises ✅

---

### v1.0.0 — Production (Q1 2027) ✅

**Goal:** Enterprise-grade, federation-ready, certified

#### Federation ✅
- [x] **SERVICE clause** — SPARQL 1.1 Federated Query ✅
- [x] **Cross-instance sync** — Replicate between RDF-StarBase instances ✅
- [x] **Distributed query planning** — Push filters to remote endpoints ✅

#### Enterprise Operations ✅
- [x] **Multi-tenancy** — Isolated namespaces with resource quotas ✅
- [x] **Kubernetes manifests** — Helm chart with StatefulSet, PVC, Ingress ✅
- [x] **Prometheus endpoint** — `/metrics` in Prometheus exposition format ✅
- [x] **OpenTelemetry tracing** — Distributed traces for query execution ✅
- [x] **Health checks** — `/health`, `/ready` endpoints for orchestration ✅

#### Certification (Deferred to v1.1.0)
- [ ] **W3C SPARQL 1.1 test suite** — Full compliance verification
- [ ] **RDF-Star Working Group tests** — Quoted triple edge cases
- [ ] **Security audit** — Third-party review of auth, data isolation

---

### v1.1.0 — Governance & Ontologies (Q2 2027)

**Goal:** Guide users toward standard ontologies without blocking power users

#### Ontology Packs
- [ ] **PROV-O pack** — Agent, Activity, Entity templates
- [ ] **DCAT pack** — Dataset, Distribution, Catalog templates
- [ ] **PAV pack** — Provenance, Authoring, Versioning shortcuts
- [ ] **Enable/disable per repo** — Ontology packs as optional repo config

#### Schema Guidance
- [ ] **SHACL validation** — Validate on import, show warnings/errors
- [ ] **Auto-complete from ranges** — Suggest types based on predicate domains
- [ ] **"Annotate statement" wizard** — UI for adding RDF-Star metadata
- [ ] **Template forms** — Create PROV entities via form, not raw SPARQL

---

## Non-Goals (By Design)

These are **not planned** because they conflict with our architecture:

| Feature | Reason |
|---------|--------|
| **Disk-based triple store** | We're columnar-first; use Parquet/DuckDB for persistence |
| **Full-text search built-in** | Integrate with external FTS (Elasticsearch, Typesense) |
| **OWL 2 Full reasoning** | Computational complexity; we support useful OWL subsets |
| **Heavy ORM abstraction** | Direct SPARQL/SQL access is the interface; no magic |
| **Distributed consensus** | Single-node focus; use external coordination if needed |

---

## Implementation Priority Matrix

### v0.3.0 Sprint Plan

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1-2 | **Backup/Restore** | `snapshot()`, `restore()`, snapshot manifest format |
| 3-4 | **Clone + Versioning** | `clone()`, repo UUIDs, schema version field |
| 5-6 | **Import Staging** | Staging manifest, preview stats, validation errors |
| 7-8 | **Undo Import** | Per-commit rollback, import job tracking |
| 9-10 | **Saved Queries** | Query persistence, history, UI integration |
| 11-12 | **Observability** | Metrics collection, stats API, admin dashboard |

### Success Criteria

| Feature | Metric |
|---------|--------|
| Backup/Restore | Round-trip 1M triples in < 30s |
| Import Staging | Preview 100K triples in < 5s |
| Saved Queries | Persist/load 1000 queries |
| Observability | < 1% overhead from metrics collection |

---

## How to Contribute

### Priority Areas for v0.3.0
1. **Backup/restore testing** — Large dataset round-trips, corruption recovery
2. **Import edge cases** — Malformed files, encoding issues, huge files
3. **UI/UX feedback** — Admin dashboard design, query workspace usability
4. **Observability integration** — Prometheus/Grafana dashboards, alerting rules

### General Contributions
1. **Bug reports** — File issues with reproducible examples
2. **SPARQL edge cases** — Complex queries that fail or produce wrong results
3. **Performance regressions** — Queries that are slower than expected
4. **Documentation** — Tutorials, how-tos, and examples

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all 746+ tests pass
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
