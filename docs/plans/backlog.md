# Backlog

> Ideas and features not currently scheduled

## v2.0.0 — Data Integration Platform (Scheduled Q4 2027)

### Starchart - Visual RML/R2RML Mapper (HIGH) — IN PROGRESS
- [x] Standalone mapping UI at `/mapper`
- [x] Column-to-property drag-drop mapping
- [x] RML/R2RML generation from visual design
- [ ] Mapping templates for common patterns
- [ ] Mapping validation against sample data

### ONTOP - Virtualized Data (HIGH)
- [ ] Virtual SPARQL endpoint for external sources
- [ ] PostgreSQL connector
- [ ] MySQL connector
- [ ] Connection management with credential storage
- [ ] Query pushdown optimization

### RDFMap - Materialize Semistructured Data (HIGH)
- [ ] ETL pipeline to execute RML mappings
- [ ] Incremental update detection
- [ ] Source freshness monitoring
- [ ] Batch job scheduling
- [ ] Error quarantine and retry

### Governance Framework (HIGH)
- [ ] Governance policies (access, retention, quality)
- [ ] Provenance graphs (PROV-O based)
- [ ] Auditability dashboard
- [ ] Virtualization vs materialization policies
- [ ] Change management workflows
- [ ] Agent safety guardrails

### Protégé-like Ontology Editor (MEDIUM)
- [ ] Visual class/property editor
- [ ] Ontology graph visualization
- [ ] Provenance tracking for edits
- [ ] Version control with diff/rollback
- [ ] OWL/RDFS/SKOS import/export

### Materialized vs Virtualized Understanding (MEDIUM)
- [ ] Data source annotations
- [ ] Access pattern tracking
- [ ] Performance hints / materialization recommendations
- [ ] Hybrid query support

### Embeddings & Search (MEDIUM)
- [ ] Vector index for literals/URIs
- [ ] Semantic similarity search
- [ ] Hybrid SPARQL + vector retrieval
- [ ] Provider support (OpenAI, Cohere, local)

---

## High Priority (Future Sprints)

### Authentication & Security
- [x] OAuth2/OIDC integration ✅ (v1.2.0)
- [ ] SAML support for enterprise SSO
- [ ] Fine-grained graph-level permissions
- [ ] Encryption at rest

### Performance
- [ ] Query plan caching
- [ ] Vectorized join execution
- [ ] Bloom filter indexes
- [ ] Parallel query execution

### Developer Experience
- [ ] GraphQL auto-generated API
- [ ] VS Code extension for SPARQL
- [ ] Jupyter/notebook integration
- [ ] CLI bulk import/export tools

---

## Medium Priority

### Integrations
- [ ] Elasticsearch full-text search bridge
- [ ] Apache Kafka change data capture
- [ ] Airflow DAG templates
- [ ] dbt integration for transformations

### UI Improvements
- [ ] Dark mode theme
- [ ] Query result visualization charts
- [ ] Schema diff viewer
- [ ] Collaborative editing (WebSocket)

### Operations
- [ ] Automated backup scheduling
- [ ] Point-in-time recovery
- [ ] Cross-region replication
- [ ] Blue-green deployment support

---

## Low Priority / Exploration

### Research
- [ ] GPU-accelerated reasoning
- [x] Vector embeddings for semantic search ✅ (scheduled v2.0.0)
- [ ] LLM-powered query assistance
- [ ] Property graph compatibility layer

### Experimental
- [ ] WebAssembly client library
- [ ] Edge deployment mode
- [ ] Time-series optimizations
- [ ] Geospatial extensions

---

## Explicitly Deferred

These were considered but explicitly deferred:

| Item | Reason | Revisit |
|------|--------|---------|

| OWL 2 Full | Computational complexity | Never |
| Built-in FTS | Use external systems | Integration only |
| Distributed consensus | Complexity vs. value | When needed |

---

## How to Propose Features

1. Open a GitHub issue with the `feature-request` label
2. Include: use case, expected behavior, alternatives considered
3. Features that align with roadmap themes get prioritized
