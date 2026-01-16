# RDF★ Benchmark Plan (Polars Engine vs RDFox/Jena)

> Repo artifact: `benchmarks.md`  
> Scope: **datasets + query suite + runbook + metrics** to measure RDF★ performance and compare against RDFox and Jena TDB2/Fuseki.

---

## 1. Benchmark objectives

You want to answer:
1. **Ingestion**: How many base triples / quoted triples / metadata triples per second?
2. **RDF★ query latency**: How fast are quoted triple joins and metadata-driven expansions?
3. **Cold vs warm**: How much depends on cache?
4. **Compaction cost**: What is write amplification and background maintenance overhead?
5. **Scaling**: How do results change from 1M → 50M → 500M facts?

---

## 2. Engines under test

### Required
- **Your Polars RDF★ engine**
- **RDFox** (target bar)
- **Apache Jena TDB2 + Fuseki** (OSS baseline for RDF★)

### Optional
- GraphDB (if you want an enterprise commercial comparator)

---

## 3. Hardware and environment (must record)

Record these in every benchmark report:
- CPU model and cores (physical + logical)
- RAM size
- Disk type (NVMe/SATA) and filesystem
- OS version
- JVM version (for Jena)
- RDFox version
- Your engine commit hash / build flags
- Thread counts:
  - your engine worker threads
  - Polars thread settings
  - JVM `-Xmx` for Jena
- Whether CPU frequency scaling is disabled/enabled

**Important:** pin thread counts across engines where possible.

---

## 4. Datasets

You need 3 tiers to cover engineering, product reality, and real-world shape.

### Tier A: Synthetic controlled (engineering signal)
Purpose:
- isolate physical layout and join performance
- stress `quote()` throughput

Parameters (example):
- `N_entities = 1,000,000`
- `N_predicates = 200` (Zipf distribution)
- `N_facts = 50,000,000`
- `metadata_rate = 0.30` (30% of facts get metadata)
- `metadata_per_fact = 2..5` (random within range)

Metadata predicates (suggested):
- `prov:wasDerivedFrom`
- `prov:generatedAtTime`
- `prov:wasGeneratedBy`
- `ex:confidence`

### Tier B: “Receipts per run” (product reality)
Purpose:
- model ETL/inference runs as `prov:Activity`
- link produced statements via RDF★

Structure per run:
- one `prov:Activity` node
- `prov:used` input artifacts (entities)
- for each produced base triple:
  - create quoted triple term
  - attach:
    - `prov:wasGeneratedBy` run
    - `prov:generatedAtTime` timestamp
    - `prov:wasDerivedFrom` artifact
    - optional `ex:confidence`

Scale:
- `N_runs = 10,000` (or 1,000 for smaller)
- `facts_per_run = 5,000` (tune)
- yields 50M base facts at 10k×5k

### Tier C: Real dataset + injected metadata (shape realism)
Pick one:
- DBpedia subset (domain-friendly)
- Wikidata slice (if feasible)
- LUBM / BSBM (standard benchmarks)

Then inject metadata for a subset (e.g., 20–30%) using the Tier B scheme.

---

## 5. Data formats and loaders

### Canonical exchange format
Use **Turtle-star / TriG-star** where engines support it.

If an engine cannot ingest RDF★ directly (rare for these targets), provide:
- a fallback reification loader (only for compatibility testing)
- but keep RDF★ as the primary benchmark mode

### Loader requirements
Each engine should expose:
- bulk load time (including index build)
- time-to-first-query

For your engine:
- ingestion should be measured in **batch mode**
- record `quote()` rate and collisions encountered

---

## 6. Query suite

Store the query suite in `sparql-star-suite.sparql` (provided in this repo).

Run each query:
- **cold**: fresh process, no caches
- **warm**: after 5 warm-up runs
- measure p50/p95/p99 over 30 measured runs

---

## 7. Query categories (what you’re testing)

### A. Baseline RDF patterns
These should be “good enough” but not necessarily your differentiation.

- SPO exact
- predicate constrained scans
- `rdf:type` fanout
- 2-hop join

### B. RDF★-specific patterns (must-win area)
Your engine should shine here:
- fetch metadata for a specific quoted triple
- expand quoted triple subjects into base `(s,p,o)`
- filter quoted triples by metadata predicates and then expand

### C. Aggregations
- counts per source
- top sources
- metadata cardinalities per run

---

## 8. Metrics to collect

### Ingestion
- base facts: `facts/sec`
- metadata facts: `meta_facts/sec`
- quoted triples created: `qt/sec`
- total ingest wall time
- peak RSS during ingest
- on-disk size after ingest (before compaction)

### Compaction
- compaction wall time
- write amplification: bytes_written / bytes_ingested
- post-compaction on-disk size
- delta backlog size over time (if continuous ingest)

### Query
For each query:
- cold p50/p95/p99
- warm p50/p95/p99
- rows scanned (if available)
- bytes read (if available)
- peak RSS during query

### Correctness
- result row count
- stable hash of result bindings (sorted + hashed) for cross-engine comparison

---

## 9. Runbook

### 9.1 Standard run sequence per engine per dataset
1. Prepare dataset files (Tier A/B/C).
2. Load dataset (bulk load).
3. Record load stats.
4. Run query suite cold.
5. Run query suite warm.
6. Run compaction (if applicable).
7. Repeat query suite warm post-compaction.

### 9.2 Reporting
Write a JSON report per run:
- engine, version, commit hash
- dataset tier, parameters
- ingestion metrics
- compaction metrics
- query metrics (cold/warm, p50/p95/p99)
- environment metadata

Suggested paths:
- `reports/<engine>/<dataset>/<timestamp>.json`

---

## 10. Interpretation guidance

### If you’re behind RDFox on baseline scans
That’s expected early. Focus on:
- partition pruning effectiveness
- joins on int columns
- hot predicate caching

### If you’re behind RDFox on RDF★ joins
That’s actionable; likely culprits:
- slow `quote()` interning path
- missing `qt_id` join optimization
- predicate partitions not tight enough
- too much string materialization during execution

### You should “win” (or be close) on:
- provenance/receipt workloads where predicates are hot and partitioned
- statement expansion joins: `?qt -> (s,p,o)` when qt dict is indexed

---

## 11. Next steps after first benchmark pass
Once you have baseline results:
- add optional projections (`facts_by_s`, `facts_by_o`) only if needed
- tune partition cardinality thresholds
- tune compaction schedule (size-based vs time-based)
- add query planner heuristics (predicate selectivity first)
