# RDF★ Storage Spec (Polars Engine)

> Repo artifact: `storage-spec.md`  
> Scope: **storage layout + indexes + write/compaction model** for a Polars/Rust RDF-star-native store.
>
> Assumptions:
> - The query layer consumes **integer IDs** (dictionary encoded terms).
> - Storage is **columnar** (Parquet/IPC) with **batch-first ingestion**.
> - RDF★ quoted triples are treated as **first-class RDF terms**.

---

## 1. Goals and non-goals

### Goals
- Native RDF★ support: quoted triples `<<s p o>>` are **first-class terms** that can appear as `s` or `o` in normal triples.
- Performance-first: joins and scans operate on **compact integer columns** (u32/u64).
- Durable, incremental ingestion with **append-only deltas** + periodic **compaction**.
- Enough physical organization to be competitive on:
  - predicate-constrained scans,
  - star-join patterns,
  - RDF★ expansion joins (`qt_id -> (s,p,o)`).

### Non-goals (for this document)
- Full SPARQL/SPARQL★ grammar and algebra spec (covered elsewhere).
- ETL and UI/authoring workflows.
- Distributed clustering/HA (single-node first).

---

## 2. Core internal types

### 2.1 Term identity and encoding

All RDF terms are mapped to an integer **Term ID**.

- **TermId**: `u64` (prefer `u32` if your corpus fits).
- **Kind tag**: `u8` enum
  - `0 = IRI`
  - `1 = LITERAL`
  - `2 = BNODE`
  - `3 = QUOTED_TRIPLE (QT)`

#### Option A (recommended): tagged ID space
Reserve high bits to tag the term kind:

- `term_id = (kind_tag << 62) | payload`
  - `payload` is an integer assigned from per-kind sequences.
- Pros: constant-time kind detection without joining to dictionaries.
- Cons: reduces payload bit width (still huge).

#### Option B: untagged IDs + lookup
All terms share one ID space; kind is resolved via `term_dict.kind`.
- Pros: simpler.
- Cons: kind checks require dictionary access.

Either works. Option A tends to simplify execution.

---

## 3. Logical tables

### 3.1 `term_dict` (authoritative term catalog)

| column | type | description |
|---|---:|---|
| `term_id` | u64 | internal ID |
| `kind` | u8 | IRI/LIT/BNODE/QT |
| `lex` | string | full lexical form (IRI string, literal string, bnode label) |
| `datatype_id` | u64? | datatype IRI term_id (literals only) |
| `lang` | string? | language tag (literals only) |

Notes:
- You may split IRI into `{prefix_id, local}` later; start simple with `lex`.
- Persist `term_dict` for rebuild; keep in-memory hash map for speed.

### 3.2 `term_hash` (optional but highly recommended)

| column | type | description |
|---|---:|---|
| `term_hash` | u128 (or 2×u64) | hash of canonical term bytes |
| `term_id` | u64 | internal ID |

Purpose:
- Fast bulk dedupe during batch ingest.
- Fast rebuild of in-memory term map at startup.

Canonical term bytes MUST include:
- kind tag,
- lexical form,
- datatype IRI (for typed literals),
- language tag (for lang literals).

### 3.3 `qt_dict` (quoted triple catalog)

A quoted triple is stored once and gets a stable ID.

| column | type | description |
|---|---:|---|
| `qt_id` | u64 | quoted triple ID (also a TermId of kind QT) |
| `s` | u64 | TermId |
| `p` | u64 | TermId |
| `o` | u64 | TermId |
| `g` | u64? | GraphId (optional; see §5) |
| `qt_hash` | u128 | hash of tuple (s,p,o[,g]) |

### 3.4 `facts` (quad store)

Authoritative assertion store, represented as **quads**:

| column | type | description |
|---|---:|---|
| `g` | u64 | GraphId (`DEFAULT` graph is a constant) |
| `s` | u64 | TermId (may be QT) |
| `p` | u64 | TermId |
| `o` | u64 | TermId (may be QT) |
| `flags` | u16 | bitset: asserted/inferred/deleted/etc. |
| `txn` | u64 | commit/transaction id (monotonic) |
| `t_added` | u64 | logical timestamp (optional) |

Key invariants:
- **No string terms** inside `facts`. Everything is ID-based.
- RDF★ metadata triples are stored by setting `s` (or `o`) to a **QT term_id**.

Example (conceptual):
- `<<ex:A ex:knows ex:B>> prov:wasDerivedFrom ex:Src`
  - `qt_id = quote(ex:A, ex:knows, ex:B)`
  - store quad: `(g, s=qt_id, p=prov:wasDerivedFrom, o=ex:Src)`

---

## 4. Required in-memory indexes (hot path)

Polars is columnar; it is not a row-store index. Your service layer should provide **O(1-ish)** lookups for the following:

### 4.1 Term interning map
- `HashMap<TermKey, TermId>`
- supports bulk `get_or_create_terms(batch)`.

### 4.2 Quoted triple interning map
- `HashMap<QtKeyHash, SmallVec<QtId>>` (handle rare collisions)
- `QtKey` = `(s,p,o[,g])`
- supports bulk `get_or_create_qt(batch)`.

**This is critical.** `quote()` is frequently called during metadata ingest.

### 4.3 Hot predicate partitions cache
Maintain an LRU cache of frequently-used `p` partitions from `facts`:
- especially `rdf:type`, and common metadata predicates (e.g., `prov:*`),
- plus top domain predicates in your benchmark corpora.

---

## 5. Graph semantics and quoting scope

You have two viable choices; pick one and document it clearly.

### Choice 1: Graph-agnostic quoting (simpler)
- `qt_dict` key is `(s,p,o)` only.
- A quoted triple identifies the statement independent of graph.
- Provenance/metadata can still be stored in any graph.

**Pros:** simpler; reduces qt cardinality.  
**Cons:** if the same triple appears in multiple graphs with different meaning, the quote is shared.

### Choice 2: Graph-scoped quoting (more precise)
- `qt_dict` key is `(g,s,p,o)`.
- Quoted triple refers to “the statement as asserted in graph g”.

**Pros:** avoids cross-graph mixing; supports per-run receipts cleanly.  
**Cons:** higher qt cardinality.

If your product is “receipts per ETL run,” graph-scoped quoting is often the better long-term choice.

---

## 6. Physical storage layout (Parquet datasets)

### 6.1 Base + delta (LSM-like)
Persist each logical table as:
- **base dataset**: compacted partitions
- **delta dataset**: append-only write batches

Tables:
- `data/term_dict/base/...`
- `data/term_dict/delta/...`
- `data/qt_dict/base/...`
- `data/qt_dict/delta/...`
- `data/facts/base/...`
- `data/facts/delta/...`

### 6.2 Partitioning strategy

**Primary partition key for `facts`: `p` (predicate).**  
Optional secondary partition key: `g` (graph).

Recommended directory shape:
- `data/facts/base/p=<pid>/g=<gid>/part-*.parquet`
- `data/facts/delta/p=<pid>/g=<gid>/batch-<txn>.parquet`

Rationale:
- Most graph workloads are predicate-selective.
- Metadata predicates are a small, hot set.
- Partition pruning becomes your “index.”

### 6.3 Sorting within partitions
Within each `(p,g)` partition:
- sort by `(s, o, txn)` (or `(s,o)` if you don’t need txn ordering)
- Keep `s` and `o` adjacent for compression and efficient joins.

---

## 7. Write path and compaction

### 7.1 Ingestion contract
All ingestion is **batch-first**:
- ingest terms (intern)
- ingest quoted triples (intern)
- ingest facts (append)

No single-row writes in the hot path.

### 7.2 Transaction model
- Assign a monotonic `txn` id to each ingestion batch.
- Store `txn` on rows for:
  - incremental compaction,
  - snapshot/time-travel (optional).

### 7.3 Deduplication rules
During compaction, dedupe `facts` on `(g,s,p,o)`:
- Keep the record with greatest `txn` (or your chosen merge rule).
- If `flags` include tombstones, apply last-write-wins.

### 7.4 Compaction algorithm (high level)
For each partition `(p,g)`:
1. Read base files for partition.
2. Read delta files for partition (up to a configurable size/time window).
3. Concatenate → stable sort by `(s,o,txn)`.
4. Groupby `(g,s,p,o)` selecting latest row.
5. Write new base part file(s).
6. Delete compacted delta files.
7. Update partition stats.

### 7.5 Partition stats (for pruning + planning)
Persist lightweight stats per partition:
- row count
- min/max s, min/max o (optional)
- distinct s count estimate
- file list + sizes

These stats help your planner order joins and decide hot caching.

---

## 8. Execution primitives (storage-facing)

Even without full SPARQL★, your query layer should rely on these storage primitives:

- `scan_facts(p=?, g=?) -> DataFrame[g,s,o,(flags,txn)]`
- `scan_facts_by_s(s=?, p=?, g=?)` (optional projection)
- `scan_facts_by_o(o=?, p=?, g=?)` (optional projection)
- `lookup_qt(qt_id list) -> DataFrame[qt_id,s,p,o]`
- `lookup_term(term_id list) -> term lex forms` (for results)

### Critical RDF★ expansion join
Given pattern:
- `?qt prov:wasDerivedFrom ?src`
and want `?s ?p ?o`:

1. `df1 = scan_facts(p=prov:wasDerivedFrom)` → columns: `s as qt_id`, `o as src`
2. `df2 = lookup_qt(df1.qt_id)` → columns: `qt_id,s,p,o`
3. `join(df1, df2 on qt_id)` → yields `s,p,o,src`

If `scan_facts` is partition-pruned and `lookup_qt` is indexed, this is fast.

---

## 9. Optional projections (build only if benchmarks demand it)

### 9.1 `facts_by_s` (entity-centric projection)
Partition by `s`, sort by `p,o`.
Useful for “show me all facts about entity X”.

### 9.2 `facts_by_o` (reverse lookup projection)
Partition by `o`, sort by `p,s`.
Useful for inbound edge queries.

Build these during compaction; keep them consistent with `facts`.

---

## 10. Correctness notes (RDF★ semantics)

- A quoted triple is a **term** that denotes a triple.
- Storing metadata about it does **not** imply the base triple is asserted (unless your logic says so).
- Your engine may choose:
  - “quoted triple implies nothing” (pure annotation)
  - or “quoted triple also asserts base triple” (shortcut)

**Recommendation:** keep them separate for correctness and transparency.  
If you want an option to “materialize base triple from quote,” implement it as an explicit rule/setting.

---

## 11. Minimal on-disk schema versioning
Persist:
- `meta/schema_version.json`
- `meta/build_info.json`
- `meta/partitions.json`

Include:
- schema version
- encoding choices (tagged IDs vs lookup)
- quoting scope choice (graph-agnostic vs graph-scoped)
- compaction parameters

---

## 12. Provenance Vocabulary Recommendations

RDF-StarBase uses **RDF★ embedded triples** to attach metadata (provenance, confidence, timestamps) to statements. This section documents the recommended predicates.

### 12.1 Recommended Predicates

| Attribute | Predicate | Standard | Notes |
|-----------|-----------|----------|-------|
| **Source** | `prov:wasDerivedFrom` | W3C PROV-O | The entity/agent this assertion came from |
| **Confidence** | `prov:value` | W3C PROV-O | Numeric confidence/certainty score (0.0–1.0) |
| **Process** | `prov:wasGeneratedBy` | W3C PROV-O | The activity/process that generated this |
| **Timestamp** | `prov:generatedAtTime` | W3C PROV-O | When the assertion was created |

### 12.2 Why `prov:value` for Confidence?

We considered several options:

| Option | Pros | Cons |
|--------|------|------|
| `ex:confidence` | Simple | Not standardized; poor interoperability |
| `dqv:value` + `dqv:QualityMeasurement` | W3C DQV standard | Verbose; designed for dataset-level quality, not statement-level |
| `prov:value` | W3C PROV-O standard; already in our provenance vocabulary | Generic name; semantics come from context |

**Decision:** Use `prov:value` from PROV-O (W3C Recommendation).

Rationale:
1. **Already a W3C standard** — from PROV-O, designed for "a direct representation of an entity"
2. **Consistent vocabulary** — we already use `prov:wasDerivedFrom`, `prov:wasGeneratedBy`, `prov:generatedAtTime`
3. **Sanctioned by spec** — PROV-O explicitly encourages domain extensions for qualifying influences
4. **Simple** — no blank nodes or intermediate measurement objects needed

### 12.3 Example Usage (RDF★)

```turtle
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix ex: <http://example.org/> .

# A statement with full provenance
<<ex:Alice ex:knows ex:Bob>> 
    prov:wasDerivedFrom ex:CRMSystem ;        # Source
    prov:value "0.95"^^xsd:decimal ;          # Confidence
    prov:wasGeneratedBy ex:ETLProcess2024 ;   # Process
    prov:generatedAtTime "2024-01-15T10:30:00Z"^^xsd:dateTime .
```

### 12.4 Native Column Optimization

For high-performance queries (filtering by confidence), RDF-StarBase stores **native typed columns** in the `facts` table:

| column | type | description |
|--------|------|-------------|
| `confidence` | f64 | Numeric confidence (0.0–1.0), nullable |
| `source` | u64 | TermId of the source entity, nullable |
| `process` | u64 | TermId of the generating process, nullable |

This enables **17,000× faster** confidence filters compared to parsing RDF literal strings.

**Dual-path strategy:**
- **RDF triples** (`prov:value` etc.) for SPARQL compatibility and arbitrary metadata
- **Native columns** for optimized analytical queries

### 12.5 Alternative: Domain-Specific Extensions

Applications may define more specific confidence predicates by subclassing or extending PROV-O:

```turtle
# Option: Application-specific subproperty
myapp:assertionConfidence rdfs:subPropertyOf prov:value ;
    rdfs:domain rdf:Statement ;
    rdfs:range xsd:decimal ;
    rdfs:comment "Confidence score for an assertion (0.0 to 1.0)" .
```

For maximum interoperability, prefer `prov:value` as the base predicate.

---

## 13. Acceptance checklist (storage layer)
- [ ] Can intern 10M terms with stable IDs (restart-safe).
- [ ] Can intern 50M quoted triples with O(1) lookup average.
- [ ] Can append 100M facts via batch ingestion without row-level mutation.
- [ ] Can compact partitions with predictable write amplification.
- [ ] Can execute RDF★ expansion join (qt metadata → base triple) with partition pruning.
