# Patent Guidance for Development

> Ensuring enhancements align with and strengthen RDF-StarBase intellectual property

## Overview

RDF-StarBase has a pending patent covering its core architecture. This document helps contributors understand what aspects are protected and how to develop features that strengthen (not conflict with) the patent claims.

**Reference:** [rdf-starbase-invention-disclosure.md](rdf-starbase-invention-disclosure.md)

---

## Core Patent Claims

### Core Invention A: Conditional RDF* Semantic Materialization

**What's protected:**
- RDF* metadata stored **separately** from base triples
- Queries not referencing RDF* semantics **bypass metadata entirely**
- RDF* semantics resolved **only when explicitly referenced**
- "Pay-as-you-go" semantic enrichment

**Implementation touchpoints:**
- `storage/facts.py` — Base triple storage
- `storage/quoted_triples.py` — Separate RDF* metadata storage
- `executor.py` — Conditional resolution logic

**When developing:**
- ✅ Keep RDF* metadata in separate columnar structures
- ✅ Ensure non-RDF* queries have zero overhead from RDF* features
- ✅ Add metrics showing performance difference between RDF and RDF* queries
- ❌ Don't merge RDF* annotations into base triple storage
- ❌ Don't materialize all quoted triples at load time

---

### Core Invention B: Analytical Execution of Graph Queries

**What's protected:**
- SPARQL patterns transformed to **columnar analytical operations**
- Vectorized joins, batch filtering instead of pointer chasing
- Graph semantics via analytical operators (not graph traversal)

**Implementation touchpoints:**
- `executor.py` — Query plan generation
- `storage/indexing.py` — B-tree sorted indexes
- Polars DataFrame operations throughout

**When developing:**
- ✅ Express graph operations as DataFrame joins/filters
- ✅ Use vectorized operations (Polars lazy evaluation)
- ✅ Benchmark against graph-native implementations
- ❌ Don't introduce pointer-based graph traversal
- ❌ Don't add row-by-row execution paths

---

### Extension 1: Ontology & Provenance Indexing

**What's protected:**
- Ontology/provenance data in **columnar physical indexes**
- Semantic metadata as **first-class execution inputs**
- Provenance-aware filtering without reification

**Implementation touchpoints:**
- `storage/trust.py` — Trust scoring, confidence
- `storage/reasoner.py` — RDFS/OWL entailment
- Provenance predicates (prov:*, dct:*)

**When developing:**
- ✅ Store provenance in indexed columnar structures
- ✅ Apply ontology constraints during execution (not post-processing)
- ✅ Enable provenance filtering in query plans

---

### Extension 2: Unified SPARQL/SQL Interface

**What's protected:**
- Both SPARQL and SQL mapped to **same columnar engine**
- No separate relational materialization
- RDF semantics preserved during SQL execution

**Implementation touchpoints:**
- `storage/duckdb.py` — SQL interface
- `executor.py` — SPARQL execution
- Shared term dictionary

**When developing:**
- ✅ Keep single physical storage for both interfaces
- ✅ Ensure SQL queries use same term encoding
- ❌ Don't create separate SQL-specific storage
- ❌ Don't duplicate data for relational queries

---

## Feature Development Checklist

Before implementing a new feature, consider:

### Does this feature strengthen the patent?

| Question | Good Answer |
|----------|-------------|
| Does it use columnar/vectorized execution? | Yes |
| Does it maintain separation of RDF* metadata? | Yes |
| Does it add conditional (pay-as-you-go) behavior? | Yes |
| Does it unify graph and analytical semantics? | Yes |
| Can we benchmark it against prior art? | Yes |

### Could this feature conflict with the patent?

| Warning Sign | Action |
|--------------|--------|
| Introduces graph traversal primitives | Reconsider approach |
| Materializes all RDF* at load time | Use conditional resolution |
| Creates separate storage for SQL | Use unified substrate |
| Adds row-by-row processing | Use vectorized operations |

---

## Benchmarking for Patent Support

Performance benchmarks strengthen patent claims. Ensure benchmarks demonstrate:

1. **RDF* overhead isolation**
   - Compare query time: RDF-only vs RDF* queries
   - Show near-zero overhead for non-RDF* workloads

2. **Analytical vs graph-native performance**
   - Compare against rdflib, Jena, Virtuoso
   - Show vectorized execution advantages

3. **Unified interface efficiency**
   - Compare SPARQL vs SQL on same data
   - Show no performance penalty for interface choice

4. **Provenance/ontology selectivity**
   - Show cost scales with usage, not presence
   - Benchmark with/without semantic constraints

---

## Documentation for Patent

When documenting features, include:

1. **Technical mechanism** — How it works internally
2. **Prior art differentiation** — Why this is novel
3. **Performance characteristics** — Benchmarks, complexity
4. **Conditional behavior** — When overhead is incurred

---

## Key Terms (Patent Language)

Use consistent terminology in code and docs:

| Term | Meaning |
|------|---------|
| Columnar substrate | Polars DataFrames / Parquet storage |
| Conditional materialization | Resolve RDF* only when queried |
| Analytical operators | Vectorized joins, batch filters |
| Unified execution engine | Single engine for SPARQL + SQL |
| Physical indexing | Columnar indexes (not logical overlays) |
| Semantic enrichment | Ontology/provenance participation |

---

## Contact

Questions about patent-sensitive development should be directed to the project leads before implementation.
