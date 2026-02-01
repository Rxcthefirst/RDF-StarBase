# RDF-StarBase: Rust/Columnar Performance Roadmap

## Current State vs. Oxigraph Analysis

### What Oxigraph Does Differently

Based on [Oxigraph's Architecture](https://github.com/oxigraph/oxigraph/wiki/Architecture):

| Component | Oxigraph | RDF-StarBase | Gap |
|-----------|----------|--------------|-----|
| **Language** | 96.6% Rust | 100% Python + Polars | ~10-50x overhead for hot paths |
| **Parsing** | Rust lexer/parser (oxttl) | Python TurtleParser | ~17x slower (measured) |
| **Term Encoding** | 128-bit hash, inline small strings <16B | Python dict + integer IDs | Similar concept, Python overhead |
| **Storage** | RocksDB (LSMT) or in-memory | Polars DataFrames (Rust) | **Already fast!** |
| **Indexes** | SPOG, POSG, OSPG + 6 more | Polars lazy scans | Could add B-tree indexes |
| **Query Eval** | Volcano iterator model | Python pattern matching | Significant overhead |

### Where We ARE Fast (Already Rust)

Our Polars-based FactStore is **already Rust under the hood**:
- `add_facts_columnar`: **10.5M triples/sec** - faster than Oxigraph's insert!
- DataFrame concat and filtering: Rust-speed vectorized operations
- Memory layout: Arrow-compatible columnar format

### Where We ARE Slow (Pure Python)

1. **Parsing** (~68% of load time): Pure Python lexer/parser
2. **Term interning** (~14% of load time): Python dict lookups
3. **SPARQL execution**: Python pattern matching and joins
4. **Triple iteration**: Python object creation overhead

## Strategic Options

### Option 1: Hybrid Architecture (Recommended)

Keep our Polars columnar core, use Rust for hot paths:

```
┌─────────────────────────────────────────────────────┐
│                    Python API                        │
│  (User-facing: TripleStore, SPARQL, load_graph)     │
├─────────────────────────────────────────────────────┤
│              Rust Extensions (PyO3)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Parsing    │  │    Term     │  │   SPARQL    │  │
│  │  (oxttl)    │  │  Interning  │  │   Eval      │  │
│  │  468K/sec   │  │  HashBrown  │  │   Volcano   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────┤
│              Polars/Arrow (Already Rust)             │
│  ┌─────────────────────────────────────────────────┐│
│  │     FactStore: UInt64 columnar storage          ││
│  │     10.5M triples/sec insert rate               ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Option 2: Use Oxigraph as Backend

Replace our storage entirely with pyoxigraph:

```python
from pyoxigraph import Store

class TripleStore:
    def __init__(self):
        self._store = Store()  # Use Oxigraph directly
```

**Pros**: Immediate full Rust performance
**Cons**: Lose our RDF-Star metadata (confidence, source, timestamps)

### Option 3: Build Custom Rust Extension

Write a PyO3 extension specifically for RDF-StarBase:

```rust
// rdf_starbase_core/src/lib.rs
use pyo3::prelude::*;

#[pyclass]
struct TermDict {
    iri_to_id: hashbrown::HashMap<String, u64>,
    id_to_iri: Vec<String>,
}

#[pymethods]
impl TermDict {
    fn intern_batch(&mut self, terms: Vec<String>) -> Vec<u64> {
        // Rust-speed batch interning
    }
}
```

## Recommended Roadmap

### Phase 1: Quick Wins (Done ✅)

- [x] Integrate pyoxigraph for parsing (17x speedup)
- [x] Use `add_triples_columnar` for bulk loading
- [x] Streaming chunk loader for large files

**Result**: 92K triples/sec (up from 40K)

### Phase 2: Oxigraph SPARQL Integration (1-2 days)

Use Oxigraph's SPARQL engine for query evaluation:

```python
def execute_sparql_fast(store: TripleStore, query: str):
    # Export to Oxigraph, run query, return results
    ox_store = pyoxigraph.Store()
    for quad in store.iter_quads():
        ox_store.add(quad)
    return ox_store.query(query)
```

**Expected**: 10-50x faster SPARQL execution

### Phase 3: Rust Term Dictionary (1 week)

Build PyO3 extension for term interning:

```toml
# Cargo.toml
[package]
name = "rdf_starbase_core"

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
hashbrown = "0.14"
polars = "0.35"
```

**Expected**: 5-10x faster term interning

### Phase 4: Full Rust Query Engine (2-4 weeks)

Implement Volcano-style iterator in Rust with Polars integration:

```rust
enum QueryPlan {
    Scan { pattern: TriplePattern },
    Filter { child: Box<QueryPlan>, expr: Expr },
    Join { left: Box<QueryPlan>, right: Box<QueryPlan> },
    Project { child: Box<QueryPlan>, vars: Vec<Var> },
}

fn execute(plan: QueryPlan, facts: &DataFrame) -> DataFrame {
    // Push down to Polars lazy expressions
}
```

## Performance Targets

| Metric | Current | With Oxigraph Parse | Full Rust |
|--------|---------|---------------------|-----------|
| Parse rate | 27K/sec | 468K/sec ✅ | 468K/sec |
| Bulk insert | 40K/sec | 150K/sec | 500K/sec |
| Term intern | 500K/sec | 500K/sec | 5M/sec |
| SPARQL simple | 50K/sec | 200K/sec | 1M/sec |
| SPARQL complex | 10K/sec | 100K/sec | 500K/sec |

## Competitor Comparison (Measured)

Benchmark run on `data/sample/data.ttl` (5,393 triples):

| Metric | RDF-StarBase | Oxigraph | rdflib | vs rdflib | vs Oxigraph |
|--------|-------------|----------|--------|-----------|-------------|
| **Load Rate** | 84K/sec | 276K/sec | 17K/sec | **5x faster** | 3.3x slower |
| **COUNT Query** | 5.95ms | 1.35ms | 97ms | **16x faster** | 4.4x slower |
| **Pattern Query** | 1.60ms | 0.90ms | 43ms | **27x faster** | 1.8x slower |
| **Filter Query** | 2.45ms | 2.15ms | 202ms | **82x faster** | 1.1x slower |

### Analysis

**Good News:**
- We massively outperform rdflib (pure Python)
- Our filter queries are nearly as fast as Oxigraph (Polars doing work!)
- Architecture is sound - overhead is in Python glue code

**Gaps to Close:**
1. **Loading (3.3x gap)**: Term interning in Python dicts
2. **COUNT query (4.4x gap)**: Python SPARQL execution overhead
3. **Pattern query (1.8x gap)**: Small gap, nearly competitive

### Path to Parity

To match Oxigraph performance:

## Key Insight

**Our architecture is sound.** The Polars FactStore at 10.5M triples/sec proves the columnar approach works. The bottleneck is Python overhead in:

1. Parsing (solved with Oxigraph)
2. Term interning (solvable with PyO3)
3. Query execution (solvable with Oxigraph or custom Rust)

We don't need to rewrite everything - we need to **wrap Rust libraries for hot paths** while keeping our Python API for flexibility and our unique RDF-Star metadata features.

## Next Steps

1. **Immediate**: Run competitor benchmarks (rdflib, Oxigraph Store)
2. **This week**: Integrate Oxigraph SPARQL for query execution
3. **Next sprint**: Evaluate PyO3 for term dictionary
4. **Future**: Consider full Rust extension for query engine
