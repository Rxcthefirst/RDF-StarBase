# RDF-StarBase: Technical Architecture & Innovation

**A Provenance-Native RDF-Star Knowledge Engine for AI Grounding**

---

## Executive Summary

RDF-StarBase is a next-generation RDF-Star knowledge store that achieves **10-70x performance gains** over traditional RDF libraries while providing **native provenance tracking** for AI applications. Built on columnar storage (Polars), it combines the semantic richness of RDF with the analytical power of modern data engines.

### Key Innovations

| Innovation | Impact |
|------------|--------|
| Columnar RDF Storage | 72x faster FILTER queries |
| Dictionary-Encoded Terms | O(1) term comparison |
| Provenance-Native Design | First-class trust & attribution |
| AI Grounding API | Sub-100ms fact retrieval for LLMs |
| Drop-in rdflib Compatibility | Zero migration cost |

---

## Part 1: Performance Architecture

### 1.1 The Problem with Traditional RDF Stores

Traditional RDF libraries like **rdflib** use row-based, object-oriented storage:

```
┌─────────────────────────────────────────────────────────┐
│  Triple Object → Triple Object → Triple Object → ...    │
│  (Python dict)    (Python dict)    (Python dict)        │
└─────────────────────────────────────────────────────────┘
     ↓                  ↓                  ↓
  Heap allocation    Pointer chase     Cache miss
```

**Performance Consequences:**
- Each triple is a separate Python object (~200 bytes overhead)
- Iteration requires pointer chasing (cache unfriendly)
- String comparisons for every filter operation
- No vectorization possible

### 1.2 RDF-StarBase Columnar Architecture

RDF-StarBase uses **dictionary-encoded columnar storage**:

```
┌──────────────────────────────────────────────────────────┐
│  TermDict (String → Integer ID)                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ "http://xmlns.com/foaf/0.1/Person" → 42             │ │
│  │ "http://example.org/Alice"         → 1001           │ │
│  │ "Alice Johnson"                    → 5000           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  FactStore (Columnar Integer Arrays)                     │
│  ┌───────┬───────┬───────┬───────┬────────────┬───────┐ │
│  │   s   │   p   │   o   │   g   │ confidence │ source│ │
│  ├───────┼───────┼───────┼───────┼────────────┼───────┤ │
│  │ 1001  │  42   │ 5000  │   0   │    0.95    │  101  │ │
│  │ 1002  │  42   │ 5001  │   0   │    0.88    │  101  │ │
│  │ ...   │ ...   │ ...   │ ...   │    ...     │  ...  │ │
│  └───────┴───────┴───────┴───────┴────────────┴───────┘ │
└──────────────────────────────────────────────────────────┘
```

**Performance Advantages:**

| Aspect | rdflib | RDF-StarBase | Improvement |
|--------|--------|--------------|-------------|
| Storage per triple | ~200 bytes | ~48 bytes | 4x smaller |
| Term comparison | String (O(n)) | Integer (O(1)) | 10x faster |
| Filter scan | Python loop | SIMD vectorized | 50-100x faster |
| Memory locality | Random access | Sequential | Cache-friendly |

### 1.3 Vectorized Query Execution

RDF-StarBase leverages **Polars** for query execution, enabling:

#### SIMD Operations
Modern CPUs process 4-8 integers per cycle using AVX2/AVX-512:

```python
# rdflib: Python loop (1 comparison per iteration)
for triple in graph:
    if triple.predicate == rdf_type:  # String compare + Python overhead
        results.append(triple)

# RDF-StarBase: Polars vectorized (8+ comparisons per cycle)
df.filter(pl.col("p") == rdf_type_id)  # Compiles to SIMD instructions
```

#### Lazy Evaluation & Optimization
Polars builds a query plan and optimizes before execution:
- **Predicate pushdown**: Filters applied at scan time
- **Column pruning**: Only reads needed columns
- **Parallel execution**: Automatic multi-core utilization

### 1.4 Benchmark Results

**Test Configuration:**
- Dataset: 60,000 triples (20,000 entities × 3 predicates)
- Hardware: AMD Ryzen 12-core, 64GB RAM
- Polars 1.35.2 with 12 threads

| Query Type | rdflib | RDF-StarBase | Speedup |
|------------|--------|--------------|---------|
| **Parse Turtle (30K triples)** | 1.78s | 0.67s | **2.7x** |
| **Simple SELECT** | 1,231ms | 127ms | **9.7x** |
| **FILTER (numeric comparison)** | 2,096ms | 29ms | **72x** |
| **COUNT aggregation** | 283ms | 7ms | **42x** |
| **Serialize N-Triples** | 90ms | 91ms | 1.0x |

#### Why FILTER is 72x Faster

The `FILTER(?age > 80)` query demonstrates the full architectural advantage:

1. **Pre-typed Values**: Numeric literals stored as native Float64 in `object_value` column
2. **No Parsing**: rdflib parses `"80"^^xsd:integer` strings at query time
3. **SIMD Comparison**: Polars compares entire column in parallel
4. **Zero Object Creation**: No URIRef/Literal objects until final output

```sql
-- This SPARQL query:
SELECT ?person ?age WHERE {
    ?person foaf:age ?age .
    FILTER(?age > 80)
}

-- Translates to this Polars operation:
df.filter(pl.col("age_value") > 80)  # Single vectorized operation
```

---

## Part 2: Provenance-Native Design

### 2.1 RDF-Star: Statements About Statements

Traditional RDF can only make statements about things:
```turtle
ex:Alice foaf:age 30 .
```

RDF-Star allows statements about statements (quoted triples):
```turtle
<< ex:Alice foaf:age 30 >> prov:wasAttributedTo "HR_System" .
<< ex:Alice foaf:age 30 >> prov:value 0.95 .
<< ex:Alice foaf:age 30 >> prov:generatedAtTime "2026-01-15T10:00:00Z" .
```

### 2.2 Provenance as First-Class Columns

RDF-StarBase doesn't store provenance as separate triples. It's **built into the fact schema**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FactStore Schema                                 │
├─────────┬─────────┬─────────┬─────────┬────────────┬────────────────────┤
│ subject │ predic. │ object  │  graph  │ confidence │ source │ timestamp │
├─────────┼─────────┼─────────┼─────────┼────────────┼────────┼───────────┤
│ u64     │ u64     │ u64     │ u64     │ f64        │ u64    │ datetime  │
└─────────┴─────────┴─────────┴─────────┴────────────┴────────┴───────────┘
```

**Advantages:**
- No extra joins to get provenance
- Filter by confidence in the same scan
- Zero storage overhead for provenance metadata

### 2.3 Competing Claims Model

Unlike traditional databases that overwrite, RDF-StarBase **preserves competing assertions**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Subject: ex:Customer/123                                                │
│ Predicate: foaf:age                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Claim 1: age = 34  │ Source: CRM      │ Confidence: 0.95 │ 2026-01-15  │
│ Claim 2: age = 36  │ Source: DataLake │ Confidence: 0.72 │ 2025-12-01  │
└─────────────────────────────────────────────────────────────────────────┘
```

The AI Grounding API can:
- Return the highest-confidence claim
- Surface conflicts for human review
- Apply trust policies per source

---

## Part 3: AI Grounding API

### 3.1 The Problem: Ungrounded AI

Large Language Models hallucinate because they lack:
- **Source attribution**: Where did this fact come from?
- **Confidence scores**: How reliable is this information?
- **Temporal context**: Is this still current?
- **Conflict awareness**: Are there competing claims?

### 3.2 Grounded Facts for RAG

The AI Grounding API provides facts with full provenance chain:

```python
# Request
POST /ai/query
{
    "subject": "http://example.org/customer/123",
    "min_confidence": "high",  # >= 0.9
    "max_age_days": 30,
    "include_inferred": true
}

# Response
{
    "facts": [
        {
            "subject": "http://example.org/customer/123",
            "predicate": "http://xmlns.com/foaf/0.1/name",
            "object": "Alice Johnson",
            "citation": {
                "source": "CRM_System",
                "confidence": 0.95,
                "timestamp": "2026-01-15T10:30:00Z",
                "fact_hash": "a1b2c3..."
            }
        }
    ],
    "sources_used": ["CRM_System", "ERP"],
    "retrieval_timestamp": "2026-01-18T14:22:00Z"
}
```

### 3.3 Claim Verification

Before an LLM asserts something, it can verify against the knowledge base:

```python
POST /ai/verify
{
    "subject": "http://example.org/customer/123",
    "predicate": "http://example.org/hasStatus",
    "expected_object": "Premium",
    "min_confidence": "medium"
}

# Response
{
    "claim_supported": true,
    "confidence": 0.92,
    "supporting_facts": [...],
    "contradicting_facts": [],
    "has_conflicts": false,
    "recommendation": "Claim is well-supported by CRM_System with high confidence."
}
```

### 3.4 Entity Context for Grounding

Get complete context about an entity for LLM consumption:

```python
GET /ai/context/http://example.org/customer/123

# Response includes:
# - All facts about the entity
# - Related entities (graph neighborhood)
# - Source breakdown
# - Confidence distribution
```

### 3.5 Inference Materialization

RDF-StarBase includes an RDFS/OWL reasoner that materializes inferences **with provenance**:

```python
POST /ai/materialize
{
    "enable_rdfs": true,
    "enable_owl": true,
    "max_iterations": 100
}

# Inferred facts are marked:
{
    "subject": "http://example.org/Alice",
    "predicate": "rdf:type",
    "object": "http://xmlns.com/foaf/0.1/Agent",  # Inferred from foaf:Person
    "is_inferred": true,
    "citation": {
        "source": "RDFS_Reasoner",
        "confidence": 1.0,  # Logical entailment
        "process": "rdfs:subClassOf transitivity"
    }
}
```

---

## Part 4: Drop-in rdflib Compatibility

### 4.1 Migration Path

RDF-StarBase provides a compatibility layer that mirrors the rdflib API:

```python
# Before (rdflib)
from rdflib import Graph, URIRef, Literal, Namespace

g = Graph()
g.parse("data.ttl", format="turtle")
for s, p, o in g.triples((None, RDF.type, FOAF.Person)):
    print(s)

# After (RDF-StarBase) - just change the import
from rdf_starbase.compat.rdflib import Graph, URIRef, Literal, Namespace

g = Graph()
g.parse("data.ttl", format="turtle")
for s, p, o in g.triples((None, RDF.type, FOAF.Person)):
    print(s)  # Same API, 10-70x faster
```

### 4.2 What's Compatible

| Feature | Status |
|---------|--------|
| `Graph.parse()` (Turtle, N-Triples, RDF/XML) | ✅ |
| `Graph.serialize()` | ✅ |
| `Graph.triples()` | ✅ |
| `Graph.query()` (SPARQL) | ✅ |
| `URIRef`, `Literal`, `BNode` | ✅ |
| `Namespace`, `NamespaceManager` | ✅ |
| `RDF`, `RDFS`, `OWL`, `FOAF`, `XSD` | ✅ |

### 4.3 What's Added

Beyond rdflib compatibility, RDF-StarBase adds:

```python
from rdf_starbase.compat.rdflib import Graph

g = Graph()
g.parse("data.ttl")

# Provenance-aware queries (RDF-StarBase extension)
results = g.query("""
    SELECT ?person ?name ?confidence
    WHERE {
        ?person foaf:name ?name .
        << ?person foaf:name ?name >> prov:value ?confidence .
    }
    FILTER(?confidence > 0.8)
""")

# Direct access to underlying Polars DataFrame
df = g._store._df  # Full analytical power
```

---

## Part 5: Use Cases

### 5.1 AI-Powered Knowledge Retrieval

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   LLM/RAG   │────▶│ AI Grounding API │────▶│  RDF-StarBase   │
│   System    │◀────│  (sub-100ms)     │◀────│ (Columnar RDF)  │
└─────────────┘     └──────────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │ Grounded Facts   │
                    │ with Citations   │
                    └──────────────────┘
```

### 5.2 Enterprise Data Governance

- **Competing Claims**: See when systems disagree
- **Provenance Timeline**: Track assertion history
- **Trust Policies**: Filter by source reliability
- **Audit Trail**: Full lineage for compliance

### 5.3 Knowledge Graph Analytics

- **SPARQL at Scale**: 72x faster than rdflib
- **Columnar Analytics**: Direct Polars integration
- **Inference Materialization**: RDFS/OWL reasoning

---

## Part 6: Technical Specifications

### 6.1 Storage Format

```
TermDict:
  - term_id: UInt64 (tagged: 2 bits kind + 62 bits payload)
  - kind: {IRI=0, LITERAL=1, BNODE=2, QUOTED_TRIPLE=3}
  - lex: String (lexical form)
  - datatype_id: UInt64 (for typed literals)
  - lang: String (for language-tagged literals)

FactStore:
  - g, s, p, o: UInt64 (term IDs)
  - flags: UInt8 (ASSERTED, INFERRED, DELETED)
  - t_added: UInt64 (microseconds since epoch)
  - confidence: Float64
  - source, process: UInt64 (term IDs)
```

### 6.2 Query Execution Pipeline

```
SPARQL Query
     │
     ▼
┌─────────────┐
│   Parser    │  SPARQLStarParser → AST
└─────────────┘
     │
     ▼
┌─────────────┐
│  Executor   │  AST → Polars LazyFrame operations
└─────────────┘
     │
     ▼
┌─────────────┐
│   Polars    │  Query optimization + vectorized execution
│   Engine    │  (predicate pushdown, parallel scan)
└─────────────┘
     │
     ▼
┌─────────────┐
│   Result    │  DataFrame → URIRef/Literal objects
│ Conversion  │  (only at output boundary)
└─────────────┘
```

### 6.3 Test Coverage

- **491 tests passing**
- **72% code coverage**
- **42 rdflib compatibility tests**

---

## Conclusion

RDF-StarBase represents a fundamental rethinking of RDF storage for the AI era:

1. **Performance**: Columnar architecture delivers analytical database speeds
2. **Provenance**: Every fact carries its trust chain
3. **AI-Ready**: Purpose-built API for grounded knowledge retrieval
4. **Compatible**: Drop-in replacement for existing rdflib code

**RDF-StarBase is not just faster RDF. It's RDF designed for a world where AI needs to know what to trust.**

---

## Appendix: Running the Benchmarks

```bash
# Install
pip install -e .
pip install rdflib  # For comparison

# Run benchmark
python benchmarks/vs_rdflib.py

# Run full test suite
pytest tests/ -q
```

---

*RDF-StarBase — A Product of Ontus*
