# RDF-StarBase Benchmark Results

## Summary

**Date**: 2026-01-16  
**Engine**: RDF-StarBase v0.1.0  
**Hardware**: AMD Ryzen 9 (6 cores/12 threads), 64GB RAM  
**Polars**: 1.35.2 (12 threads)

---

## 🚀 Latest Results: Optimized Fast Benchmark

### Medium Scale (1.9M total facts)

| Metric | Value |
|--------|-------|
| **Base Facts** | 1,000,000 |
| **Metadata Facts** | 901,376 |
| **Quoted Triples** | 300,444 |
| **Total Terms** | 325,179 |
| **Ingestion Time** | 19.3 sec |
| **Base Facts/sec** | **51,843** |
| **Total Facts/sec** | **98,573** |
| **Peak Memory** | 463 MB |

**Query Performance** (warm p50):
| Query | Time (ms) | Rows |
|-------|-----------|------|
| Q1 (SPO exact) | 3.10 | 12,290 |
| Q4 (predicate fanout) | 6.05 | 143,437 |
| Q6 (QT metadata) | 3.71 | 537 |
| Q7 (by source) | 34.51 | 5,121 |
| Q11 (count by source) | 18.24 | 100 |
| Q12 (count by run) | 16.58 | 50 |

### Large Scale (19M total facts)

| Metric | Value |
|--------|-------|
| **Base Facts** | 10,000,000 |
| **Metadata Facts** | 8,996,820 |
| **Quoted Triples** | 2,998,768 |
| **Total Terms** | 2,672,073 |
| **Ingestion Time** | 186.8 sec (3.1 min) |
| **Base Facts/sec** | **53,523** |
| **Total Facts/sec** | **101,677** |
| **Peak Memory** | 2,995 MB (3 GB) |

**Query Performance** (warm p50):
| Query | Time (ms) | Rows |
|-------|-----------|------|
| Q1 (SPO exact) | 20.89 | 119,012 |
| Q4 (predicate fanout) | 47.26 | 1,431,011 |
| Q6 (QT metadata) | 31.20 | 5,805 |
| Q7 (by source) | 1,094 | 261,711 |
| Q11 (count by source) | 145 | 100 |
| Q12 (count by run) | 144 | 50 |

### XLarge Scale (95M total facts) 🔥

| Metric | Value |
|--------|-------|
| **Base Facts** | 50,000,000 |
| **Metadata Facts** | 44,996,807 |
| **Quoted Triples** | 14,999,392 |
| **Total Terms** | 10,461,914 |
| **Ingestion Time** | 1080.4 sec (18 min) |
| **Base Facts/sec** | **46,280** |
| **Total Facts/sec** | **87,929** |
| **Peak Memory** | 13,528 MB (13.2 GB) |

**Query Performance** (warm p50):
| Query | Time (ms) | Rows |
|-------|-----------|------|
| Q1 (SPO exact) | 126.45 | 591,093 |
| Q4 (predicate fanout) | 305.34 | 7,155,941 |
| Q6 (QT metadata) | 163.04 | 26,301 |
| Q7 (by source) | 26,331 | 6,084,394 |
| Q11 (count by source) | 793.39 | 100 |
| Q12 (count by run) | 784.74 | 50 |

---

## Performance Comparison

### Ingestion Throughput

| Benchmark | Scale | Total Facts | Time | Facts/sec |
|-----------|-------|-------------|------|-----------|
| Original | 1.9M | 1.9M | 23 min | 1,382 |
| **Optimized** | Medium | 1.9M | 19s | **98,573** |
| **Optimized** | Large | 19M | 3.1m | **101,677** |
| **Optimized** | XLarge | 95M | 18m | **87,929** |

### Target Goals (from benchmarks.md)

| Goal | Target | Achieved | Status |
|------|--------|----------|--------|
| Ingestion | 50K-100K facts/sec | **88K-102K facts/sec** | ✅ |
| Q7 (by source) | <100ms at 1M | 35ms at 1.9M | ✅ |
| Memory | <4GB at 10M | 3GB at 19M facts | ✅ |
| Scale | 50M+ facts | **95M facts** | ✅ |

### Industry Comparison

| Engine | Expected Rate | RDF-StarBase |
|--------|---------------|--------------|
| RDFox | 100K-500K facts/sec | **102K ✓** |
| Jena TDB2 | 10K-50K facts/sec | **102K ✓** |

---

## Previous Results (Original Benchmark)

### Small Scale (100K base facts)

| Metric | Tier A | Tier B |
|--------|--------|--------|
| Base Facts | 100,000 | 50,000 |
| Metadata Facts | 89,789 | 200,000 |
| Quoted Triples | 29,942 | 50,000 |
| Total Terms | 30,999 | 92,894 |
| Ingestion Time | 23.6s | 3.6s |
| Facts/sec | 4,233 | 13,895 |
| Meta/sec | 3,801 | 55,580 |
| Peak Memory | 115 MB | 151 MB |

### Query Performance (Small Scale, Warm p50 in ms)

| Query | Description | Tier A | Tier B |
|-------|-------------|--------|--------|
| Q1 | SPO exact match | 0.98 | 0.89 |
| Q4 | Predicate fanout | 1.28 | 0.93 |
| Q6 | QT metadata fetch | 1.25 | 0.05 |
| Q7 | Expand by source | 3.88 | 0.04 |
| Q11 | Count by source | 3.43 | 15.12 |
| Q12 | Count by run | 3.51 | 5.51 |

### Medium Scale (1M base facts)

| Metric | Value |
|--------|-------|
| Base Facts | 1,000,000 |
| Metadata Facts | 899,525 |
| Quoted Triples | 299,779 |
| Total Terms | 288,888 |
| Ingestion Time | 1375s (~23 min) |
| Facts/sec | 727 |
| Peak Memory | 420 MB |

---

## Analysis & Optimizations Applied

### Key Optimizations (71× speedup)

1. **Pre-interning entities**: Intern all entity URIs upfront before building facts
2. **Vectorized data generation**: Use NumPy for random index generation (Zipf distribution)
3. **Batch fact ingestion**: Build fact tuples in chunks of 50K and batch insert
4. **Reduced Python overhead**: Minimize per-fact dictionary lookups

### Remaining Opportunities

1. **Q7 at large scale**: 1.1 sec at 19M facts - could use predicate partitioning
2. **LSMStorage**: Enable Parquet persistence for predicate-pruned scans
3. **Columnar confidence**: Store as f64 for native filtering instead of string literals
