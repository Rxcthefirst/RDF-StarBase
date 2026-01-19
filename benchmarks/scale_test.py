"""
Scale test for RDF-StarBase.

Tests ingestion and query performance at different scales.
"""

import time
import gc
import polars as pl
from datetime import datetime, timezone

# Direct import to bypass any overhead
import sys
sys.path.insert(0, "src")

from rdf_starbase.storage.terms import TermDict, TermKind
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.facts import FactStore, FactFlags


def benchmark_ingestion_pure_polars(n_triples: int) -> dict:
    """
    Benchmark ingestion using pure Polars columnar construction.
    
    This is the theoretical maximum speed - build columns directly.
    """
    print(f"\n{'='*60}")
    print(f"Pure Polars Ingestion: {n_triples:,} triples")
    print('='*60)
    
    gc.collect()
    
    # Build columns directly (the Polars way)
    start = time.perf_counter()
    
    # Pre-allocate column data
    subjects = [f"http://example.org/entity/{i % 100000}" for i in range(n_triples)]
    predicates = [f"http://example.org/pred/{i % 50}" for i in range(n_triples)]
    objects = [f"value_{i}" for i in range(n_triples)]
    sources = ["benchmark"] * n_triples
    confidences = [0.95] * n_triples
    
    gen_time = time.perf_counter() - start
    print(f"  Data generation: {gen_time:.2f}s")
    
    # Create DataFrame from columns (fast path)
    start = time.perf_counter()
    df = pl.DataFrame({
        "subject": subjects,
        "predicate": predicates, 
        "object": objects,
        "source": sources,
        "confidence": confidences,
    })
    df_time = time.perf_counter() - start
    print(f"  DataFrame creation: {df_time:.2f}s")
    print(f"  Memory: {df.estimated_size('mb'):.1f} MB")
    
    # Simple filter query
    start = time.perf_counter()
    result = df.filter(pl.col("subject") == "http://example.org/entity/42")
    query_time = time.perf_counter() - start
    print(f"  Filter query: {query_time*1000:.2f}ms ({len(result)} rows)")
    
    # Aggregation
    start = time.perf_counter()
    agg = df.group_by("predicate").len()
    agg_time = time.perf_counter() - start
    print(f"  Group-by count: {agg_time*1000:.2f}ms")
    
    return {
        "n_triples": n_triples,
        "gen_time": gen_time,
        "df_time": df_time,
        "query_time": query_time,
        "agg_time": agg_time,
        "memory_mb": df.estimated_size('mb'),
        "triples_per_sec": n_triples / (gen_time + df_time),
    }


def benchmark_factstore_ingestion(n_triples: int) -> dict:
    """
    Benchmark ingestion through FactStore with term interning.
    
    This tests the actual storage layer overhead.
    """
    print(f"\n{'='*60}")
    print(f"FactStore Ingestion: {n_triples:,} triples")
    print('='*60)
    
    gc.collect()
    
    # Initialize storage
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    # Batch intern terms first
    start = time.perf_counter()
    
    # Build unique term sets
    n_entities = min(n_triples, 100000)
    n_preds = 50
    n_objects = min(n_triples, 50000)  # Realistic: limited unique object values
    
    entity_ids = [
        term_dict.intern_iri(f"http://example.org/entity/{i}")
        for i in range(n_entities)
    ]
    pred_ids = [
        term_dict.intern_iri(f"http://example.org/pred/{i}")
        for i in range(n_preds)
    ]
    # Pre-intern objects (realistic scenario)
    object_ids = [
        term_dict.intern_literal(f"value_{i}")
        for i in range(n_objects)
    ]
    source_id = term_dict.intern_literal("benchmark")
    
    intern_time = time.perf_counter() - start
    print(f"  Term interning ({n_entities + n_preds + n_objects} unique): {intern_time:.2f}s")
    
    # Build facts as integer tuples (now just lookups - fast!)
    start = time.perf_counter()
    
    facts = []
    for i in range(n_triples):
        s_id = entity_ids[i % n_entities]
        p_id = pred_ids[i % n_preds]
        o_id = object_ids[i % n_objects]  # Reuse pre-interned objects
        facts.append((0, s_id, p_id, o_id))  # g=0 (default graph)
    
    build_time = time.perf_counter() - start
    print(f"  Fact tuple building: {build_time:.2f}s")
    
    # Batch insert
    start = time.perf_counter()
    fact_store.add_facts_batch(
        facts,
        flags=FactFlags.ASSERTED,
        source=source_id,
        confidence=0.95,
    )
    insert_time = time.perf_counter() - start
    print(f"  Batch insert: {insert_time:.2f}s")
    
    total_time = intern_time + build_time + insert_time
    print(f"  TOTAL: {total_time:.2f}s ({n_triples/total_time:,.0f} triples/sec)")
    print(f"  Memory: {fact_store._df.estimated_size('mb'):.1f} MB")
    
    # Query performance
    start = time.perf_counter()
    target_s = entity_ids[42]
    result = fact_store._df.filter(pl.col("s") == target_s)
    query_time = time.perf_counter() - start
    print(f"  Filter query: {query_time*1000:.2f}ms ({len(result)} rows)")
    
    return {
        "n_triples": n_triples,
        "intern_time": intern_time,
        "build_time": build_time,
        "insert_time": insert_time,
        "total_time": total_time,
        "query_time": query_time,
        "memory_mb": fact_store._df.estimated_size('mb'),
        "triples_per_sec": n_triples / total_time,
    }


def benchmark_optimized_factstore(n_triples: int) -> dict:
    """
    Benchmark with optimized columnar construction.
    """
    print(f"\n{'='*60}")
    print(f"FactStore Ingestion: {n_triples:,} triples")
    print('='*60)
    
    gc.collect()
    
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    start = time.perf_counter()
    
    # Build unique term sets
    n_entities = min(n_triples, 100000)
    n_preds = 50
    n_objects = min(n_triples, 50000)
    
    entity_ids = [
        term_dict.intern_iri(f"http://example.org/entity/{i}")
        for i in range(n_entities)
    ]
    pred_ids = [
        term_dict.intern_iri(f"http://example.org/pred/{i}")
        for i in range(n_preds)
    ]
    object_ids = [
        term_dict.intern_literal(f"value_{i}")
        for i in range(n_objects)
    ]
    source_id = term_dict.intern_literal("benchmark")
    
    intern_time = time.perf_counter() - start
    print(f"  Term interning ({n_entities + n_preds + n_objects} unique): {intern_time:.2f}s")
    
    # Build columns using list comprehensions (faster than append loop)
    start = time.perf_counter()
    
    # Vectorized index computation
    indices = range(n_triples)
    s_col = [entity_ids[i % n_entities] for i in indices]
    p_col = [pred_ids[i % n_preds] for i in indices]
    o_col = [object_ids[i % n_objects] for i in indices]
    
    build_time = time.perf_counter() - start
    print(f"  Column building (list comp): {build_time:.2f}s")
    
    # Direct DataFrame construction (the Polars way)
    start = time.perf_counter()
    
    txn = fact_store._allocate_txn()
    t_added = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
    
    new_df = pl.DataFrame({
        "g": pl.Series([0] * n_triples, dtype=pl.UInt64),
        "s": pl.Series(s_col, dtype=pl.UInt64),
        "p": pl.Series(p_col, dtype=pl.UInt64),
        "o": pl.Series(o_col, dtype=pl.UInt64),
        "flags": pl.Series([int(FactFlags.ASSERTED)] * n_triples, dtype=pl.UInt16),
        "txn": pl.Series([txn] * n_triples, dtype=pl.UInt64),
        "t_added": pl.Series([t_added] * n_triples, dtype=pl.UInt64),
        "source": pl.Series([source_id] * n_triples, dtype=pl.UInt64),
        "confidence": pl.Series([0.95] * n_triples, dtype=pl.Float64),
        "process": pl.Series([0] * n_triples, dtype=pl.UInt64),
    })
    fact_store._df = pl.concat([fact_store._df, new_df], how="vertical")
    
    insert_time = time.perf_counter() - start
    print(f"  DataFrame insert: {insert_time:.2f}s")
    
    total_time = intern_time + build_time + insert_time
    print(f"  TOTAL: {total_time:.2f}s ({n_triples/total_time:,.0f} triples/sec)")
    print(f"  Memory: {fact_store._df.estimated_size('mb'):.1f} MB")
    
    # Query
    start = time.perf_counter()
    target_s = entity_ids[42]
    result = fact_store._df.filter(pl.col("s") == target_s)
    query_time = time.perf_counter() - start
    print(f"  Filter query: {query_time*1000:.2f}ms ({len(result)} rows)")
    
    return {
        "n_triples": n_triples,
        "total_time": total_time,
        "triples_per_sec": n_triples / total_time,
        "memory_mb": fact_store._df.estimated_size('mb'),
    }


if __name__ == "__main__":
    # Scale tests
    scales = [100_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]
    
    print("\n" + "="*60)
    print("RDF-StarBase Scale Benchmark")
    print("="*60)
    
    results = []
    
    for n in scales:
        try:
            # Test optimized path
            r = benchmark_optimized_factstore(n)
            results.append(r)
        except MemoryError:
            print(f"  ❌ MemoryError at {n:,} triples")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Triples':>12} | {'Time (s)':>10} | {'Triples/sec':>15} | {'Memory (MB)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['n_triples']:>12,} | {r['total_time']:>10.2f} | {r['triples_per_sec']:>15,.0f} | {r['memory_mb']:>12.1f}")
