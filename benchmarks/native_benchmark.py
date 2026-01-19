"""
Native Column Benchmark - Demonstrates native confidence/source columns.

This benchmark stores provenance (confidence, source) in native columns
instead of as separate RDF triples, showing the performance difference.

Usage:
    python -m benchmarks.native_benchmark --scale medium
"""

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List
import statistics

import numpy as np
import polars as pl
import psutil

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase.storage import (
    TermDict, QtDict, FactStore, ExpansionPatterns, DEFAULT_GRAPH_ID,
)


SCALE_CONFIGS = {
    "tiny": {"entities": 1_000, "facts": 10_000},
    "small": {"entities": 10_000, "facts": 100_000},
    "medium": {"entities": 100_000, "facts": 1_000_000},
    "large": {"entities": 500_000, "facts": 10_000_000},
}


def get_memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def generate_and_ingest_native(
    n_entities: int,
    n_facts: int,
    n_predicates: int = 200,
    n_sources: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate and ingest facts with NATIVE confidence/source columns.
    
    Each fact gets:
    - confidence: random float 0.5-1.0 stored in native column
    - source: random source IRI stored in native column
    """
    print(f"  Generating {n_facts:,} facts with native provenance...")
    start = time.perf_counter()
    
    # Initialize storage
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    rng = np.random.default_rng(seed)
    
    # Pre-intern entities
    print("    Pre-interning terms...")
    entity_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Entity_{i:06d}")
        for i in range(n_entities)
    ], dtype=np.uint64)
    
    pred_ids = np.array([
        term_dict.intern_iri(f"http://example.org/pred_{i:03d}")
        for i in range(n_predicates)
    ], dtype=np.uint64)
    
    source_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Source_{i:03d}")
        for i in range(n_sources)
    ], dtype=np.uint64)
    
    # Zipf weights
    entity_weights = 1.0 / (np.arange(1, n_entities + 1) ** 1.2)
    entity_weights /= entity_weights.sum()
    pred_weights = 1.0 / (np.arange(1, n_predicates + 1) ** 1.5)
    pred_weights /= pred_weights.sum()
    
    # Generate indices
    s_indices = rng.choice(n_entities, size=n_facts, p=entity_weights)
    o_indices = rng.choice(n_entities, size=n_facts, p=entity_weights)
    p_indices = rng.choice(n_predicates, size=n_facts, p=pred_weights)
    source_indices = rng.integers(0, n_sources, size=n_facts)
    confidences = rng.random(n_facts) * 0.5 + 0.5  # 0.5 to 1.0
    
    gen_time = time.perf_counter() - start
    print(f"    Generated data in {gen_time:.2f}s")
    
    # Ingest in chunks using native provenance columns
    print("    Ingesting with native columns...")
    ingest_start = time.perf_counter()
    peak_memory = get_memory_mb()
    chunk_size = 50_000
    
    for chunk_start in range(0, n_facts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_facts)
        
        # Build facts with per-fact provenance: (g, s, p, o, source, confidence, process)
        facts = []
        for i in range(chunk_start, chunk_end):
            facts.append((
                DEFAULT_GRAPH_ID,
                int(entity_ids[s_indices[i]]),
                int(pred_ids[p_indices[i]]),
                int(entity_ids[o_indices[i]]),
                int(source_ids[source_indices[i]]),  # source
                float(confidences[i]),                # confidence
                None,                                 # process
            ))
        
        fact_store.add_facts_with_provenance(facts)
        
        current_memory = get_memory_mb()
        peak_memory = max(peak_memory, current_memory)
        
        if chunk_end % 200_000 == 0 or chunk_end == n_facts:
            elapsed = time.perf_counter() - ingest_start
            rate = chunk_end / elapsed
            print(f"      {chunk_end:,}/{n_facts:,} ({rate:,.0f} facts/sec)")
    
    elapsed = time.perf_counter() - start
    
    return {
        "term_dict": term_dict,
        "qt_dict": qt_dict,
        "fact_store": fact_store,
        "n_facts": n_facts,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory,
        "facts_per_second": n_facts / elapsed,
    }


def benchmark_q9_comparison(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    n_warm: int = 10,
) -> Dict[str, Any]:
    """
    Compare Q9 implementations:
    1. Native column (new fast path)
    2. Join-based (current implementation for RDF triples)
    """
    patterns = ExpansionPatterns(term_dict, qt_dict, fact_store)
    results = {}
    
    # Q9 Native (uses native confidence column)
    print("  Q9 Native (confidence column)...")
    times = []
    for _ in range(n_warm):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q9_native_filter_by_confidence(0.8, expand_lex=False)
        times.append((time.perf_counter() - start) * 1000)
    
    results["q9_native_ids"] = {
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(max(times), 2),
        "rows": len(df),
    }
    
    # Q9 Native count
    print("  Q9 Native count...")
    times = []
    for _ in range(n_warm):
        start = time.perf_counter()
        count = patterns.q9_native_count(0.8)
        times.append((time.perf_counter() - start) * 1000)
    
    results["q9_native_count"] = {
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(max(times), 2),
        "count": count,
    }
    
    # Scan by source (native column)
    print("  Scan by source (native column)...")
    source_id = term_dict.lookup_iri("http://example.org/Source_001")
    times = []
    for _ in range(n_warm):
        start = time.perf_counter()
        df = fact_store.scan_by_source(source_id)
        times.append((time.perf_counter() - start) * 1000)
    
    results["scan_by_source"] = {
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(max(times), 2),
        "rows": len(df),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Native column benchmark")
    parser.add_argument("--scale", choices=list(SCALE_CONFIGS.keys()), default="small")
    args = parser.parse_args()
    
    config = SCALE_CONFIGS[args.scale]
    
    print("=" * 70)
    print(f"Native Column Benchmark: Scale {args.scale}")
    print("=" * 70)
    print(f"Config: {config}")
    print()
    
    # Generate and ingest
    result = generate_and_ingest_native(
        n_entities=config["entities"],
        n_facts=config["facts"],
    )
    
    print(f"\n  Ingestion Summary:")
    print(f"    Facts:          {result['n_facts']:>12,}")
    print(f"    Time:           {result['elapsed_seconds']:>12.1f} sec")
    print(f"    Facts/sec:      {result['facts_per_second']:>12,.0f}")
    print(f"    Peak memory:    {result['peak_memory_mb']:>12.0f} MB")
    
    # Run Q9 comparison
    print("\n  Query Benchmarks:")
    query_results = benchmark_q9_comparison(
        result["term_dict"],
        result["qt_dict"],
        result["fact_store"],
    )
    
    print(f"\n  Results:")
    for name, metrics in query_results.items():
        count_or_rows = metrics.get("count", metrics.get("rows", 0))
        print(f"    {name:<25} {metrics['p50_ms']:>8.2f} ms ({count_or_rows:,} results)")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
