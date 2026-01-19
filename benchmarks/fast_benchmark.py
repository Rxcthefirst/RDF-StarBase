"""
Fast RDF★ Benchmark with Optimized Batch Ingestion.

Uses vectorized operations for much faster data generation and ingestion.

Usage:
    python -m benchmarks.fast_benchmark --tier A --scale medium
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

import numpy as np
import polars as pl
import psutil
import platform

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase.storage import (
    TermDict, QtDict, FactStore, LSMStorage,
    ExpansionPatterns, DEFAULT_GRAPH_ID, TermKind,
)
from rdf_starbase.storage.terms import Term


# =============================================================================
# Scale Configurations
# =============================================================================

SCALE_CONFIGS = {
    "tiny": {"tier_a": {"entities": 1_000, "facts": 10_000}, "tier_b": {"runs": 10, "facts_per_run": 100}},
    "small": {"tier_a": {"entities": 10_000, "facts": 100_000}, "tier_b": {"runs": 100, "facts_per_run": 500}},
    "medium": {"tier_a": {"entities": 100_000, "facts": 1_000_000}, "tier_b": {"runs": 1_000, "facts_per_run": 500}},
    "large": {"tier_a": {"entities": 500_000, "facts": 10_000_000}, "tier_b": {"runs": 5_000, "facts_per_run": 1_000}},
    "xlarge": {"tier_a": {"entities": 1_000_000, "facts": 50_000_000}, "tier_b": {"runs": 10_000, "facts_per_run": 5_000}},
}


def get_memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def get_environment_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "polars_version": pl.__version__,
        "polars_threads": pl.thread_pool_size(),
    }


# =============================================================================
# Fast Batch Ingestion
# =============================================================================

def fast_generate_tier_a(
    n_entities: int,
    n_facts: int,
    n_predicates: int = 200,
    metadata_rate: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate Tier A data using vectorized NumPy operations.
    
    Returns raw arrays for bulk processing.
    """
    print(f"  Generating {n_facts:,} base facts (vectorized)...")
    start = time.perf_counter()
    
    rng = np.random.default_rng(seed)
    
    # Zipf distribution for predicates
    pred_weights = 1.0 / (np.arange(1, n_predicates + 1) ** 1.5)
    pred_weights /= pred_weights.sum()
    
    # Entity weights
    entity_weights = 1.0 / (np.arange(1, n_entities + 1) ** 1.2)
    entity_weights /= entity_weights.sum()
    
    # Generate all indices at once (vectorized!)
    s_indices = rng.choice(n_entities, size=n_facts, p=entity_weights)
    o_indices = rng.choice(n_entities, size=n_facts, p=entity_weights)
    p_indices = rng.choice(n_predicates, size=n_facts, p=pred_weights)
    
    # Metadata selection
    has_metadata = rng.random(n_facts) < metadata_rate
    n_metadata = has_metadata.sum()
    
    # For metadata facts, generate random counts (2-4)
    meta_counts = rng.integers(2, 5, size=n_metadata)
    total_meta = meta_counts.sum()
    
    elapsed = time.perf_counter() - start
    print(f"    Generated indices in {elapsed:.2f}s")
    
    return {
        "s_indices": s_indices,
        "o_indices": o_indices,
        "p_indices": p_indices,
        "has_metadata": has_metadata,
        "meta_counts": meta_counts,
        "n_entities": n_entities,
        "n_predicates": n_predicates,
        "n_facts": n_facts,
        "n_metadata_facts": total_meta,
    }


def fast_ingest_tier_a(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ingest pre-generated data using optimized batch operations.
    """
    print(f"  Ingesting {data['n_facts']:,} base facts + ~{data['n_metadata_facts']:,} metadata...")
    
    start = time.perf_counter()
    peak_memory = get_memory_mb()
    
    n_entities = data["n_entities"]
    n_predicates = data["n_predicates"]
    n_facts = data["n_facts"]
    
    # Pre-intern all entity URIs in batch
    print("    Pre-interning entities...")
    entity_ids = []
    for i in range(n_entities):
        entity_ids.append(term_dict.intern_iri(f"http://example.org/Entity_{i:06d}"))
    entity_ids = np.array(entity_ids, dtype=np.uint64)
    
    # Pre-intern predicates
    print("    Pre-interning predicates...")
    pred_ids = []
    for i in range(n_predicates):
        pred_ids.append(term_dict.intern_iri(f"http://example.org/pred_{i:03d}"))
    pred_ids = np.array(pred_ids, dtype=np.uint64)
    
    # Metadata predicates
    derived_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
    time_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
    gen_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasGeneratedBy")
    conf_pred = term_dict.intern_iri("http://example.org/confidence")
    
    # Sources and processes
    n_sources = min(100, n_entities // 100 + 1)
    n_processes = min(50, n_entities // 200 + 1)
    source_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Source_{i:03d}")
        for i in range(n_sources)
    ], dtype=np.uint64)
    process_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Process_{i:03d}")
        for i in range(n_processes)
    ], dtype=np.uint64)
    
    rng = np.random.default_rng(42)
    
    # Build fact tuples in chunks
    print("    Building and ingesting facts...")
    chunk_size = 50000
    base_facts_count = 0
    meta_facts_count = 0
    qt_count = 0
    
    s_indices = data["s_indices"]
    o_indices = data["o_indices"]
    p_indices = data["p_indices"]
    has_metadata = data["has_metadata"]
    
    for chunk_start in range(0, n_facts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_facts)
        
        batch_facts = []
        
        for i in range(chunk_start, chunk_end):
            s = entity_ids[s_indices[i]]
            p = pred_ids[p_indices[i]]
            o = entity_ids[o_indices[i]]
            
            # Base fact
            batch_facts.append((DEFAULT_GRAPH_ID, s, p, o))
            base_facts_count += 1
            
            # Metadata?
            if has_metadata[i]:
                qt_id = qt_dict.get_or_create(s, p, o)
                qt_count += 1
                
                # Random metadata predicates
                n_meta = rng.integers(2, 5)
                meta_preds = [derived_pred, time_pred, gen_pred, conf_pred]
                rng.shuffle(meta_preds)
                
                for mp in meta_preds[:n_meta]:
                    if mp == derived_pred:
                        mo = source_ids[rng.integers(0, len(source_ids))]
                    elif mp == gen_pred:
                        mo = process_ids[rng.integers(0, len(process_ids))]
                    elif mp == time_pred:
                        ts = datetime(2025, 1, 1) + timedelta(seconds=int(rng.integers(0, 31536000)))
                        mo = term_dict.intern_literal(ts.isoformat())
                    elif mp == conf_pred:
                        mo = term_dict.intern_literal(str(round(rng.random() * 0.5 + 0.5, 3)))
                    else:
                        continue
                    
                    batch_facts.append((DEFAULT_GRAPH_ID, qt_id, mp, mo))
                    meta_facts_count += 1
        
        # Ingest batch
        fact_store.add_facts_batch(batch_facts)
        
        current_memory = get_memory_mb()
        peak_memory = max(peak_memory, current_memory)
        
        if chunk_end % 200000 == 0 or chunk_end == n_facts:
            elapsed = time.perf_counter() - start
            rate = (base_facts_count + meta_facts_count) / elapsed
            print(f"      {chunk_end:,}/{n_facts:,} ({rate:,.0f} facts/sec, {current_memory:.0f} MB)")
    
    elapsed = time.perf_counter() - start
    
    return {
        "base_facts": base_facts_count,
        "metadata_facts": meta_facts_count,
        "quoted_triples": qt_count,
        "total_terms": len(term_dict),
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory,
        "facts_per_second": base_facts_count / elapsed,
        "total_per_second": (base_facts_count + meta_facts_count) / elapsed,
    }


def fast_generate_tier_b(
    n_runs: int,
    facts_per_run: int,
    n_entities: int = 100_000,
    n_predicates: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate Tier B structure."""
    print(f"  Preparing Tier B: {n_runs:,} runs × {facts_per_run:,} facts...")
    
    total_facts = n_runs * facts_per_run
    rng = np.random.default_rng(seed)
    
    # For each fact, we need: run_idx, fact_idx, s_idx, p_idx, o_idx, input_idx
    run_indices = np.repeat(np.arange(n_runs), facts_per_run)
    fact_indices = np.tile(np.arange(facts_per_run), n_runs)
    
    # Entities per run (50 input entities)
    entities_per_run = 50
    
    return {
        "n_runs": n_runs,
        "facts_per_run": facts_per_run,
        "total_facts": total_facts,
        "n_entities": n_entities,
        "n_predicates": n_predicates,
        "entities_per_run": entities_per_run,
        "run_indices": run_indices,
        "fact_indices": fact_indices,
    }


def fast_ingest_tier_b(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Ingest Tier B data."""
    print(f"  Ingesting Tier B ({data['total_facts']:,} base facts, 4× metadata)...")
    
    start = time.perf_counter()
    peak_memory = get_memory_mb()
    
    n_runs = data["n_runs"]
    facts_per_run = data["facts_per_run"]
    n_entities = data["n_entities"]
    n_predicates = data["n_predicates"]
    entities_per_run = data["entities_per_run"]
    
    rng = np.random.default_rng(42)
    
    # Pre-intern entities
    print("    Pre-interning entities...")
    entity_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Entity_{i:06d}")
        for i in range(n_entities)
    ], dtype=np.uint64)
    
    # Predicates
    pred_ids = np.array([
        term_dict.intern_iri(f"http://example.org/pred_{i:03d}")
        for i in range(n_predicates)
    ], dtype=np.uint64)
    
    # Run URIs
    run_ids = np.array([
        term_dict.intern_iri(f"http://example.org/Run_{i:06d}")
        for i in range(n_runs)
    ], dtype=np.uint64)
    
    # Metadata predicates
    gen_by_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasGeneratedBy")
    time_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
    derived_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
    conf_pred = term_dict.intern_iri("http://example.org/confidence")
    
    base_time = datetime(2025, 1, 1)
    
    base_facts = 0
    meta_facts = 0
    qt_count = 0
    
    chunk_size = 100  # Runs per chunk
    
    for run_chunk_start in range(0, n_runs, chunk_size):
        run_chunk_end = min(run_chunk_start + chunk_size, n_runs)
        
        batch_facts = []
        
        for run_idx in range(run_chunk_start, run_chunk_end):
            run_id = run_ids[run_idx]
            run_time = base_time + timedelta(hours=run_idx)
            
            # Input entities for this run
            input_entity_indices = rng.choice(n_entities, size=entities_per_run, replace=False)
            
            for fact_idx in range(facts_per_run):
                s = entity_ids[input_entity_indices[rng.integers(0, entities_per_run)]]
                p = pred_ids[rng.integers(0, n_predicates)]
                o = entity_ids[rng.integers(0, n_entities)]
                
                # Base fact
                batch_facts.append((DEFAULT_GRAPH_ID, s, p, o))
                base_facts += 1
                
                # Quoted triple
                qt_id = qt_dict.get_or_create(s, p, o)
                qt_count += 1
                
                # 4 metadata facts
                batch_facts.append((DEFAULT_GRAPH_ID, qt_id, gen_by_pred, run_id))
                meta_facts += 1
                
                fact_time = run_time + timedelta(seconds=fact_idx)
                time_lit = term_dict.intern_literal(fact_time.isoformat())
                batch_facts.append((DEFAULT_GRAPH_ID, qt_id, time_pred, time_lit))
                meta_facts += 1
                
                derived_from = entity_ids[input_entity_indices[rng.integers(0, entities_per_run)]]
                batch_facts.append((DEFAULT_GRAPH_ID, qt_id, derived_pred, derived_from))
                meta_facts += 1
                
                conf = round(rng.random() * 0.4 + 0.6, 3)
                conf_lit = term_dict.intern_literal(str(conf))
                batch_facts.append((DEFAULT_GRAPH_ID, qt_id, conf_pred, conf_lit))
                meta_facts += 1
        
        fact_store.add_facts_batch(batch_facts)
        
        current_memory = get_memory_mb()
        peak_memory = max(peak_memory, current_memory)
        
        if run_chunk_end % 500 == 0 or run_chunk_end == n_runs:
            elapsed = time.perf_counter() - start
            rate = (base_facts + meta_facts) / elapsed
            print(f"      {run_chunk_end:,}/{n_runs:,} runs ({rate:,.0f} facts/sec)")
    
    elapsed = time.perf_counter() - start
    
    return {
        "base_facts": base_facts,
        "metadata_facts": meta_facts,
        "quoted_triples": qt_count,
        "total_terms": len(term_dict),
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory,
        "facts_per_second": base_facts / elapsed,
        "total_per_second": (base_facts + meta_facts) / elapsed,
    }


# =============================================================================
# Query Benchmarks
# =============================================================================

def run_query_suite(
    patterns: ExpansionPatterns,
    term_dict: TermDict,
    fact_store: FactStore,
    n_cold: int = 3,
    n_warm: int = 10,
) -> List[Dict[str, Any]]:
    """Run query benchmarks."""
    
    results = []
    sample_entity = "http://example.org/Entity_000001"
    sample_pred = "http://example.org/pred_001"
    
    def bench_query(name, fn, n_cold=n_cold, n_warm=n_warm):
        cold_times = []
        warm_times = []
        result_count = 0
        
        for _ in range(n_cold):
            gc.collect()
            start = time.perf_counter()
            df = fn()
            result_count = len(df) if df is not None else 0
            cold_times.append((time.perf_counter() - start) * 1000)
        
        for _ in range(n_warm):
            start = time.perf_counter()
            fn()
            warm_times.append((time.perf_counter() - start) * 1000)
        
        return {
            "name": name,
            "cold_p50_ms": round(statistics.median(cold_times), 2),
            "cold_p95_ms": round(max(cold_times), 2),
            "warm_p50_ms": round(statistics.median(warm_times), 2),
            "warm_p95_ms": round(max(warm_times), 2),
            "result_count": result_count,
        }
    
    print("  Running query suite...")
    
    # Q1: SPO exact
    s_id = term_dict.lookup_iri(sample_entity)
    p_id = term_dict.lookup_iri(sample_pred)
    results.append(bench_query("Q1_spo_exact", lambda: (
        fact_store.scan_facts().filter((pl.col("s") == s_id) & (pl.col("p") == p_id)) if s_id and p_id else pl.DataFrame()
    )))
    
    # Q4: Predicate fanout
    results.append(bench_query("Q4_pred_fanout", lambda: (
        fact_store.scan_facts().filter(pl.col("p") == p_id) if p_id else pl.DataFrame()
    )))
    
    # Q6: QT metadata
    results.append(bench_query("Q6_qt_metadata", lambda: (
        patterns.q6_metadata_for_triple(sample_entity, sample_pred, "http://example.org/Entity_000002")
    )))
    
    # Q7: Expand by source
    results.append(bench_query("Q7_by_source", lambda: (
        patterns.q7_expand_by_source("http://example.org/Source_001")
    )))
    
    # Q9: Filter by confidence (fast - returns term IDs)
    results.append(bench_query("Q9_confidence_ids", lambda: (
        patterns.q9_filter_by_confidence(0.8, expand_lex=False)
    )))
    
    # Q9 count variant (very fast)
    results.append(bench_query("Q9_confidence_count", lambda: (
        pl.DataFrame({"count": [patterns.q9_count_by_confidence(0.8)]})
    )))
    
    # Q11: Count by source
    results.append(bench_query("Q11_count_source", lambda: patterns.q11_count_by_source()))
    
    # Q12: Count by run
    results.append(bench_query("Q12_count_run", lambda: patterns.q12_count_by_run()))
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast RDF★ benchmark")
    parser.add_argument("--tier", choices=["A", "B"], default="A")
    parser.add_argument("--scale", choices=list(SCALE_CONFIGS.keys()), default="small")
    parser.add_argument("--output", type=Path)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Fast RDF★ Benchmark: Tier {args.tier}, Scale {args.scale}")
    print("=" * 70)
    
    config = SCALE_CONFIGS[args.scale][f"tier_{args.tier.lower()}"]
    env = get_environment_info()
    
    print(f"\nEnvironment: {env['cpu_model']}, {env['cpu_count']} cores, {env['ram_total_gb']}GB RAM")
    print(f"Polars: {env['polars_version']} ({env['polars_threads']} threads)")
    print(f"\nConfig: {config}")
    
    # Initialize storage
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    # Generate and ingest
    if args.tier == "A":
        data = fast_generate_tier_a(
            n_entities=config["entities"],
            n_facts=config["facts"],
        )
        metrics = fast_ingest_tier_a(term_dict, qt_dict, fact_store, data)
    else:
        data = fast_generate_tier_b(
            n_runs=config["runs"],
            facts_per_run=config["facts_per_run"],
        )
        metrics = fast_ingest_tier_b(term_dict, qt_dict, fact_store, data)
    
    print(f"\n  Ingestion Summary:")
    print(f"    Base facts:     {metrics['base_facts']:>12,}")
    print(f"    Metadata facts: {metrics['metadata_facts']:>12,}")
    print(f"    Quoted triples: {metrics['quoted_triples']:>12,}")
    print(f"    Total terms:    {metrics['total_terms']:>12,}")
    print(f"    Time:           {metrics['elapsed_seconds']:>12.1f} sec")
    print(f"    Base facts/sec: {metrics['facts_per_second']:>12,.0f}")
    print(f"    Total/sec:      {metrics['total_per_second']:>12,.0f}")
    print(f"    Peak memory:    {metrics['peak_memory_mb']:>12.0f} MB")
    
    # Queries
    patterns = ExpansionPatterns(term_dict, qt_dict, fact_store)
    query_results = run_query_suite(patterns, term_dict, fact_store)
    
    print(f"\n  Query Results (warm p50 ms):")
    for q in query_results:
        print(f"    {q['name']:<20} {q['warm_p50_ms']:>8.2f} ms ({q['result_count']:,} rows)")
    
    # Save report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": args.tier,
            "scale": args.scale,
            "environment": env,
            "config": config,
            "ingestion": metrics,
            "queries": query_results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report: {args.output}")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
