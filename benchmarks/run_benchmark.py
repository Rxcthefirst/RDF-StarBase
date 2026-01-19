"""
RDF★ Benchmark Runner.

Measures ingestion and query performance for the new storage layer.
Reports metrics per benchmarks.md specifications.

Usage:
    python -m benchmarks.run_benchmark --tier A --scale small
    python -m benchmarks.run_benchmark --tier B --scale medium --output results.json
"""

import argparse
import gc
import json
import os
import platform
import psutil
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict

import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase.storage import (
    TermDict, QtDict, FactStore, LSMStorage,
    ExpansionPatterns, DEFAULT_GRAPH_ID,
)


# =============================================================================
# Scale Configurations
# =============================================================================

SCALE_CONFIGS = {
    "tiny": {
        "tier_a": {"entities": 1_000, "facts": 10_000, "metadata_rate": 0.3},
        "tier_b": {"runs": 10, "facts_per_run": 100},
    },
    "small": {
        "tier_a": {"entities": 10_000, "facts": 100_000, "metadata_rate": 0.3},
        "tier_b": {"runs": 100, "facts_per_run": 500},
    },
    "medium": {
        "tier_a": {"entities": 100_000, "facts": 1_000_000, "metadata_rate": 0.3},
        "tier_b": {"runs": 1_000, "facts_per_run": 500},
    },
    "large": {
        "tier_a": {"entities": 500_000, "facts": 10_000_000, "metadata_rate": 0.3},
        "tier_b": {"runs": 5_000, "facts_per_run": 1_000},
    },
    "xlarge": {
        "tier_a": {"entities": 1_000_000, "facts": 50_000_000, "metadata_rate": 0.3},
        "tier_b": {"runs": 10_000, "facts_per_run": 5_000},
    },
}


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class IngestionMetrics:
    """Metrics from data ingestion phase."""
    base_facts: int = 0
    metadata_facts: int = 0
    quoted_triples: int = 0
    total_terms: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    
    @property
    def facts_per_second(self) -> float:
        return self.base_facts / self.elapsed_seconds if self.elapsed_seconds > 0 else 0
    
    @property
    def meta_per_second(self) -> float:
        return self.metadata_facts / self.elapsed_seconds if self.elapsed_seconds > 0 else 0
    
    @property
    def qt_per_second(self) -> float:
        return self.quoted_triples / self.elapsed_seconds if self.elapsed_seconds > 0 else 0


@dataclass
class QueryMetrics:
    """Metrics from a single query type."""
    query_name: str
    cold_times_ms: List[float] = field(default_factory=list)
    warm_times_ms: List[float] = field(default_factory=list)
    result_count: int = 0
    
    @property
    def cold_p50(self) -> float:
        return statistics.median(self.cold_times_ms) if self.cold_times_ms else 0
    
    @property
    def cold_p95(self) -> float:
        return statistics.quantiles(self.cold_times_ms, n=20)[18] if len(self.cold_times_ms) >= 20 else max(self.cold_times_ms, default=0)
    
    @property
    def warm_p50(self) -> float:
        return statistics.median(self.warm_times_ms) if self.warm_times_ms else 0
    
    @property
    def warm_p95(self) -> float:
        return statistics.quantiles(self.warm_times_ms, n=20)[18] if len(self.warm_times_ms) >= 20 else max(self.warm_times_ms, default=0)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    engine: str = "RDF-StarBase"
    version: str = "0.1.0"
    timestamp: str = ""
    tier: str = ""
    scale: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    ingestion: Optional[IngestionMetrics] = None
    queries: List[QueryMetrics] = field(default_factory=list)
    storage_stats: Dict[str, Any] = field(default_factory=dict)


def get_environment_info() -> Dict[str, Any]:
    """Collect environment information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "polars_version": pl.__version__,
        "polars_threads": pl.thread_pool_size(),
    }


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


# =============================================================================
# Data Generation (inline for benchmark)
# =============================================================================

def generate_tier_a_data(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    n_entities: int,
    n_facts: int,
    metadata_rate: float,
    n_predicates: int = 200,
    seed: int = 42,
) -> IngestionMetrics:
    """Generate and ingest Tier A data directly."""
    import numpy as np
    
    metrics = IngestionMetrics()
    start_time = time.perf_counter()
    start_memory = get_memory_mb()
    peak_memory = start_memory
    
    rng = np.random.default_rng(seed)
    
    # Pre-intern predicates
    predicates = [
        term_dict.intern_iri(f"http://example.org/pred_{i:03d}")
        for i in range(n_predicates)
    ]
    
    # Metadata predicates
    derived_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
    time_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
    gen_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasGeneratedBy")
    conf_pred = term_dict.intern_iri("http://example.org/confidence")
    meta_preds = [derived_pred, time_pred, gen_pred, conf_pred]
    
    # Sources and processes
    n_sources = min(100, n_entities // 100)
    n_processes = min(50, n_entities // 200)
    sources = [term_dict.intern_iri(f"http://example.org/Source_{i:03d}") for i in range(n_sources)]
    processes = [term_dict.intern_iri(f"http://example.org/Process_{i:03d}") for i in range(n_processes)]
    
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    # Zipf-like distribution for predicates
    pred_weights = 1.0 / (np.arange(1, n_predicates + 1) ** 1.5)
    pred_weights /= pred_weights.sum()
    
    # Entity weights (less skewed)
    entity_weights = 1.0 / (np.arange(1, n_entities + 1) ** 1.2)
    entity_weights /= entity_weights.sum()
    
    batch_size = 10000
    batch_facts = []
    
    print(f"  Generating {n_facts:,} base facts with {metadata_rate:.0%} metadata rate...")
    
    for i in range(n_facts):
        # Sample indices
        s_idx = rng.choice(n_entities, p=entity_weights)
        o_idx = rng.choice(n_entities, p=entity_weights)
        p_idx = rng.choice(n_predicates, p=pred_weights)
        
        # Intern terms
        s = term_dict.intern_iri(f"http://example.org/Entity_{s_idx:06d}")
        p = predicates[p_idx]
        o = term_dict.intern_iri(f"http://example.org/Entity_{o_idx:06d}")
        
        batch_facts.append((DEFAULT_GRAPH_ID, s, p, o))
        metrics.base_facts += 1
        
        # Add metadata?
        if rng.random() < metadata_rate:
            qt_id = qt_dict.get_or_create(s, p, o)
            metrics.quoted_triples += 1
            
            # Select 2-4 metadata predicates
            n_meta = rng.integers(2, 5)
            rng.shuffle(meta_preds)
            
            for mp in meta_preds[:n_meta]:
                if mp == derived_pred:
                    mo = sources[rng.integers(0, len(sources))]
                elif mp == gen_pred:
                    mo = processes[rng.integers(0, len(processes))]
                elif mp == time_pred:
                    delta = timedelta(seconds=int(rng.integers(0, 86400 * 365)))
                    ts = base_time + delta
                    mo = term_dict.intern_literal(ts.isoformat())
                elif mp == conf_pred:
                    conf = round(rng.random() * 0.5 + 0.5, 3)
                    mo = term_dict.intern_literal(str(conf))
                else:
                    continue
                
                batch_facts.append((DEFAULT_GRAPH_ID, qt_id, mp, mo))
                metrics.metadata_facts += 1
        
        # Flush batch
        if len(batch_facts) >= batch_size:
            fact_store.add_facts_batch(batch_facts)
            batch_facts = []
            
            current_memory = get_memory_mb()
            peak_memory = max(peak_memory, current_memory)
            
            if (i + 1) % 100000 == 0:
                elapsed = time.perf_counter() - start_time
                rate = (i + 1) / elapsed
                print(f"    {i+1:,} facts ({rate:,.0f}/sec), memory: {current_memory:.0f} MB")
    
    # Final batch
    if batch_facts:
        fact_store.add_facts_batch(batch_facts)
    
    metrics.elapsed_seconds = time.perf_counter() - start_time
    metrics.total_terms = len(term_dict)
    metrics.peak_memory_mb = peak_memory
    
    return metrics


def generate_tier_b_data(
    term_dict: TermDict,
    qt_dict: QtDict,
    fact_store: FactStore,
    n_runs: int,
    facts_per_run: int,
    n_entities: int = 100_000,
    n_predicates: int = 100,
    seed: int = 42,
) -> IngestionMetrics:
    """Generate and ingest Tier B receipts-per-run data."""
    import numpy as np
    
    metrics = IngestionMetrics()
    start_time = time.perf_counter()
    start_memory = get_memory_mb()
    peak_memory = start_memory
    
    rng = np.random.default_rng(seed)
    
    # Pre-intern predicates
    predicates = [
        term_dict.intern_iri(f"http://example.org/pred_{i:03d}")
        for i in range(n_predicates)
    ]
    
    # Metadata predicates
    gen_by_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasGeneratedBy")
    time_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#generatedAtTime")
    derived_pred = term_dict.intern_iri("http://www.w3.org/ns/prov#wasDerivedFrom")
    conf_pred = term_dict.intern_iri("http://example.org/confidence")
    
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    entities_per_run = 50
    
    batch_facts = []
    batch_size = 10000
    
    print(f"  Generating {n_runs:,} runs × {facts_per_run:,} facts = {n_runs * facts_per_run:,} base facts...")
    
    for run_idx in range(n_runs):
        run_uri = term_dict.intern_iri(f"http://example.org/Run_{run_idx:06d}")
        run_time = base_time + timedelta(hours=run_idx)
        
        # Input entities for this run
        input_entities = [
            term_dict.intern_iri(f"http://example.org/Entity_{rng.integers(0, n_entities):06d}")
            for _ in range(entities_per_run)
        ]
        
        for fact_idx in range(facts_per_run):
            s = input_entities[rng.integers(0, len(input_entities))]
            p = predicates[rng.integers(0, len(predicates))]
            o = term_dict.intern_iri(f"http://example.org/Entity_{rng.integers(0, n_entities):06d}")
            
            # Base fact
            batch_facts.append((DEFAULT_GRAPH_ID, s, p, o))
            metrics.base_facts += 1
            
            # Quoted triple for metadata
            qt_id = qt_dict.get_or_create(s, p, o)
            metrics.quoted_triples += 1
            
            # 4 metadata facts per base fact
            # wasGeneratedBy
            batch_facts.append((DEFAULT_GRAPH_ID, qt_id, gen_by_pred, run_uri))
            metrics.metadata_facts += 1
            
            # generatedAtTime
            fact_time = run_time + timedelta(seconds=fact_idx)
            time_lit = term_dict.intern_literal(fact_time.isoformat())
            batch_facts.append((DEFAULT_GRAPH_ID, qt_id, time_pred, time_lit))
            metrics.metadata_facts += 1
            
            # wasDerivedFrom
            derived_from = input_entities[rng.integers(0, len(input_entities))]
            batch_facts.append((DEFAULT_GRAPH_ID, qt_id, derived_pred, derived_from))
            metrics.metadata_facts += 1
            
            # confidence
            conf = round(rng.random() * 0.4 + 0.6, 3)
            conf_lit = term_dict.intern_literal(str(conf))
            batch_facts.append((DEFAULT_GRAPH_ID, qt_id, conf_pred, conf_lit))
            metrics.metadata_facts += 1
        
        # Flush batch
        if len(batch_facts) >= batch_size:
            fact_store.add_facts_batch(batch_facts)
            batch_facts = []
            
            current_memory = get_memory_mb()
            peak_memory = max(peak_memory, current_memory)
        
        if (run_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            total_facts = metrics.base_facts + metrics.metadata_facts
            rate = total_facts / elapsed
            print(f"    {run_idx+1:,} runs, {total_facts:,} total facts ({rate:,.0f}/sec)")
    
    # Final batch
    if batch_facts:
        fact_store.add_facts_batch(batch_facts)
    
    metrics.elapsed_seconds = time.perf_counter() - start_time
    metrics.total_terms = len(term_dict)
    metrics.peak_memory_mb = peak_memory
    
    return metrics


# =============================================================================
# Query Benchmarks
# =============================================================================

def run_query_benchmarks(
    patterns: ExpansionPatterns,
    term_dict: TermDict,
    fact_store: FactStore,
    n_cold: int = 3,
    n_warm: int = 10,
) -> List[QueryMetrics]:
    """Run the SPARQL★ query suite and collect metrics."""
    
    results = []
    
    # Q1: SPO exact match (baseline scan)
    print("  Running Q1: SPO exact match...")
    q1 = QueryMetrics(query_name="Q1_spo_exact")
    
    # Find a real entity to query
    sample_entity = "http://example.org/Entity_000001"
    sample_pred = "http://example.org/pred_001"
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        s_id = term_dict.lookup_iri(sample_entity)
        p_id = term_dict.lookup_iri(sample_pred)
        if s_id and p_id:
            df = fact_store.scan_facts().filter(
                (pl.col("s") == s_id) & (pl.col("p") == p_id)
            )
            q1.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q1.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        s_id = term_dict.lookup_iri(sample_entity)
        p_id = term_dict.lookup_iri(sample_pred)
        if s_id and p_id:
            df = fact_store.scan_facts().filter(
                (pl.col("s") == s_id) & (pl.col("p") == p_id)
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        q1.warm_times_ms.append(elapsed_ms)
    
    results.append(q1)
    
    # Q4: rdf:type fanout (simulated with predicate scan)
    print("  Running Q4: Predicate fanout...")
    q4 = QueryMetrics(query_name="Q4_predicate_fanout")
    pred_to_scan = "http://example.org/pred_001"
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        p_id = term_dict.lookup_iri(pred_to_scan)
        if p_id:
            df = fact_store.scan_facts().filter(pl.col("p") == p_id)
            q4.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q4.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        p_id = term_dict.lookup_iri(pred_to_scan)
        if p_id:
            df = fact_store.scan_facts().filter(pl.col("p") == p_id)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q4.warm_times_ms.append(elapsed_ms)
    
    results.append(q4)
    
    # Q6: Metadata for specific quoted triple
    print("  Running Q6: Metadata for quoted triple...")
    q6 = QueryMetrics(query_name="Q6_qt_metadata")
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q6_metadata_for_triple(
            sample_entity,
            sample_pred,
            "http://example.org/Entity_000002"
        )
        q6.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q6.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        df = patterns.q6_metadata_for_triple(
            sample_entity,
            sample_pred,
            "http://example.org/Entity_000002"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        q6.warm_times_ms.append(elapsed_ms)
    
    results.append(q6)
    
    # Q7: Expand by source
    print("  Running Q7: Expand by source...")
    q7 = QueryMetrics(query_name="Q7_expand_by_source")
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q7_expand_by_source("http://example.org/Source_001")
        q7.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q7.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        df = patterns.q7_expand_by_source("http://example.org/Source_001")
        elapsed_ms = (time.perf_counter() - start) * 1000
        q7.warm_times_ms.append(elapsed_ms)
    
    results.append(q7)
    
    # Q9: Filter by confidence
    print("  Running Q9: Filter by confidence...")
    q9 = QueryMetrics(query_name="Q9_confidence_filter")
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q9_filter_by_confidence(0.8)
        q9.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q9.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        df = patterns.q9_filter_by_confidence(0.8)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q9.warm_times_ms.append(elapsed_ms)
    
    results.append(q9)
    
    # Q11: Count by source
    print("  Running Q11: Count by source...")
    q11 = QueryMetrics(query_name="Q11_count_by_source")
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q11_count_by_source()
        q11.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q11.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        df = patterns.q11_count_by_source()
        elapsed_ms = (time.perf_counter() - start) * 1000
        q11.warm_times_ms.append(elapsed_ms)
    
    results.append(q11)
    
    # Q12: Count by run (Tier B)
    print("  Running Q12: Count by run...")
    q12 = QueryMetrics(query_name="Q12_count_by_run")
    
    for _ in range(n_cold):
        gc.collect()
        start = time.perf_counter()
        df = patterns.q12_count_by_run()
        q12.result_count = len(df)
        elapsed_ms = (time.perf_counter() - start) * 1000
        q12.cold_times_ms.append(elapsed_ms)
    
    for _ in range(n_warm):
        start = time.perf_counter()
        df = patterns.q12_count_by_run()
        elapsed_ms = (time.perf_counter() - start) * 1000
        q12.warm_times_ms.append(elapsed_ms)
    
    results.append(q12)
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark(tier: str, scale: str, output_path: Optional[Path] = None) -> BenchmarkReport:
    """Run complete benchmark suite."""
    
    print("=" * 70)
    print(f"RDF★ Benchmark: Tier {tier}, Scale {scale}")
    print("=" * 70)
    
    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        tier=tier,
        scale=scale,
        environment=get_environment_info(),
    )
    
    config = SCALE_CONFIGS[scale][f"tier_{tier.lower()}"]
    report.config = config
    
    print(f"\nEnvironment:")
    for k, v in report.environment.items():
        print(f"  {k}: {v}")
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    
    # Create storage
    print("\n[1/3] Initializing storage...")
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    # Generate data
    print("\n[2/3] Generating and ingesting data...")
    
    if tier == "A":
        ingestion = generate_tier_a_data(
            term_dict, qt_dict, fact_store,
            n_entities=config["entities"],
            n_facts=config["facts"],
            metadata_rate=config["metadata_rate"],
        )
    else:  # Tier B
        ingestion = generate_tier_b_data(
            term_dict, qt_dict, fact_store,
            n_runs=config["runs"],
            facts_per_run=config["facts_per_run"],
        )
    
    report.ingestion = ingestion
    
    print(f"\n  Ingestion Results:")
    print(f"    Base facts:     {ingestion.base_facts:>12,}")
    print(f"    Metadata facts: {ingestion.metadata_facts:>12,}")
    print(f"    Quoted triples: {ingestion.quoted_triples:>12,}")
    print(f"    Total terms:    {ingestion.total_terms:>12,}")
    print(f"    Elapsed time:   {ingestion.elapsed_seconds:>12.2f} sec")
    print(f"    Facts/sec:      {ingestion.facts_per_second:>12,.0f}")
    print(f"    Meta/sec:       {ingestion.meta_per_second:>12,.0f}")
    print(f"    Peak memory:    {ingestion.peak_memory_mb:>12.0f} MB")
    
    # Storage stats
    report.storage_stats = {
        "term_dict": term_dict.stats(),
        "qt_dict": {"count": len(qt_dict), "collisions": qt_dict.collision_count},
        "fact_store": {"count": len(fact_store)},
    }
    
    # Run queries
    print("\n[3/3] Running query benchmarks...")
    patterns = ExpansionPatterns(term_dict, qt_dict, fact_store)
    
    query_results = run_query_benchmarks(patterns, term_dict, fact_store)
    report.queries = query_results
    
    print(f"\n  Query Results:")
    print(f"  {'Query':<25} {'Cold p50':>10} {'Cold p95':>10} {'Warm p50':>10} {'Warm p95':>10} {'Rows':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for q in query_results:
        print(f"  {q.query_name:<25} {q.cold_p50:>10.2f} {q.cold_p95:>10.2f} {q.warm_p50:>10.2f} {q.warm_p95:>10.2f} {q.result_count:>10,}")
    
    # Save report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable dict
        report_dict = {
            "engine": report.engine,
            "version": report.version,
            "timestamp": report.timestamp,
            "tier": report.tier,
            "scale": report.scale,
            "environment": report.environment,
            "config": report.config,
            "ingestion": {
                "base_facts": ingestion.base_facts,
                "metadata_facts": ingestion.metadata_facts,
                "quoted_triples": ingestion.quoted_triples,
                "total_terms": ingestion.total_terms,
                "elapsed_seconds": ingestion.elapsed_seconds,
                "facts_per_second": ingestion.facts_per_second,
                "meta_per_second": ingestion.meta_per_second,
                "peak_memory_mb": ingestion.peak_memory_mb,
            },
            "queries": [
                {
                    "name": q.query_name,
                    "cold_p50_ms": q.cold_p50,
                    "cold_p95_ms": q.cold_p95,
                    "warm_p50_ms": q.warm_p50,
                    "warm_p95_ms": q.warm_p95,
                    "result_count": q.result_count,
                }
                for q in query_results
            ],
            "storage_stats": report.storage_stats,
        }
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\n  Report saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run RDF★ benchmarks")
    parser.add_argument("--tier", choices=["A", "B"], default="A", help="Data tier")
    parser.add_argument("--scale", choices=list(SCALE_CONFIGS.keys()), default="small", help="Scale preset")
    parser.add_argument("--output", type=Path, help="Output JSON report path")
    
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"benchmarks/reports/tier_{args.tier.lower()}_{args.scale}_{timestamp}.json")
    
    run_benchmark(args.tier, args.scale, args.output)


if __name__ == "__main__":
    main()
