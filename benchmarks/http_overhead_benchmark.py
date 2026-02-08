"""
HTTP Overhead Benchmark - Compare Embedded vs HTTP API Performance

This benchmark demonstrates the performance difference between:
1. Embedded mode: Direct Python calls (no network overhead)
2. HTTP mode: REST API calls (realistic HTTP overhead)

This provides an apples-to-apples comparison with Virtuoso/GraphDB which
are always accessed via HTTP SPARQL endpoints.

Usage:
    python benchmarks/http_overhead_benchmark.py
    python benchmarks/http_overhead_benchmark.py --embedded-only
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_benchmark(embedded_only: bool = False):
    """
    Run the HTTP overhead benchmark.
    
    Args:
        embedded_only: If True, only run embedded benchmarks
    """
    print("=" * 70)
    print("RDF-StarBase HTTP Overhead Benchmark")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Setup: Load data into embedded store
    # =========================================================================
    print("Setting up...")
    
    from rdf_starbase import TripleStore, execute_sparql
    from bulk_loader import bulk_load_turtle_oneshot
    
    store = TripleStore()
    
    # Check if large benchmark file exists
    data_file = "data/sample/benchmark_10M.ttl"
    if not os.path.exists(data_file):
        print(f"Benchmark file not found: {data_file}")
        print("Please generate it first with: python benchmarks/data_generator.py")
        return None
    
    count = bulk_load_turtle_oneshot(store, data_file)
    print(f"Loaded {count:,} triples")
    print()
    
    # Test queries
    pattern_query = """
    SELECT ?student ?advisor WHERE {
        ?student <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://benchmark.example.org/GraduateStudent> .
        ?student <http://benchmark.example.org/advisor> ?advisor .
    }
    """
    
    count_query = """
    SELECT (COUNT(*) as ?c) WHERE {
        ?student <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://benchmark.example.org/GraduateStudent> .
        ?student <http://benchmark.example.org/advisor> ?advisor .
    }
    """
    
    simple_count = "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"
    
    results = {"embedded": {}, "http": {}}
    
    # =========================================================================
    # EMBEDDED MODE
    # =========================================================================
    print("EMBEDDED MODE (Direct Python Calls)")
    print("-" * 70)
    
    # Pattern query
    execute_sparql(store, pattern_query)  # warmup
    times = []
    for _ in range(10):
        t0 = time.time()
        r = execute_sparql(store, pattern_query)
        times.append(time.time() - t0)
    results["embedded"]["select"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000, "rows": len(r)}
    print(f"SPARQL SELECT:      {results['embedded']['select']['min']:.1f}ms min ({results['embedded']['select']['rows']:,} rows)")
    
    # COUNT pattern
    execute_sparql(store, count_query)  # warmup
    times = []
    for _ in range(10):
        t0 = time.time()
        execute_sparql(store, count_query)
        times.append(time.time() - t0)
    results["embedded"]["count_pattern"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000}
    print(f"COUNT(*) pattern:   {results['embedded']['count_pattern']['min']:.1f}ms min")
    
    # COUNT all
    execute_sparql(store, simple_count)  # warmup
    times = []
    for _ in range(10):
        t0 = time.time()
        execute_sparql(store, simple_count)
        times.append(time.time() - t0)
    results["embedded"]["count_all"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000}
    print(f"COUNT(*) all:       {results['embedded']['count_all']['min']:.1f}ms min")
    
    if embedded_only:
        return results
    
    # =========================================================================
    # HTTP MODE (using TestClient to simulate HTTP stack without network)
    # =========================================================================
    print()
    print("HTTP MODE (REST API via TestClient)")
    print("-" * 70)
    
    from api.web import create_app
    from starlette.testclient import TestClient
    
    app = create_app()
    app.state.store = store  # Share the same store
    
    client = TestClient(app)
    
    # Warmup
    client.post("/sparql", json={"query": simple_count})
    
    # Pattern query
    times = []
    for _ in range(10):
        t0 = time.time()
        resp = client.post("/sparql", json={"query": pattern_query})
        times.append(time.time() - t0)
    http_rows = resp.json().get("count", 0)
    results["http"]["select"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000, "rows": http_rows}
    print(f"SPARQL SELECT:      {results['http']['select']['min']:.1f}ms min ({http_rows:,} rows)")
    
    # COUNT pattern
    times = []
    for _ in range(10):
        t0 = time.time()
        client.post("/sparql", json={"query": count_query})
        times.append(time.time() - t0)
    results["http"]["count_pattern"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000}
    print(f"COUNT(*) pattern:   {results['http']['count_pattern']['min']:.1f}ms min")
    
    # COUNT all
    times = []
    for _ in range(10):
        t0 = time.time()
        client.post("/sparql", json={"query": simple_count})
        times.append(time.time() - t0)
    results["http"]["count_all"] = {"min": min(times)*1000, "avg": sum(times)/len(times)*1000}
    print(f"COUNT(*) all:       {results['http']['count_all']['min']:.1f}ms min")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY: HTTP Overhead Analysis")
    print("=" * 70)
    print()
    print(f"                    Embedded     HTTP        Overhead    Slowdown")
    print(f"                    --------     ----        --------    --------")
    
    emb = results["embedded"]["select"]["min"]
    http = results["http"]["select"]["min"]
    print(f"SPARQL SELECT:      {emb:6.1f}ms    {http:6.1f}ms    +{http-emb:.1f}ms     {http/emb:.1f}x")
    
    emb = results["embedded"]["count_pattern"]["min"]
    http = results["http"]["count_pattern"]["min"]
    print(f"COUNT(*) pattern:   {emb:6.1f}ms    {http:6.1f}ms    +{http-emb:.1f}ms      {http/emb:.1f}x")
    
    emb = results["embedded"]["count_all"]["min"]
    http = results["http"]["count_all"]["min"]
    print(f"COUNT(*) all:       {emb:6.1f}ms    {http:6.1f}ms    +{http-emb:.1f}ms      {http/emb:.1f}x")
    
    print()
    print("Key Insight:")
    print("  For large result sets, HTTP JSON serialization dominates.")
    print("  For aggregates (COUNT), HTTP overhead is minimal.")
    print("  Embedded mode eliminates serialization entirely.")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HTTP Overhead Benchmark")
    parser.add_argument("--embedded-only", action="store_true", help="Only run embedded benchmarks")
    args = parser.parse_args()
    
    run_benchmark(embedded_only=args.embedded_only)
