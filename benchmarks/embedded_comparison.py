"""
Comprehensive Embedded RDF Store Benchmark

Compares RDF-StarBase against other embedded Python RDF stores:
- RDF-StarBase (this project) - Polars-based columnar execution
- pyoxigraph - Rust-based RDF store with Python bindings
- rdflib - Pure Python RDF library (standard, slow)
- Virtuoso (HTTP) - For reference against non-embedded
- GraphDB (HTTP) - For reference against non-embedded

Note: Only RDF-StarBase natively supports RDF-Star. Others are tested
with standard RDF triples only.

Usage:
    python benchmarks/embedded_comparison.py
    python benchmarks/embedded_comparison.py --scale large
"""

import time
import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_benchmark_data(num_triples: int, output_file: str):
    """Generate synthetic benchmark data."""
    print(f"Generating {num_triples:,} triples to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("@prefix ex: <http://benchmark.example.org/> .\n")
        f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n")
        
        entity_id = 0
        written = 0
        
        while written < num_triples:
            # Create entities with multiple properties
            f.write(f"ex:entity_{entity_id} rdf:type ex:Entity ;\n")
            f.write(f"    ex:name \"Entity {entity_id}\" ;\n")
            f.write(f"    ex:value {entity_id} ;\n")
            
            # Add relationships
            for i in range(min(10, num_triples - written - 3)):
                target = (entity_id + i + 1) % (num_triples // 10)
                if i < 9 and written + 4 + i < num_triples:
                    f.write(f"    ex:relatesTo ex:entity_{target} ;\n")
                else:
                    f.write(f"    ex:relatesTo ex:entity_{target} .\n")
                    break
            
            written += 4 + min(10, num_triples - written - 3)
            entity_id += 1
            
            if entity_id % 10000 == 0:
                print(f"  Generated {written:,} triples...")
    
    print(f"  Done: {output_file}")


def benchmark_rdfstarbase(file_path: str, iterations: int = 10) -> dict:
    """Benchmark RDF-StarBase."""
    from rdf_starbase import TripleStore, execute_sparql
    from bulk_loader import bulk_load_turtle_oneshot
    
    results = {"system": "RDF-StarBase", "embedded": True}
    
    # Load
    store = TripleStore()
    t0 = time.time()
    count = bulk_load_turtle_oneshot(store, file_path)
    t1 = time.time()
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    # Warmup
    execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    
    # COUNT query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
        times.append(time.time() - t0)
    results["count_min"] = min(times) * 1000
    results["count_avg"] = sum(times) / len(times) * 1000
    
    # Pattern query (LIMIT 1000)
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
        times.append(time.time() - t0)
    results["pattern_min"] = min(times) * 1000
    results["pattern_avg"] = sum(times) / len(times) * 1000
    
    # Type pattern query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, """
            SELECT ?s WHERE { 
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://benchmark.example.org/Entity>
            }
        """)
        times.append(time.time() - t0)
    results["type_pattern_min"] = min(times) * 1000
    results["type_pattern_avg"] = sum(times) / len(times) * 1000
    results["type_pattern_rows"] = len(result)
    
    # Filter query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = execute_sparql(store, """
            SELECT ?s ?o WHERE { 
                ?s <http://benchmark.example.org/value> ?o
                FILTER(?o > 1000)
            } LIMIT 1000
        """)
        times.append(time.time() - t0)
    results["filter_min"] = min(times) * 1000
    results["filter_avg"] = sum(times) / len(times) * 1000
    
    return results


def benchmark_pyoxigraph(file_path: str, iterations: int = 10) -> dict:
    """Benchmark pyoxigraph Store."""
    try:
        from pyoxigraph import Store, RdfFormat
    except ImportError:
        return {"system": "pyoxigraph", "error": "Not installed (pip install pyoxigraph)"}
    
    results = {"system": "pyoxigraph", "embedded": True}
    
    # Load
    store = Store()
    t0 = time.time()
    with open(file_path, 'rb') as f:
        store.load(f, RdfFormat.TURTLE, base_iri="http://example.org/")
    t1 = time.time()
    
    count = len(list(store))
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    # Warmup
    list(store.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
    
    # COUNT query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(store.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
        times.append(time.time() - t0)
    results["count_min"] = min(times) * 1000
    results["count_avg"] = sum(times) / len(times) * 1000
    
    # Pattern query (LIMIT 1000)
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(store.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000"))
        times.append(time.time() - t0)
    results["pattern_min"] = min(times) * 1000
    results["pattern_avg"] = sum(times) / len(times) * 1000
    
    # Type pattern query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(store.query("""
            SELECT ?s WHERE { 
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://benchmark.example.org/Entity>
            }
        """))
        times.append(time.time() - t0)
    results["type_pattern_min"] = min(times) * 1000
    results["type_pattern_avg"] = sum(times) / len(times) * 1000
    results["type_pattern_rows"] = len(result)
    
    # Filter query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(store.query("""
            SELECT ?s ?o WHERE { 
                ?s <http://benchmark.example.org/value> ?o
                FILTER(?o > 1000)
            } LIMIT 1000
        """))
        times.append(time.time() - t0)
    results["filter_min"] = min(times) * 1000
    results["filter_avg"] = sum(times) / len(times) * 1000
    
    return results


def benchmark_rdflib(file_path: str, iterations: int = 10, max_triples: int = 100000) -> dict:
    """Benchmark rdflib (with size limit due to slowness)."""
    try:
        import rdflib
    except ImportError:
        return {"system": "rdflib", "error": "Not installed (pip install rdflib)"}
    
    results = {"system": "rdflib", "embedded": True}
    
    # Check file size
    import os
    file_size = os.path.getsize(file_path)
    if file_size > 5_000_000:  # > 5MB
        results["warning"] = f"File too large for rdflib ({file_size/1e6:.1f}MB), using subset"
        # rdflib is too slow for large files, skip or use smaller iterations
        iterations = 3
    
    # Load
    g = rdflib.Graph()
    t0 = time.time()
    g.parse(file_path, format="turtle")
    t1 = time.time()
    
    count = len(g)
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    if count > max_triples:
        results["note"] = f"Reduced iterations to {iterations} due to dataset size"
    
    # Warmup
    list(g.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
    
    # COUNT query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(g.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
        times.append(time.time() - t0)
    results["count_min"] = min(times) * 1000
    results["count_avg"] = sum(times) / len(times) * 1000
    
    # Pattern query (LIMIT 1000)
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(g.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000"))
        times.append(time.time() - t0)
    results["pattern_min"] = min(times) * 1000
    results["pattern_avg"] = sum(times) / len(times) * 1000
    
    # Type pattern query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(g.query("""
            SELECT ?s WHERE { 
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://benchmark.example.org/Entity>
            }
        """))
        times.append(time.time() - t0)
    results["type_pattern_min"] = min(times) * 1000
    results["type_pattern_avg"] = sum(times) / len(times) * 1000
    results["type_pattern_rows"] = len(result)
    
    # Filter query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = list(g.query("""
            SELECT ?s ?o WHERE { 
                ?s <http://benchmark.example.org/value> ?o
                FILTER(?o > 1000)
            } LIMIT 1000
        """))
        times.append(time.time() - t0)
    results["filter_min"] = min(times) * 1000
    results["filter_avg"] = sum(times) / len(times) * 1000
    
    return results


def benchmark_virtuoso_http(file_path: str, endpoint: str = "http://localhost:8890/sparql", 
                            iterations: int = 10) -> dict:
    """Benchmark Virtuoso via HTTP (for reference)."""
    import urllib.request
    import urllib.parse
    import json
    
    results = {"system": "Virtuoso (HTTP)", "embedded": False}
    
    # Check if reachable
    try:
        test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(test_query)}&format=json",
            method="GET"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        return {"system": "Virtuoso (HTTP)", "error": f"Not reachable: {e}"}
    
    # Data should already be loaded
    results["load_time"] = "N/A (pre-loaded)"
    results["load_triples"] = "N/A"
    
    def query_virtuoso(sparql: str) -> list:
        url = f"{endpoint}?query={urllib.parse.quote(sparql)}&format=json"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get("results", {}).get("bindings", [])
    
    # Warmup
    query_virtuoso("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    
    # COUNT query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = query_virtuoso("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
        times.append(time.time() - t0)
    results["count_min"] = min(times) * 1000
    results["count_avg"] = sum(times) / len(times) * 1000
    
    # Pattern query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = query_virtuoso("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
        times.append(time.time() - t0)
    results["pattern_min"] = min(times) * 1000
    results["pattern_avg"] = sum(times) / len(times) * 1000
    
    return results


def benchmark_graphdb_http(file_path: str, endpoint: str = "http://localhost:7200/repositories/benchmark",
                           iterations: int = 10) -> dict:
    """Benchmark GraphDB via HTTP (for reference)."""
    import urllib.request
    import urllib.parse
    import json
    
    results = {"system": "GraphDB (HTTP)", "embedded": False}
    
    # Check if reachable
    try:
        test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(test_query)}",
            method="GET",
            headers={"Accept": "application/sparql-results+json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        return {"system": "GraphDB (HTTP)", "error": f"Not reachable: {e}"}
    
    results["load_time"] = "N/A (pre-loaded)"
    results["load_triples"] = "N/A"
    
    def query_graphdb(sparql: str) -> list:
        url = f"{endpoint}?query={urllib.parse.quote(sparql)}"
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get("results", {}).get("bindings", [])
    
    # Warmup
    query_graphdb("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    
    # COUNT query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = query_graphdb("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
        times.append(time.time() - t0)
    results["count_min"] = min(times) * 1000
    results["count_avg"] = sum(times) / len(times) * 1000
    
    # Pattern query
    times = []
    for _ in range(iterations):
        t0 = time.time()
        result = query_graphdb("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
        times.append(time.time() - t0)
    results["pattern_min"] = min(times) * 1000
    results["pattern_avg"] = sum(times) / len(times) * 1000
    
    return results


def print_results(results: list[dict], scale: str):
    """Pretty print benchmark results."""
    print()
    print("=" * 90)
    print(f"EMBEDDED RDF STORE BENCHMARK RESULTS ({scale})")
    print("=" * 90)
    print()
    
    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    if errors:
        print("Unavailable systems:")
        for r in errors:
            print(f"  - {r['system']}: {r['error']}")
        print()
    
    if not valid:
        print("No valid results!")
        return
    
    # Load performance
    print("LOAD PERFORMANCE")
    print("-" * 90)
    print(f"{'System':<20} {'Triples':>12} {'Time (s)':>10} {'Rate (t/s)':>15} {'Embedded':>10}")
    print("-" * 90)
    for r in valid:
        triples = r.get("load_triples", "N/A")
        load_time = r.get("load_time", "N/A")
        rate = r.get("load_rate", "N/A")
        embedded = "Yes" if r.get("embedded") else "No"
        
        if isinstance(triples, int):
            triples = f"{triples:,}"
        if isinstance(load_time, float):
            load_time = f"{load_time:.3f}"
        if isinstance(rate, float):
            rate = f"{rate:,.0f}"
        
        print(f"{r['system']:<20} {triples:>12} {load_time:>10} {rate:>15} {embedded:>10}")
    
    print()
    print("QUERY PERFORMANCE (milliseconds, min of 10 iterations)")
    print("-" * 90)
    print(f"{'System':<20} {'COUNT(*)':>12} {'Pattern':>12} {'Type Query':>12} {'Filter':>12}")
    print("-" * 90)
    for r in valid:
        count = r.get("count_min", "N/A")
        pattern = r.get("pattern_min", "N/A")
        type_q = r.get("type_pattern_min", "N/A")
        filter_q = r.get("filter_min", "N/A")
        
        if isinstance(count, float):
            count = f"{count:.2f}"
        if isinstance(pattern, float):
            pattern = f"{pattern:.2f}"
        if isinstance(type_q, float):
            type_q = f"{type_q:.2f}"
        if isinstance(filter_q, float):
            filter_q = f"{filter_q:.2f}"
        
        print(f"{r['system']:<20} {count:>12} {pattern:>12} {type_q:>12} {filter_q:>12}")
    
    # Calculate speedups vs rdflib
    rdflib_result = next((r for r in valid if r["system"] == "rdflib"), None)
    rdfstarbase_result = next((r for r in valid if r["system"] == "RDF-StarBase"), None)
    
    if rdflib_result and rdfstarbase_result:
        print()
        print("SPEEDUP vs rdflib")
        print("-" * 90)
        
        if "count_min" in rdflib_result and "count_min" in rdfstarbase_result:
            if rdfstarbase_result["count_min"] > 0.01:
                speedup = rdflib_result["count_min"] / rdfstarbase_result["count_min"]
                print(f"  COUNT(*):     {speedup:.1f}x faster")
            else:
                print(f"  COUNT(*):     {rdflib_result['count_min']:.1f}ms vs <0.01ms (effectively instant)")
        
        if "pattern_min" in rdflib_result and "pattern_min" in rdfstarbase_result:
            if rdfstarbase_result["pattern_min"] > 0.01:
                speedup = rdflib_result["pattern_min"] / rdfstarbase_result["pattern_min"]
                print(f"  Pattern:      {speedup:.1f}x faster")
        
        if "type_pattern_min" in rdflib_result and "type_pattern_min" in rdfstarbase_result:
            if rdfstarbase_result["type_pattern_min"] > 0.01:
                speedup = rdflib_result["type_pattern_min"] / rdfstarbase_result["type_pattern_min"]
                print(f"  Type query:   {speedup:.1f}x faster")
        
        if "load_rate" in rdflib_result and "load_rate" in rdfstarbase_result:
            if isinstance(rdflib_result["load_rate"], (int, float)) and isinstance(rdfstarbase_result["load_rate"], (int, float)):
                speedup = rdfstarbase_result["load_rate"] / rdflib_result["load_rate"]
                print(f"  Load rate:    {speedup:.1f}x faster")
    
    print()
    print("NOTES:")
    print("  - RDF-StarBase is the ONLY system with native RDF-Star support")
    print("  - Virtuoso/GraphDB times include HTTP overhead (not embedded)")
    print("  - rdflib is pure Python; others use native code")
    print("  - pyoxigraph uses Rust; RDF-StarBase uses Polars (also Rust-based)")


def run_benchmark(scale: str = "small"):
    """Run the full benchmark suite."""
    
    if scale == "small":
        num_triples = 50000
        data_file = "data/sample/benchmark_50k.ttl"
    elif scale == "medium":
        num_triples = 500000
        data_file = "data/sample/benchmark_500k.ttl"
    else:  # large
        num_triples = 2500000
        data_file = "data/sample/benchmark_10M.ttl"  # Use existing large file
    
    print("=" * 90)
    print("EMBEDDED RDF STORE BENCHMARK")
    print("=" * 90)
    print(f"Scale: {scale} ({num_triples:,} target triples)")
    print()
    
    # Generate data if needed
    if not os.path.exists(data_file):
        if scale != "large":
            generate_benchmark_data(num_triples, data_file)
        else:
            print(f"ERROR: Large benchmark file not found: {data_file}")
            print("Please generate with: python benchmarks/data_generator.py")
            return []
    
    print(f"Using data file: {data_file}")
    print()
    
    results = []
    
    # Run embedded benchmarks
    print("Benchmarking RDF-StarBase...")
    results.append(benchmark_rdfstarbase(data_file))
    
    print("Benchmarking pyoxigraph...")
    results.append(benchmark_pyoxigraph(data_file))
    
    print("Benchmarking rdflib...")
    if scale == "large":
        print("  (Skipping rdflib for large scale - too slow)")
        results.append({"system": "rdflib", "error": "Skipped for large scale (>10min expected)"})
    else:
        results.append(benchmark_rdflib(data_file))
    
    # Optional: HTTP-based systems
    print("Checking Virtuoso (HTTP)...")
    results.append(benchmark_virtuoso_http(data_file))
    
    print("Checking GraphDB (HTTP)...")
    results.append(benchmark_graphdb_http(data_file))
    
    # Print results
    print_results(results, f"{scale} - {num_triples:,} triples")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embedded RDF Store Benchmark")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small",
                        help="Benchmark scale (small=50K, medium=500K, large=2.5M triples)")
    args = parser.parse_args()
    
    run_benchmark(args.scale)
