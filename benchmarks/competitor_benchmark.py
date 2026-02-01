"""
Competitor benchmark comparing RDF-StarBase vs Oxigraph vs rdflib.

Tests parsing, loading, and query performance.
"""
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data
TEST_FILE = "data/sample/data.ttl"


def benchmark_rdfstarbase(file_path: str) -> dict:
    """Benchmark RDF-StarBase with optimized Oxigraph direct loading."""
    from src.rdf_starbase import TripleStore, execute_sparql
    from bulk_loader import bulk_load_turtle_oneshot, OXIGRAPH_AVAILABLE
    
    results = {"system": "RDF-StarBase", "oxigraph_parser": OXIGRAPH_AVAILABLE}
    
    # Load using ONESHOT path (fastest for files that fit in memory)
    store = TripleStore()
    t0 = time.time()
    count = bulk_load_turtle_oneshot(store, file_path)
    t1 = time.time()
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    # Warm up the parser and cache (not included in timing)
    execute_sparql(store, "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    
    # Simple query
    t0 = time.time()
    for _ in range(10):
        result = execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        result = execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        result = execute_sparql(store, """
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """)
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def benchmark_oxigraph_store(file_path: str) -> dict:
    """Benchmark Oxigraph as a full store."""
    try:
        from pyoxigraph import Store, RdfFormat
    except ImportError:
        return {"system": "Oxigraph Store", "error": "pyoxigraph not installed"}
    
    results = {"system": "Oxigraph Store"}
    
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
    
    # Simple query
    t0 = time.time()
    for _ in range(10):
        result = list(store.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        result = list(store.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000"))
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        result = list(store.query("""
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """))
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def benchmark_rdflib(file_path: str) -> dict:
    """Benchmark rdflib."""
    try:
        import rdflib
    except ImportError:
        return {"system": "rdflib", "error": "rdflib not installed"}
    
    results = {"system": "rdflib"}
    
    # Load
    g = rdflib.Graph()
    t0 = time.time()
    g.parse(file_path, format="turtle")
    t1 = time.time()
    
    count = len(g)
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    # Simple query
    t0 = time.time()
    for _ in range(10):
        result = list(g.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"))
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        result = list(g.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000"))
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        result = list(g.query("""
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """))
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def print_results(results: list[dict]):
    """Print benchmark results as a comparison table."""
    print("\n" + "=" * 80)
    print("COMPETITOR BENCHMARK RESULTS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Metric':<25} ", end="")
    for r in results:
        print(f"{r['system']:<20} ", end="")
    print()
    print("-" * (25 + 21 * len(results)))
    
    # Load metrics
    print(f"{'Load Time (s)':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['load_time']:<20.3f} ", end="")
    print()
    
    print(f"{'Load Rate (triples/sec)':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['load_rate']:<20,.0f} ", end="")
    print()
    
    print(f"{'Triples Loaded':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['load_triples']:<20,} ", end="")
    print()
    
    # Query metrics
    print("-" * (25 + 21 * len(results)))
    
    print(f"{'COUNT Query (ms)':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['count_query_time']*1000:<20.2f} ", end="")
    print()
    
    print(f"{'Pattern Query (ms)':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['pattern_query_time']*1000:<20.2f} ", end="")
    print()
    
    print(f"{'Filter Query (ms)':<25} ", end="")
    for r in results:
        if "error" in r:
            print(f"{'N/A':<20} ", end="")
        else:
            print(f"{r['filter_query_time']*1000:<20.2f} ", end="")
    print()
    
    print("=" * 80)
    
    # Summary
    print("\n📊 ANALYSIS:")
    
    # Find fastest loader
    valid = [r for r in results if "error" not in r and "load_rate" in r]
    if valid:
        fastest = max(valid, key=lambda x: x["load_rate"])
        print(f"  🏆 Fastest loader: {fastest['system']} ({fastest['load_rate']:,.0f} triples/sec)")
    
    # Find fastest query
    if valid:
        fastest = min(valid, key=lambda x: x["count_query_time"])
        print(f"  🏆 Fastest COUNT query: {fastest['system']} ({fastest['count_query_time']*1000:.2f}ms)")
        
        fastest = min(valid, key=lambda x: x["pattern_query_time"])
        print(f"  🏆 Fastest pattern query: {fastest['system']} ({fastest['pattern_query_time']*1000:.2f}ms)")


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else TEST_FILE
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Benchmarking with: {file_path}")
    print(f"File size: {Path(file_path).stat().st_size:,} bytes")
    
    results = []
    
    # Benchmark each system
    print("\n--- RDF-StarBase ---")
    results.append(benchmark_rdfstarbase(file_path))
    
    print("\n--- Oxigraph Store ---")
    results.append(benchmark_oxigraph_store(file_path))
    
    print("\n--- rdflib ---")
    results.append(benchmark_rdflib(file_path))
    
    # Print comparison
    print_results(results)


if __name__ == "__main__":
    main()
