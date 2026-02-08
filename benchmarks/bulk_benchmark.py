"""
Bulk loading benchmark comparing RDF-StarBase vs Virtuoso vs GraphDB.

Uses native bulk loaders for fair comparison.
"""
import time
import sys
import subprocess
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test files (relative to workspace, mapped to /data in Virtuoso)
TEST_FILES = {
    "small": "data/sample/data.ttl",           # ~5K triples
    "medium": "data/sample/customers.ttl",     # ~10K triples  
    "large": "data/sample/FIBO_cleaned.ttl",   # ~100K triples
}

DEFAULT_SIZE = "small"


def benchmark_rdfstarbase(file_path: str) -> dict:
    """Benchmark RDF-StarBase with optimized Oxigraph bulk loading."""
    from src.rdf_starbase import TripleStore, execute_sparql
    from bulk_loader import bulk_load_turtle_oneshot, OXIGRAPH_AVAILABLE
    
    results = {"system": "RDF-StarBase", "oxigraph_parser": OXIGRAPH_AVAILABLE}
    
    store = TripleStore()
    t0 = time.time()
    count = bulk_load_turtle_oneshot(store, file_path)
    t1 = time.time()
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    # Warm up
    execute_sparql(store, "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    
    # COUNT query
    t0 = time.time()
    for _ in range(10):
        execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        execute_sparql(store, """
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """)
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def benchmark_virtuoso_bulk(file_path: str, graph: str = "http://benchmark.example.org/") -> dict:
    """
    Benchmark Virtuoso using native bulk loader via iSQL.
    
    Requires:
    - Virtuoso container named 'virtuoso'
    - /data mounted with DirsAllowed configured
    """
    results = {"system": "Virtuoso (bulk)"}
    
    # Convert Windows path to container path
    container_path = "/data/" + "/".join(Path(file_path).parts[1:])
    
    # Check if Virtuoso is reachable
    endpoint = "http://localhost:8890/sparql"
    try:
        test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(test_query)}&format=json"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        return {"system": "Virtuoso (bulk)", "error": f"Not reachable: {e}"}
    
    def run_isql(command: str) -> tuple[bool, str]:
        """Run iSQL command and return success status and output."""
        try:
            result = subprocess.run(
                ["docker", "exec", "virtuoso", "isql", "1111", "dba", "dba123", f"EXEC={command}"],
                capture_output=True, text=True, timeout=120
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    # Clear existing data
    run_isql(f"SPARQL CLEAR GRAPH <{graph}>;")
    run_isql("DELETE FROM DB.DBA.load_list;")
    
    # Register file for bulk loading
    success, output = run_isql(f"ld_dir('/data/sample', '{Path(file_path).name}', '{graph}');")
    if not success:
        return {"system": "Virtuoso (bulk)", "error": f"ld_dir failed: {output}"}
    
    # Run bulk loader with timing
    t0 = time.time()
    success, output = run_isql("rdf_loader_run();")
    t1 = time.time()
    
    if not success:
        return {"system": "Virtuoso (bulk)", "error": f"rdf_loader_run failed: {output}"}
    
    # Checkpoint to ensure data is committed
    run_isql("checkpoint;")
    
    # Get triple count
    count_query = f"SELECT (COUNT(*) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
    try:
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(count_query)}&format=json"
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        count = int(data["results"]["bindings"][0]["c"]["value"])
    except Exception as e:
        return {"system": "Virtuoso (bulk)", "error": f"Count failed: {e}"}
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    def run_query(query: str) -> list:
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(query)}&format=json"
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["results"]["bindings"]
    
    # COUNT query
    t0 = time.time()
    for _ in range(10):
        run_query(f"SELECT (COUNT(*) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        run_query(f"SELECT ?s ?p ?o WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }} LIMIT 1000")
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        run_query(f"""
            SELECT ?s ?p ?o WHERE {{ 
                GRAPH <{graph}> {{
                    ?s ?p ?o 
                    FILTER(isIRI(?o))
                }}
            }} LIMIT 1000
        """)
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def benchmark_graphdb(file_path: str, endpoint: str = "http://localhost:7200", repo: str = "graphdb-benchmark") -> dict:
    """
    Benchmark GraphDB via SPARQL HTTP endpoint.
    """
    results = {"system": "GraphDB"}
    
    repo_url = f"{endpoint}/repositories/{repo}"
    statements_url = f"{repo_url}/statements"
    
    # Check if GraphDB is reachable
    try:
        req = urllib.request.Request(f"{endpoint}/rest/repositories", method="GET")
        req.add_header("Accept", "application/json")
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        return {"system": "GraphDB", "error": f"Not reachable: {e}"}
    
    # Clear existing data in repository
    try:
        req = urllib.request.Request(statements_url, method="DELETE")
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass
    
    # Load data via SPARQL Graph Store Protocol
    with open(file_path, 'rb') as f:
        data = f.read()
    
    t0 = time.time()
    try:
        req = urllib.request.Request(statements_url, data=data, method="POST")
        req.add_header("Content-Type", "text/turtle")
        urllib.request.urlopen(req, timeout=300)
    except Exception as e:
        return {"system": "GraphDB", "error": f"Load failed: {e}"}
    t1 = time.time()
    
    # Get triple count
    count_query = "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"
    try:
        req = urllib.request.Request(
            f"{repo_url}?query={urllib.parse.quote(count_query)}"
        )
        req.add_header("Accept", "application/sparql-results+json")
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        count = int(data["results"]["bindings"][0]["c"]["value"])
    except Exception as e:
        return {"system": "GraphDB", "error": f"Count query failed: {e}"}
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    def run_query(query: str) -> list:
        req = urllib.request.Request(
            f"{repo_url}?query={urllib.parse.quote(query)}"
        )
        req.add_header("Accept", "application/sparql-results+json")
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["results"]["bindings"]
    
    # COUNT query
    t0 = time.time()
    for _ in range(10):
        run_query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    t1 = time.time()
    results["count_query_time"] = (t1 - t0) / 10
    
    # Pattern query
    t0 = time.time()
    for _ in range(10):
        run_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
    t1 = time.time()
    results["pattern_query_time"] = (t1 - t0) / 10
    
    # Filter query
    t0 = time.time()
    for _ in range(10):
        run_query("""
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """)
    t1 = time.time()
    results["filter_query_time"] = (t1 - t0) / 10
    
    return results


def print_results(results: list[dict], file_info: str):
    """Print benchmark results as a comparison table."""
    print("\n" + "=" * 90)
    print(f"BULK LOADING BENCHMARK RESULTS - {file_info}")
    print("=" * 90)
    
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
            print(f"{'ERROR':<20} ", end="")
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
    
    # Triple count discrepancy analysis
    valid_counts = [(r['system'], r['load_triples']) for r in results if "error" not in r]
    if len(valid_counts) > 1:
        counts = [c[1] for c in valid_counts]
        if len(set(counts)) > 1:
            print(f"\n⚠️  TRIPLE COUNT DISCREPANCY DETECTED:")
            baseline = valid_counts[0]
            for system, count in valid_counts:
                diff = count - baseline[1]
                pct = (diff / baseline[1] * 100) if baseline[1] > 0 else 0
                sign = "+" if diff > 0 else ""
                print(f"   {system}: {count:,} triples ({sign}{diff:,}, {sign}{pct:.1f}% vs {baseline[0]})")
            print(f"   Note: Differences may be due to blank node handling, RDF-Star syntax, or OWL imports")
    
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
    
    print("=" * 90)
    
    # Print any errors
    for r in results:
        if "error" in r:
            print(f"\n⚠️  {r['system']}: {r['error']}")
    
    # Summary
    print("\n📊 ANALYSIS:")
    valid = [r for r in results if "error" not in r and "load_rate" in r]
    if valid:
        fastest_load = max(valid, key=lambda x: x["load_rate"])
        print(f"  🏆 Fastest loader: {fastest_load['system']} ({fastest_load['load_rate']:,.0f} triples/sec)")
        
        fastest_count = min(valid, key=lambda x: x["count_query_time"])
        print(f"  🏆 Fastest COUNT query: {fastest_count['system']} ({fastest_count['count_query_time']*1000:.2f}ms)")
        
        fastest_pattern = min(valid, key=lambda x: x["pattern_query_time"])
        print(f"  🏆 Fastest pattern query: {fastest_pattern['system']} ({fastest_pattern['pattern_query_time']*1000:.2f}ms)")
        
        fastest_filter = min(valid, key=lambda x: x["filter_query_time"])
        print(f"  🏆 Fastest filter query: {fastest_filter['system']} ({fastest_filter['filter_query_time']*1000:.2f}ms)")


def main():
    # Parse arguments
    size = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIZE
    
    if size in TEST_FILES:
        file_path = TEST_FILES[size]
    else:
        file_path = size  # Assume it's a path
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        print(f"Available sizes: {', '.join(TEST_FILES.keys())}")
        sys.exit(1)
    
    file_size = Path(file_path).stat().st_size
    file_info = f"{file_path} ({file_size:,} bytes)"
    print(f"Benchmarking with: {file_info}")
    
    results = []
    
    print("\n--- RDF-StarBase ---")
    results.append(benchmark_rdfstarbase(file_path))
    
    print("\n--- Virtuoso (bulk loader) ---")
    results.append(benchmark_virtuoso_bulk(file_path))
    
    print("\n--- GraphDB ---")
    results.append(benchmark_graphdb(file_path))
    
    print_results(results, file_info)
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
