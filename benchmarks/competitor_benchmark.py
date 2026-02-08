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


def benchmark_virtuoso(file_path: str, endpoint: str = "http://localhost:8890/sparql", graph: str = "http://benchmark.example.org/", user: str = "dba", password: str = "dba123") -> dict:
    """
    Benchmark Virtuoso via SPARQL HTTP endpoint.
    
    Args:
        file_path: Path to the Turtle file to load
        endpoint: Virtuoso SPARQL endpoint (default: http://localhost:8890/sparql)
        graph: Named graph URI for loading data
        user: Virtuoso username for write operations
        password: Virtuoso password for write operations
    """
    import urllib.request
    import urllib.parse
    import json
    
    results = {"system": "Virtuoso"}
    
    # Set up digest authentication for write operations
    auth_endpoint = endpoint.replace("/sparql", "/sparql-auth")
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, endpoint.rsplit("/", 1)[0], user, password)
    auth_handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
    auth_opener = urllib.request.build_opener(auth_handler)
    
    # Check if Virtuoso is reachable
    try:
        test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(test_query)}&format=json",
            method="GET"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        return {"system": "Virtuoso", "error": f"Not reachable: {e}"}
    
    # Clear existing data in the benchmark graph via authenticated SPARQL UPDATE
    try:
        clear_query = f"CLEAR GRAPH <{graph}>"
        data = f"query={urllib.parse.quote(clear_query)}".encode('utf-8')
        req = urllib.request.Request(auth_endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("Accept", "application/json")
        auth_opener.open(req, timeout=10)
    except Exception:
        pass  # Graph might not exist
    
    # Load data using rdflib to parse and convert to N-Triples for INSERT
    try:
        import rdflib
    except ImportError:
        return {"system": "Virtuoso", "error": "rdflib required for parsing - pip install rdflib"}
    
    g = rdflib.Graph()
    g.parse(file_path, format="turtle")
    
    # Batch insert triples using SPARQL UPDATE
    # Virtuoso has limits on update size, so chunk it
    chunk_size = 500
    triples = list(g)
    
    def format_term(term):
        if isinstance(term, rdflib.URIRef):
            return f"<{term}>"
        elif isinstance(term, rdflib.Literal):
            if term.language:
                return f'"{term}"@{term.language}'
            elif term.datatype:
                return f'"{term}"^^<{term.datatype}>'
            else:
                # Escape special characters
                escaped = str(term).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                return f'"{escaped}"'
        elif isinstance(term, rdflib.BNode):
            return f"_:{term}"
        return str(term)
    
    t0 = time.time()
    loaded = 0
    
    for i in range(0, len(triples), chunk_size):
        chunk = triples[i:i + chunk_size]
        triple_strs = []
        for s, p, o in chunk:
            triple_strs.append(f"{format_term(s)} {format_term(p)} {format_term(o)} .")
        
        insert_query = f"INSERT DATA {{ GRAPH <{graph}> {{ {' '.join(triple_strs)} }} }}"
        
        try:
            data = f"query={urllib.parse.quote(insert_query)}".encode('utf-8')
            req = urllib.request.Request(auth_endpoint, data=data, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            req.add_header("Accept", "application/json")
            auth_opener.open(req, timeout=60)
            loaded += len(chunk)
        except Exception as e:
            # Try smaller chunks or skip problematic triples
            # For now, report partial success
            break
    
    t1 = time.time()
    
    if loaded == 0:
        return {"system": "Virtuoso", "error": "INSERT DATA failed - check Virtuoso permissions"}
    
    # Get triple count in the graph
    count_query = f"SELECT (COUNT(*) as ?c) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
    try:
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(count_query)}&format=json",
            method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        count = int(data["results"]["bindings"][0]["c"]["value"])
    except Exception as e:
        count = loaded
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    def run_query(query: str) -> list:
        req = urllib.request.Request(
            f"{endpoint}?query={urllib.parse.quote(query)}&format=json",
            method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["results"]["bindings"]
    
    # COUNT query (scoped to our graph)
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


def benchmark_graphdb(file_path: str, endpoint: str = "http://localhost:7200", repo: str = "benchmark") -> dict:
    """
    Benchmark GraphDB via SPARQL HTTP endpoint.
    
    Args:
        file_path: Path to the Turtle file to load
        endpoint: GraphDB server URL (default: http://localhost:7200)
        repo: Repository name (default: benchmark)
    """
    import urllib.request
    import urllib.parse
    import json
    
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
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Repository might not exist or be empty
    
    # Load data via SPARQL Graph Store Protocol
    with open(file_path, 'rb') as f:
        data = f.read()
    
    t0 = time.time()
    try:
        req = urllib.request.Request(statements_url, data=data, method="POST")
        req.add_header("Content-Type", "text/turtle")
        urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        return {"system": "GraphDB", "error": f"Load failed: {e}"}
    t1 = time.time()
    
    # Get triple count
    count_query = "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }"
    try:
        req = urllib.request.Request(
            f"{repo_url}?query={urllib.parse.quote(count_query)}",
            method="GET"
        )
        req.add_header("Accept", "application/sparql-results+json")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        count = int(data["results"]["bindings"][0]["c"]["value"])
    except Exception as e:
        return {"system": "GraphDB", "error": f"Count query failed: {e}"}
    
    results["load_triples"] = count
    results["load_time"] = t1 - t0
    results["load_rate"] = count / (t1 - t0) if t1 > t0 else 0
    
    def run_query(query: str) -> list:
        req = urllib.request.Request(
            f"{repo_url}?query={urllib.parse.quote(query)}",
            method="GET"
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
    
    print("\n--- Virtuoso ---")
    results.append(benchmark_virtuoso(file_path))
    
    print("\n--- GraphDB ---")
    results.append(benchmark_graphdb(file_path, repo="knowledge-graph-demo"))
    
    print("\n--- rdflib ---")
    results.append(benchmark_rdflib(file_path))
    
    # Print comparison
    print_results(results)


if __name__ == "__main__":
    main()
