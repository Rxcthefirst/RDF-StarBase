"""
Performance benchmarks for RDF-StarBase query execution.

Run with: python benchmarks/bench_query.py
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql.executor import execute_sparql


def prov(source="benchmark"):
    return ProvenanceContext(source=source, confidence=1.0)


def create_test_data(store: TripleStore, num_entities: int = 10000):
    """Create test data with various patterns."""
    print(f"Creating {num_entities} entities with relationships...")
    start = time.perf_counter()
    
    subjects = []
    predicates = []
    objects = []
    
    for i in range(num_entities):
        # Type triple
        subjects.append(f"http://example.org/person/{i}")
        predicates.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        objects.append("http://xmlns.com/foaf/0.1/Person")
        
        # Name triple
        subjects.append(f"http://example.org/person/{i}")
        predicates.append("http://xmlns.com/foaf/0.1/name")
        objects.append(f"Person {i}")
        
        # Age triple
        subjects.append(f"http://example.org/person/{i}")
        predicates.append("http://xmlns.com/foaf/0.1/age")
        objects.append(str(20 + (i % 60)))
        
        # Knows relationship (chain)
        if i > 0:
            subjects.append(f"http://example.org/person/{i}")
            predicates.append("http://xmlns.com/foaf/0.1/knows")
            objects.append(f"http://example.org/person/{i-1}")
    
    # Use columnar insert for speed
    store.add_triples_columnar(
        subjects=subjects,
        predicates=predicates,
        objects=objects,
        source="benchmark",
        confidence=1.0
    )
    
    elapsed = time.perf_counter() - start
    total_triples = len(subjects)
    print(f"  Created {total_triples:,} triples in {elapsed:.3f}s ({total_triples/elapsed:,.0f} triples/sec)")
    return total_triples


def benchmark_simple_select(store: TripleStore, iterations: int = 10):
    """Benchmark simple SELECT query."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name WHERE {
            ?person foaf:name ?name .
        }
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    rows = len(result)
    print(f"  Simple SELECT: {avg*1000:.2f}ms avg, {rows:,} rows")
    return avg


def benchmark_filtered_select(store: TripleStore, iterations: int = 10):
    """Benchmark SELECT with FILTER on string."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name WHERE {
            ?person foaf:name ?name .
            FILTER(STRSTARTS(?name, "Person 1"))
        }
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    rows = len(result)
    print(f"  Filtered SELECT: {avg*1000:.2f}ms avg, {rows:,} rows")
    return avg


def benchmark_join_select(store: TripleStore, iterations: int = 10):
    """Benchmark SELECT with join (2-hop)."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person1 ?person2 WHERE {
            ?person1 foaf:knows ?person2 .
            ?person2 a foaf:Person .
        }
        LIMIT 1000
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    rows = len(result)
    print(f"  Join SELECT: {avg*1000:.2f}ms avg, {rows:,} rows")
    return avg


def benchmark_exists(store: TripleStore, iterations: int = 10):
    """Benchmark EXISTS filter."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person WHERE {
            ?person a foaf:Person .
            FILTER EXISTS { ?person foaf:knows ?other }
        }
        LIMIT 1000
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    rows = len(result)
    print(f"  EXISTS filter: {avg*1000:.2f}ms avg, {rows:,} rows")
    return avg


def benchmark_df_materialization(store: TripleStore, iterations: int = 5):
    """Benchmark DataFrame materialization (cache rebuild)."""
    times = []
    for _ in range(iterations):
        # Invalidate cache
        store._invalidate_cache()
        
        start = time.perf_counter()
        df = store._df  # Force materialization
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    print(f"  DataFrame materialization: {avg*1000:.2f}ms avg")
    return avg


def benchmark_aggregation(store: TripleStore, iterations: int = 10):
    """Benchmark aggregation query."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT (COUNT(?person) AS ?count) WHERE {
            ?person a foaf:Person .
        }
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    print(f"  Aggregation: {avg*1000:.2f}ms avg")
    return avg


def benchmark_string_filter(store: TripleStore, iterations: int = 10):
    """Benchmark string function filters."""
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name WHERE {
            ?person foaf:name ?name .
            FILTER(CONTAINS(?name, "100"))
        }
    """
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = execute_sparql(store, query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    rows = len(result)
    print(f"  String FILTER: {avg*1000:.2f}ms avg, {rows:,} rows")
    return avg


def benchmark_parallel_vs_sequential(store: TripleStore, iterations: int = 5):
    """
    Benchmark parallel vs sequential pattern execution.
    
    Uses a query with independent pattern groups (e.g., looking up unrelated data).
    Note: For very large independent groups, cross joins can explode - use LIMIT.
    """
    # Query with 3 independent patterns (each with limit to avoid cartesian explosion)
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name WHERE {
            ?person foaf:name ?name .
            ?person foaf:age ?age .
            ?person a foaf:Person .
        }
        LIMIT 1000
    """
    
    from rdf_starbase.sparql.executor import SPARQLExecutor
    from rdf_starbase.sparql.parser import parse_query
    
    parsed = parse_query(query)
    
    # Benchmark parallel execution
    executor_parallel = SPARQLExecutor(store, parallel=True)
    times_parallel = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = executor_parallel.execute(parsed)
        elapsed = time.perf_counter() - start
        times_parallel.append(elapsed)
    
    avg_parallel = sum(times_parallel) / len(times_parallel)
    
    # Benchmark sequential execution
    executor_sequential = SPARQLExecutor(store, parallel=False)
    times_sequential = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = executor_sequential.execute(parsed)
        elapsed = time.perf_counter() - start
        times_sequential.append(elapsed)
    
    avg_sequential = sum(times_sequential) / len(times_sequential)
    
    speedup = avg_sequential / avg_parallel if avg_parallel > 0 else 1.0
    print(f"  Parallel execution: {avg_parallel*1000:.2f}ms avg")
    print(f"  Sequential execution: {avg_sequential*1000:.2f}ms avg")
    print(f"  Speedup: {speedup:.2f}x")
    return avg_parallel, avg_sequential


def main():
    print("=" * 60)
    print("RDF-StarBase Query Performance Benchmark")
    print("=" * 60)
    
    # Small scale test
    print("\n--- Small Scale (10K entities, ~40K triples) ---")
    store = TripleStore()
    total = create_test_data(store, num_entities=10000)
    print()
    
    print("Running benchmarks (10 iterations each)...")
    benchmark_df_materialization(store)
    benchmark_simple_select(store)
    benchmark_filtered_select(store)
    benchmark_join_select(store)
    benchmark_exists(store)
    benchmark_aggregation(store)
    benchmark_string_filter(store)
    benchmark_parallel_vs_sequential(store)
    
    print()
    
    # Medium scale test
    print("--- Medium Scale (50K entities, ~200K triples) ---")
    store2 = TripleStore()
    total2 = create_test_data(store2, num_entities=50000)
    print()
    
    print("Running benchmarks (5 iterations each)...")
    benchmark_df_materialization(store2, iterations=3)
    benchmark_simple_select(store2, iterations=5)
    benchmark_filtered_select(store2, iterations=5)
    benchmark_join_select(store2, iterations=5)
    benchmark_aggregation(store2, iterations=5)
    benchmark_parallel_vs_sequential(store2)
    
    print()
    print("=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
