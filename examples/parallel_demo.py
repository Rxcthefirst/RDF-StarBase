"""
Demonstration of parallel pattern execution.

This shows how RDF-StarBase parallelizes independent pattern groups
for improved query performance.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdf_starbase import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql.executor import SPARQLExecutor, _PARALLEL_THRESHOLD
from rdf_starbase.sparql.parser import parse_query


def main():
    # Create test data with distinct entity types
    store = TripleStore()

    print("Creating test data...")
    start = time.perf_counter()
    
    n = 10000  # Dataset size per entity type (larger to see parallel benefits)
    
    # Build all data as columnar lists for batch insert
    subjects = []
    predicates = []
    objects = []
    
    # People
    for i in range(n):
        subjects.append(f"http://ex.org/person/{i}")
        predicates.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        objects.append("http://ex.org/Person")
        
        subjects.append(f"http://ex.org/person/{i}")
        predicates.append("http://ex.org/name")
        objects.append(f'"Person {i}"')
    
    # Companies (independent)
    for i in range(n):
        subjects.append(f"http://ex.org/company/{i}")
        predicates.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        objects.append("http://ex.org/Company")
        
        subjects.append(f"http://ex.org/company/{i}")
        predicates.append("http://ex.org/revenue")
        objects.append(f'"{i * 1000}"')
    
    # Products (independent)
    for i in range(n):
        subjects.append(f"http://ex.org/product/{i}")
        predicates.append("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        objects.append("http://ex.org/Product")
        
        subjects.append(f"http://ex.org/product/{i}")
        predicates.append("http://ex.org/price")
        objects.append(f'"{i * 10}"')
    
    # Use columnar batch insert (much faster!)
    store.add_triples_columnar(subjects, predicates, objects)
    
    elapsed = time.perf_counter() - start
    stats = store.stats()
    print(f"Created {stats['total_assertions']:,} triples in {elapsed:.2f}s ({stats['total_assertions']/elapsed:,.0f} triples/sec)")
    print()

    # Query with 3 independent pattern groups (no shared variables)
    # NOTE: This creates a cartesian product, so use LIMIT
    query_cross = """
    SELECT ?person ?pname ?company ?revenue ?product ?price WHERE {
        ?person a <http://ex.org/Person> .
        ?person <http://ex.org/name> ?pname .
        
        ?company a <http://ex.org/Company> .
        ?company <http://ex.org/revenue> ?revenue .
        
        ?product a <http://ex.org/Product> .
        ?product <http://ex.org/price> ?price .
    }
    LIMIT 100
    """
    
    # More practical: UNION query (independent branches, no cross join)
    query = """
    SELECT ?entity ?value WHERE {
        { ?entity a <http://ex.org/Person> . ?entity <http://ex.org/name> ?value . }
        UNION
        { ?entity a <http://ex.org/Company> . ?entity <http://ex.org/revenue> ?value . }
        UNION
        { ?entity a <http://ex.org/Product> . ?entity <http://ex.org/price> ?value . }
    }
    """

    parsed = parse_query(query)

    # Identify groups
    print("=== Pattern Group Analysis ===")
    executor_parallel = SPARQLExecutor(store, parallel=True)
    
    # For UNION queries, each branch is an independent group
    print(f"Query uses UNION with 3 branches")
    print(f"Each UNION branch can be executed independently")
    print()

    # Benchmark parallel execution
    print("=== Parallel Execution ===")
    times_parallel = []
    result = None
    for _ in range(5):
        start = time.perf_counter()
        result = executor_parallel.execute(parsed)
        times_parallel.append(time.perf_counter() - start)
    avg_p = sum(times_parallel) / len(times_parallel)
    print(f"Parallel: {avg_p*1000:.2f}ms avg ({len(result)} rows)")
    print(f"Sample results:")
    print(result.head(5))

    # Benchmark sequential execution
    print("=== Sequential Execution ===")
    executor_seq = SPARQLExecutor(store, parallel=False)
    times_seq = []
    for _ in range(5):
        start = time.perf_counter()
        result = executor_seq.execute(parsed)
        times_seq.append(time.perf_counter() - start)
    avg_s = sum(times_seq) / len(times_seq)
    print(f"Sequential: {avg_s*1000:.2f}ms avg ({len(result)} rows)")

    print()
    print("=== Result ===")
    speedup = avg_s / avg_p if avg_p > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("✅ Parallel execution is faster!")
    else:
        print("Note: Python's GIL limits benefits for CPU-bound Polars operations.")
        print("      Parallel execution benefits I/O-bound operations (e.g., SERVICE clauses).")
        print("      Polars already parallelizes internally at the C++/Rust level.")


if __name__ == "__main__":
    main()
