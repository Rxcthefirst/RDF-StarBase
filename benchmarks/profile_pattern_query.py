#!/usr/bin/env python3
"""Profile pattern query execution to find bottlenecks."""

import time
import cProfile
import pstats
from io import StringIO

# Add src to path
import sys
sys.path.insert(0, "src")

from rdf_starbase import TripleStore
from rdf_starbase.sparql.parser import parse_query
from rdf_starbase.sparql.executor import SPARQLExecutor, execute_sparql


def create_test_store(n_triples: int = 100_000) -> TripleStore:
    """Create a store with n triples for testing."""
    store = TripleStore()
    
    # Batch insert for speed
    subjects = [f"http://example.org/s{i}" for i in range(n_triples)]
    predicates = [f"http://example.org/p{i % 100}" for i in range(n_triples)]
    objects = [f"http://example.org/o{i}" for i in range(n_triples)]
    
    store.add_triples_columnar(subjects, predicates, objects)
    print(f"Created store with {len(store)} triples")
    return store


def profile_pattern_query():
    """Profile pattern query execution."""
    store = create_test_store(100_000)
    
    query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000"
    
    # Warm up
    execute_sparql(store, query)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):
        result = execute_sparql(store, query)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())
    
    # Also time individual components
    print("\n" + "=" * 60)
    print("COMPONENT TIMING")
    print("=" * 60)
    
    executor = SPARQLExecutor(store)
    
    # Time parsing
    t0 = time.perf_counter()
    for _ in range(100):
        parsed = parse_query(query)
    parse_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"Parse query:        {parse_time:.3f} ms")
    
    # Time execution (without parsing)
    parsed = parse_query(query)
    t0 = time.perf_counter()
    for _ in range(100):
        result = executor.execute(parsed)
    exec_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"Execute query:      {exec_time:.3f} ms")
    
    # Time just getting the dataframe view
    t0 = time.perf_counter()
    for _ in range(100):
        df = store._df.head(1000)
    df_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"DataFrame head:     {df_time:.3f} ms")
    
    # Time term lookup
    t0 = time.perf_counter()
    for _ in range(100):
        df = store._df.head(1000)
        # Just get the columns
        s_vals = df["subject"].to_list()
        p_vals = df["predicate"].to_list()
        o_vals = df["object"].to_list()
    col_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"Get columns:        {col_time:.3f} ms")
    
    print("\n" + "=" * 60)
    print("BREAKDOWN ANALYSIS")
    print("=" * 60)
    total = parse_time + exec_time
    print(f"Parse:    {parse_time:.2f} ms ({parse_time/total*100:.1f}%)")
    print(f"Execute:  {exec_time:.2f} ms ({exec_time/total*100:.1f}%)")
    print(f"Total:    {total:.2f} ms")


if __name__ == "__main__":
    profile_pattern_query()
