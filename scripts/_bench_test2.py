"""Benchmark test2 repository query performance."""
import time
from rdf_starbase.repositories import RepositoryManager
from rdf_starbase import execute_sparql

mgr = RepositoryManager('./data/repositories')

# Phase 1: Load time
print("=== Phase 1: Store Load ===")
t0 = time.perf_counter()
store = mgr.get_store('test2')
t1 = time.perf_counter()
print(f"Load time: {t1-t0:.3f}s")

# Phase 2: First stats (uncached)
print("\n=== Phase 2: First stats() ===")
t0 = time.perf_counter()
stats = store.stats()
t1 = time.perf_counter()
print(f"Stats time: {t1-t0:.3f}s")
print(f"  active: {stats['active_assertions']:,}")

# Phase 3: Simple LIMIT query
print("\n=== Phase 3: SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000 ===")
t0 = time.perf_counter()
result = execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Result rows: {len(result)}")

# Phase 4: Same query again (indexes warm)
print("\n=== Phase 4: Same query again (warm) ===")
t0 = time.perf_counter()
result2 = execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")

# Phase 5: LIMIT 10
print("\n=== Phase 5: LIMIT 10 ===")
t0 = time.perf_counter()
result3 = execute_sparql(store, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Result rows: {len(result3)}")

# Phase 6: Count query
print("\n=== Phase 6: SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } ===")
t0 = time.perf_counter()
result4 = execute_sparql(store, "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Count: {result4}")

# Phase 7: Filtered query
print("\n=== Phase 7: SELECT ?s ?o WHERE { ?s a ?o } LIMIT 100 ===")
t0 = time.perf_counter()
result5 = execute_sparql(store, "SELECT ?s ?o WHERE { ?s a ?o } LIMIT 100")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Result rows: {len(result5)}")

# Phase 8: Schema browser queries
print("\n=== Phase 8: Schema browser - classes ===")
t0 = time.perf_counter()
result6 = execute_sparql(store, """
SELECT ?class (COUNT(?s) AS ?count)
WHERE { ?s a ?class }
GROUP BY ?class
ORDER BY DESC(?count)
LIMIT 50
""")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Result rows: {len(result6)}")

print("\n=== Phase 9: Schema browser - properties ===")
t0 = time.perf_counter()
result7 = execute_sparql(store, """
SELECT ?prop (COUNT(*) AS ?count)
WHERE { ?s ?prop ?o }
GROUP BY ?prop
ORDER BY DESC(?count)
LIMIT 100
""")
t1 = time.perf_counter()
print(f"Query time: {t1-t0:.3f}s")
print(f"Result rows: {len(result7)}")
