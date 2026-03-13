"""Profile where time is spent in slow queries."""
import time
import polars as pl
from rdf_starbase.repositories import RepositoryManager
from rdf_starbase.sparql.parser import parse_query

mgr = RepositoryManager('./data/repositories')
store = mgr.get_store('test2')
print(f"Facts shape: {store._fact_store._df.shape}")
print(f"Term dict size: {len(store._term_dict)}")

# Check rdf:type
rdf_type_iri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
rdf_type_term = store._term_dict.get_by_lex(rdf_type_iri)
print(f"rdf:type in dict: {rdf_type_term}")

from rdf_starbase.sparql.executor import SPARQLExecutor
executor = SPARQLExecutor(store)

# Test 1: Can integer executor handle ?s a ?o LIMIT 100 ?
query1_str = "SELECT ?s ?o WHERE { ?s a ?o } LIMIT 100"
query1 = parse_query(query1_str)
print(f"\n=== Query 1: ?s a ?o LIMIT 100 ===")
print(f"can_use_integer_executor: {executor._can_use_integer_executor(query1)}")

# Time just the integer executor part
from rdf_starbase.sparql.integer_executor import IntegerExecutor
if not hasattr(store, '_integer_executor') or store._integer_executor is None:
    store._integer_executor = IntegerExecutor(store)
int_exec = store._integer_executor

print("\n--- resolve_term for rdf:type ---")
from rdf_starbase.sparql.parser import IRI
t0 = time.perf_counter()
tid = int_exec.resolve_term(IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), {})
t1 = time.perf_counter()
print(f"resolve_term result: {tid}, time: {t1-t0:.3f}s")

if tid is not None:
    # Try index lookup
    print("\n--- Index lookup on p ---")
    t0 = time.perf_counter()
    result = store._fact_store.lookup_by_index("p", tid)
    t1 = time.perf_counter()
    print(f"Index lookup result: {result.shape if result is not None else 'None'}, time: {t1-t0:.3f}s")

    # Direct filter
    print("\n--- Direct Polars filter ---")
    t0 = time.perf_counter()
    filtered = store._fact_store._df.filter(pl.col('p') == tid)
    t1 = time.perf_counter()
    print(f"Filter result: {filtered.shape}, time: {t1-t0:.3f}s")

# Test 2: GROUP BY query
query2_str = """SELECT ?prop (COUNT(*) AS ?count)
WHERE { ?s ?prop ?o }
GROUP BY ?prop
ORDER BY DESC(?count)
LIMIT 100"""
query2 = parse_query(query2_str)
print(f"\n=== Query 2: GROUP BY ?prop ===")
print(f"can_use_integer_executor: {executor._can_use_integer_executor(query2)}")

# Time just the integer executor part
print("\n--- execute_where (integer) for all triples ---")
t0 = time.perf_counter()
bindings = int_exec.execute_where(query2.where, query2.prefixes)
t1 = time.perf_counter()
print(f"execute_where time: {t1-t0:.3f}s, shape: {bindings.df.shape if not bindings.is_empty else 'empty'}")

if not bindings.is_empty:
    print(f"Dtypes: {bindings.df.dtypes}")
    
    # Time materialization
    print("\n--- materialize_strings_batch ---")
    t0 = time.perf_counter()
    materialized = int_exec.materialize_strings_batch(bindings)
    t1 = time.perf_counter()
    print(f"Materialize time: {t1-t0:.3f}s, shape: {materialized.shape}")
    
    # Try GROUP BY on integers instead
    print("\n--- GROUP BY on integer p column directly ---")
    t0 = time.perf_counter()
    grouped = bindings.df.group_by('prop').agg(pl.len().alias('count'))
    t1 = time.perf_counter()
    print(f"Integer GROUP BY time: {t1-t0:.3f}s, groups: {len(grouped)}")

# Profile the full query execution with timing
print("\n\n=== Full execute_sparql timing ===")
from rdf_starbase import execute_sparql

print("\n--- Query 1: ?s a ?o LIMIT 100 ---")
t0 = time.perf_counter()
r1 = execute_sparql(store, query1_str)
t1 = time.perf_counter()
print(f"Total time: {t1-t0:.3f}s, rows: {len(r1)}")

print("\n--- Query 2: GROUP BY ?prop ---")
t0 = time.perf_counter()
r2 = execute_sparql(store, query2_str)
t1 = time.perf_counter()
print(f"Total time: {t1-t0:.3f}s, rows: {len(r2)}")
