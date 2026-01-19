"""Profile triples() method."""
import time
import gc

# Setup
n_entities = 20000
lines = ['@prefix ex: <http://example.org/> .', '@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '']
for i in range(n_entities):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:name "Person {i}" ; foaf:age {i % 100} .')
ttl = '\n'.join(lines)

from rdf_starbase.compat.rdflib import Graph, RDF, FOAF

g = Graph()
g.parse(data=ttl, format="turtle")
print(f"Dataset: {len(g):,} triples")

# Test 1: Just the store query
gc.collect()
start = time.perf_counter()
results = g._store.get_triples(
    predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    obj="http://xmlns.com/foaf/0.1/Person"
)
t1 = time.perf_counter() - start
print(f"Store query (Polars filter): {t1*1000:.1f}ms ({len(results):,} rows)")

# Test 2: Just iteration
gc.collect()
start = time.perf_counter()
count = 0
for row in results.iter_rows():
    count += 1
t2 = time.perf_counter() - start
print(f"Row iteration only: {t2*1000:.1f}ms ({count:,} rows)")

# Test 3: Iteration with named=True
gc.collect()
start = time.perf_counter()
count = 0
for row in results.iter_rows(named=True):
    count += 1
t3 = time.perf_counter() - start
print(f"Row iteration (named=True): {t3*1000:.1f}ms ({count:,} rows)")

# Test 4: Full conversion
from rdf_starbase.compat.rdflib import URIRef, Literal, BNode
gc.collect()
start = time.perf_counter()
count = 0
for row in results.iter_rows(named=True):
    s = URIRef(row['subject'])
    p = URIRef(row['predicate'])
    o = URIRef(row['object']) if row['object'].startswith('http') else Literal(row['object'])
    count += 1
t4 = time.perf_counter() - start
print(f"Full term conversion: {t4*1000:.1f}ms ({count:,} rows)")

# Test 5: List conversion instead of iter
gc.collect()
start = time.perf_counter()
triples = []
for row in results.iter_rows(named=True):
    triples.append((
        URIRef(row['subject']),
        URIRef(row['predicate']),
        URIRef(row['object']) if row['object'].startswith('http') else Literal(row['object'])
    ))
t5 = time.perf_counter() - start
print(f"Build list of triples: {t5*1000:.1f}ms ({len(triples):,} triples)")

# For comparison: rdflib
try:
    import rdflib
    g2 = rdflib.Graph()
    g2.parse(data=ttl, format="turtle")
    
    gc.collect()
    start = time.perf_counter()
    results2 = list(g2.triples((None, rdflib.RDF.type, rdflib.FOAF.Person)))
    t6 = time.perf_counter() - start
    print(f"rdflib triples(): {t6*1000:.1f}ms ({len(results2):,} triples)")
except ImportError:
    pass
