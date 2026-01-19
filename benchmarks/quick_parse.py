"""Quick parse benchmark."""
import time
import gc
from rdf_starbase.compat.rdflib import Graph

# Generate turtle data
n = 10000
lines = ['@prefix ex: <http://example.org/> .', '@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '']
for i in range(n):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:name "Person {i}" ; foaf:age {i % 100} .')
ttl = '\n'.join(lines)

gc.collect()
start = time.perf_counter()
g = Graph()
g.parse(data=ttl, format='turtle')
t = time.perf_counter() - start
print(f'Parsed {n*3:,} triples in {t:.3f}s = {n*3/t:,.0f} triples/sec')
