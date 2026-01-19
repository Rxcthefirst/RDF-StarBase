"""Compare parsing speed of our parser vs rdflib."""
import time
import gc

# Generate turtle data first
n = 10000
lines = ['@prefix ex: <http://example.org/> .', '@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '']
for i in range(n):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:name "Person {i}" ; foaf:age {i % 100} .')
ttl = '\n'.join(lines)
print(f"Generated {n*3:,} triples worth of Turtle")

# Our parser
gc.collect()
start = time.perf_counter()
from rdf_starbase.formats.turtle import parse_turtle
doc = parse_turtle(ttl)
t1 = time.perf_counter() - start
print(f"Our Turtle parser: {t1:.3f}s ({n*3/t1:,.0f} triples/sec)")

# rdflib parser (if available)
try:
    import rdflib
    gc.collect()
    start = time.perf_counter()
    g = rdflib.Graph()
    g.parse(data=ttl, format='turtle')
    t2 = time.perf_counter() - start
    print(f"rdflib parser: {t2:.3f}s ({n*3/t2:,.0f} triples/sec)")
    print(f"Ratio: rdflib is {t1/t2:.1f}x {'faster' if t2 < t1 else 'slower'}")
except ImportError:
    print("rdflib not installed")
