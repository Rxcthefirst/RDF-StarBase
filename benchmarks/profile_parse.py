"""Profile where the time goes in parsing."""
import time
import gc

# Generate turtle data first
n = 10000
lines = ['@prefix ex: <http://example.org/> .', '@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '']
for i in range(n):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:name "Person {i}" ; foaf:age {i % 100} .')
ttl = '\n'.join(lines)
print(f"Generated {n*3:,} triples worth of Turtle")

# Step 1: Parse only
gc.collect()
start = time.perf_counter()
from rdf_starbase.formats.turtle import parse_turtle
doc = parse_turtle(ttl)
t1 = time.perf_counter() - start
print(f"Step 1 - Parse Turtle: {t1:.3f}s ({n*3/t1:,.0f} triples/sec)")

# Step 2: Extract columns
gc.collect()
start = time.perf_counter()
subjects = [t.subject for t in doc.triples]
predicates = [t.predicate for t in doc.triples]
objects = [t.object for t in doc.triples]
t2 = time.perf_counter() - start
print(f"Step 2 - Extract columns: {t2:.3f}s")

# Step 3: Intern terms (this is the suspected bottleneck)
gc.collect()
start = time.perf_counter()
from rdf_starbase import TripleStore
store = TripleStore()
store.add_triples_columnar(subjects, predicates, objects, source="test")
t3 = time.perf_counter() - start
print(f"Step 3 - Intern + Insert: {t3:.3f}s ({n*3/t3:,.0f} triples/sec)")

print(f"\nTotal: {t1+t2+t3:.3f}s ({n*3/(t1+t2+t3):,.0f} triples/sec)")
print(f"Breakdown: Parse={t1/(t1+t2+t3)*100:.0f}% Columns={t2/(t1+t2+t3)*100:.0f}% Intern={t3/(t1+t2+t3)*100:.0f}%")
