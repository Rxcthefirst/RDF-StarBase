"""Debug FILTER and COUNT issues."""
from rdf_starbase.compat.rdflib import Graph

# Generate turtle
lines = ['@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '@prefix ex: <http://example.org/> .', '']
for i in range(100):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:name "Person {i}" ; foaf:age {i % 100} .')
ttl = '\n'.join(lines)

g = Graph()
g.parse(data=ttl, format='turtle')
print(f"Loaded {len(g)} triples")

# Test FILTER query
query1 = '''
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?age WHERE {
        ?person a foaf:Person .
        ?person foaf:age ?age .
        FILTER(?age > 80)
    }
'''
print("\nQuery 1: FILTER(?age > 80)")
results1 = list(g.query(query1))
print(f"Results: {len(results1)}")
for r in results1[:5]:
    print(f"  {r}")

# Test without FILTER
query2 = '''
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?age WHERE {
        ?person a foaf:Person .
        ?person foaf:age ?age .
    }
'''
print("\nQuery 2: No FILTER")
results2 = list(g.query(query2))
print(f"Results: {len(results2)}")
for r in results2[:3]:
    print(f"  {r}")

# Test COUNT
query3 = '''
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT (COUNT(?person) as ?count) WHERE {
        ?person a foaf:Person .
    }
'''
print("\nQuery 3: COUNT")
results3 = list(g.query(query3))
print(f"Results: {results3}")

# Check what age values look like
print("\nChecking age values:")
for s, p, o in list(g.triples((None, None, None)))[:10]:
    if 'age' in str(p):
        print(f"  {s} {p} {o} (type: {type(o)})")
