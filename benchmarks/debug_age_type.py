"""Check how ages are stored in the store."""
from rdf_starbase.compat.rdflib import Graph

lines = ['@prefix foaf: <http://xmlns.com/foaf/0.1/> .', '@prefix ex: <http://example.org/> .', '']
for i in range(10):
    lines.append(f'ex:e{i} a foaf:Person ; foaf:age {i} .')
ttl = '\n'.join(lines)

g = Graph()
g.parse(data=ttl, format='turtle')

# Check the raw store
import polars as pl
df = g._store._df
print("Store DataFrame:")
print(df.filter(pl.col("predicate").str.contains("age")))

# Check what the executor sees
from rdf_starbase.sparql import execute_sparql
query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?age WHERE { ?p foaf:age ?age }
"""
result = execute_sparql(g._store, query)
print("\nSPARQL result:")
print(result)
print("\nColumn dtypes:", result.dtypes)
