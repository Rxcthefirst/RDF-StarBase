"""Debug COUNT aggregation."""
from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql.parser import SPARQLStarParser
from rdf_starbase.sparql.executor import SPARQLExecutor
from rdf_starbase.compat.rdflib import Graph

# Create store and add some data via compat layer
g = Graph()

for i in range(100):
    from rdf_starbase.compat.rdflib import URIRef, RDF, FOAF
    g.add((
        URIRef(f'http://example.org/person{i}'),
        RDF.type,
        FOAF.Person
    ))

# Simple COUNT
query = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT (COUNT(?person) as ?count) WHERE {
    ?person a foaf:Person .
}
"""
print("Running COUNT query via compat layer")
result = g.query(query)

for row in result:
    print("Row:", row)
    print("Row type:", type(row))
    print("Row._row:", row._row if hasattr(row, '_row') else "N/A")
    print("row[0]:", row[0] if hasattr(row, '__getitem__') else "N/A")
    print("tuple(row):", tuple(row))
