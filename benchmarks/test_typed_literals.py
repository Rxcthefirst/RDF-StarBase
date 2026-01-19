"""Test typed literal storage and FILTER comparisons."""
from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql.parser import SPARQLStarParser
from rdf_starbase.sparql.executor import SPARQLExecutor

# Create store and add some age data
store = TripleStore()
prov = ProvenanceContext(source='test', confidence=1.0)

# Add triples with typed literals (simulating what Turtle parser produces)
for i in range(100):
    age = 70 + (i % 30)  # Ages 70-99
    age_literal = f'"{age}"^^<http://www.w3.org/2001/XMLSchema#integer>'
    store.add_triple(
        f'http://example.org/person{i}',
        'http://example.org/age',
        age_literal,
        prov
    )

# Check how ages are stored
print('DataFrame sample:')
print(store._df.select(['object', 'object_value']).head(5))
print()

# Run FILTER query
parser = SPARQLStarParser()
executor = SPARQLExecutor(store)

query = '''
PREFIX ex: <http://example.org/>
SELECT ?person ?age
WHERE {
    ?person ex:age ?age .
    FILTER(?age > 80)
}
'''
print("Running FILTER query: ?age > 80")
ast = parser.parse(query)
result = executor.execute(ast)
print(f'People with age > 80: {len(result)}')
print(result.head(5))

# Test with equality
query2 = '''
PREFIX ex: <http://example.org/>
SELECT ?person ?age
WHERE {
    ?person ex:age ?age .
    FILTER(?age = 85)
}
'''
print("\nRunning FILTER query: ?age = 85")
ast2 = parser.parse(query2)
result2 = executor.execute(ast2)
print(f'People with age = 85: {len(result2)}')
print(result2.head(5))
