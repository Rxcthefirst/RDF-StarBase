"""Test real SPARQL-Star queries with quoted triple syntax."""
from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext
from rdf_starbase.sparql import parse_query
from rdf_starbase.sparql.executor import SPARQLExecutor

store = TripleStore()
store.add_triple(
    '<http://example.org/alice>', 
    '<http://xmlns.com/foaf/0.1/name>', 
    'Alice',
    provenance=ProvenanceContext(source='dbpedia', confidence=0.95, process='etl')
)
store.add_triple(
    '<http://example.org/bob>', 
    '<http://xmlns.com/foaf/0.1/name>', 
    'Bob',
    provenance=ProvenanceContext(source='wikidata', confidence=0.8)
)

executor = SPARQLExecutor(store)

# Test 1: Variable predicate - get ALL provenance metadata
print("=" * 60)
print("Test 1: Query ALL metadata using variable predicate ?mp ?mo")
query_str = '''
SELECT ?s ?p ?o ?mp ?mo WHERE {
  << ?s ?p ?o >> ?mp ?mo .
}
'''
print("Query:", query_str)
try:
    query = parse_query(query_str)
    result = executor.execute(query)
    print("Result:")
    print(result)
except Exception as e:
    import traceback
    traceback.print_exc()

# Test 2: Specific predicate for comparison
print("\n" + "=" * 60)
print("Test 2: Query specific predicate (prov:value) for comparison")
query_str2 = '''
PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?s ?p ?o ?conf WHERE {
  << ?s ?p ?o >> prov:value ?conf .
}
'''
print("Query:", query_str2)
try:
    query2 = parse_query(query_str2)
    result2 = executor.execute(query2)
    print("Result:")
    print(result2)
except Exception as e:
    import traceback
    traceback.print_exc()

