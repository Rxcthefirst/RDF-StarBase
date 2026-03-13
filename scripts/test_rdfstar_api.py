"""Test RDF-Star querying in repository API."""
import requests
import json

API_BASE = "http://localhost:8000/api/v1"

# First, create a test repo
print("Creating test repository...")
response = requests.post(f"{API_BASE}/repositories", json={"name": "test-rdfstar", "description": "Test RDF-Star"})
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

# Import sample RDF-Star data
print("\nImporting RDF-Star data...")
data = """
@prefix ex: <http://example.org/> .
@prefix prov: <http://www.w3.org/ns/prov#> .

ex:person1 ex:name "Alice" .
<<ex:person1 ex:name "Alice">> prov:wasDerivedFrom ex:source1 .
<<ex:person1 ex:name "Alice">> prov:value "0.95" .
"""

response = requests.post(
    f"{API_BASE}/repositories/test-rdfstar/import",
    json={"data": data, "format": "turtle"}
)
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

# Query all triples
print("\nQuerying all triples...")
query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
response = requests.post(
    f"{API_BASE}/repositories/test-rdfstar/sparql",
    json={"query": query}
)
result = response.json()
print(f"  Found {len(result.get('results', []))} triples:")
for row in result.get('results', []):
    print(f"    {row}")

# Query with RDF-Star pattern
print("\nQuerying RDF-Star provenance...")
query = """
PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?source WHERE {
    << <http://example.org/person1> <http://example.org/name> "Alice" >> prov:wasDerivedFrom ?source .
}
"""
response = requests.post(
    f"{API_BASE}/repositories/test-rdfstar/sparql",
    json={"query": query}
)
print(f"  Status: {response.status_code}")
result = response.json()
print(f"  Result: {result}")

# Check what format the quoted triples are stored in
print("\nChecking stored format for quoted triples...")
query = """
SELECT ?s ?p ?o WHERE { 
    ?s ?p ?o .
    FILTER(STRSTARTS(STR(?s), "<<"))
}
"""
response = requests.post(
    f"{API_BASE}/repositories/test-rdfstar/sparql",
    json={"query": query}
)
result = response.json()
print(f"  Quoted triple assertions: {result.get('results', [])}")

# Cleanup
print("\nCleaning up...")
requests.delete(f"{API_BASE}/repositories/test-rdfstar")
print("Done!")
