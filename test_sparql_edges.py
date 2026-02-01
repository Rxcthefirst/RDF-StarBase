"""
Test SPARQL protocol edge cases for RDF-StarBase.

This test file exercises the boundaries of SPARQL 1.1 and SPARQL-Star support
to identify gaps in protocol coverage.

Run with: python test_sparql_edges.py
"""
from pathlib import Path
from rdf_starbase.store import TripleStore
from rdf_starbase.sparql import SPARQLExecutor, parse_query

# Load the test data first
store = TripleStore()
executor = SPARQLExecutor(store)
query_text = Path('data/insert_queries/test_insert.rq').read_text()
parsed = parse_query(query_text)
result = executor._execute_insert_data(parsed)
print(f"Loaded {result['count']} triples into store")
print("=" * 70)

# Test battery - SPARQL 1.1 Query
tests = [
    # Basic patterns
    ("Basic SELECT", "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"),
    ("SELECT DISTINCT", "PREFIX ex: <https://example.org/> SELECT DISTINCT ?type WHERE { ?s a ?type }"),
    
    # OPTIONAL
    ("OPTIONAL pattern", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p ?label WHERE { 
            ?p a ex:Person . 
            OPTIONAL { ?p rdfs:label ?label } 
        }
    """),
    
    # FILTER expressions  
    ("FILTER regex", 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?s ?label WHERE { ?s rdfs:label ?label FILTER(regex(?label, "Customer")) }'),
    ("FILTER BOUND", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p ?label WHERE { 
            ?p a ex:Person . 
            OPTIONAL { ?p rdfs:label ?label } 
            FILTER(BOUND(?label))
        }
    """),
    ("FILTER CONTAINS", """
        PREFIX ex: <https://example.org/>
        SELECT ?s ?conf WHERE { 
            ?s ex:confidence ?conf
            FILTER(CONTAINS(STR(?conf), "0.9"))
        } LIMIT 5
    """),
    
    # ORDER BY
    ("ORDER BY ASC", "PREFIX dqv: <http://www.w3.org/ns/dqv#> SELECT ?m ?v WHERE { ?m dqv:value ?v } ORDER BY ?v"),
    ("ORDER BY DESC", "PREFIX dqv: <http://www.w3.org/ns/dqv#> SELECT ?m ?v WHERE { ?m dqv:value ?v } ORDER BY DESC(?v)"),
    
    # LIMIT / OFFSET
    ("LIMIT + OFFSET", "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 3 OFFSET 5"),
    
    # Aggregates
    ("COUNT", "PREFIX ex: <https://example.org/> SELECT (COUNT(?p) as ?count) WHERE { ?p a ex:Person }"),
    ("SUM", "PREFIX dqv: <http://www.w3.org/ns/dqv#> SELECT (SUM(?v) as ?total) WHERE { ?m dqv:value ?v }"),
    ("AVG", "PREFIX dqv: <http://www.w3.org/ns/dqv#> SELECT (AVG(?v) as ?avg) WHERE { ?m dqv:value ?v }"),
    ("MIN/MAX", "PREFIX dqv: <http://www.w3.org/ns/dqv#> SELECT (MIN(?v) as ?min) (MAX(?v) as ?max) WHERE { ?m dqv:value ?v }"),
    
    # GROUP BY
    ("GROUP BY", """
        PREFIX ex: <https://example.org/>
        SELECT ?type (COUNT(?s) as ?cnt) WHERE { ?s a ?type } 
        GROUP BY ?type ORDER BY DESC(?cnt) LIMIT 5
    """),
    
    # UNION
    ("UNION", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?entity ?label WHERE {
            { ?entity a ex:Person ; rdfs:label ?label }
            UNION
            { ?entity a ex:Organization ; rdfs:label ?label }
        }
    """),
    
    # ASK
    ("ASK true", "PREFIX ex: <https://example.org/> ASK { ?s a ex:Person }"),
    ("ASK false", "PREFIX ex: <https://example.org/> ASK { ?s a ex:Unicorn }"),
    
    # CONSTRUCT
    ("CONSTRUCT", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        CONSTRUCT { ?p ex:name ?label } 
        WHERE { ?p a ex:Person ; rdfs:label ?label }
    """),
    
    # DESCRIBE
    ("DESCRIBE", "PREFIX ex: <https://example.org/> DESCRIBE ex:Chris"),
    
    # Property paths
    ("Property path /", """
        PREFIX ex: <https://example.org/>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        SELECT ?catalog ?dist WHERE { 
            ?catalog dcat:dataset/dcat:distribution ?dist 
        } LIMIT 5
    """),
    ("Property path |", """
        PREFIX ex: <https://example.org/>
        PREFIX dct: <http://purl.org/dc/terms/>
        SELECT ?s ?agent WHERE {
            ?s (dct:creator|dct:publisher) ?agent
        } LIMIT 5
    """),
    
    # BIND
    ("BIND", """
        PREFIX ex: <https://example.org/>
        SELECT ?p ?fullName WHERE { 
            ?p a ex:Person . 
            BIND(CONCAT("Person: ", STR(?p)) AS ?fullName)
        }
    """),
    
    # VALUES
    ("VALUES inline", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p ?label WHERE {
            VALUES ?p { ex:Chris ex:Ada }
            ?p rdfs:label ?label
        }
    """),
    
    # Subquery
    ("Subquery", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?person ?label WHERE {
            {
                SELECT ?person WHERE { ?person a ex:Person } LIMIT 2
            }
            ?person rdfs:label ?label
        }
    """),
    
    # Named graph query
    ("GRAPH pattern", """
        PREFIX ex: <https://example.org/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?activity ?label WHERE {
            GRAPH ex:g_prov {
                ?activity a prov:Activity ;
                          rdfs:label ?label
            }
        }
    """),
    
    # FROM / FROM NAMED
    ("FROM graph", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?label 
        FROM ex:g_domain
        WHERE { ?s rdfs:label ?label } LIMIT 5
    """),
]

# SPARQL-Star specific tests
sparql_star_tests = [
    # Quoted triple in subject position - WORKS
    ("RDF-star: quoted triple subject", """
        PREFIX ex: <https://example.org/>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT ?confidence WHERE {
            << ex:Dataset_Customers dct:publisher ex:AcmeBank >> prov:value ?confidence
        }
    """),
    
    # All annotations on a specific quoted triple - WORKS
    ("RDF-star: all annotations on triple", """
        PREFIX ex: <https://example.org/>
        PREFIX dct: <http://purl.org/dc/terms/>
        SELECT ?p ?o WHERE {
            << ex:Dataset_Customers dct:publisher ex:AcmeBank >> ?p ?o
        }
    """),
    
    # Find all things with confidence - WORKS
    ("RDF-star: find annotated subjects", """
        PREFIX ex: <https://example.org/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT ?subj ?conf WHERE {
            ?subj prov:value ?conf
        } LIMIT 5
    """),
    
    # Variable in quoted triple pattern (EDGE CASE)
    ("RDF-star: variable in quoted triple", """
        PREFIX ex: <https://example.org/>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT ?dataset ?conf WHERE {
            << ?dataset dct:publisher ex:AcmeBank >> prov:value ?conf
        }
    """),
    
    # Nested quoted triples (EDGE CASE)
    # Query: "Find what verified the annotation about Dataset_Customers conforming to schema"
    ("RDF-star: nested quoted triple", """
        PREFIX ex: <https://example.org/>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        SELECT ?verifier ?certainty WHERE {
            << << ex:Dataset_Customers dct:conformsTo ex:Schema_CustomerV1 >> prov:value "0.95"^^<http://www.w3.org/2001/XMLSchema#decimal> >>
                ex:verifiedBy ?verifier ;
                ex:certainty ?certainty .
        }
    """),
]

# SPARQL Update tests
update_tests = [
    # DELETE DATA - Should parse
    ("DELETE DATA simple", """
        PREFIX ex: <https://example.org/>
        DELETE DATA {
            ex:TestSubject ex:testProp "test_value"
        }
    """),
    
    # INSERT WHERE - Should parse  
    ("INSERT WHERE", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        INSERT { ?p ex:displayName ?label }
        WHERE { ?p a ex:Person ; rdfs:label ?label }
    """),
    
    # DELETE WHERE - Should parse
    ("DELETE WHERE", """
        PREFIX ex: <https://example.org/>
        DELETE { ?s ex:displayName ?o }
        WHERE { ?s ex:displayName ?o }
    """),
    
    # DELETE/INSERT combo
    ("DELETE/INSERT combo", """
        PREFIX ex: <https://example.org/>
        DELETE { ?s ex:oldProp ?o }
        INSERT { ?s ex:newProp ?o }
        WHERE { ?s ex:oldProp ?o }
    """),
    
    # CLEAR (EDGE CASE - may not be supported)
    ("CLEAR GRAPH", "PREFIX ex: <https://example.org/> CLEAR GRAPH ex:temp_graph"),
    
    # DROP (EDGE CASE - may not be supported)
    ("DROP GRAPH", "PREFIX ex: <https://example.org/> DROP SILENT GRAPH ex:temp_graph"),
    
    # CREATE (EDGE CASE - may not be supported)
    ("CREATE GRAPH", "PREFIX ex: <https://example.org/> CREATE GRAPH ex:new_graph"),
    
    # COPY (EDGE CASE - may not be supported)
    ("COPY GRAPH", "PREFIX ex: <https://example.org/> COPY ex:src TO ex:dst"),
    
    # MOVE (EDGE CASE - may not be supported)
    ("MOVE GRAPH", "PREFIX ex: <https://example.org/> MOVE ex:src TO ex:dst"),
    
    # ADD (EDGE CASE - may not be supported)
    ("ADD GRAPH", "PREFIX ex: <https://example.org/> ADD ex:src TO ex:dst"),
]

# Missing SPARQL 1.1 Query features to test
edge_query_tests = [
    # FILTER numeric comparison (stored as strings)
    ("FILTER numeric >", """
        PREFIX dqv: <http://www.w3.org/ns/dqv#>
        SELECT ?m ?v WHERE { ?m dqv:value ?v FILTER(?v > 0.92) }
    """),
    
    # DATATYPE function  
    ("DATATYPE function", """
        PREFIX dqv: <http://www.w3.org/ns/dqv#>
        SELECT ?v (DATATYPE(?v) as ?dt) WHERE { ?m dqv:value ?v }
    """),
    
    # LANG function
    ("LANG function", """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label (LANG(?label) as ?l) WHERE { ?s rdfs:label ?label } LIMIT 3
    """),
    
    # Property path * (zero or more)
    ("Property path *", """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?type WHERE { 
            ?s a/rdfs:subClassOf* ?type 
        } LIMIT 5
    """),
    
    # Property path + (one or more)
    ("Property path +", """
        PREFIX ex: <https://example.org/>
        SELECT ?s ?target WHERE { 
            ?s ex:memberOf+ ?target
        } LIMIT 5
    """),
    
    # Property path ? (zero or one)
    ("Property path ?", """
        PREFIX ex: <https://example.org/>
        SELECT ?s ?target WHERE { 
            ?s ex:memberOf? ?target
        } LIMIT 5
    """),
    
    # Inverse property path
    ("Property path ^", """
        PREFIX ex: <https://example.org/>
        SELECT ?org ?member WHERE { 
            ?org ^ex:memberOf ?member
        } LIMIT 5
    """),
    
    # HAVING clause
    ("HAVING clause", """
        PREFIX ex: <https://example.org/>
        SELECT ?type (COUNT(?s) as ?cnt) WHERE { ?s a ?type }
        GROUP BY ?type
        HAVING (COUNT(?s) > 2)
    """),
    
    # EXISTS filter
    ("FILTER EXISTS", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p WHERE {
            ?p a ex:Person .
            FILTER EXISTS { ?p ex:memberOf ?org }
        }
    """),
    
    # NOT EXISTS filter
    ("FILTER NOT EXISTS", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?p WHERE {
            ?p a ex:Person .
            FILTER NOT EXISTS { ?p ex:email ?email }
        }
    """),
    
    # MINUS
    ("MINUS pattern", """
        PREFIX ex: <https://example.org/>
        SELECT ?s WHERE {
            ?s a ex:Person .
            MINUS { ?s ex:memberOf ex:Ontus }
        }
    """),
    
    # SERVICE (federated query - edge case)
    ("SERVICE federated", """
        PREFIX dbpedia: <http://dbpedia.org/resource/>
        SELECT ?s WHERE {
            SERVICE <http://dbpedia.org/sparql> {
                ?s a dbpedia:Person
            }
        } LIMIT 1
    """),
    
    # IF conditional
    ("IF conditional", """
        PREFIX ex: <https://example.org/>
        SELECT ?p (IF(BOUND(?label), ?label, "Unknown") as ?status) WHERE {
            ?p a ex:Person .
            OPTIONAL { ?p ex:name ?label }
        }
    """),
    
    # COALESCE
    ("COALESCE", """
        PREFIX ex: <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s (COALESCE(?label, "Unknown") as ?name) WHERE {
            ?s a ex:Person .
            OPTIONAL { ?s rdfs:label ?label }
        }
    """),
    
    # SAMPLE aggregate
    ("SAMPLE aggregate", """
        PREFIX ex: <https://example.org/>
        SELECT ?type (SAMPLE(?s) as ?example) WHERE { ?s a ?type }
        GROUP BY ?type LIMIT 5
    """),
    
    # GROUP_CONCAT
    ("GROUP_CONCAT", """
        PREFIX ex: <https://example.org/>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        SELECT ?ds (GROUP_CONCAT(?kw; separator=", ") as ?keywords) WHERE {
            ?ds dcat:keyword ?kw
        } GROUP BY ?ds
    """),
]

print("\n=== SPARQL 1.1 Query Tests ===\n")
passed = 0
failed = 0
failures = []
for name, query in tests:
    try:
        result = executor.execute(parse_query(query))
        if hasattr(result, 'height'):
            print(f"✓ {name} -> {result.height} rows")
        elif isinstance(result, bool):
            print(f"✓ {name} -> {result}")
        elif isinstance(result, list):
            print(f"✓ {name} -> {len(result)} triples")
        else:
            print(f"✓ {name} -> {type(result).__name__}")
        passed += 1
    except Exception as e:
        print(f"✗ {name} -> {type(e).__name__}: {str(e)[:60]}")
        failures.append((name, e))
        failed += 1

print(f"\nQuery tests: {passed} passed, {failed} failed")

print("\n=== SPARQL-Star Tests ===\n")
star_passed = 0
star_failed = 0
for name, query in sparql_star_tests:
    try:
        result = executor.execute(parse_query(query))
        if hasattr(result, 'height'):
            print(f"✓ {name} -> {result.height} rows")
        else:
            print(f"✓ {name} -> {result}")
        star_passed += 1
    except Exception as e:
        print(f"✗ {name} -> {type(e).__name__}: {str(e)[:60]}")
        failures.append((name, e))
        star_failed += 1

print(f"\nSPARQL-Star tests: {star_passed} passed, {star_failed} failed")

print("\n=== SPARQL 1.1 Update Tests ===\n")
update_passed = 0
update_failed = 0
for name, query in update_tests:
    try:
        parsed = parse_query(query)
        print(f"✓ {name} -> parsed as {type(parsed).__name__}")
        update_passed += 1
    except Exception as e:
        print(f"✗ {name} -> {type(e).__name__}: {str(e)[:60]}")
        failures.append((name, e))
        update_failed += 1

print(f"\nUpdate tests: {update_passed} passed, {update_failed} failed")

print("\n=== Edge Case Query Tests ===\n")
edge_passed = 0
edge_failed = 0
for name, query in edge_query_tests:
    try:
        result = executor.execute(parse_query(query))
        if hasattr(result, 'height'):
            print(f"✓ {name} -> {result.height} rows")
        else:
            print(f"✓ {name} -> {result}")
        edge_passed += 1
    except Exception as e:
        print(f"✗ {name} -> {type(e).__name__}: {str(e)[:60]}")
        failures.append((name, e))
        edge_failed += 1

print(f"\nEdge query tests: {edge_passed} passed, {edge_failed} failed")

print("\n" + "=" * 70)
total_passed = passed + star_passed + update_passed + edge_passed
total_failed = failed + star_failed + update_failed + edge_failed
print(f"TOTAL: {total_passed} passed, {total_failed} failed")

if failures:
    print("\n=== Failure Summary ===")
    for name, e in failures:
        print(f"  • {name}: {type(e).__name__}")
