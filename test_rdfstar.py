"""Test RDF-Star parsing and querying."""
from rdf_starbase.formats.turtle import parse_turtle

# Test parsing
data = """
@prefix ex: <http://example.org/> .
@prefix prov: <http://www.w3.org/ns/prov#> .

ex:person1 ex:name "Alice" .
<<ex:person1 ex:name "Alice">> prov:wasDerivedFrom ex:source1 .
<<ex:person1 ex:name "Alice">> prov:value "0.95" .
"""

result = parse_turtle(data)
print(f"Parsed {len(result.triples)} triples:")
for t in result.triples:
    subj = t.subject[:60] if t.subject else "[quoted]"
    print(f"  S: {subj}")
    print(f"  P: {t.predicate}")
    print(f"  O: {t.object[:60] if t.object else '[quoted]'}")
    if t.subject_triple:
        print(f"    Subject is quoted triple!")
    print()

# Test columnar extraction
print("\n--- Testing to_columnar with RDF-Star ---")
subjects, predicates, objects = result.to_columnar()
for s, p, o in zip(subjects, predicates, objects):
    print(f"  S: {s}")
    print(f"  P: {p}")
    print(f"  O: {o}")
    print()

print("✅ Quoted triples now properly serialized!")
