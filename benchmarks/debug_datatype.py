"""Debug datatype storage."""
from rdf_starbase.store import TripleStore
from rdf_starbase.models import ProvenanceContext

store = TripleStore()
prov = ProvenanceContext(source='test', confidence=1.0)

# Check the XSD datatype IDs
print("XSD Integer ID:", store._term_dict.xsd_integer_id)
print("XSD Decimal ID:", store._term_dict.xsd_decimal_id)
print("XSD Double ID:", store._term_dict.xsd_double_id)
print("XSD Boolean ID:", store._term_dict.xsd_boolean_id)
print()

# Add a typed literal
age_literal = '"70"^^<http://www.w3.org/2001/XMLSchema#integer>'
store.add_triple(
    'http://example.org/person1',
    'http://example.org/age',
    age_literal,
    prov
)

# Check term storage
print("Terms in dictionary:")
for tid, term in list(store._term_dict._id_to_term.items())[:20]:
    print(f"  {tid}: kind={term.kind.name}, lex='{term.lex}', datatype_id={term.datatype_id}")

print()
print("Raw fact store:")
print(store._fact_store._df)

print()
print("Materialized _df:")
df = store._df
print(df.columns)
print(df.select(['object', 'object_value', 'object_type']))
