"""Test the columnar ETL engine."""
from rdf_starbase.etl_engine import ETLEngine, YARRRMLParser, OutputFormat, ColumnarRDFGenerator
import polars as pl
import io

# Sample data
csv_data = '''id,name,email,age
1,Alice Smith,alice@example.com,30
2,Bob Jones,bob@example.com,25
3,Carol White,carol@example.com,35'''

# Create DataFrame
df = pl.read_csv(io.StringIO(csv_data))
print('Data loaded:', df.shape)
print(df)

# Sample YARRRML mapping
yarrrml = '''
prefixes:
  ex: "http://example.org/"
  foaf: "http://xmlns.com/foaf/0.1/"
  xsd: "http://www.w3.org/2001/XMLSchema#"

mappings:
  persons:
    sources:
      - [data.csv~csv]
    s: ex:person/$(id)
    po:
      - [foaf:name, $(name)]
      - [foaf:mbox, $(email)]
      - [foaf:age, $(age)]
'''

# Parse and generate
parser = YARRRMLParser()
mapping = parser.parse(yarrrml)
print('\nMapping parsed:', len(mapping.mappings), 'triples maps')
print('Prefixes:', list(mapping.prefixes.keys()))
print('Subject template:', mapping.mappings[0].subject.template)
print('Column refs:', mapping.mappings[0].subject.columns)
print('Predicate-objects:', len(mapping.mappings[0].predicate_objects))

for po in mapping.mappings[0].predicate_objects:
    print(f'  {po.source_column} -> {po.predicate}')

# Generate RDF
gen = ColumnarRDFGenerator()

print('\n=== N-Triples ===')
rdf_nt = gen.generate(df, mapping, OutputFormat.NTRIPLES)
print(rdf_nt)
print(f'Triple count: {gen.triple_count}')

print('\n=== Turtle ===')
gen2 = ColumnarRDFGenerator()
rdf_ttl = gen2.generate(df, mapping, OutputFormat.TURTLE)
print(rdf_ttl)
print(f'Triple count: {gen2.triple_count}')

print('\n=== JSON-LD ===')
gen3 = ColumnarRDFGenerator()
rdf_jsonld = gen3.generate(df, mapping, OutputFormat.JSONLD)
print(rdf_jsonld)
print(f'Triple count: {gen3.triple_count}')

# Test full ETL engine
print('\n=== Full ETL Engine Test ===')
engine = ETLEngine()

# Save test CSV
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / 'test.csv'
    csv_path.write_text(csv_data)
    
    # Using YARRRML string
    rdf, report = engine.transform(
        data_file=csv_path,
        mapping_config=yarrrml,
        output_format=OutputFormat.TURTLE,
    )
    
    print('Transform complete!')
    print(f"Triples: {report['triple_count']}")
    print(f"Rows: {report['row_count']}")
    print(f"Warnings: {report['warnings']}")

print('\nAll tests passed!')
