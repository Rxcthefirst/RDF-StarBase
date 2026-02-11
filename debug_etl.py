"""Debug ETL output generation."""
from rdf_starbase.etl_engine import ETLEngine, OutputFormat
import tempfile
import os

csv_data = '''customer_id,first_name,last_name,email_address
CUST00001,Alice,Smith,alice@example.com
CUST00002,Bob,Jones,bob@example.com'''

# Simulating Starchart mapping config with ecom prefix BUT PREFIX NOT DECLARED
# This simulates what happens when the ontology API doesn't return the ecom prefix
mapping_config = {
    'prefixes': {
        # NOTE: ecom is NOT included - this is the bug!
        'owl': 'http://www.w3.org/2002/07/owl#',
        'xsd': 'http://www.w3.org/2001/XMLSchema#',
    },
    'sources': [{'file': 'test.csv'}],
    'subject_template': 'http://example.org/customer/$(customer_id)',
    'mappings': [
        {'source_column': 'first_name', 'predicate': 'ecom:firstName'},
        {'source_column': 'last_name', 'predicate': 'ecom:lastName'},
        {'source_column': 'email_address', 'predicate': 'ecom:email'},
    ],
    'class_type': 'ecom:Customer',
    'add_owl_individual': True,
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(csv_data)
    temp_path = f.name

try:
    engine = ETLEngine()
    rdf, report = engine.transform(temp_path, mapping_config, OutputFormat.TURTLE)
    print('=== GENERATED TURTLE ===')
    print(rdf)
    print()
    
    # Show character positions around 474
    if len(rdf) >= 474:
        print(f'=== AROUND POSITION 474 ===')
        print(f'Chars 460-500: {repr(rdf[460:500])}')
        print(f'Chars 470-480: {repr(rdf[470:480])}')
    
    # Try to parse with rdflib to see if there's an issue
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(data=rdf, format='turtle')
        print(f'\n✓ RDFLib parsed successfully! {len(g)} triples')
    except Exception as e:
        print(f'\n✗ RDFLib parse error: {e}')
    
    # Try to parse with OUR parser
    try:
        from rdf_starbase.formats.turtle import TurtleParser
        from io import StringIO
        parser = TurtleParser()
        result = parser.parse(StringIO(rdf))
        print(f'✓ Our parser: {len(result.triples)} triples')
    except Exception as e:
        print(f'✗ Our parser error: {e}')
        # Show context around failure
        import re
        match = re.search(r'position (\d+)', str(e))
        if match:
            pos = int(match.group(1))
            print(f'Context at position {pos}:')
            print(f'  Before: {repr(rdf[max(0,pos-50):pos])}')
            print(f'  At:     {repr(rdf[pos:pos+10])}')
            print(f'  After:  {repr(rdf[pos:min(len(rdf),pos+50)])}')
        
finally:
    os.unlink(temp_path)
