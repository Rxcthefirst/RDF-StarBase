"""Test ETL flow with UI-style input."""
import sys
sys.path.insert(0, 'src')
from rdf_starbase.etl_engine import ETLEngine, OutputFormat
import tempfile
from pathlib import Path

# Create a small test dataset
csv_data = '''productId,productName
PROD001,Widget
PROD002,Gadget'''

# Create the engine  
engine = ETLEngine()

# Test 1: With ecom prefix - should work
print("=" * 60)
print("TEST 1: With ecom prefix")
print("=" * 60)
mapping_config = {
    'sources': [{'type': 'csv', 'file': 'test.csv'}],
    'subject_template': 'http://example.org/resource/$(productId)',
    'mappings': [
        {'source_column': 'productId', 'predicate': 'ecom:productId'},
    ],
    'prefixes': {
        'ecom': 'http://example.org/ecommerce#',
    },
    'class_type': 'ecom:Product',
}

# Test 2: WITHOUT ecom prefix - simulates empty/incomplete ontology
print("\n" + "=" * 60)
print("TEST 2: WITHOUT ecom prefix (simulates empty ontology)")
print("=" * 60)
mapping_config_no_prefix = {
    'sources': [{'type': 'csv', 'file': 'test.csv'}],
    'subject_template': 'http://example.org/resource/$(productId)',
    'mappings': [
        {'source_column': 'productId', 'predicate': 'ecom:productId'},
    ],
    'prefixes': {},  # No prefixes!
    'class_type': None,
}

# Test 3: With full URI predicates (what might happen with fallback)
print("\n" + "=" * 60)
print("TEST 3: With full URI predicates")
print("=" * 60)
mapping_config_full_uri = {
    'sources': [{'type': 'csv', 'file': 'test.csv'}],
    'subject_template': 'http://example.org/resource/$(productId)',
    'mappings': [
        {'source_column': 'productId', 'predicate': 'http://example.org/ecommerce#productId'},
    ],
    'prefixes': {},
    'class_type': None,
}

# Run all tests
for name, config in [
    ("Test 1 (with ecom prefix)", mapping_config),
    ("Test 2 (no prefix)", mapping_config_no_prefix),
    ("Test 3 (full URI)", mapping_config_full_uri),
]:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test.csv'
        csv_path.write_text(csv_data)
        
        rdf_output, report = engine.transform(
            data_file=csv_path,
            mapping_config=config,
            output_format=OutputFormat.TURTLE,
        )
        
        print(f"\n--- {name} ---")
        print(rdf_output[:500])  # First 500 chars

# Write CSV to temp file
with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / 'test.csv'
    csv_path.write_text(csv_data)
    
    rdf_output, report = engine.transform(
        data_file=csv_path,
        mapping_config=mapping_config,
        output_format=OutputFormat.TURTLE,
    )
    
    print('Generated Turtle output:')
    print(rdf_output)
    print()
    print('Report:', report)
