"""
Tests for JSON-LD parser and serializer.
"""

import pytest
import json
from rdf_starbase.formats.jsonld import (
    JSONLDParser, JSONLDSerializer, 
    parse_jsonld, serialize_jsonld,
    Triple
)


class TestJSONLDParser:
    """Test JSON-LD parsing."""
    
    @pytest.fixture
    def parser(self):
        return JSONLDParser()
    
    def test_parse_simple(self, parser):
        """Test parsing a simple JSON-LD document."""
        jsonld = {
            "@id": "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name": "Alice"
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/alice"
        assert t.predicate == "http://xmlns.com/foaf/0.1/name"
        assert '"Alice"' in t.object
    
    def test_parse_with_context(self, parser):
        """Test parsing with @context."""
        jsonld = {
            "@context": {
                "foaf": "http://xmlns.com/foaf/0.1/",
                "name": "foaf:name"
            },
            "@id": "http://example.org/alice",
            "name": "Alice"
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].predicate == "http://xmlns.com/foaf/0.1/name"
    
    def test_parse_type(self, parser):
        """Test parsing @type."""
        jsonld = {
            "@context": {
                "foaf": "http://xmlns.com/foaf/0.1/"
            },
            "@id": "http://example.org/alice",
            "@type": "foaf:Person"
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        assert doc.triples[0].object == "http://xmlns.com/foaf/0.1/Person"
    
    def test_parse_multiple_types(self, parser):
        """Test parsing multiple @type values."""
        jsonld = {
            "@id": "http://example.org/alice",
            "@type": ["http://xmlns.com/foaf/0.1/Person", "http://schema.org/Person"]
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 2
    
    def test_parse_nested_object(self, parser):
        """Test parsing nested objects."""
        jsonld = {
            "@context": {
                "foaf": "http://xmlns.com/foaf/0.1/",
                "knows": {"@id": "foaf:knows", "@type": "@id"}
            },
            "@id": "http://example.org/alice",
            "knows": {
                "@id": "http://example.org/bob",
                "foaf:name": "Bob"
            }
        }
        doc = parser.parse(jsonld)
        
        # Should have: alice knows bob, bob name "Bob"
        assert len(doc.triples) == 2
    
    def test_parse_value_object(self, parser):
        """Test parsing @value objects."""
        jsonld = {
            "@id": "http://example.org/alice",
            "http://example.org/label": {
                "@value": "Alice",
                "@language": "en"
            }
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 1
        assert '@en' in doc.triples[0].object
    
    def test_parse_typed_value(self, parser):
        """Test parsing typed values."""
        jsonld = {
            "@id": "http://example.org/alice",
            "http://example.org/age": {
                "@value": "30",
                "@type": "http://www.w3.org/2001/XMLSchema#integer"
            }
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 1
        assert "integer" in doc.triples[0].object
    
    def test_parse_json_types(self, parser):
        """Test automatic type detection for JSON types."""
        jsonld = {
            "@id": "http://example.org/thing",
            "http://example.org/count": 42,
            "http://example.org/price": 19.99,
            "http://example.org/active": True
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 3
        
        # Check for proper datatypes
        objects = {t.predicate: t.object for t in doc.triples}
        assert "integer" in objects["http://example.org/count"]
        assert "double" in objects["http://example.org/price"]
        assert "boolean" in objects["http://example.org/active"]
    
    def test_parse_list(self, parser):
        """Test parsing @list."""
        jsonld = {
            "@id": "http://example.org/thing",
            "http://example.org/items": {
                "@list": ["a", "b", "c"]
            }
        }
        doc = parser.parse(jsonld)
        
        # Should create RDF list structure
        assert len(doc.triples) > 3  # list nodes + connections
    
    def test_parse_graph(self, parser):
        """Test parsing @graph."""
        jsonld = {
            "@graph": [
                {"@id": "http://example.org/a", "http://example.org/p": "1"},
                {"@id": "http://example.org/b", "http://example.org/p": "2"}
            ]
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 2
    
    def test_parse_string(self, parser):
        """Test parsing from JSON string."""
        jsonld_str = '{"@id": "http://example.org/x", "http://example.org/p": "value"}'
        doc = parser.parse(jsonld_str)
        
        assert len(doc.triples) == 1
    
    def test_parse_array_values(self, parser):
        """Test parsing arrays as multiple values."""
        jsonld = {
            "@id": "http://example.org/thing",
            "http://example.org/tags": ["a", "b", "c"]
        }
        doc = parser.parse(jsonld)
        
        assert len(doc.triples) == 3


class TestJSONLDSerializer:
    """Test JSON-LD serialization."""
    
    def test_serialize_simple(self):
        """Test serializing simple triples."""
        triples = [
            Triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", '"Alice"')
        ]
        
        result = serialize_jsonld(triples)
        data = json.loads(result)
        
        assert data["@id"] == "http://example.org/alice"
        assert "name" in data or "foaf:name" in data
    
    def test_serialize_with_context(self):
        """Test serializing with context."""
        triples = [
            Triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/name", '"Alice"')
        ]
        
        context = {
            "foaf": "http://xmlns.com/foaf/0.1/",
            "name": {"@id": "foaf:name"}
        }
        
        result = serialize_jsonld(triples, context)
        data = json.loads(result)
        
        assert "@context" in data
    
    def test_serialize_type(self):
        """Test serializing rdf:type as @type."""
        triples = [
            Triple(
                "http://example.org/alice",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/Person"
            )
        ]
        
        result = serialize_jsonld(triples)
        data = json.loads(result)
        
        assert "@type" in data
    
    def test_serialize_multiple_values(self):
        """Test serializing multiple values for same predicate."""
        triples = [
            Triple("http://example.org/thing", "http://example.org/tag", '"a"'),
            Triple("http://example.org/thing", "http://example.org/tag", '"b"'),
            Triple("http://example.org/thing", "http://example.org/tag", '"c"'),
        ]
        
        result = serialize_jsonld(triples)
        data = json.loads(result)
        
        # Should have array of tags
        tag_key = [k for k in data.keys() if "tag" in k.lower()]
        assert len(tag_key) == 1
        tags = data[tag_key[0]]
        assert isinstance(tags, list)
        assert len(tags) == 3
    
    def test_serialize_graph(self):
        """Test serializing multiple subjects as @graph."""
        triples = [
            Triple("http://example.org/a", "http://example.org/p", '"1"'),
            Triple("http://example.org/b", "http://example.org/p", '"2"'),
        ]
        
        result = serialize_jsonld(triples)
        data = json.loads(result)
        
        assert "@graph" in data
        assert len(data["@graph"]) == 2
    
    def test_roundtrip(self):
        """Test parse -> serialize -> parse roundtrip."""
        original = {
            "@context": {
                "foaf": "http://xmlns.com/foaf/0.1/"
            },
            "@id": "http://example.org/alice",
            "@type": "foaf:Person",
            "foaf:name": "Alice",
            "foaf:age": 30
        }
        
        # Parse
        doc = parse_jsonld(original)
        
        # Serialize
        serialized = serialize_jsonld(doc.triples, {"foaf": "http://xmlns.com/foaf/0.1/"})
        
        # Parse again
        doc2 = parse_jsonld(serialized)
        
        # Should have same number of triples
        assert len(doc2.triples) == len(doc.triples)


class TestJSONLDConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_jsonld_function(self):
        """Test parse_jsonld function."""
        doc = parse_jsonld({"@id": "http://x", "http://p": "v"})
        assert len(doc.triples) == 1
    
    def test_serialize_jsonld_function(self):
        """Test serialize_jsonld function."""
        result = serialize_jsonld([
            Triple("http://x", "http://p", '"v"')
        ])
        assert "http://x" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
