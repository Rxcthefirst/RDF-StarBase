"""
Tests for RDF/XML parser and serializer.
"""

import pytest
from rdf_starbase.formats.rdfxml import (
    RDFXMLParser, RDFXMLSerializer,
    parse_rdfxml, serialize_rdfxml,
    Triple, RDF_NS
)


class TestRDFXMLParser:
    """Tests for RDF/XML parsing."""
    
    def test_simple_description(self):
        """Test parsing a simple rdf:Description."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:name>Alice</foaf:name>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/alice"
        assert t.predicate == "http://xmlns.com/foaf/0.1/name"
        assert t.object == '"Alice"'
    
    def test_multiple_properties(self):
        """Test parsing multiple properties on a resource."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:name>Alice</foaf:name>
                <foaf:age rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">30</foaf:age>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 2
        
        names = [t for t in doc.triples if "name" in t.predicate]
        assert len(names) == 1
        assert names[0].object == '"Alice"'
        
        ages = [t for t in doc.triples if "age" in t.predicate]
        assert len(ages) == 1
        assert "30" in ages[0].object
        assert "integer" in ages[0].object
    
    def test_rdf_resource(self):
        """Test parsing rdf:resource attribute."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:knows rdf:resource="http://example.org/bob"/>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/alice"
        assert t.predicate == "http://xmlns.com/foaf/0.1/knows"
        assert t.object == "http://example.org/bob"
    
    def test_typed_node(self):
        """Test parsing typed nodes (not rdf:Description)."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <foaf:Person rdf:about="http://example.org/alice">
                <foaf:name>Alice</foaf:name>
            </foaf:Person>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 2
        
        # Should have type triple
        type_triples = [t for t in doc.triples if "type" in t.predicate]
        assert len(type_triples) == 1
        assert type_triples[0].object == "http://xmlns.com/foaf/0.1/Person"
        
        # And name triple
        name_triples = [t for t in doc.triples if "name" in t.predicate]
        assert len(name_triples) == 1
    
    def test_language_tag(self):
        """Test parsing xml:lang attribute."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
            <rdf:Description rdf:about="http://example.org/cat">
                <rdfs:label xml:lang="en">cat</rdfs:label>
                <rdfs:label xml:lang="fr">chat</rdfs:label>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 2
        
        en = [t for t in doc.triples if "@en" in t.object]
        assert len(en) == 1
        assert '"cat"@en' == en[0].object
        
        fr = [t for t in doc.triples if "@fr" in t.object]
        assert len(fr) == 1
        assert '"chat"@fr' == fr[0].object
    
    def test_datatype(self):
        """Test parsing rdf:datatype attribute."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <rdf:Description rdf:about="http://example.org/item">
                <ex:price rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">19.99</ex:price>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert '"19.99"^^<http://www.w3.org/2001/XMLSchema#decimal>' == t.object
    
    def test_blank_node_nodeid(self):
        """Test parsing rdf:nodeID for blank nodes."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:nodeID="alice">
                <foaf:name>Alice</foaf:name>
            </rdf:Description>
            <rdf:Description rdf:about="http://example.org/knows">
                <foaf:knows rdf:nodeID="alice"/>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        
        # Find alice's name triple
        name_triples = [t for t in doc.triples if "name" in t.predicate]
        assert len(name_triples) == 1
        assert name_triples[0].subject == "_:alice"
        
        # Find knows triple
        knows_triples = [t for t in doc.triples if "knows" in t.predicate]
        assert len(knows_triples) == 1
        assert knows_triples[0].object == "_:alice"
    
    def test_parse_type_resource(self):
        """Test parsing rdf:parseType='Resource'."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:address rdf:parseType="Resource">
                    <foaf:city>New York</foaf:city>
                    <foaf:zip>10001</foaf:zip>
                </foaf:address>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        
        # Should have 3 triples: alice->address->blank, blank->city->NY, blank->zip->10001
        assert len(doc.triples) == 3
        
        # Find address triple
        addr_triples = [t for t in doc.triples if "address" in t.predicate]
        assert len(addr_triples) == 1
        blank = addr_triples[0].object
        assert blank.startswith("_:")
        
        # Find city triple
        city_triples = [t for t in doc.triples if "city" in t.predicate]
        assert len(city_triples) == 1
        assert city_triples[0].subject == blank
        assert '"New York"' == city_triples[0].object
    
    def test_nested_description(self):
        """Test parsing nested rdf:Description elements."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:knows>
                    <rdf:Description rdf:about="http://example.org/bob">
                        <foaf:name>Bob</foaf:name>
                    </rdf:Description>
                </foaf:knows>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        
        # Should have 2 triples: alice knows bob, bob name "Bob"
        assert len(doc.triples) == 2
        
        knows_triples = [t for t in doc.triples if "knows" in t.predicate]
        assert len(knows_triples) == 1
        assert knows_triples[0].subject == "http://example.org/alice"
        assert knows_triples[0].object == "http://example.org/bob"
        
        name_triples = [t for t in doc.triples if "name" in t.predicate]
        assert len(name_triples) == 1
        assert name_triples[0].subject == "http://example.org/bob"
    
    def test_parse_type_collection(self):
        """Test parsing rdf:parseType='Collection'."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <rdf:Description rdf:about="http://example.org/list">
                <ex:items rdf:parseType="Collection">
                    <rdf:Description rdf:about="http://example.org/item1"/>
                    <rdf:Description rdf:about="http://example.org/item2"/>
                </ex:items>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        
        # Should create list structure
        # list->items->blank, blank->first->item1, blank->rest->blank2,
        # blank2->first->item2, blank2->rest->nil
        items_triples = [t for t in doc.triples if "items" in t.predicate]
        assert len(items_triples) == 1
        
        first_triples = [t for t in doc.triples if "first" in t.predicate]
        assert len(first_triples) == 2
        
        rest_triples = [t for t in doc.triples if "rest" in t.predicate]
        assert len(rest_triples) == 2
        
        # Last rest should be nil
        nil_triples = [t for t in doc.triples if "nil" in t.object]
        assert len(nil_triples) == 1
    
    def test_rdf_id(self):
        """Test parsing rdf:ID attribute."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/"
                 xml:base="http://example.org/">
            <rdf:Description rdf:ID="alice">
                <foaf:name>Alice</foaf:name>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/#alice"
    
    def test_empty_literal(self):
        """Test parsing empty literals."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <rdf:Description rdf:about="http://example.org/item">
                <ex:note></ex:note>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        assert doc.triples[0].object == '""'
    
    def test_escaped_characters(self):
        """Test parsing strings with special characters."""
        rdfxml = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:ex="http://example.org/">
            <rdf:Description rdf:about="http://example.org/item">
                <ex:desc>Line 1
Line 2</ex:desc>
            </rdf:Description>
        </rdf:RDF>
        """
        doc = parse_rdfxml(rdfxml)
        assert len(doc.triples) == 1
        # Newline should be escaped
        assert "\\n" in doc.triples[0].object


class TestRDFXMLSerializer:
    """Tests for RDF/XML serialization."""
    
    def test_simple_serialization(self):
        """Test basic serialization."""
        triples = [
            Triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/name",
                '"Alice"'
            )
        ]
        
        xml = serialize_rdfxml(triples)
        assert '<?xml version' in xml
        assert 'rdf:Description' in xml
        assert 'rdf:about="http://example.org/alice"' in xml
        assert 'Alice' in xml
    
    def test_multiple_triples(self):
        """Test serializing multiple triples."""
        triples = [
            Triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/name",
                '"Alice"'
            ),
            Triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/knows",
                "http://example.org/bob"
            )
        ]
        
        xml = serialize_rdfxml(triples)
        assert 'rdf:resource="http://example.org/bob"' in xml
    
    def test_blank_node_serialization(self):
        """Test serializing blank nodes."""
        triples = [
            Triple(
                "_:b1",
                "http://xmlns.com/foaf/0.1/name",
                '"Anonymous"'
            )
        ]
        
        xml = serialize_rdfxml(triples)
        assert 'rdf:nodeID="b1"' in xml
    
    def test_language_serialization(self):
        """Test serializing language-tagged literals."""
        triples = [
            Triple(
                "http://example.org/cat",
                "http://www.w3.org/2000/01/rdf-schema#label",
                '"cat"@en'
            )
        ]
        
        xml = serialize_rdfxml(triples)
        assert 'xml:lang="en"' in xml
        assert '>cat<' in xml
    
    def test_datatype_serialization(self):
        """Test serializing typed literals."""
        triples = [
            Triple(
                "http://example.org/item",
                "http://example.org/price",
                '"19.99"^^<http://www.w3.org/2001/XMLSchema#decimal>'
            )
        ]
        
        xml = serialize_rdfxml(triples)
        assert 'rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal"' in xml
        assert '>19.99<' in xml


class TestRoundTrip:
    """Test round-trip parsing and serialization."""
    
    def test_roundtrip_simple(self):
        """Test round-trip of simple data."""
        original = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:foaf="http://xmlns.com/foaf/0.1/">
            <rdf:Description rdf:about="http://example.org/alice">
                <foaf:name>Alice</foaf:name>
                <foaf:knows rdf:resource="http://example.org/bob"/>
            </rdf:Description>
        </rdf:RDF>
        """
        
        # Parse
        doc = parse_rdfxml(original)
        assert len(doc.triples) == 2
        
        # Serialize and re-parse
        serialized = serialize_rdfxml(doc.triples)
        doc2 = parse_rdfxml(serialized)
        
        # Should have same triples
        assert len(doc2.triples) == 2
        
        subjects = {t.subject for t in doc2.triples}
        assert "http://example.org/alice" in subjects
    
    def test_roundtrip_typed_literals(self):
        """Test round-trip of typed literals."""
        triples = [
            Triple(
                "http://example.org/item",
                "http://example.org/count",
                '"42"^^<http://www.w3.org/2001/XMLSchema#integer>'
            )
        ]
        
        xml = serialize_rdfxml(triples)
        doc = parse_rdfxml(xml)
        
        assert len(doc.triples) == 1
        assert "42" in doc.triples[0].object
        assert "integer" in doc.triples[0].object
