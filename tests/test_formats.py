"""Tests for RDF format parsers and serializers."""

import pytest
from rdf_starbase.formats.turtle import TurtleParser, TurtleSerializer, Triple, parse_turtle, serialize_turtle
from rdf_starbase.formats.ntriples import NTriplesParser, NTriplesSerializer, parse_ntriples, serialize_ntriples


class TestTurtleParser:
    """Tests for Turtle parser."""
    
    def test_parse_simple_triple(self):
        """Test parsing a simple triple."""
        ttl = '<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .'
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/alice"
        assert t.predicate == "http://example.org/knows"
        assert t.object == "http://example.org/bob"
    
    def test_parse_with_prefix(self):
        """Test parsing with prefix declarations."""
        ttl = """
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        
        ex:alice foaf:name "Alice" .
        """
        doc = parse_turtle(ttl)
        
        assert "ex" in doc.prefixes
        assert doc.prefixes["ex"] == "http://example.org/"
        assert len(doc.triples) == 1
        assert doc.triples[0].subject == "http://example.org/alice"
        assert doc.triples[0].predicate == "http://xmlns.com/foaf/0.1/name"
    
    def test_parse_sparql_style_prefix(self):
        """Test parsing SPARQL-style PREFIX declarations."""
        ttl = """
        PREFIX ex: <http://example.org/>
        ex:alice ex:knows ex:bob .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].subject == "http://example.org/alice"
    
    def test_parse_literal_with_language(self):
        """Test parsing literals with language tags."""
        ttl = '<http://example.org/alice> <http://example.org/name> "Alice"@en .'
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].object == '"Alice"@en'
    
    def test_parse_literal_with_datatype(self):
        """Test parsing literals with datatypes."""
        ttl = '<http://example.org/alice> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .'
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        assert "integer" in doc.triples[0].object
    
    def test_parse_numeric_literals(self):
        """Test parsing numeric literals."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:item ex:count 42 .
        ex:item ex:price 19.99 .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 2
    
    def test_parse_boolean_literals(self):
        """Test parsing boolean literals."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:item ex:available true .
        ex:item ex:discontinued false .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 2
    
    def test_parse_blank_node(self):
        """Test parsing blank nodes."""
        ttl = """
        @prefix ex: <http://example.org/> .
        _:b1 ex:name "Anonymous" .
        ex:alice ex:knows _:b1 .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 2
        assert doc.triples[0].subject == "_:b1"
    
    def test_parse_blank_node_property_list(self):
        """Test parsing blank node property lists."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:alice ex:knows [ ex:name "Bob" ; ex:age 30 ] .
        """
        doc = parse_turtle(ttl)
        
        # Should create multiple triples
        assert len(doc.triples) >= 2
    
    def test_parse_predicate_object_list(self):
        """Test parsing predicate-object lists (semicolon)."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:alice ex:name "Alice" ;
                 ex:age 30 ;
                 ex:knows ex:bob .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 3
        subjects = [t.subject for t in doc.triples]
        assert all(s == "http://example.org/alice" for s in subjects)
    
    def test_parse_object_list(self):
        """Test parsing object lists (comma)."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:alice ex:knows ex:bob, ex:charlie, ex:david .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 3
        predicates = [t.predicate for t in doc.triples]
        assert all(p == "http://example.org/knows" for p in predicates)
    
    def test_parse_rdf_type_shorthand(self):
        """Test parsing 'a' as rdf:type shorthand."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:alice a ex:Person .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    
    def test_parse_collection(self):
        """Test parsing RDF collections."""
        ttl = """
        @prefix ex: <http://example.org/> .
        ex:list ex:items (ex:a ex:b ex:c) .
        """
        doc = parse_turtle(ttl)
        
        # Collections create multiple triples (rdf:first, rdf:rest)
        assert len(doc.triples) >= 4
    
    def test_parse_quoted_triple(self):
        """Test parsing RDF-Star quoted triples."""
        ttl = """
        @prefix ex: <http://example.org/> .
        << ex:alice ex:knows ex:bob >> ex:confidence "0.9" .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject_triple is not None
        assert t.subject_triple.subject == "http://example.org/alice"
        assert t.subject_triple.predicate == "http://example.org/knows"
        assert t.subject_triple.object == "http://example.org/bob"
    
    def test_parse_nested_quoted_triple(self):
        """Test parsing nested quoted triples."""
        ttl = """
        @prefix ex: <http://example.org/> .
        << << ex:alice ex:knows ex:bob >> ex:source ex:wikipedia >> ex:verified true .
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject_triple is not None
        assert t.subject_triple.subject_triple is not None
    
    def test_parse_comments(self):
        """Test that comments are ignored."""
        ttl = """
        # This is a comment
        @prefix ex: <http://example.org/> .
        ex:alice ex:knows ex:bob . # inline comment
        """
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
    
    def test_parse_multiline_string(self):
        """Test parsing multiline strings."""
        ttl = '''
        @prefix ex: <http://example.org/> .
        ex:alice ex:bio """This is a
        multiline
        biography.""" .
        '''
        doc = parse_turtle(ttl)
        
        assert len(doc.triples) == 1
        assert "multiline" in doc.triples[0].object


class TestTurtleSerializer:
    """Tests for Turtle serializer."""
    
    def test_serialize_simple_triple(self):
        """Test serializing a simple triple."""
        triples = [Triple("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")]
        ttl = serialize_turtle(triples)
        
        assert "<http://example.org/alice>" in ttl
        assert "<http://example.org/knows>" in ttl
        assert "<http://example.org/bob>" in ttl
    
    def test_serialize_with_prefix(self):
        """Test serializing with prefix compression."""
        triples = [Triple("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")]
        prefixes = {"ex": "http://example.org/"}
        ttl = serialize_turtle(triples, prefixes)
        
        assert "@prefix ex:" in ttl
        assert "ex:alice" in ttl
        assert "ex:knows" in ttl
        assert "ex:bob" in ttl
    
    def test_serialize_literal(self):
        """Test serializing literals."""
        triples = [Triple("http://example.org/alice", "http://example.org/name", '"Alice"@en')]
        ttl = serialize_turtle(triples)
        
        assert '"Alice"@en' in ttl
    
    def test_serialize_rdf_type(self):
        """Test serializing rdf:type as 'a'."""
        triples = [Triple("http://example.org/alice", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://example.org/Person")]
        prefixes = {"ex": "http://example.org/"}
        ttl = serialize_turtle(triples, prefixes)
        
        assert " a " in ttl
    
    def test_serialize_quoted_triple(self):
        """Test serializing RDF-Star quoted triples."""
        inner = Triple("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
        outer = Triple("", "http://example.org/confidence", '"0.9"', subject_triple=inner)
        ttl = serialize_turtle([outer])
        
        assert "<<" in ttl
        assert ">>" in ttl
    
    def test_roundtrip(self):
        """Test parsing then serializing produces equivalent triples."""
        original = """
        @prefix ex: <http://example.org/> .
        ex:alice ex:knows ex:bob .
        ex:alice ex:name "Alice" .
        """
        doc = parse_turtle(original)
        ttl = serialize_turtle(doc.triples, doc.prefixes)
        doc2 = parse_turtle(ttl)
        
        assert len(doc.triples) == len(doc2.triples)


class TestNTriplesParser:
    """Tests for N-Triples parser."""
    
    def test_parse_simple_triple(self):
        """Test parsing a simple N-Triple."""
        nt = '<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .'
        doc = parse_ntriples(nt)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject == "http://example.org/alice"
        assert t.predicate == "http://example.org/knows"
        assert t.object == "http://example.org/bob"
    
    def test_parse_literal(self):
        """Test parsing literal objects."""
        nt = '<http://example.org/alice> <http://example.org/name> "Alice" .'
        doc = parse_ntriples(nt)
        
        assert len(doc.triples) == 1
        assert doc.triples[0].object == '"Alice"'
    
    def test_parse_literal_with_language(self):
        """Test parsing literals with language tags."""
        nt = '<http://example.org/alice> <http://example.org/name> "Alice"@en .'
        doc = parse_ntriples(nt)
        
        assert doc.triples[0].object == '"Alice"@en'
    
    def test_parse_literal_with_datatype(self):
        """Test parsing literals with datatypes."""
        nt = '<http://example.org/alice> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .'
        doc = parse_ntriples(nt)
        
        assert "integer" in doc.triples[0].object
    
    def test_parse_blank_node(self):
        """Test parsing blank nodes."""
        nt = '_:b1 <http://example.org/name> "Test" .'
        doc = parse_ntriples(nt)
        
        assert doc.triples[0].subject == "_:b1"
    
    def test_parse_multiple_lines(self):
        """Test parsing multiple lines."""
        nt = """<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
<http://example.org/bob> <http://example.org/knows> <http://example.org/charlie> .
<http://example.org/charlie> <http://example.org/name> "Charlie" ."""
        doc = parse_ntriples(nt)
        
        assert len(doc.triples) == 3
    
    def test_parse_comments(self):
        """Test that comments are ignored."""
        nt = """# Comment line
<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
# Another comment"""
        doc = parse_ntriples(nt)
        
        assert len(doc.triples) == 1
    
    def test_parse_escape_sequences(self):
        """Test parsing escape sequences."""
        nt = '<http://example.org/alice> <http://example.org/bio> "Line1\\nLine2\\tTabbed" .'
        doc = parse_ntriples(nt)
        
        assert "\\n" in doc.triples[0].object or "\n" in doc.triples[0].object
    
    def test_parse_quoted_triple(self):
        """Test parsing RDF-Star quoted triples."""
        nt = '<< <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> >> <http://example.org/confidence> "0.9" .'
        doc = parse_ntriples(nt)
        
        assert len(doc.triples) == 1
        t = doc.triples[0]
        assert t.subject_triple is not None
        assert t.subject_triple.subject == "http://example.org/alice"


class TestNTriplesSerializer:
    """Tests for N-Triples serializer."""
    
    def test_serialize_simple_triple(self):
        """Test serializing a simple triple."""
        triples = [Triple("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")]
        nt = serialize_ntriples(triples)
        
        assert "<http://example.org/alice>" in nt
        assert "<http://example.org/knows>" in nt
        assert "<http://example.org/bob>" in nt
        assert nt.strip().endswith(".")
    
    def test_serialize_literal(self):
        """Test serializing literals."""
        triples = [Triple("http://example.org/alice", "http://example.org/name", '"Alice"')]
        nt = serialize_ntriples(triples)
        
        assert '"Alice"' in nt
    
    def test_serialize_blank_node(self):
        """Test serializing blank nodes."""
        triples = [Triple("_:b1", "http://example.org/name", '"Test"')]
        nt = serialize_ntriples(triples)
        
        assert "_:b1" in nt
    
    def test_serialize_quoted_triple(self):
        """Test serializing RDF-Star quoted triples."""
        inner = Triple("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")
        outer = Triple("", "http://example.org/confidence", '"0.9"', subject_triple=inner)
        nt = serialize_ntriples([outer])
        
        assert "<<" in nt
        assert ">>" in nt
    
    def test_roundtrip(self):
        """Test parsing then serializing produces equivalent triples."""
        original = """<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
<http://example.org/alice> <http://example.org/name> "Alice" ."""
        doc = parse_ntriples(original)
        nt = serialize_ntriples(doc.triples)
        doc2 = parse_ntriples(nt)
        
        assert len(doc.triples) == len(doc2.triples)


class TestFormatInterop:
    """Test interoperability between formats."""
    
    def test_turtle_to_ntriples(self):
        """Test converting Turtle to N-Triples."""
        ttl = """
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        
        ex:alice foaf:name "Alice" ;
                 foaf:knows ex:bob .
        """
        doc = parse_turtle(ttl)
        nt = serialize_ntriples(doc.triples)
        doc2 = parse_ntriples(nt)
        
        assert len(doc.triples) == len(doc2.triples)
    
    def test_ntriples_to_turtle(self):
        """Test converting N-Triples to Turtle."""
        nt = """<http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
<http://example.org/alice> <http://example.org/name> "Alice" ."""
        doc = parse_ntriples(nt)
        prefixes = {"ex": "http://example.org/"}
        ttl = serialize_turtle(doc.triples, prefixes)
        doc2 = parse_turtle(ttl)
        
        assert len(doc.triples) == len(doc2.triples)
