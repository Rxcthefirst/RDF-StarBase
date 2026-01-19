"""
Tests for rdflib compatibility layer.

These tests verify that RDF-StarBase provides a drop-in replacement for rdflib.
"""

import pytest
from io import StringIO
from pathlib import Path
import tempfile

from rdf_starbase.compat.rdflib import (
    Graph, URIRef, Literal, BNode, Namespace,
    RDF, RDFS, OWL, XSD, FOAF
)


class TestURIRef:
    """Test URIRef class."""
    
    def test_create_uriref(self):
        uri = URIRef("http://example.org/test")
        assert str(uri) == "http://example.org/test"
    
    def test_uriref_is_string(self):
        uri = URIRef("http://example.org/test")
        assert isinstance(uri, str)
    
    def test_uriref_equality(self):
        uri1 = URIRef("http://example.org/test")
        uri2 = URIRef("http://example.org/test")
        assert uri1 == uri2
    
    def test_uriref_hash(self):
        uri1 = URIRef("http://example.org/test")
        uri2 = URIRef("http://example.org/test")
        assert hash(uri1) == hash(uri2)
        assert {uri1, uri2} == {uri1}
    
    def test_uriref_n3(self):
        uri = URIRef("http://example.org/test")
        assert uri.n3() == "<http://example.org/test>"
    
    def test_uriref_with_base(self):
        uri = URIRef("local", base="http://example.org/")
        assert str(uri) == "http://example.org/local"


class TestLiteral:
    """Test Literal class."""
    
    def test_create_string_literal(self):
        lit = Literal("hello")
        assert str(lit) == "hello"
        assert lit.datatype == XSD.string
    
    def test_create_integer_literal(self):
        lit = Literal(42)
        assert lit.toPython() == 42
        assert lit.datatype == XSD.integer
    
    def test_create_float_literal(self):
        lit = Literal(3.14)
        assert lit.toPython() == 3.14
        assert lit.datatype == XSD.double
    
    def test_create_boolean_literal(self):
        lit = Literal(True)
        assert lit.toPython() == True
        assert lit.datatype == XSD.boolean
    
    def test_create_lang_literal(self):
        lit = Literal("hello", lang="en")
        assert lit.language == "en"
        assert lit.datatype is None
    
    def test_literal_n3(self):
        lit = Literal("hello")
        assert lit.n3() == '"hello"'
        
        lit_lang = Literal("bonjour", lang="fr")
        assert lit_lang.n3() == '"bonjour"@fr'
        
        lit_int = Literal(42)
        assert '"42"' in lit_int.n3()
    
    def test_literal_equality(self):
        lit1 = Literal("hello")
        lit2 = Literal("hello")
        assert lit1 == lit2
    
    def test_lang_literal_no_datatype(self):
        """Cannot have both lang and datatype."""
        with pytest.raises(TypeError):
            Literal("hello", lang="en", datatype=XSD.string)


class TestBNode:
    """Test BNode class."""
    
    def test_create_bnode(self):
        node = BNode()
        assert str(node).startswith("N")
    
    def test_create_bnode_with_id(self):
        node = BNode("mynode")
        assert str(node) == "mynode"
    
    def test_bnode_n3(self):
        node = BNode("test")
        assert node.n3() == "_:test"
    
    def test_bnode_equality(self):
        node1 = BNode("same")
        node2 = BNode("same")
        assert node1 == node2
        
        node3 = BNode("different")
        assert node1 != node3


class TestNamespace:
    """Test Namespace class."""
    
    def test_namespace_attribute(self):
        EX = Namespace("http://example.org/")
        assert str(EX.person) == "http://example.org/person"
    
    def test_namespace_getitem(self):
        EX = Namespace("http://example.org/")
        assert str(EX["person"]) == "http://example.org/person"
    
    def test_well_known_namespaces(self):
        assert str(RDF.type) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        assert str(RDFS.label) == "http://www.w3.org/2000/01/rdf-schema#label"
        assert str(FOAF.name) == "http://xmlns.com/foaf/0.1/name"
        assert str(XSD.string) == "http://www.w3.org/2001/XMLSchema#string"


class TestGraphBasics:
    """Test Graph basic operations."""
    
    def test_create_empty_graph(self):
        g = Graph()
        assert len(g) == 0
    
    def test_add_triple(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        p = RDF.type
        o = FOAF.Person
        
        g.add((s, p, o))
        assert len(g) == 1
    
    def test_triple_in_graph(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        p = RDF.type
        o = FOAF.Person
        
        g.add((s, p, o))
        assert (s, p, o) in g
    
    def test_iterate_triples(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        
        g.add((s, RDF.type, FOAF.Person))
        g.add((s, FOAF.name, Literal("Alice")))
        
        count = 0
        for triple in g:
            count += 1
        assert count == 2
    
    def test_triples_pattern(self):
        g = Graph()
        s1 = URIRef("http://example.org/person/1")
        s2 = URIRef("http://example.org/person/2")
        
        g.add((s1, RDF.type, FOAF.Person))
        g.add((s2, RDF.type, FOAF.Person))
        g.add((s1, FOAF.name, Literal("Alice")))
        
        # Match all with type
        matches = list(g.triples((None, RDF.type, None)))
        assert len(matches) == 2
        
        # Match specific subject
        matches = list(g.triples((s1, None, None)))
        assert len(matches) == 2


class TestGraphQueries:
    """Test Graph query helpers."""
    
    def test_subjects(self):
        g = Graph()
        s1 = URIRef("http://example.org/person/1")
        s2 = URIRef("http://example.org/person/2")
        
        g.add((s1, RDF.type, FOAF.Person))
        g.add((s2, RDF.type, FOAF.Person))
        
        subjects = list(g.subjects(RDF.type, FOAF.Person))
        assert len(subjects) == 2
    
    def test_objects(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        
        g.add((s, FOAF.name, Literal("Alice")))
        g.add((s, FOAF.name, Literal("Ali")))
        
        names = list(g.objects(s, FOAF.name))
        assert len(names) == 2
    
    def test_value(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        
        g.add((s, FOAF.name, Literal("Alice")))
        
        name = g.value(s, FOAF.name)
        assert str(name) == "Alice"
    
    def test_predicate_objects(self):
        g = Graph()
        s = URIRef("http://example.org/person/1")
        
        g.add((s, RDF.type, FOAF.Person))
        g.add((s, FOAF.name, Literal("Alice")))
        
        po = list(g.predicate_objects(s))
        assert len(po) == 2


class TestGraphParsing:
    """Test Graph parsing."""
    
    def test_parse_turtle_string(self):
        ttl = """
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix ex: <http://example.org/> .
        
        ex:alice a foaf:Person ;
            foaf:name "Alice" .
        """
        
        g = Graph()
        g.parse(data=ttl, format="turtle")
        
        assert len(g) >= 2
    
    def test_parse_ntriples_string(self):
        nt = """
        <http://example.org/alice> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
        <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        """
        
        g = Graph()
        g.parse(data=nt, format="nt")
        
        assert len(g) == 2
    
    def test_parse_from_file(self):
        ttl = """
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        <http://example.org/bob> a foaf:Person ;
            foaf:name "Bob" .
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
            f.write(ttl)
            path = f.name
        
        try:
            g = Graph()
            g.parse(path)
            assert len(g) >= 2
        finally:
            Path(path).unlink()


class TestGraphSerialization:
    """Test Graph serialization."""
    
    def test_serialize_turtle(self):
        g = Graph()
        s = URIRef("http://example.org/alice")
        
        g.add((s, RDF.type, FOAF.Person))
        g.add((s, FOAF.name, Literal("Alice")))
        
        ttl = g.serialize(format="turtle")
        assert "alice" in ttl
        assert "Alice" in ttl
    
    def test_serialize_ntriples(self):
        g = Graph()
        s = URIRef("http://example.org/alice")
        
        g.add((s, RDF.type, FOAF.Person))
        
        nt = g.serialize(format="nt")
        assert "<http://example.org/alice>" in nt
        assert "." in nt
    
    def test_serialize_to_file(self):
        g = Graph()
        s = URIRef("http://example.org/alice")
        g.add((s, FOAF.name, Literal("Alice")))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
            path = f.name
        
        try:
            g.serialize(destination=path, format="turtle")
            content = Path(path).read_text()
            assert "Alice" in content
        finally:
            Path(path).unlink()


class TestGraphSPARQL:
    """Test Graph SPARQL queries."""
    
    def test_sparql_select(self):
        g = Graph()
        s = URIRef("http://example.org/alice")
        
        g.add((s, RDF.type, FOAF.Person))
        g.add((s, FOAF.name, Literal("Alice")))
        
        results = g.query("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name WHERE {
                ?s foaf:name ?name
            }
        """)
        
        names = [row['name'] for row in results]
        assert len(names) == 1
        assert str(names[0]) == "Alice"
    
    def test_sparql_with_initns(self):
        g = Graph()
        s = URIRef("http://example.org/alice")
        g.add((s, FOAF.name, Literal("Alice")))
        
        results = g.query(
            "SELECT ?name WHERE { ?s foaf:name ?name }",
            initNs={"foaf": FOAF}
        )
        
        names = list(results)
        assert len(names) == 1


class TestNamespaceManager:
    """Test NamespaceManager."""
    
    def test_bind_namespace(self):
        g = Graph()
        EX = Namespace("http://example.org/")
        g.bind("ex", EX)
        
        namespaces = dict(g.namespaces())
        assert "ex" in namespaces
        assert str(namespaces["ex"]) == "http://example.org/"
    
    def test_default_namespaces(self):
        g = Graph()
        namespaces = dict(g.namespaces())
        
        assert "rdf" in namespaces
        assert "rdfs" in namespaces
        assert "xsd" in namespaces
        assert "owl" in namespaces


class TestRdflibCompatibility:
    """
    Tests to verify drop-in compatibility with rdflib usage patterns.
    """
    
    def test_common_pattern_add_and_query(self):
        """Common pattern: add triples and query."""
        g = Graph()
        
        EX = Namespace("http://example.org/")
        
        g.add((EX.alice, RDF.type, FOAF.Person))
        g.add((EX.alice, FOAF.name, Literal("Alice")))
        g.add((EX.alice, FOAF.age, Literal(30)))
        g.add((EX.alice, FOAF.knows, EX.bob))
        
        g.add((EX.bob, RDF.type, FOAF.Person))
        g.add((EX.bob, FOAF.name, Literal("Bob")))
        
        # Query pattern
        for person in g.subjects(RDF.type, FOAF.Person):
            name = g.value(person, FOAF.name)
            print(f"{person} is named {name}")
        
        assert len(g) == 6
    
    def test_common_pattern_parse_query_serialize(self):
        """Common pattern: parse, query, serialize."""
        ttl = """
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix ex: <http://example.org/> .
        
        ex:alice a foaf:Person ;
            foaf:name "Alice" ;
            foaf:knows ex:bob .
        
        ex:bob a foaf:Person ;
            foaf:name "Bob" .
        """
        
        g = Graph()
        g.parse(data=ttl, format="turtle")
        
        # Add inferred data
        for s, o in g.subject_objects(FOAF.knows):
            g.add((o, FOAF.knows, s))  # Symmetric
        
        output = g.serialize(format="turtle")
        assert "alice" in output
        assert "bob" in output
