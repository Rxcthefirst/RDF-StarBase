"""
Tests for SPARQL Property Path support.

Property paths allow navigation through RDF graphs with path expressions:
- foaf:knows+ (one or more)
- foaf:knows* (zero or more)
- foaf:knows? (zero or one)
- ^foaf:knows (inverse)
- foaf:knows/foaf:name (sequence)
- foaf:knows|foaf:friend (alternative)
- !(foaf:hates) (negated property set)
"""

import pytest
from rdf_starbase.sparql.parser import SPARQLStarParser
from rdf_starbase.sparql.ast import (
    TriplePattern, PathMod, PathSequence, PathAlternative,
    PathInverse, PathNegatedPropertySet, PathIRI, PathFixedLength,
    PropertyPathModifier, IRI
)


class TestPropertyPathParsing:
    """Test parsing of property path expressions."""
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def get_first_predicate(self, parser, query: str):
        """Parse query and return the predicate of the first pattern."""
        full_query = f"PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n{query}"
        result = parser.parse(full_query)
        return result.where.patterns[0].predicate
    
    def test_simple_iri_not_path(self, parser):
        """Plain IRI predicates should NOT be property paths."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows ?o }")
        assert isinstance(pred, IRI)
        # Prefix may or may not be expanded depending on parser implementation
        assert "knows" in pred.value
    
    def test_one_or_more(self, parser):
        """Test foaf:knows+ (one or more)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows+ ?o }")
        assert isinstance(pred, PathMod)
        assert pred.modifier == PropertyPathModifier.ONE_OR_MORE
        assert isinstance(pred.path, PathIRI)
    
    def test_zero_or_more(self, parser):
        """Test foaf:knows* (zero or more)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows* ?o }")
        assert isinstance(pred, PathMod)
        assert pred.modifier == PropertyPathModifier.ZERO_OR_MORE
    
    def test_zero_or_one(self, parser):
        """Test foaf:knows? (zero or one)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows? ?o }")
        assert isinstance(pred, PathMod)
        assert pred.modifier == PropertyPathModifier.ZERO_OR_ONE
    
    def test_inverse(self, parser):
        """Test ^foaf:knows (inverse path)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s ^foaf:knows ?o }")
        assert isinstance(pred, PathInverse)
        assert isinstance(pred.path, PathIRI)
    
    def test_sequence(self, parser):
        """Test foaf:knows/foaf:name (sequence path)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows/foaf:name ?o }")
        assert isinstance(pred, PathSequence)
        assert len(pred.paths) == 2
        assert all(isinstance(p, PathIRI) for p in pred.paths)
    
    def test_alternative(self, parser):
        """Test foaf:knows|foaf:friend (alternative path)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows|foaf:friend ?o }")
        assert isinstance(pred, PathAlternative)
        assert len(pred.paths) == 2
    
    def test_complex_path(self, parser):
        """Test complex path: foaf:knows+/foaf:name."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows+/foaf:name ?o }")
        assert isinstance(pred, PathSequence)
        assert len(pred.paths) == 2
        assert isinstance(pred.paths[0], PathMod)
        assert pred.paths[0].modifier == PropertyPathModifier.ONE_OR_MORE
    
    @pytest.mark.skip(reason="Grouped path with modifier not yet supported")
    def test_grouped_path(self, parser):
        """Test grouped path: (foaf:knows|foaf:friend)+."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s (foaf:knows|foaf:friend)+ ?o }")
        # This should be a PathMod wrapping a PathAlternative
        # Note: Current grammar may not support this fully
        # This is an aspirational test
    
    def test_negated_property_set(self, parser):
        """Test negated property set: !(foaf:hates|foaf:dislikes)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s !(foaf:hates|foaf:dislikes) ?o }")
        assert isinstance(pred, PathNegatedPropertySet)
        assert len(pred.iris) == 2
    
    def test_fixed_length_exact(self, parser):
        """Test fixed-length path: foaf:knows{2} (exactly 2 hops)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows{2} ?o }")
        assert isinstance(pred, PathFixedLength)
        assert pred.min_length == 2
        assert pred.max_length == 2
    
    def test_fixed_length_range(self, parser):
        """Test fixed-length path: foaf:knows{2,4} (2 to 4 hops)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows{2,4} ?o }")
        assert isinstance(pred, PathFixedLength)
        assert pred.min_length == 2
        assert pred.max_length == 4
    
    def test_fixed_length_unbounded(self, parser):
        """Test fixed-length path: foaf:knows{2,} (2 or more hops)."""
        pred = self.get_first_predicate(parser, "SELECT * WHERE { ?s foaf:knows{2,} ?o }")
        assert isinstance(pred, PathFixedLength)
        assert pred.min_length == 2
        assert pred.max_length is None  # Unbounded
    
    def test_multiple_paths_in_query(self, parser):
        """Test query with multiple path patterns."""
        query = """
        SELECT ?name WHERE {
            ?person foaf:knows+ ?friend .
            ?friend foaf:name ?name .
        }
        """
        full_query = f"PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n{query}"
        result = parser.parse(full_query)
        
        assert len(result.where.patterns) == 2
        # First pattern has a path
        assert isinstance(result.where.patterns[0].predicate, PathMod)
        # Second pattern is a simple IRI
        assert isinstance(result.where.patterns[1].predicate, IRI)
    
    def test_path_with_full_iri(self, parser):
        """Test property path with full IRI syntax."""
        query = "SELECT * WHERE { ?s <http://example.org/knows>+ ?o }"
        result = parser.parse(query)
        pred = result.where.patterns[0].predicate
        assert isinstance(pred, PathMod)


class TestPropertyPathExecution:
    """Test execution of property path queries."""
    
    @pytest.fixture
    def executor(self):
        """Create an executor with test data for path queries."""
        from rdf_starbase.storage import TermDict, QtDict, FactStore, StorageExecutor
        from rdf_starbase.storage.facts import DEFAULT_GRAPH_ID
        
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Create entities
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        carol = td.intern_iri("http://example.org/carol")
        dave = td.intern_iri("http://example.org/dave")
        
        # Create predicates
        knows = td.intern_iri("http://xmlns.com/foaf/0.1/knows")
        name = td.intern_iri("http://xmlns.com/foaf/0.1/name")
        likes = td.intern_iri("http://example.org/likes")
        
        # Create names
        alice_name = td.intern_literal("Alice")
        bob_name = td.intern_literal("Bob")
        carol_name = td.intern_literal("Carol")
        dave_name = td.intern_literal("Dave")
        
        g = DEFAULT_GRAPH_ID
        
        # Create a social network graph: Alice -> Bob -> Carol -> Dave
        fs.add_fact(alice, knows, bob, g=g)
        fs.add_fact(bob, knows, carol, g=g)
        fs.add_fact(carol, knows, dave, g=g)
        
        # Names
        fs.add_fact(alice, name, alice_name, g=g)
        fs.add_fact(bob, name, bob_name, g=g)
        fs.add_fact(carol, name, carol_name, g=g)
        fs.add_fact(dave, name, dave_name, g=g)
        
        # Alice likes Dave directly
        fs.add_fact(alice, likes, dave, g=g)
        
        return StorageExecutor(td, qd, fs)
    
    @pytest.fixture
    def parser(self):
        from rdf_starbase.sparql.parser import SPARQLStarParser
        return SPARQLStarParser()
    
    def test_one_or_more_execution(self, executor, parser):
        """Test executing foaf:knows+ query (transitive closure)."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person WHERE {
            <http://example.org/alice> foaf:knows+ ?person .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Alice knows+ should give: Bob, Carol, Dave (transitive closure)
        people = results["person"].to_list()
        assert len(people) == 3
        assert "http://example.org/bob" in people
        assert "http://example.org/carol" in people
        assert "http://example.org/dave" in people
    
    def test_zero_or_more_execution(self, executor, parser):
        """Test executing foaf:knows* query (includes self)."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person WHERE {
            <http://example.org/alice> foaf:knows* ?person .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Alice knows* should include Alice herself plus Bob, Carol, Dave
        people = results["person"].to_list()
        assert "http://example.org/alice" in people
        assert "http://example.org/bob" in people
        assert len(people) >= 4
    
    def test_sequence_execution(self, executor, parser):
        """Test executing foaf:knows/foaf:name query."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name WHERE {
            <http://example.org/alice> foaf:knows/foaf:name ?name .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Alice knows Bob, Bob's name is "Bob"
        names = results["name"].to_list()
        assert "Bob" in names
    
    def test_inverse_execution(self, executor, parser):
        """Test executing ^foaf:knows query."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person WHERE {
            <http://example.org/bob> ^foaf:knows ?person .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Who knows Bob? -> Alice
        people = results["person"].to_list()
        assert "http://example.org/alice" in people
    
    def test_alternative_execution(self, executor, parser):
        """Test executing foaf:knows|ex:likes query."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/>
        SELECT ?obj WHERE {
            <http://example.org/alice> foaf:knows|ex:likes ?obj .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Alice knows Bob OR likes Dave
        objects = results["obj"].to_list()
        assert "http://example.org/bob" in objects
        assert "http://example.org/dave" in objects
    
    def test_double_sequence(self, executor, parser):
        """Test executing foaf:knows/foaf:knows (two hops)."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person WHERE {
            <http://example.org/alice> foaf:knows/foaf:knows ?person .
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        
        # Alice -> Bob -> Carol
        people = results["person"].to_list()
        assert "http://example.org/carol" in people


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
