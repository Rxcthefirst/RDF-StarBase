"""
Tests for CONSTRUCT and DESCRIBE query support.
"""

import pytest
from rdf_starbase.sparql.parser import SPARQLStarParser


class TestConstructParsing:
    """Test parsing of CONSTRUCT queries."""
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def test_simple_construct(self, parser):
        """Test parsing a simple CONSTRUCT query."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT { ?s foaf:name ?name }
        WHERE { ?s foaf:firstName ?name }
        """
        result = parser.parse(query)
        
        from rdf_starbase.sparql.ast import ConstructQuery
        assert isinstance(result, ConstructQuery)
        assert len(result.template) == 1
        assert len(result.where.patterns) == 1
    
    def test_construct_multiple_triples(self, parser):
        """Test CONSTRUCT with multiple template triples."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT {
            ?s foaf:name ?name .
            ?s foaf:knows ?o .
        }
        WHERE { 
            ?s foaf:firstName ?name .
            ?s foaf:friendOf ?o .
        }
        """
        result = parser.parse(query)
        assert len(result.template) == 2


class TestDescribeParsing:
    """Test parsing of DESCRIBE queries."""
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def test_describe_iri(self, parser):
        """Test DESCRIBE with a concrete IRI."""
        query = """
        DESCRIBE <http://example.org/alice>
        """
        result = parser.parse(query)
        
        from rdf_starbase.sparql.ast import DescribeQuery, IRI
        assert isinstance(result, DescribeQuery)
        assert len(result.resources) == 1
        assert isinstance(result.resources[0], IRI)
    
    def test_describe_variable(self, parser):
        """Test DESCRIBE with a variable."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        DESCRIBE ?person
        WHERE { ?person foaf:name "Alice" }
        """
        result = parser.parse(query)
        
        from rdf_starbase.sparql.ast import DescribeQuery, Variable
        assert isinstance(result, DescribeQuery)
        assert len(result.resources) == 1
        assert isinstance(result.resources[0], Variable)
        assert result.where is not None


class TestConstructExecution:
    """Test execution of CONSTRUCT queries."""
    
    @pytest.fixture
    def executor(self):
        """Create an executor with test data."""
        from rdf_starbase.storage import TermDict, QtDict, FactStore, StorageExecutor
        from rdf_starbase.storage.facts import DEFAULT_GRAPH_ID
        
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Create entities
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        
        # Create predicates
        first_name = td.intern_iri("http://xmlns.com/foaf/0.1/firstName")
        last_name = td.intern_iri("http://xmlns.com/foaf/0.1/lastName")
        knows = td.intern_iri("http://xmlns.com/foaf/0.1/knows")
        
        # Create literals
        alice_fn = td.intern_literal("Alice")
        alice_ln = td.intern_literal("Smith")
        bob_fn = td.intern_literal("Bob")
        
        g = DEFAULT_GRAPH_ID
        
        fs.add_fact(alice, first_name, alice_fn, g=g)
        fs.add_fact(alice, last_name, alice_ln, g=g)
        fs.add_fact(alice, knows, bob, g=g)
        fs.add_fact(bob, first_name, bob_fn, g=g)
        
        return StorageExecutor(td, qd, fs)
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def test_construct_transforms_data(self, executor, parser):
        """Test CONSTRUCT transforms data according to template."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT { ?s foaf:name ?fname }
        WHERE { ?s foaf:firstName ?fname }
        """
        ast = parser.parse(query)
        triples = executor.execute(ast)
        
        # Should produce 2 triples (Alice and Bob's names)
        assert len(triples) == 2
        
        # Each triple should have foaf:name as predicate
        for s, p, o in triples:
            assert p == "http://xmlns.com/foaf/0.1/name"
    
    def test_construct_with_constants(self, executor, parser):
        """Test CONSTRUCT with constant terms in template."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        CONSTRUCT { ?s rdf:type foaf:Person }
        WHERE { ?s foaf:firstName ?name }
        """
        ast = parser.parse(query)
        triples = executor.execute(ast)
        
        # Should produce 2 triples (Alice and Bob are Persons)
        assert len(triples) == 2
        
        for s, p, o in triples:
            assert p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            assert o == "http://xmlns.com/foaf/0.1/Person"


class TestDescribeExecution:
    """Test execution of DESCRIBE queries."""
    
    @pytest.fixture
    def executor(self):
        """Create an executor with test data."""
        from rdf_starbase.storage import TermDict, QtDict, FactStore, StorageExecutor
        from rdf_starbase.storage.facts import DEFAULT_GRAPH_ID
        
        td = TermDict()
        qd = QtDict(td)
        fs = FactStore(td, qd)
        
        # Create entities
        alice = td.intern_iri("http://example.org/alice")
        bob = td.intern_iri("http://example.org/bob")
        
        # Create predicates
        name = td.intern_iri("http://xmlns.com/foaf/0.1/name")
        knows = td.intern_iri("http://xmlns.com/foaf/0.1/knows")
        
        # Create literals
        alice_name = td.intern_literal("Alice")
        bob_name = td.intern_literal("Bob")
        
        g = DEFAULT_GRAPH_ID
        
        fs.add_fact(alice, name, alice_name, g=g)
        fs.add_fact(alice, knows, bob, g=g)
        fs.add_fact(bob, name, bob_name, g=g)
        
        return StorageExecutor(td, qd, fs)
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def test_describe_returns_all_triples(self, executor, parser):
        """Test DESCRIBE returns all triples about a resource."""
        query = """
        DESCRIBE <http://example.org/alice>
        """
        ast = parser.parse(query)
        triples = executor.execute(ast)
        
        # Should return: alice name, alice knows bob, bob knows alice (reverse)
        assert len(triples) >= 2
        
        # Check alice appears in subjects
        subjects = [s for s, p, o in triples]
        assert "http://example.org/alice" in subjects
    
    def test_describe_with_where(self, executor, parser):
        """Test DESCRIBE with WHERE clause to bind variable."""
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        DESCRIBE ?person
        WHERE { ?person foaf:name "Alice" }
        """
        ast = parser.parse(query)
        triples = executor.execute(ast)
        
        # Should describe Alice
        subjects = [s for s, p, o in triples]
        assert "http://example.org/alice" in subjects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
