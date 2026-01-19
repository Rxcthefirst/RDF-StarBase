"""
Tests for SPARQL MINUS pattern support.

MINUS performs set difference - it removes solutions where the
MINUS pattern matches.
"""

import pytest
from rdf_starbase.sparql.parser import SPARQLStarParser
from rdf_starbase.sparql.ast import MinusPattern


class TestMinusParsing:
    """Test parsing of MINUS patterns."""
    
    @pytest.fixture
    def parser(self):
        return SPARQLStarParser()
    
    def test_simple_minus(self, parser):
        """Test simple MINUS pattern."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { ?person ex:retired true }
        }
        """
        result = parser.parse(query)
        
        assert len(result.where.patterns) == 1
        assert len(result.where.minus_patterns) == 1
        assert isinstance(result.where.minus_patterns[0], MinusPattern)
    
    def test_minus_with_multiple_patterns(self, parser):
        """Test MINUS with multiple patterns inside."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { 
                ?person ex:status ex:Inactive .
                ?person ex:deleted true .
            }
        }
        """
        result = parser.parse(query)
        
        assert len(result.where.minus_patterns) == 1
        minus = result.where.minus_patterns[0]
        assert len(minus.patterns) == 2
    
    def test_multiple_minus_clauses(self, parser):
        """Test multiple MINUS clauses."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { ?person ex:retired true }
            MINUS { ?person ex:deceased true }
        }
        """
        result = parser.parse(query)
        
        assert len(result.where.minus_patterns) == 2
    
    def test_minus_with_filter(self, parser):
        """Test MINUS with FILTER inside."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            ?person ex:age ?age .
            MINUS { 
                ?person ex:age ?age .
                FILTER(?age < 18)
            }
        }
        """
        result = parser.parse(query)
        
        assert len(result.where.minus_patterns) == 1
        minus = result.where.minus_patterns[0]
        assert len(minus.filters) == 1
    
    def test_minus_combined_with_optional(self, parser):
        """Test MINUS combined with OPTIONAL."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?email WHERE {
            ?person ex:type ex:Person .
            OPTIONAL { ?person ex:email ?email }
            MINUS { ?person ex:private true }
        }
        """
        result = parser.parse(query)
        
        assert len(result.where.optional_patterns) == 1
        assert len(result.where.minus_patterns) == 1


class TestMinusExecution:
    """Test execution of MINUS queries."""
    
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
        carol = td.intern_iri("http://example.org/carol")
        
        # Create predicates and types
        type_pred = td.intern_iri("http://example.org/type")
        person = td.intern_iri("http://example.org/Person")
        retired = td.intern_iri("http://example.org/retired")
        deceased = td.intern_iri("http://example.org/deceased")
        true_val = td.intern_literal("true")
        
        g = DEFAULT_GRAPH_ID
        
        # People
        fs.add_fact(alice, type_pred, person, g=g)
        fs.add_fact(bob, type_pred, person, g=g)
        fs.add_fact(carol, type_pred, person, g=g)
        
        # Alice is retired
        fs.add_fact(alice, retired, true_val, g=g)
        
        # Bob is active (not retired)
        
        # Carol is retired and deceased
        fs.add_fact(carol, retired, true_val, g=g)
        fs.add_fact(carol, deceased, true_val, g=g)
        
        return StorageExecutor(td, qd, fs)
    
    @pytest.fixture
    def parser(self):
        from rdf_starbase.sparql.parser import SPARQLStarParser
        return SPARQLStarParser()
    
    def test_minus_excludes_matching(self, executor, parser):
        """Test that MINUS excludes matching solutions."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { ?person ex:retired "true" }
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        people = results["person"].to_list()
        
        # Only Bob should remain (not retired)
        assert len(people) == 1
        assert "http://example.org/bob" in people
    
    def test_multiple_minus(self, executor, parser):
        """Test multiple MINUS clauses."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { ?person ex:retired "true" }
            MINUS { ?person ex:deceased "true" }
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        people = results["person"].to_list()
        
        # Only Bob - Alice is retired, Carol is both
        assert len(people) == 1
        assert "http://example.org/bob" in people
    
    def test_minus_no_shared_vars(self, executor, parser):
        """Test MINUS with no shared variables has no effect."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:type ex:Person .
            MINUS { ?other ex:retired "true" }
        }
        """
        ast = parser.parse(query)
        results = executor.execute(ast)
        people = results["person"].to_list()
        
        # All three should remain - MINUS has no effect with no shared vars
        assert len(people) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
