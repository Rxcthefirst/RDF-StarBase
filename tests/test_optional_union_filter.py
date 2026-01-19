"""
Tests for OPTIONAL, UNION, FILTER, and Aggregate execution in StorageExecutor.
"""

import pytest
import polars as pl

from rdf_starbase.storage import (
    TermDict,
    FactStore,
    StorageExecutor,
    Term,
    TermKind,
    DEFAULT_GRAPH_ID,
)
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.sparql.parser import SPARQLStarParser


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def executor():
    """Create a fresh executor with sample data."""
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    
    # Add sample data
    def add(s: str, p: str, o: str):
        s_term = Term(kind=TermKind.IRI, lex=s)
        p_term = Term(kind=TermKind.IRI, lex=p)
        # Detect if object is a literal (number or string with quotes)
        if o.isdigit():
            o_term = Term(kind=TermKind.LITERAL, lex=o)
        elif o.startswith('"') and o.endswith('"'):
            o_term = Term(kind=TermKind.LITERAL, lex=o[1:-1])
        else:
            o_term = Term(kind=TermKind.IRI, lex=o)
        
        s_id = term_dict.get_or_create(s_term)
        p_id = term_dict.get_or_create(p_term)
        o_id = term_dict.get_or_create(o_term)
        
        fact_store.add_facts_batch([(DEFAULT_GRAPH_ID, s_id, p_id, o_id)])
    
    # People
    add("http://ex/alice", "http://ex/name", '"Alice"')
    add("http://ex/alice", "http://ex/age", "30")
    add("http://ex/alice", "http://ex/email", '"alice@example.org"')
    add("http://ex/alice", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person")
    
    add("http://ex/bob", "http://ex/name", '"Bob"')
    add("http://ex/bob", "http://ex/age", "25")
    add("http://ex/bob", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person")
    
    add("http://ex/charlie", "http://ex/name", '"Charlie"')
    add("http://ex/charlie", "http://ex/age", "35")
    add("http://ex/charlie", "http://ex/phone", '"555-1234"')
    add("http://ex/charlie", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://ex/Person")
    
    # Relationships
    add("http://ex/alice", "http://ex/knows", "http://ex/bob")
    add("http://ex/bob", "http://ex/knows", "http://ex/charlie")
    
    # Items with prices
    add("http://ex/item1", "http://ex/price", "100")
    add("http://ex/item2", "http://ex/price", "200")
    add("http://ex/item3", "http://ex/price", "150")
    
    return StorageExecutor(term_dict, qt_dict, fact_store)


@pytest.fixture
def parser():
    return SPARQLStarParser()


# =============================================================================
# OPTIONAL Tests
# =============================================================================

class TestOptionalExecution:
    """Test OPTIONAL pattern execution."""
    
    def test_optional_adds_binding_when_present(self, executor, parser):
        """OPTIONAL should add binding when pattern matches."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?person ?email
        WHERE {
            ?person a ex:Person .
            OPTIONAL { ?person ex:email ?email }
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # All 3 people should be returned
        assert len(result) == 3
        
        # Check that email column exists
        assert "email" in result.columns
        
        # Alice has email, bob and charlie don't (charlie has phone)
        emails = result.filter(pl.col("email").is_not_null())
        assert len(emails) >= 1  # At least Alice
    
    def test_optional_preserves_rows_without_match(self, executor, parser):
        """OPTIONAL should keep rows even when pattern doesn't match."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?person ?phone
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:phone ?phone }
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # All 3 people should be returned
        assert len(result) == 3
        
        # Only Charlie has phone
        with_phone = result.filter(pl.col("phone").is_not_null())
        assert len(with_phone) == 1


# =============================================================================
# UNION Tests
# =============================================================================

class TestUnionExecution:
    """Test UNION pattern execution."""
    
    def test_union_combines_alternatives(self, executor, parser):
        """UNION should combine results from multiple patterns."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?contact
        WHERE {
            { ex:alice ex:email ?contact }
            UNION
            { ex:charlie ex:phone ?contact }
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # Should have alice's email and charlie's phone
        assert len(result) >= 2


# =============================================================================
# FILTER Tests
# =============================================================================

class TestFilterExecution:
    """Test FILTER expression execution."""
    
    def test_filter_numeric_comparison(self, executor, parser):
        """FILTER should work with numeric comparisons."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?item ?price
        WHERE {
            ?item ex:price ?price .
            FILTER(?price > 100)
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # item2 (200) and item3 (150) have price > 100
        assert len(result) == 2
    
    def test_filter_equality(self, executor, parser):
        """FILTER with equality check."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?item
        WHERE {
            ?item ex:price ?price .
            FILTER(?price = 200)
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # Only item2 has price 200
        assert len(result) == 1
    
    def test_filter_logical_and(self, executor, parser):
        """FILTER with AND logic."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?item
        WHERE {
            ?item ex:price ?price .
            FILTER(?price >= 100 && ?price <= 150)
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # item1 (100) and item3 (150) are in range
        assert len(result) == 2


# =============================================================================
# Aggregate Tests
# =============================================================================

class TestAggregateExecution:
    """Test aggregate function execution."""
    
    def test_count_all(self, executor, parser):
        """COUNT(*) should count all rows."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT (COUNT(*) AS ?count)
        WHERE {
            ?person a ex:Person .
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        assert len(result) == 1
        assert "count" in result.columns
        count_val = result["count"][0]
        assert count_val == 3  # 3 people
    
    def test_count_variable(self, executor, parser):
        """COUNT(?var) should count non-null values."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT (COUNT(?email) AS ?email_count)
        WHERE {
            ?person a ex:Person .
            OPTIONAL { ?person ex:email ?email }
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        assert len(result) == 1
        # Only alice has email
        assert result["email_count"][0] >= 1
    
    def test_sum(self, executor, parser):
        """SUM should sum numeric values."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT (SUM(?price) AS ?total)
        WHERE {
            ?item ex:price ?price .
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        assert len(result) == 1
        # 100 + 200 + 150 = 450
        assert result["total"][0] == 450.0
    
    def test_group_by(self, executor, parser):
        """GROUP BY should group results."""
        query = """
        PREFIX ex: <http://ex/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?type (COUNT(?person) AS ?count)
        WHERE {
            ?person rdf:type ?type .
        }
        GROUP BY ?type
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # All people have same type, so 1 group with count 3
        assert len(result) == 1
        assert result["count"][0] == 3


# =============================================================================
# Combined Tests
# =============================================================================

class TestCombinedPatterns:
    """Test combinations of patterns."""
    
    def test_optional_with_filter(self, executor, parser):
        """OPTIONAL combined with FILTER."""
        query = """
        PREFIX ex: <http://ex/>
        SELECT ?person ?age
        WHERE {
            ?person ex:name ?name .
            OPTIONAL { ?person ex:age ?age }
            FILTER(?age > 25)
        }
        """
        ast = parser.parse(query)
        result = executor.execute(ast)
        
        # Alice (30) and Charlie (35) have age > 25
        assert len(result) == 2
