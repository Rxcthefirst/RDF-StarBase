"""
Tests for DuckDB SQL Interface.

Tests the SQL interface for querying the columnar RDF store.
"""

import pytest
from rdf_starbase import TripleStore
from rdf_starbase.storage.duckdb import (
    DuckDBInterface,
    SQLQueryResult,
    create_sql_interface,
    get_cached_interface,
    check_duckdb_available,
)


@pytest.fixture
def sample_store():
    """Create a store with sample data."""
    store = TripleStore()
    
    # Add some test triples
    EX = "http://example.org/"
    
    triples = [
        {"subject": f"{EX}alice", "predicate": f"{EX}name", "object": "Alice", "source": "test"},
        {"subject": f"{EX}alice", "predicate": f"{EX}age", "object": "30", "source": "test"},
        {"subject": f"{EX}alice", "predicate": f"{EX}worksAt", "object": f"{EX}acme", "source": "test"},
        {"subject": f"{EX}bob", "predicate": f"{EX}name", "object": "Bob", "source": "test"},
        {"subject": f"{EX}bob", "predicate": f"{EX}age", "object": "25", "source": "test"},
        {"subject": f"{EX}bob", "predicate": f"{EX}worksAt", "object": f"{EX}acme", "source": "test"},
        {"subject": f"{EX}acme", "predicate": f"{EX}name", "object": "Acme Corp", "source": "test"},
        {"subject": f"{EX}acme", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": f"{EX}Company", "source": "test"},
    ]
    
    store.add_triples_batch(triples)
    
    return store


class TestDuckDBAvailability:
    """Test DuckDB availability check."""
    
    def test_duckdb_available(self):
        """DuckDB should be available after installation."""
        assert check_duckdb_available() is True


class TestDuckDBInterface:
    """Test DuckDB interface functionality."""
    
    def test_create_interface(self, sample_store):
        """Should create interface from store."""
        interface = create_sql_interface(sample_store)
        assert isinstance(interface, DuckDBInterface)
        interface.close()
    
    def test_context_manager(self, sample_store):
        """Should work as context manager."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT 1 as x")
            assert result.rows == [[1]]
    
    def test_select_all_triples(self, sample_store):
        """Should query all triples."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT * FROM triples LIMIT 10")
            
            assert result.row_count > 0
            assert "subject" in result.columns
            assert "predicate" in result.columns
            assert "object" in result.columns
    
    def test_count_triples(self, sample_store):
        """Should count triples."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT COUNT(*) as cnt FROM triples")
            
            assert result.row_count == 1
            assert result.rows[0][0] == 8  # 8 triples
    
    def test_filter_by_predicate(self, sample_store):
        """Should filter by predicate."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute(
                "SELECT subject, object FROM triples WHERE predicate LIKE '%name%'"
            )
            
            # Should get alice, bob, and acme names
            assert result.row_count == 3
    
    def test_group_by_predicate(self, sample_store):
        """Should group by predicate."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("""
                SELECT predicate, COUNT(*) as count 
                FROM triples 
                GROUP BY predicate 
                ORDER BY count DESC
            """)
            
            assert result.row_count > 0
            # name should appear 3 times (alice, bob, acme)
            for row in result.rows:
                if "name" in row[0]:
                    assert row[1] == 3
    
    def test_distinct_subjects(self, sample_store):
        """Should get distinct subjects."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute(
                "SELECT DISTINCT subject FROM triples"
            )
            
            # alice, bob, acme
            assert result.row_count == 3
    
    def test_rdf_types_view(self, sample_store):
        """Should query rdf_types view."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT * FROM rdf_types")
            
            # Should have 1 type (acme -> Company)
            assert result.row_count == 1
    
    def test_terms_table(self, sample_store):
        """Should query terms dictionary."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute(
                "SELECT * FROM terms WHERE kind_name = 'IRI' LIMIT 5"
            )
            
            assert result.row_count > 0
            assert "term_id" in result.columns
            assert "lex" in result.columns
    
    def test_list_tables(self, sample_store):
        """Should list available tables."""
        with create_sql_interface(sample_store) as sql:
            tables = sql.list_tables()
            
            table_names = [t.name for t in tables]
            assert "triples" in table_names
            assert "terms" in table_names
            assert "facts" in table_names
    
    def test_get_schema(self, sample_store):
        """Should get table schema."""
        with create_sql_interface(sample_store) as sql:
            schema = sql.get_schema("triples")
            
            assert "subject" in schema
            assert "predicate" in schema
            assert "object" in schema
    
    def test_sample(self, sample_store):
        """Should sample rows from table."""
        with create_sql_interface(sample_store) as sql:
            result = sql.sample("triples", n=3)
            
            assert result.row_count <= 3
    
    def test_aggregate_helper(self, sample_store):
        """Should use aggregate helper method."""
        with create_sql_interface(sample_store) as sql:
            result = sql.aggregate(
                group_by="predicate",
                aggregations={
                    "count": "COUNT(*)",
                    "subjects": "COUNT(DISTINCT subject)",
                },
                order_by="count DESC",
                limit=5,
            )
            
            assert "count" in result.columns
            assert "subjects" in result.columns
    
    def test_read_only_mode(self, sample_store):
        """Should prevent write operations in read-only mode."""
        with create_sql_interface(sample_store, read_only=True) as sql:
            with pytest.raises(ValueError):
                sql.execute("DROP TABLE triples")
    
    def test_limit_parameter(self, sample_store):
        """Should respect limit parameter."""
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT * FROM triples", limit=2)
            
            assert result.row_count == 2
    
    def test_to_polars(self, sample_store):
        """Should convert result to Polars DataFrame."""
        import polars as pl
        
        with create_sql_interface(sample_store) as sql:
            result = sql.execute("SELECT subject, predicate, object FROM triples LIMIT 5")
            df = result.to_polars()
            
            assert isinstance(df, pl.DataFrame)
            assert len(df) == result.row_count


class TestSQLQueryResult:
    """Test SQLQueryResult functionality."""
    
    def test_to_dict(self):
        """Should convert to dict."""
        result = SQLQueryResult(
            columns=["a", "b"],
            rows=[[1, 2], [3, 4]],
            row_count=2,
            execution_time_ms=1.5,
        )
        
        d = result.to_dict()
        assert d["columns"] == ["a", "b"]
        assert d["rows"] == [[1, 2], [3, 4]]
        assert d["row_count"] == 2
    
    def test_empty_result_to_polars(self):
        """Should handle empty result."""
        result = SQLQueryResult(
            columns=["x", "y"],
            rows=[],
            row_count=0,
            execution_time_ms=0.5,
        )
        
        df = result.to_polars()
        assert len(df) == 0
        assert list(df.columns) == ["x", "y"]


class TestCachedInterface:
    """Test cached interface functionality for performance optimization."""
    
    def test_get_cached_interface(self, sample_store):
        """Should return cached interface for same store."""
        sql1 = get_cached_interface(sample_store)
        sql2 = get_cached_interface(sample_store)
        
        # Same interface instance should be returned
        assert sql1 is sql2
        
        # Both should work correctly
        result = sql1.execute("SELECT COUNT(*) as cnt FROM facts")
        assert result.rows[0][0] == 8  # 8 facts added
    
    def test_cached_interface_connection_reuse(self, sample_store):
        """Should reuse DuckDB connection across queries."""
        sql = get_cached_interface(sample_store)
        
        # First query
        result1 = sql.execute("SELECT COUNT(*) FROM facts")
        conn1 = sql._conn
        
        # Second query should use same connection
        result2 = sql.execute("SELECT COUNT(*) FROM terms")
        conn2 = sql._conn
        
        assert conn1 is conn2
    
    def test_facts_table_no_materialization(self, sample_store):
        """Queries on 'facts' table should not trigger string materialization."""
        sql = get_cached_interface(sample_store)
        
        # Query facts (integer-based) without triggering triples registration
        result = sql.execute("SELECT COUNT(*) FROM facts")
        
        # Triples table should NOT be registered yet (lazy loading)
        assert not sql._triples_registered
        
        # But facts should be registered
        assert sql._facts_registered
    
    def test_triples_table_lazy_registration(self, sample_store):
        """Triples table should be lazily registered on first use."""
        sql = get_cached_interface(sample_store)
        
        # Initially, triples should not be registered
        # (note: fresh interface needed)
        from rdf_starbase.storage.duckdb import DuckDBInterface
        fresh_sql = DuckDBInterface(sample_store)
        fresh_sql._ensure_connection()
        
        # After just connecting, triples should not be registered
        assert not fresh_sql._triples_registered
        
        # Query triples table
        fresh_sql.execute("SELECT * FROM triples LIMIT 1")
        
        # Now triples should be registered
        assert fresh_sql._triples_registered
        
        fresh_sql.close()
    
    def test_query_needs_triples_detection(self, sample_store):
        """Should correctly detect queries that need triples table."""
        sql = get_cached_interface(sample_store)
        
        # Queries that don't need triples
        assert not sql._query_needs_triples_table("SELECT * FROM facts")
        assert not sql._query_needs_triples_table("SELECT * FROM terms")
        assert not sql._query_needs_triples_table("SELECT COUNT(*) FROM facts WHERE s > 0")
        
        # Queries that need triples
        assert sql._query_needs_triples_table("SELECT * FROM triples")
        assert sql._query_needs_triples_table("SELECT subject FROM TRIPLES")
        assert sql._query_needs_triples_table("SELECT * FROM provenance")
        assert sql._query_needs_triples_table("SELECT * FROM named_graphs")
        assert sql._query_needs_triples_table("SELECT * FROM rdf_types")
