"""
DuckDB SQL Interface for RDF-StarBase.

Provides a SQL interface to the columnar RDF storage, enabling:
- Direct SQL queries on triple data
- Analytical workloads (aggregations, window functions)
- Join with external data sources
- Export to various formats

DuckDB reads Parquet files directly without copying data into memory,
making it ideal for analytical workloads on large datasets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, TYPE_CHECKING
import json

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

import polars as pl

if TYPE_CHECKING:
    from rdf_starbase.store import TripleStore


@dataclass
class SQLQueryResult:
    """Result of a SQL query execution."""
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    execution_time_ms: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "warnings": self.warnings,
        }
    
    def to_polars(self) -> pl.DataFrame:
        """Convert result to a Polars DataFrame."""
        if not self.rows:
            return pl.DataFrame({col: [] for col in self.columns})
        return pl.DataFrame(self.rows, schema=self.columns, orient="row")


@dataclass
class TableInfo:
    """Information about a registered table."""
    name: str
    columns: List[str]
    row_count: int
    description: str = ""


class DuckDBInterface:
    """
    DuckDB SQL interface for RDF-StarBase storage.
    
    Provides SQL access to the columnar RDF data with automatic
    registration of triple store tables.
    
    Tables available:
    - triples: Main triple data (subject, predicate, object, graph)
    - terms: Term dictionary (term_id, kind, lex)
    - facts: Raw integer-encoded facts
    - provenance: Provenance metadata
    
    Example:
        interface = DuckDBInterface(store)
        result = interface.execute('''
            SELECT subject, predicate, object 
            FROM triples 
            WHERE predicate LIKE '%type%'
            LIMIT 10
        ''')
    """
    
    def __init__(self, store: "TripleStore", read_only: bool = True):
        """
        Initialize the DuckDB interface.
        
        Args:
            store: The TripleStore to expose via SQL
            read_only: If True, only read operations are allowed
            
        Raises:
            ImportError: If duckdb is not installed
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError(
                "DuckDB is required for SQL interface. "
                "Install with: pip install duckdb"
            )
        
        self._store = store
        self._read_only = read_only
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._tables_registered = False
    
    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure DuckDB connection is established."""
        if self._conn is None:
            # Create in-memory database
            self._conn = duckdb.connect(":memory:", read_only=False)
            # Register the Polars DataFrames as views
            self._register_tables()
        return self._conn
    
    def _register_tables(self) -> None:
        """Register TripleStore DataFrames as DuckDB tables."""
        if self._tables_registered:
            return
        
        conn = self._conn
        if conn is None:
            return
        
        # Register the string-based triple view
        triples_df = self._store._df
        conn.register("triples", triples_df.to_arrow())
        
        # Register the raw facts table (integer-encoded)
        facts_df = self._store._fact_store._df
        conn.register("facts", facts_df.to_arrow())
        
        # Register the term dictionary
        term_rows = [
            {
                "term_id": tid,
                "kind": term.kind.value,
                "kind_name": term.kind.name,
                "lex": term.lex,
            }
            for tid, term in self._store._term_dict._id_to_term.items()
        ]
        if term_rows:
            terms_df = pl.DataFrame(term_rows)
            conn.register("terms", terms_df.to_arrow())
        else:
            # Empty terms table
            terms_df = pl.DataFrame({
                "term_id": pl.Series([], dtype=pl.UInt64),
                "kind": pl.Series([], dtype=pl.UInt8),
                "kind_name": pl.Series([], dtype=pl.Utf8),
                "lex": pl.Series([], dtype=pl.Utf8),
            })
            conn.register("terms", terms_df.to_arrow())
        
        # Create a provenance view (filtered facts with source info)
        conn.execute("""
            CREATE VIEW provenance AS
            SELECT 
                assertion_id,
                subject,
                predicate,
                object,
                source,
                confidence,
                process,
                timestamp
            FROM triples
            WHERE source IS NOT NULL OR confidence < 1.0
        """)
        
        # Create a named_graphs view
        conn.execute("""
            CREATE VIEW named_graphs AS
            SELECT DISTINCT graph
            FROM triples
            WHERE graph IS NOT NULL
        """)
        
        # Create an rdf_type view for easy class queries
        conn.execute("""
            CREATE VIEW rdf_types AS
            SELECT 
                subject,
                object AS type,
                graph
            FROM triples
            WHERE predicate LIKE '%type%'
        """)
        
        self._tables_registered = True
    
    def refresh_tables(self) -> None:
        """
        Refresh table registrations after store modifications.
        
        Call this after adding/removing triples to update the SQL views.
        """
        if self._conn is not None:
            # Drop existing views and tables
            self._conn.execute("DROP VIEW IF EXISTS provenance")
            self._conn.execute("DROP VIEW IF EXISTS named_graphs")
            self._conn.execute("DROP VIEW IF EXISTS rdf_types")
            
            # Unregister tables
            try:
                self._conn.unregister("triples")
                self._conn.unregister("facts")
                self._conn.unregister("terms")
            except Exception:
                pass  # Tables might not exist
            
            self._tables_registered = False
            self._register_tables()
    
    def execute(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> SQLQueryResult:
        """
        Execute a SQL query against the triple store.
        
        Args:
            sql: SQL query string
            params: Optional parameter dict for prepared statements
            limit: Optional row limit (added as LIMIT clause if not present)
            
        Returns:
            SQLQueryResult with columns, rows, and metadata
            
        Raises:
            ValueError: If read_only mode and query is not SELECT
            duckdb.Error: If SQL is invalid
        """
        import time
        
        conn = self._ensure_connection()
        
        # Check read-only mode
        sql_upper = sql.strip().upper()
        if self._read_only and not sql_upper.startswith("SELECT"):
            if not (sql_upper.startswith("SHOW") or 
                    sql_upper.startswith("DESCRIBE") or
                    sql_upper.startswith("EXPLAIN")):
                raise ValueError(
                    "Only SELECT/SHOW/DESCRIBE/EXPLAIN queries allowed in read-only mode"
                )
        
        # Add limit if not present
        if limit is not None and "LIMIT" not in sql_upper:
            sql = f"{sql.rstrip().rstrip(';')} LIMIT {limit}"
        
        warnings = []
        start = time.perf_counter()
        
        try:
            if params:
                result = conn.execute(sql, params)
            else:
                result = conn.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in result.description] if result.description else []
            
            # Fetch all rows
            rows = result.fetchall()
            
            # Convert to JSON-serializable types
            rows = [
                [self._serialize_value(v) for v in row]
                for row in rows
            ]
            
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise type(e)(f"SQL Error: {e}") from e
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return SQLQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=round(elapsed, 3),
            warnings=warnings,
        )
    
    def _serialize_value(self, value: Any) -> Any:
        """Convert DuckDB values to JSON-serializable types."""
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return str(value)
    
    def list_tables(self) -> List[TableInfo]:
        """List all available tables and views."""
        self._ensure_connection()
        
        result = self.execute("SHOW TABLES")
        
        tables = []
        for row in result.rows:
            table_name = row[0]
            
            # Get column info
            col_result = self.execute(f"DESCRIBE {table_name}")
            columns = [r[0] for r in col_result.rows]
            
            # Get row count
            try:
                count_result = self.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = count_result.rows[0][0] if count_result.rows else 0
            except Exception:
                row_count = 0
            
            tables.append(TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
            ))
        
        return tables
    
    def get_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get the schema (column types) for a table.
        
        Args:
            table_name: Name of the table/view
            
        Returns:
            Dict mapping column name to type
        """
        self._ensure_connection()
        result = self.execute(f"DESCRIBE {table_name}")
        return {row[0]: row[1] for row in result.rows}
    
    def sample(
        self, 
        table_name: str, 
        n: int = 10,
        columns: Optional[List[str]] = None,
    ) -> SQLQueryResult:
        """
        Get a sample of rows from a table.
        
        Args:
            table_name: Table to sample from
            n: Number of rows
            columns: Optional list of columns (default: all)
            
        Returns:
            SQLQueryResult with sample rows
        """
        col_spec = ", ".join(columns) if columns else "*"
        return self.execute(f"SELECT {col_spec} FROM {table_name} LIMIT {n}")
    
    def aggregate(
        self,
        group_by: str,
        aggregations: Dict[str, str],
        table: str = "triples",
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> SQLQueryResult:
        """
        Run an aggregation query.
        
        Args:
            group_by: Column(s) to group by
            aggregations: Dict of output_name -> aggregation expression
                Example: {"count": "COUNT(*)", "subjects": "COUNT(DISTINCT subject)"}
            table: Table to query (default: triples)
            where: Optional WHERE clause
            order_by: Optional ORDER BY clause
            limit: Optional LIMIT
            
        Returns:
            SQLQueryResult with aggregation results
        """
        agg_parts = [f"{expr} AS {name}" for name, expr in aggregations.items()]
        agg_str = ", ".join(agg_parts)
        
        sql = f"SELECT {group_by}, {agg_str} FROM {table}"
        
        if where:
            sql += f" WHERE {where}"
        
        sql += f" GROUP BY {group_by}"
        
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        if limit:
            sql += f" LIMIT {limit}"
        
        return self.execute(sql)
    
    def export_parquet(self, sql: str, output_path: str | Path) -> None:
        """
        Export query results to Parquet file.
        
        Args:
            sql: SELECT query
            output_path: Path for output Parquet file
        """
        conn = self._ensure_connection()
        conn.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET)")
    
    def export_csv(self, sql: str, output_path: str | Path) -> None:
        """
        Export query results to CSV file.
        
        Args:
            sql: SELECT query
            output_path: Path for output CSV file
        """
        conn = self._ensure_connection()
        conn.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT CSV, HEADER)")
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._tables_registered = False
    
    def __enter__(self) -> "DuckDBInterface":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def create_sql_interface(
    store: "TripleStore",
    read_only: bool = True,
) -> DuckDBInterface:
    """
    Create a DuckDB SQL interface for a TripleStore.
    
    Args:
        store: The TripleStore to expose via SQL
        read_only: If True, only read operations allowed
        
    Returns:
        DuckDBInterface instance
        
    Raises:
        ImportError: If duckdb is not installed
    """
    return DuckDBInterface(store, read_only=read_only)


def check_duckdb_available() -> bool:
    """Check if DuckDB is available for import."""
    return DUCKDB_AVAILABLE
