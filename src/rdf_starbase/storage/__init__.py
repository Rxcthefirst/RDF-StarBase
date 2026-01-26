"""
RDF-StarBase Storage Layer.

High-performance columnar storage with dictionary-encoded terms
and predicate-partitioned facts.
"""

from rdf_starbase.storage.terms import (
    TermKind,
    TermId,
    TermDict,
    Term,
)
from rdf_starbase.storage.quoted_triples import QtDict, QtId
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID
from rdf_starbase.storage.lsm import LSMStorage, PartitionStats
from rdf_starbase.storage.executor import StorageExecutor, ExpansionPatterns
from rdf_starbase.storage.reasoner import RDFSReasoner, ReasoningStats
from rdf_starbase.storage.persistence import (
    StoragePersistence,
    save_storage,
    load_storage,
)
from rdf_starbase.storage.duckdb import (
    DuckDBInterface,
    SQLQueryResult,
    TableInfo,
    create_sql_interface,
    check_duckdb_available,
)
from rdf_starbase.storage.wal import (
    WriteAheadLog,
    WALEntry,
    WALEntryType,
    InsertPayload,
    DeletePayload,
    CheckpointState,
)
from rdf_starbase.storage.transactions import (
    Transaction,
    TransactionManager,
    TransactionState,
    IsolationLevel,
)
from rdf_starbase.storage.query_context import (
    QueryContext,
    QueryState,
    QueryStats,
    CancellationToken,
    QueryCancelledException,
    QueryTimeoutException,
    ExplainPlan,
    QueryExecutionManager,
    execute_with_timeout,
    query_timeout,
)
from rdf_starbase.storage.connection_pool import (
    ConnectionPool,
    SharedExecutorPool,
    PooledConnection,
    PoolStats,
    ConnectionMode,
    PoolExhaustedError,
)
from rdf_starbase.storage.partitioning import (
    PredicatePartitioner,
    PartitionInfo,
    PartitionStats as PredicatePartitionStats,
)
from rdf_starbase.storage.indexing import (
    SortedIndex,
    IndexManager,
    IndexStats,
    IndexEntry,
    indexed_filter,
    estimate_selectivity,
    should_use_index,
)
from rdf_starbase.storage.memory_budget import (
    MemoryBudget,
    MemoryGuard,
    MemoryStats,
    MemoryPressure,
    BudgetedStore,
    estimate_size,
    configure_budget,
    get_budget,
    track,
    untrack,
    enforce,
)

__all__ = [
    "TermKind",
    "TermId",
    "TermDict",
    "Term",
    "QtDict",
    "QtId",
    "FactStore",
    "FactFlags",
    "DEFAULT_GRAPH_ID",
    "LSMStorage",
    "PartitionStats",
    "StorageExecutor",
    "ExpansionPatterns",
    "RDFSReasoner",
    "ReasoningStats",
    "StoragePersistence",
    "save_storage",
    "load_storage",
    # DuckDB SQL Interface
    "DuckDBInterface",
    "SQLQueryResult",
    "TableInfo",
    "create_sql_interface",
    "check_duckdb_available",
    # Write-Ahead Log
    "WriteAheadLog",
    "WALEntry",
    "WALEntryType",
    "InsertPayload",
    "DeletePayload",
    "CheckpointState",
    # Transactions
    "Transaction",
    "TransactionManager",
    "TransactionState",
    "IsolationLevel",
    # Query Context & Timeout
    "QueryContext",
    "QueryState",
    "QueryStats",
    "CancellationToken",
    "QueryCancelledException",
    "QueryTimeoutException",
    "ExplainPlan",
    "QueryExecutionManager",
    "execute_with_timeout",
    "query_timeout",
    # Connection Pooling
    "ConnectionPool",
    "SharedExecutorPool",
    "PooledConnection",
    "PoolStats",
    "ConnectionMode",
    "PoolExhaustedError",
    # Predicate Partitioning
    "PredicatePartitioner",
    "PartitionInfo",
    "PredicatePartitionStats",
    # Indexing
    "SortedIndex",
    "IndexManager",
    "IndexStats",
    "IndexEntry",
    "indexed_filter",
    "estimate_selectivity",
    "should_use_index",
    # Memory Budget
    "MemoryBudget",
    "MemoryGuard",
    "MemoryStats",
    "MemoryPressure",
    "BudgetedStore",
    "estimate_size",
    "configure_budget",
    "get_budget",
    "track",
    "untrack",
    "enforce",
]
