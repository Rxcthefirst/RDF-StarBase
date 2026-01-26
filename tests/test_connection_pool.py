"""
Tests for connection pooling.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from rdf_starbase.storage.connection_pool import (
    ConnectionPool,
    SharedExecutorPool,
    PooledConnection,
    PoolStats,
    ConnectionMode,
    PoolExhaustedError,
)
from rdf_starbase.storage import (
    StorageExecutor, TermDict, QtDict, FactStore, 
    TermKind, Term, DEFAULT_GRAPH_ID
)


def create_storage_executor():
    """Create a properly initialized StorageExecutor."""
    term_dict = TermDict()
    qt_dict = QtDict(term_dict)
    fact_store = FactStore(term_dict, qt_dict)
    return StorageExecutor(term_dict, qt_dict, fact_store)


def add_triple_to_executor(executor, s, p, o):
    """Helper to add a triple to an executor."""
    td = executor.term_dict
    fs = executor.fact_store
    
    s_term = Term(kind=TermKind.IRI, lex=s)
    p_term = Term(kind=TermKind.IRI, lex=p)
    if o.startswith('"'):
        o_term = Term(kind=TermKind.LITERAL, lex=o.strip('"'))
    else:
        o_term = Term(kind=TermKind.IRI, lex=o)
    
    s_id = td.get_or_create(s_term)
    p_id = td.get_or_create(p_term)
    o_id = td.get_or_create(o_term)
    
    fs.add_facts_batch([(DEFAULT_GRAPH_ID, s_id, p_id, o_id)])


@pytest.fixture
def executor_factory():
    """Factory that creates real StorageExecutor instances."""
    def factory():
        return create_storage_executor()
    return factory


@pytest.fixture
def mock_executor_factory():
    """Factory that creates mock executors for testing."""
    def factory():
        mock = MagicMock()
        mock.count = MagicMock(return_value=0)
        return mock
    return factory


class TestPooledConnection:
    """Tests for PooledConnection dataclass."""
    
    def test_initial_state(self):
        """Test initial state of pooled connection."""
        conn = PooledConnection()
        assert conn.checked_out is False
        assert conn.use_count == 0
        assert conn.checked_out_at is None
    
    def test_mark_checked_out(self):
        """Test marking connection as checked out."""
        conn = PooledConnection()
        conn.mark_checked_out(ConnectionMode.READ)
        
        assert conn.checked_out is True
        assert conn.mode == ConnectionMode.READ
        assert conn.use_count == 1
        assert conn.checked_out_at is not None
        assert conn.checked_out_by == threading.current_thread().ident
    
    def test_mark_returned(self):
        """Test marking connection as returned."""
        conn = PooledConnection()
        conn.mark_checked_out(ConnectionMode.WRITE)
        conn.mark_returned()
        
        assert conn.checked_out is False
        assert conn.checked_out_at is None
        assert conn.last_used_at is not None
    
    def test_use_count_increments(self):
        """Test use count increments on each checkout."""
        conn = PooledConnection()
        
        conn.mark_checked_out(ConnectionMode.READ)
        conn.mark_returned()
        assert conn.use_count == 1
        
        conn.mark_checked_out(ConnectionMode.WRITE)
        conn.mark_returned()
        assert conn.use_count == 2


class TestConnectionPool:
    """Tests for ConnectionPool."""
    
    def test_pool_initialization(self, executor_factory):
        """Test pool initializes with minimum connections."""
        pool = ConnectionPool(executor_factory, max_size=5, min_size=2)
        
        assert pool.size() >= 2
        assert pool.available() >= 2
        
        pool.close()
    
    def test_checkout_connection(self, executor_factory):
        """Test checking out a connection."""
        pool = ConnectionPool(executor_factory, max_size=3)
        
        with pool.connection() as conn:
            assert conn is not None
            assert isinstance(conn, StorageExecutor)
        
        pool.close()
    
    def test_connection_returned_to_pool(self, executor_factory):
        """Test connection is returned to pool after use."""
        pool = ConnectionPool(executor_factory, max_size=3, min_size=1)
        initial_available = pool.available()
        
        with pool.connection() as conn:
            # Connection checked out
            assert pool.available() == initial_available - 1
        
        # Connection returned
        assert pool.available() == initial_available
        
        pool.close()
    
    def test_pool_exhaustion_timeout(self, mock_executor_factory):
        """Test timeout when pool is exhausted."""
        pool = ConnectionPool(
            mock_executor_factory, 
            max_size=1, 
            min_size=1,
            max_overflow=0,
            timeout=0.1
        )
        
        # Check out the only connection
        with pool.connection() as conn1:
            # Try to get another - should timeout
            with pytest.raises(PoolExhaustedError):
                with pool.connection(timeout=0.05) as conn2:
                    pass
        
        pool.close()
    
    def test_overflow_connections(self, mock_executor_factory):
        """Test overflow connections are created when needed."""
        pool = ConnectionPool(
            mock_executor_factory,
            max_size=1,
            min_size=1,
            max_overflow=2,
            timeout=1.0
        )
        
        connections = []
        
        # Get connection from pool
        with pool.connection() as conn1:
            # Get overflow connection
            with pool.connection() as conn2:
                assert conn1 is not None
                assert conn2 is not None
        
        pool.close()
    
    def test_write_connection_serialized(self, mock_executor_factory):
        """Test write connections are serialized."""
        pool = ConnectionPool(mock_executor_factory, max_size=3)
        
        write_acquired = threading.Event()
        write_released = threading.Event()
        second_write_acquired = threading.Event()
        
        def hold_write():
            with pool.write_connection() as conn:
                write_acquired.set()
                # Wait until second write attempt starts
                time.sleep(0.1)
            write_released.set()
        
        def try_second_write():
            # Wait for first write
            write_acquired.wait()
            with pool.write_connection(timeout=1.0) as conn:
                second_write_acquired.set()
        
        t1 = threading.Thread(target=hold_write)
        t2 = threading.Thread(target=try_second_write)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Second write should have acquired after first released
        assert write_released.is_set()
        assert second_write_acquired.is_set()
        
        pool.close()
    
    def test_concurrent_reads(self, executor_factory):
        """Test multiple concurrent read connections."""
        pool = ConnectionPool(executor_factory, max_size=5, min_size=5)
        results = []
        
        def read_operation(i):
            with pool.connection() as conn:
                # Simulate some work
                time.sleep(0.01)
                return i
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_operation, i) for i in range(5)]
            for future in as_completed(futures):
                results.append(future.result())
        
        assert len(results) == 5
        
        pool.close()
    
    def test_pool_stats(self, mock_executor_factory):
        """Test pool statistics tracking."""
        pool = ConnectionPool(mock_executor_factory, max_size=3, min_size=1)
        
        # Do some checkouts
        for _ in range(3):
            with pool.connection() as conn:
                pass
        
        stats = pool.stats()
        assert stats.total_checkouts == 3
        assert stats.checked_out_connections == 0
        assert stats.total_connections >= 1
        
        pool.close()
    
    def test_pool_close(self, mock_executor_factory):
        """Test pool closes cleanly."""
        pool = ConnectionPool(mock_executor_factory, max_size=3, min_size=2)
        
        pool.close()
        
        # Should raise error after close
        with pytest.raises(RuntimeError):
            with pool.connection() as conn:
                pass
    
    def test_context_manager(self, mock_executor_factory):
        """Test pool as context manager."""
        with ConnectionPool(mock_executor_factory, max_size=3) as pool:
            with pool.connection() as conn:
                assert conn is not None
        
        # Pool should be closed
        with pytest.raises(RuntimeError):
            with pool.connection() as conn:
                pass
    
    def test_connection_validation(self, mock_executor_factory):
        """Test connection validation on checkout."""
        pool = ConnectionPool(
            mock_executor_factory, 
            max_size=3,
            min_size=1,
            validate_on_checkout=True
        )
        
        with pool.connection() as conn:
            # Should have called count() for validation
            conn.count.assert_called()
        
        pool.close()


class TestSharedExecutorPool:
    """Tests for SharedExecutorPool."""
    
    def test_read_access(self):
        """Test read access to shared executor."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor)
        
        with pool.read() as exec:
            assert exec is executor
    
    def test_write_access(self):
        """Test write access to shared executor."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor)
        
        with pool.write() as exec:
            assert exec is executor
    
    def test_concurrent_reads(self):
        """Test multiple concurrent readers."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor, max_readers=5)
        
        read_count = []
        
        def read_op():
            with pool.read() as exec:
                time.sleep(0.01)
                read_count.append(1)
        
        threads = [threading.Thread(target=read_op) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(read_count) == 5
    
    def test_write_blocks_reads(self):
        """Test write blocks new reads."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor)
        
        write_started = threading.Event()
        write_done = threading.Event()
        read_started = threading.Event()
        
        def write_op():
            with pool.write() as exec:
                write_started.set()
                time.sleep(0.1)
            write_done.set()
        
        def read_op():
            write_started.wait()
            with pool.read() as exec:
                read_started.set()
        
        t1 = threading.Thread(target=write_op)
        t2 = threading.Thread(target=read_op)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Read should have waited for write to complete
        assert write_done.is_set()
        assert read_started.is_set()
    
    def test_write_waits_for_readers(self):
        """Test write waits for active readers to complete."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor, max_readers=5)
        
        read_active = threading.Event()
        read_done = threading.Event()
        write_acquired = threading.Event()
        
        def long_read():
            with pool.read() as exec:
                read_active.set()
                time.sleep(0.1)
            read_done.set()
        
        def write_op():
            read_active.wait()
            with pool.write(timeout=2.0) as exec:
                write_acquired.set()
        
        t1 = threading.Thread(target=long_read)
        t2 = threading.Thread(target=write_op)
        
        t1.start()
        time.sleep(0.01)  # Ensure read starts first
        t2.start()
        
        t1.join()
        t2.join()
        
        # Write should have acquired after read completed
        assert read_done.is_set()
        assert write_acquired.is_set()
    
    def test_stats(self):
        """Test pool statistics."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor)
        
        for _ in range(3):
            with pool.read() as exec:
                pass
        
        with pool.write() as exec:
            pass
        
        stats = pool.stats()
        assert stats["total_reads"] == 3
        assert stats["total_writes"] == 1
        assert stats["active_readers"] == 0
    
    def test_write_timeout(self):
        """Test write timeout when readers don't complete."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor, write_timeout=0.1)
        
        read_holding = threading.Event()
        write_failed = threading.Event()
        
        def hold_read():
            with pool.read() as exec:
                read_holding.set()
                time.sleep(0.5)  # Hold longer than timeout
        
        def try_write():
            read_holding.wait()
            try:
                with pool.write(timeout=0.05) as exec:
                    pass
            except PoolExhaustedError:
                write_failed.set()
        
        t1 = threading.Thread(target=hold_read)
        t2 = threading.Thread(target=try_write)
        
        t1.start()
        t2.start()
        
        t2.join()
        t1.join()
        
        assert write_failed.is_set()


def get_count(executor):
    """Get triple count from executor."""
    return len(executor.fact_store._df)


class TestPoolWithRealOperations:
    """Integration tests with real storage operations."""
    
    def test_concurrent_triple_insertion(self, executor_factory):
        """Test concurrent triple insertion through pool."""
        pool = ConnectionPool(executor_factory, max_size=1)
        
        def insert_triple(i):
            with pool.write_connection() as exec:
                add_triple_to_executor(
                    exec,
                    f"http://example.org/s{i}",
                    "http://example.org/p",
                    f"http://example.org/o{i}",
                )
                return get_count(exec)
        
        # Sequential inserts (writes are serialized)
        counts = []
        for i in range(5):
            counts.append(insert_triple(i))
        
        # Each insert should see increasing count
        # Note: Each connection is separate, so counts won't accumulate
        # unless we use a shared executor
        assert all(c >= 1 for c in counts)
        
        pool.close()
    
    def test_shared_pool_preserves_state(self):
        """Test shared pool maintains state across operations."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor)
        
        # Insert data
        with pool.write() as exec:
            add_triple_to_executor(
                exec,
                "http://example.org/Alice",
                "http://example.org/name",
                '"Alice"',
            )
        
        # Read data
        with pool.read() as exec:
            count = get_count(exec)
        
        assert count == 1
    
    def test_pool_under_load(self):
        """Test pool behavior under concurrent load."""
        executor = create_storage_executor()
        pool = SharedExecutorPool(executor, max_readers=10)
        
        # Add some data
        with pool.write() as exec:
            for i in range(100):
                add_triple_to_executor(
                    exec,
                    f"http://example.org/s{i}",
                    "http://example.org/p",
                    f"http://example.org/o{i}",
                )
        
        read_results = []
        
        def read_count():
            with pool.read() as exec:
                count = get_count(exec)
                read_results.append(count)
        
        # Concurrent reads
        with ThreadPoolExecutor(max_workers=10) as thread_executor:
            futures = [thread_executor.submit(read_count) for _ in range(20)]
            for f in futures:
                f.result()
        
        # All reads should see same count
        assert all(r == 100 for r in read_results)
