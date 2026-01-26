"""
Connection pooling for thread-safe concurrent access to RDF-StarBase storage.

Provides connection management with:
- Configurable pool size
- Checkout/checkin semantics
- Timeout on pool exhaustion
- Read/write connection separation
- Context manager support
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from rdf_starbase.storage.executor import StorageExecutor


class ConnectionMode(Enum):
    """Connection access mode."""
    READ = auto()
    WRITE = auto()
    READ_WRITE = auto()


class PoolExhaustedError(Exception):
    """Raised when pool is exhausted and timeout expires."""
    pass


class ConnectionInUseError(Exception):
    """Raised when attempting to use a connection that's already checked out."""
    pass


class ConnectionNotCheckedOutError(Exception):
    """Raised when attempting to return a connection not checked out."""
    pass


@dataclass
class PooledConnection:
    """A connection wrapper that tracks usage state."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    executor: Optional["StorageExecutor"] = None
    mode: ConnectionMode = ConnectionMode.READ
    checked_out: bool = False
    checked_out_at: Optional[float] = None
    checked_out_by: Optional[int] = None  # Thread ID
    use_count: int = 0
    last_used_at: Optional[float] = None
    
    def mark_checked_out(self, mode: ConnectionMode) -> None:
        """Mark connection as checked out."""
        self.checked_out = True
        self.checked_out_at = time.time()
        self.checked_out_by = threading.current_thread().ident
        self.mode = mode
        self.use_count += 1
    
    def mark_returned(self) -> None:
        """Mark connection as returned to pool."""
        self.checked_out = False
        self.last_used_at = time.time()
        self.checked_out_at = None
        self.checked_out_by = None


@dataclass
class PoolStats:
    """Statistics for connection pool monitoring."""
    
    total_connections: int = 0
    available_connections: int = 0
    checked_out_connections: int = 0
    total_checkouts: int = 0
    total_timeouts: int = 0
    total_waits: int = 0
    avg_wait_time_ms: float = 0.0
    avg_checkout_time_ms: float = 0.0
    max_concurrent_checkouts: int = 0


class ConnectionPool:
    """
    Thread-safe connection pool for StorageExecutor instances.
    
    Features:
    - Fixed pool size with configurable limits
    - Timeout on pool exhaustion
    - Separate read and write pools (optional)
    - Connection health checking
    - Pool statistics
    
    Example:
        pool = ConnectionPool(executor_factory, max_size=10)
        
        with pool.connection() as conn:
            result = conn.triples(subject="http://example.org/Alice")
        
        # Or for writes
        with pool.write_connection() as conn:
            conn.add_triple(s, p, o, g)
    """
    
    def __init__(
        self,
        executor_factory: Callable[[], "StorageExecutor"],
        max_size: int = 10,
        min_size: int = 1,
        max_overflow: int = 5,
        timeout: float = 30.0,
        recycle_time: Optional[float] = 3600.0,  # 1 hour
        validate_on_checkout: bool = False,
        separate_write_pool: bool = False,
        max_write_connections: int = 1,
    ):
        """
        Initialize connection pool.
        
        Args:
            executor_factory: Callable that creates new StorageExecutor instances
            max_size: Maximum number of connections in pool
            min_size: Minimum connections to maintain
            max_overflow: Additional connections allowed beyond max_size
            timeout: Seconds to wait for available connection
            recycle_time: Seconds before connection is recycled (None = never)
            validate_on_checkout: Run validation query on checkout
            separate_write_pool: Use separate pool for write connections
            max_write_connections: Max connections for writes (if separate pool)
        """
        self._executor_factory = executor_factory
        self._max_size = max_size
        self._min_size = min_size
        self._max_overflow = max_overflow
        self._timeout = timeout
        self._recycle_time = recycle_time
        self._validate_on_checkout = validate_on_checkout
        self._separate_write_pool = separate_write_pool
        self._max_write_connections = max_write_connections
        
        # Read pool
        self._read_pool: Queue[PooledConnection] = Queue(maxsize=max_size)
        self._read_overflow: int = 0
        self._read_lock = threading.RLock()
        
        # Write pool (if separate)
        self._write_pool: Queue[PooledConnection] = Queue(maxsize=max_write_connections)
        self._write_lock = threading.Lock()
        self._current_writer: Optional[PooledConnection] = None
        
        # Tracking
        self._all_connections: dict[str, PooledConnection] = {}
        self._stats = PoolStats()
        self._stats_lock = threading.Lock()
        self._closed = False
        
        # Wait time tracking
        self._total_wait_time: float = 0.0
        self._total_checkout_time: float = 0.0
        
        # Initialize minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Pre-create minimum connections."""
        for _ in range(self._min_size):
            conn = self._create_connection()
            self._read_pool.put(conn)
            self._stats.total_connections += 1
            self._stats.available_connections += 1
    
    def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        executor = self._executor_factory()
        conn = PooledConnection(executor=executor)
        self._all_connections[conn.id] = conn
        return conn
    
    def _validate_connection(self, conn: PooledConnection) -> bool:
        """Validate connection is still usable."""
        if conn.executor is None:
            return False
        
        # Check recycle time
        if self._recycle_time and conn.last_used_at:
            age = time.time() - conn.last_used_at
            if age > self._recycle_time:
                return False
        
        # Optionally run validation query
        if self._validate_on_checkout:
            try:
                # Simple existence check
                _ = conn.executor.count()
                return True
            except Exception:
                return False
        
        return True
    
    def _recycle_connection(self, conn: PooledConnection) -> PooledConnection:
        """Recycle a connection by creating a new one."""
        # Remove old connection
        if conn.id in self._all_connections:
            del self._all_connections[conn.id]
        
        # Create new one
        return self._create_connection()
    
    @contextmanager
    def connection(self, timeout: Optional[float] = None):
        """
        Get a read connection from the pool.
        
        Args:
            timeout: Override default timeout
            
        Yields:
            StorageExecutor instance
            
        Raises:
            PoolExhaustedError: If no connection available within timeout
        """
        conn = self._checkout(ConnectionMode.READ, timeout)
        try:
            yield conn.executor
        finally:
            self._checkin(conn)
    
    @contextmanager
    def write_connection(self, timeout: Optional[float] = None):
        """
        Get a write connection from the pool.
        
        Uses serialized writer pattern - only one write connection at a time.
        
        Args:
            timeout: Override default timeout
            
        Yields:
            StorageExecutor instance
            
        Raises:
            PoolExhaustedError: If no connection available within timeout
        """
        conn = self._checkout(ConnectionMode.WRITE, timeout)
        try:
            yield conn.executor
        finally:
            self._checkin(conn)
    
    def _checkout(
        self, 
        mode: ConnectionMode, 
        timeout: Optional[float] = None
    ) -> PooledConnection:
        """Check out a connection from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")
        
        effective_timeout = timeout if timeout is not None else self._timeout
        start_time = time.time()
        
        # For writes, use write lock for serialization
        if mode == ConnectionMode.WRITE:
            return self._checkout_write(effective_timeout, start_time)
        
        # For reads, try to get from pool
        return self._checkout_read(effective_timeout, start_time)
    
    def _checkout_read(self, timeout: float, start_time: float) -> PooledConnection:
        """Check out a read connection."""
        with self._stats_lock:
            self._stats.total_waits += 1
        
        # Try to get existing connection
        try:
            conn = self._read_pool.get(timeout=timeout)
            wait_time = time.time() - start_time
            
            # Validate and possibly recycle
            if not self._validate_connection(conn):
                conn = self._recycle_connection(conn)
            
            conn.mark_checked_out(ConnectionMode.READ)
            
            with self._stats_lock:
                self._stats.available_connections -= 1
                self._stats.checked_out_connections += 1
                self._stats.total_checkouts += 1
                self._total_wait_time += wait_time * 1000
                self._stats.avg_wait_time_ms = (
                    self._total_wait_time / self._stats.total_checkouts
                )
                self._stats.max_concurrent_checkouts = max(
                    self._stats.max_concurrent_checkouts,
                    self._stats.checked_out_connections
                )
            
            return conn
            
        except Empty:
            # Pool exhausted, try overflow
            with self._read_lock:
                if self._read_overflow < self._max_overflow:
                    self._read_overflow += 1
                    conn = self._create_connection()
                    conn.mark_checked_out(ConnectionMode.READ)
                    
                    with self._stats_lock:
                        self._stats.total_connections += 1
                        self._stats.checked_out_connections += 1
                        self._stats.total_checkouts += 1
                    
                    return conn
            
            with self._stats_lock:
                self._stats.total_timeouts += 1
            
            raise PoolExhaustedError(
                f"No connection available after {timeout}s timeout"
            )
    
    def _checkout_write(self, timeout: float, start_time: float) -> PooledConnection:
        """Check out a write connection (serialized)."""
        acquired = self._write_lock.acquire(timeout=timeout)
        
        if not acquired:
            with self._stats_lock:
                self._stats.total_timeouts += 1
            raise PoolExhaustedError(
                f"Write connection not available after {timeout}s timeout"
            )
        
        try:
            # Get or create write connection
            if self._separate_write_pool:
                try:
                    conn = self._write_pool.get_nowait()
                except Empty:
                    conn = self._create_connection()
            else:
                # Use read pool for writes too
                try:
                    conn = self._read_pool.get_nowait()
                    with self._stats_lock:
                        self._stats.available_connections -= 1
                except Empty:
                    conn = self._create_connection()
                    with self._stats_lock:
                        self._stats.total_connections += 1
            
            if not self._validate_connection(conn):
                conn = self._recycle_connection(conn)
            
            conn.mark_checked_out(ConnectionMode.WRITE)
            self._current_writer = conn
            
            with self._stats_lock:
                self._stats.checked_out_connections += 1
                self._stats.total_checkouts += 1
                wait_time = time.time() - start_time
                self._total_wait_time += wait_time * 1000
                self._stats.avg_wait_time_ms = (
                    self._total_wait_time / self._stats.total_checkouts
                )
            
            return conn
            
        except Exception:
            self._write_lock.release()
            raise
    
    def _checkin(self, conn: PooledConnection) -> None:
        """Return a connection to the pool."""
        checkout_duration = 0.0
        if conn.checked_out_at:
            checkout_duration = (time.time() - conn.checked_out_at) * 1000
        
        was_write = conn.mode == ConnectionMode.WRITE
        conn.mark_returned()
        
        if was_write:
            self._current_writer = None
            if self._separate_write_pool:
                self._write_pool.put(conn)
            else:
                self._read_pool.put(conn)
                with self._stats_lock:
                    self._stats.available_connections += 1
            self._write_lock.release()
        else:
            # Return overflow connections vs regular
            with self._read_lock:
                if self._read_overflow > 0 and self._read_pool.full():
                    self._read_overflow -= 1
                    # Don't return to pool, let it be garbage collected
                    if conn.id in self._all_connections:
                        del self._all_connections[conn.id]
                    with self._stats_lock:
                        self._stats.total_connections -= 1
                else:
                    self._read_pool.put(conn)
                    with self._stats_lock:
                        self._stats.available_connections += 1
        
        with self._stats_lock:
            self._stats.checked_out_connections -= 1
            self._total_checkout_time += checkout_duration
            if self._stats.total_checkouts > 0:
                self._stats.avg_checkout_time_ms = (
                    self._total_checkout_time / self._stats.total_checkouts
                )
    
    def stats(self) -> PoolStats:
        """Get current pool statistics."""
        with self._stats_lock:
            return PoolStats(
                total_connections=self._stats.total_connections,
                available_connections=self._stats.available_connections,
                checked_out_connections=self._stats.checked_out_connections,
                total_checkouts=self._stats.total_checkouts,
                total_timeouts=self._stats.total_timeouts,
                total_waits=self._stats.total_waits,
                avg_wait_time_ms=self._stats.avg_wait_time_ms,
                avg_checkout_time_ms=self._stats.avg_checkout_time_ms,
                max_concurrent_checkouts=self._stats.max_concurrent_checkouts,
            )
    
    def size(self) -> int:
        """Get current pool size (total connections)."""
        with self._stats_lock:
            return self._stats.total_connections
    
    def available(self) -> int:
        """Get number of available connections."""
        with self._stats_lock:
            return self._stats.available_connections
    
    def close(self) -> None:
        """Close all connections and shut down the pool."""
        self._closed = True
        
        # Drain read pool
        while not self._read_pool.empty():
            try:
                conn = self._read_pool.get_nowait()
                # Executor cleanup if needed
            except Empty:
                break
        
        # Drain write pool
        while not self._write_pool.empty():
            try:
                conn = self._write_pool.get_nowait()
            except Empty:
                break
        
        self._all_connections.clear()
        
        with self._stats_lock:
            self._stats.total_connections = 0
            self._stats.available_connections = 0
    
    def __enter__(self) -> "ConnectionPool":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class SharedExecutorPool:
    """
    Simplified pool that wraps a single executor with read/write locking.
    
    This is the recommended pattern for embedded use where you have
    a single StorageExecutor but need thread-safe access.
    
    Example:
        executor = StorageExecutor()
        pool = SharedExecutorPool(executor)
        
        # Multiple threads can read concurrently
        with pool.read() as exec:
            results = exec.triples()
        
        # Only one thread can write at a time
        with pool.write() as exec:
            exec.add_triple(s, p, o, g)
    """
    
    def __init__(
        self, 
        executor: "StorageExecutor",
        max_readers: int = 10,
        write_timeout: float = 30.0,
    ):
        """
        Initialize shared executor pool.
        
        Args:
            executor: The StorageExecutor to wrap
            max_readers: Maximum concurrent readers
            write_timeout: Timeout for acquiring write lock
        """
        self._executor = executor
        self._max_readers = max_readers
        self._write_timeout = write_timeout
        
        # Reader-writer lock using semaphore
        self._read_semaphore = threading.Semaphore(max_readers)
        self._write_lock = threading.Lock()
        self._readers_lock = threading.Lock()
        self._active_readers = 0
        
        # Stats
        self._read_count = 0
        self._write_count = 0
    
    @contextmanager
    def read(self):
        """
        Acquire read access to the executor.
        
        Multiple threads can read concurrently up to max_readers.
        Blocks if a write is in progress.
        """
        # Wait for write to complete
        with self._write_lock:
            pass  # Just need to ensure no write in progress
        
        self._read_semaphore.acquire()
        with self._readers_lock:
            self._active_readers += 1
            self._read_count += 1
        
        try:
            yield self._executor
        finally:
            with self._readers_lock:
                self._active_readers -= 1
            self._read_semaphore.release()
    
    @contextmanager
    def write(self, timeout: Optional[float] = None):
        """
        Acquire exclusive write access to the executor.
        
        Blocks until all readers complete and no other writer is active.
        """
        effective_timeout = timeout if timeout is not None else self._write_timeout
        
        acquired = self._write_lock.acquire(timeout=effective_timeout)
        if not acquired:
            raise PoolExhaustedError(
                f"Could not acquire write lock after {effective_timeout}s"
            )
        
        try:
            # Wait for all readers to finish
            start = time.time()
            while True:
                with self._readers_lock:
                    if self._active_readers == 0:
                        break
                
                if time.time() - start > effective_timeout:
                    raise PoolExhaustedError(
                        f"Timed out waiting for readers after {effective_timeout}s"
                    )
                
                time.sleep(0.001)  # 1ms poll
            
            self._write_count += 1
            yield self._executor
            
        finally:
            self._write_lock.release()
    
    @property
    def executor(self) -> "StorageExecutor":
        """Get the underlying executor (use with caution - not thread-safe)."""
        return self._executor
    
    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._readers_lock:
            return {
                "active_readers": self._active_readers,
                "total_reads": self._read_count,
                "total_writes": self._write_count,
                "max_readers": self._max_readers,
            }
