"""
Query execution context with timeout and cancellation support.

Provides:
- Query timeout (configurable per-query or globally)
- Query cancellation via token
- Query statistics and profiling
- EXPLAIN query plan output
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from threading import Event, Lock
from typing import Any, Callable, Generator, Optional, TypeVar, Union
import functools
import time
import traceback

import polars as pl


class QueryState(IntEnum):
    """Query execution states."""
    PENDING = auto()     # Created but not started
    RUNNING = auto()     # Currently executing
    COMPLETED = auto()   # Finished successfully
    CANCELLED = auto()   # Cancelled by user
    TIMEOUT = auto()     # Exceeded timeout
    FAILED = auto()      # Failed with error


@dataclass
class QueryStats:
    """Statistics for query execution."""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    state: QueryState = QueryState.PENDING
    rows_scanned: int = 0
    rows_returned: int = 0
    pattern_count: int = 0
    join_count: int = 0
    filter_count: int = 0
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Query duration in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "duration_ms": self.duration_ms,
            "state": self.state.name,
            "rows_scanned": self.rows_scanned,
            "rows_returned": self.rows_returned,
            "pattern_count": self.pattern_count,
            "join_count": self.join_count,
            "filter_count": self.filter_count,
            "error": self.error,
        }


class CancellationToken:
    """Token for cooperative query cancellation."""
    
    def __init__(self):
        self._cancelled = Event()
    
    def cancel(self):
        """Request cancellation."""
        self._cancelled.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()
    
    def check(self):
        """Raise exception if cancelled."""
        if self._cancelled.is_set():
            raise QueryCancelledException("Query was cancelled")


class QueryCancelledException(Exception):
    """Exception raised when a query is cancelled."""
    pass


class QueryTimeoutException(Exception):
    """Exception raised when a query times out."""
    pass


@dataclass
class ExplainPlan:
    """Query execution plan for EXPLAIN."""
    query_type: str
    patterns: list[dict]
    filters: list[str] = field(default_factory=list)
    joins: list[dict] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    distinct: bool = False
    estimated_cost: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert plan to dictionary."""
        return {
            "query_type": self.query_type,
            "patterns": self.patterns,
            "filters": self.filters,
            "joins": self.joins,
            "order_by": self.order_by,
            "limit": self.limit,
            "offset": self.offset,
            "distinct": self.distinct,
            "estimated_cost": self.estimated_cost,
        }
    
    def __str__(self) -> str:
        """Pretty-print the execution plan."""
        lines = [
            f"Query Type: {self.query_type}",
            f"Estimated Cost: {self.estimated_cost:.2f}",
            "",
            "Patterns:",
        ]
        for i, p in enumerate(self.patterns, 1):
            lines.append(f"  {i}. {p['type']}: {p.get('description', 'N/A')}")
            if 'selectivity' in p:
                lines.append(f"     Selectivity: {p['selectivity']:.2%}")
        
        if self.filters:
            lines.extend(["", "Filters:"])
            for f in self.filters:
                lines.append(f"  - {f}")
        
        if self.joins:
            lines.extend(["", "Joins:"])
            for j in self.joins:
                lines.append(f"  - {j['type']}: {j.get('columns', [])}")
        
        if self.order_by:
            lines.extend(["", f"Order By: {', '.join(self.order_by)}"])
        
        if self.limit is not None:
            lines.append(f"Limit: {self.limit}")
        if self.offset is not None:
            lines.append(f"Offset: {self.offset}")
        if self.distinct:
            lines.append("Distinct: Yes")
        
        return "\n".join(lines)


@dataclass
class QueryContext:
    """
    Execution context for a query.
    
    Provides timeout, cancellation, and statistics tracking.
    """
    timeout_seconds: Optional[float] = None
    cancellation_token: Optional[CancellationToken] = None
    stats: QueryStats = field(default_factory=QueryStats)
    explain: bool = False
    
    _check_interval: int = 1000  # Check cancellation every N rows
    _row_count: int = 0
    
    def start(self):
        """Mark query as started."""
        self.stats.start_time = time.time()
        self.stats.state = QueryState.RUNNING
    
    def complete(self, rows_returned: int = 0):
        """Mark query as completed."""
        self.stats.end_time = time.time()
        self.stats.state = QueryState.COMPLETED
        self.stats.rows_returned = rows_returned
    
    def fail(self, error: str):
        """Mark query as failed."""
        self.stats.end_time = time.time()
        self.stats.state = QueryState.FAILED
        self.stats.error = error
    
    def check_cancelled(self):
        """Check if query should be cancelled."""
        if self.cancellation_token and self.cancellation_token.is_cancelled():
            self.stats.state = QueryState.CANCELLED
            raise QueryCancelledException("Query was cancelled")
    
    def check_timeout(self):
        """Check if query has exceeded timeout."""
        if self.timeout_seconds is not None and self.stats.start_time is not None:
            elapsed = time.time() - self.stats.start_time
            if elapsed > self.timeout_seconds:
                self.stats.state = QueryState.TIMEOUT
                raise QueryTimeoutException(
                    f"Query exceeded timeout of {self.timeout_seconds}s"
                )
    
    def check(self):
        """Check both cancellation and timeout."""
        self.check_cancelled()
        self.check_timeout()
    
    def count_row(self):
        """Count a processed row and periodically check cancellation."""
        self._row_count += 1
        if self._row_count % self._check_interval == 0:
            self.check()
    
    def record_pattern(self):
        """Record a pattern evaluation."""
        self.stats.pattern_count += 1
    
    def record_join(self):
        """Record a join operation."""
        self.stats.join_count += 1
    
    def record_filter(self):
        """Record a filter operation."""
        self.stats.filter_count += 1
    
    def add_scanned_rows(self, count: int):
        """Add to rows scanned count."""
        self.stats.rows_scanned += count


T = TypeVar('T')


def with_timeout(timeout_seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add timeout to a function.
    
    Uses ThreadPoolExecutor to run function with timeout.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    raise QueryTimeoutException(
                        f"Function exceeded timeout of {timeout_seconds}s"
                    )
        return wrapper
    return decorator


@contextmanager
def query_timeout(seconds: float) -> Generator[QueryContext, None, None]:
    """
    Context manager for query timeout.
    
    Usage:
        with query_timeout(5.0) as ctx:
            result = executor.execute(query)
    
    Raises QueryTimeoutException if query takes longer than specified.
    """
    ctx = QueryContext(timeout_seconds=seconds)
    ctx.start()
    try:
        yield ctx
        ctx.complete()
    except (QueryTimeoutException, QueryCancelledException):
        raise
    except Exception as e:
        ctx.fail(str(e))
        raise


def execute_with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args,
    **kwargs
) -> tuple[T, QueryStats]:
    """
    Execute a function with timeout.
    
    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (result, stats)
        
    Raises:
        QueryTimeoutException: If execution exceeds timeout
    """
    stats = QueryStats()
    stats.start_time = time.time()
    stats.state = QueryState.RUNNING
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            stats.end_time = time.time()
            stats.state = QueryState.COMPLETED
            return result, stats
        except FuturesTimeoutError:
            stats.end_time = time.time()
            stats.state = QueryState.TIMEOUT
            raise QueryTimeoutException(
                f"Query exceeded timeout of {timeout_seconds}s"
            )
        except Exception as e:
            stats.end_time = time.time()
            stats.state = QueryState.FAILED
            stats.error = str(e)
            raise


class QueryExecutionManager:
    """
    Manages query execution with timeout and cancellation.
    
    Provides:
    - Global default timeout
    - Active query tracking
    - Query cancellation by ID
    """
    
    def __init__(self, default_timeout: Optional[float] = None):
        """
        Initialize query execution manager.
        
        Args:
            default_timeout: Default timeout in seconds (None = no timeout)
        """
        self.default_timeout = default_timeout
        self._lock = Lock()
        self._next_query_id = 1
        self._active_queries: dict[int, tuple[CancellationToken, QueryContext]] = {}
    
    def execute(
        self,
        func: Callable[..., T],
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> tuple[T, int, QueryStats]:
        """
        Execute a query function with tracking.
        
        Args:
            func: Query function to execute
            timeout: Timeout in seconds (None = use default)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, query_id, stats)
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        
        with self._lock:
            query_id = self._next_query_id
            self._next_query_id += 1
            
            token = CancellationToken()
            ctx = QueryContext(
                timeout_seconds=effective_timeout,
                cancellation_token=token,
            )
            self._active_queries[query_id] = (token, ctx)
        
        try:
            ctx.start()
            
            if effective_timeout is not None:
                result, stats = execute_with_timeout(
                    func, effective_timeout, *args, **kwargs
                )
                ctx.stats = stats
            else:
                result = func(*args, **kwargs)
                ctx.complete()
            
            return result, query_id, ctx.stats
            
        finally:
            with self._lock:
                self._active_queries.pop(query_id, None)
    
    def cancel(self, query_id: int) -> bool:
        """
        Cancel a running query.
        
        Args:
            query_id: ID of query to cancel
            
        Returns:
            True if query was found and cancelled
        """
        with self._lock:
            if query_id in self._active_queries:
                token, _ = self._active_queries[query_id]
                token.cancel()
                return True
            return False
    
    def get_active_queries(self) -> list[dict]:
        """Get list of active queries with stats."""
        with self._lock:
            return [
                {
                    "query_id": qid,
                    "stats": ctx.stats.to_dict(),
                }
                for qid, (_, ctx) in self._active_queries.items()
            ]
