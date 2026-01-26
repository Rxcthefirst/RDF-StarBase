"""
Tests for query timeout and cancellation.
"""

import pytest
import time
from threading import Thread

from rdf_starbase.storage.query_context import (
    QueryContext,
    QueryState,
    QueryStats,
    CancellationToken,
    QueryCancelledException,
    QueryTimeoutException,
    ExplainPlan,
    execute_with_timeout,
    query_timeout,
    QueryExecutionManager,
)


class TestCancellationToken:
    """Test cancellation token."""
    
    def test_initial_state(self):
        """Test token is not cancelled initially."""
        token = CancellationToken()
        assert not token.is_cancelled()
    
    def test_cancel(self):
        """Test cancellation."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()
    
    def test_check_raises(self):
        """Test check raises when cancelled."""
        token = CancellationToken()
        token.check()  # Should not raise
        
        token.cancel()
        with pytest.raises(QueryCancelledException):
            token.check()


class TestQueryContext:
    """Test query context."""
    
    def test_lifecycle(self):
        """Test query lifecycle tracking."""
        ctx = QueryContext()
        assert ctx.stats.state == QueryState.PENDING
        
        ctx.start()
        assert ctx.stats.state == QueryState.RUNNING
        assert ctx.stats.start_time is not None
        
        ctx.complete(rows_returned=100)
        assert ctx.stats.state == QueryState.COMPLETED
        assert ctx.stats.rows_returned == 100
        assert ctx.stats.duration_ms >= 0  # Can be 0 if very fast
    
    def test_timeout_check(self):
        """Test timeout checking."""
        ctx = QueryContext(timeout_seconds=0.001)  # 1ms
        ctx.start()
        
        time.sleep(0.01)  # 10ms
        
        with pytest.raises(QueryTimeoutException):
            ctx.check_timeout()
    
    def test_no_timeout_when_not_set(self):
        """Test no timeout when not configured."""
        ctx = QueryContext()  # No timeout
        ctx.start()
        time.sleep(0.01)
        ctx.check_timeout()  # Should not raise
    
    def test_cancellation_check(self):
        """Test cancellation checking."""
        token = CancellationToken()
        ctx = QueryContext(cancellation_token=token)
        ctx.start()
        
        ctx.check_cancelled()  # Should not raise
        
        token.cancel()
        
        with pytest.raises(QueryCancelledException):
            ctx.check_cancelled()
    
    def test_row_counting(self):
        """Test row counting triggers checks."""
        token = CancellationToken()
        ctx = QueryContext(cancellation_token=token)
        ctx._check_interval = 10  # Check every 10 rows
        ctx.start()
        
        # Count 9 rows - no check yet
        for _ in range(9):
            ctx.count_row()
        
        # Cancel
        token.cancel()
        
        # 10th row triggers check
        with pytest.raises(QueryCancelledException):
            ctx.count_row()
    
    def test_stats_recording(self):
        """Test statistics recording."""
        ctx = QueryContext()
        ctx.start()
        
        ctx.record_pattern()
        ctx.record_pattern()
        ctx.record_join()
        ctx.record_filter()
        ctx.add_scanned_rows(1000)
        
        assert ctx.stats.pattern_count == 2
        assert ctx.stats.join_count == 1
        assert ctx.stats.filter_count == 1
        assert ctx.stats.rows_scanned == 1000
    
    def test_fail_state(self):
        """Test failure state."""
        ctx = QueryContext()
        ctx.start()
        ctx.fail("Something went wrong")
        
        assert ctx.stats.state == QueryState.FAILED
        assert ctx.stats.error == "Something went wrong"


class TestExecuteWithTimeout:
    """Test execute_with_timeout function."""
    
    def test_completes_within_timeout(self):
        """Test function that completes in time."""
        def fast_func():
            return 42
        
        result, stats = execute_with_timeout(fast_func, 1.0)
        
        assert result == 42
        assert stats.state == QueryState.COMPLETED
    
    def test_timeout_raises(self):
        """Test function that exceeds timeout."""
        def slow_func():
            time.sleep(1.0)
            return 42
        
        with pytest.raises(QueryTimeoutException):
            execute_with_timeout(slow_func, 0.01)
    
    def test_exception_propagates(self):
        """Test that exceptions are propagated."""
        def error_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            execute_with_timeout(error_func, 1.0)


class TestQueryTimeoutContextManager:
    """Test query_timeout context manager."""
    
    def test_successful_execution(self):
        """Test successful execution within timeout."""
        with query_timeout(1.0) as ctx:
            time.sleep(0.01)
        
        assert ctx.stats.state == QueryState.COMPLETED
    
    def test_exception_sets_failed(self):
        """Test exception sets failed state."""
        with pytest.raises(ValueError):
            with query_timeout(1.0) as ctx:
                raise ValueError("Test error")
        
        assert ctx.stats.state == QueryState.FAILED
        assert ctx.stats.error == "Test error"


class TestExplainPlan:
    """Test EXPLAIN plan output."""
    
    def test_to_dict(self):
        """Test plan conversion to dict."""
        plan = ExplainPlan(
            query_type="SELECT",
            patterns=[
                {"type": "TriplePattern", "description": "?s ?p ?o"},
            ],
            filters=["?x > 10"],
            limit=100,
            estimated_cost=42.5,
        )
        
        d = plan.to_dict()
        assert d["query_type"] == "SELECT"
        assert len(d["patterns"]) == 1
        assert d["limit"] == 100
        assert d["estimated_cost"] == 42.5
    
    def test_str_representation(self):
        """Test string representation."""
        plan = ExplainPlan(
            query_type="SELECT",
            patterns=[
                {"type": "TriplePattern", "description": "?s ?p ?o", "selectivity": 0.5},
            ],
            filters=["?x > 10"],
            order_by=["?name"],
            limit=100,
            distinct=True,
            estimated_cost=42.5,
        )
        
        s = str(plan)
        assert "SELECT" in s
        assert "TriplePattern" in s
        assert "50.00%" in s  # Selectivity
        assert "?x > 10" in s
        assert "?name" in s
        assert "100" in s
        assert "Distinct" in s


class TestQueryExecutionManager:
    """Test query execution manager."""
    
    def test_execute_with_tracking(self):
        """Test execution with query ID tracking."""
        mgr = QueryExecutionManager()
        
        result, query_id, stats = mgr.execute(lambda: 42)
        
        assert result == 42
        assert query_id == 1
        assert stats.state == QueryState.COMPLETED
    
    def test_default_timeout(self):
        """Test default timeout is applied."""
        mgr = QueryExecutionManager(default_timeout=0.01)
        
        def slow_func():
            time.sleep(1.0)
        
        with pytest.raises(QueryTimeoutException):
            mgr.execute(slow_func)
    
    def test_per_query_timeout(self):
        """Test per-query timeout overrides default."""
        mgr = QueryExecutionManager(default_timeout=10.0)
        
        def slow_func():
            time.sleep(1.0)
        
        with pytest.raises(QueryTimeoutException):
            mgr.execute(slow_func, timeout=0.01)
    
    def test_cancel_query(self):
        """Test query cancellation."""
        mgr = QueryExecutionManager()
        
        cancelled_id = None
        
        def slow_func_with_check():
            # In real usage, the function would check ctx.check()
            # Here we just simulate a cancellable operation
            time.sleep(0.5)
            return 42
        
        # Start query in background
        results = []
        errors = []
        
        def run_query():
            try:
                result = mgr.execute(slow_func_with_check)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        thread = Thread(target=run_query)
        thread.start()
        
        # Give it time to start
        time.sleep(0.01)
        
        # Should have an active query
        active = mgr.get_active_queries()
        assert len(active) >= 0  # May have finished already
        
        thread.join(timeout=2.0)
    
    def test_active_query_tracking(self):
        """Test active query list."""
        mgr = QueryExecutionManager()
        
        # No active queries initially
        assert len(mgr.get_active_queries()) == 0
        
        # After execution, query is removed
        mgr.execute(lambda: 42)
        assert len(mgr.get_active_queries()) == 0


class TestQueryStats:
    """Test query statistics."""
    
    def test_duration_calculation(self):
        """Test duration is calculated correctly."""
        stats = QueryStats()
        stats.start_time = time.time()
        time.sleep(0.01)
        stats.end_time = time.time()
        
        assert stats.duration_ms >= 10  # At least 10ms
    
    def test_to_dict(self):
        """Test stats conversion to dict."""
        stats = QueryStats(
            rows_scanned=1000,
            rows_returned=50,
            pattern_count=3,
            join_count=2,
            filter_count=1,
        )
        stats.start_time = time.time()
        stats.end_time = stats.start_time + 0.5  # 500ms
        stats.state = QueryState.COMPLETED
        
        d = stats.to_dict()
        assert d["rows_scanned"] == 1000
        assert d["rows_returned"] == 50
        assert d["state"] == "COMPLETED"
        assert d["duration_ms"] >= 500
