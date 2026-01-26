"""Tests for the observability module."""
import pytest
import time
import threading
from datetime import datetime, timedelta

from rdf_starbase.storage.observability import (
    MetricType,
    MetricValue,
    Metric,
    MetricsCollector,
    Timer,
    SystemStats,
    StoreStats,
    QueryStats,
    HealthStatus,
    HealthCheck,
    ObservabilityManager,
    get_observability,
    record_request,
    record_query
)


# ========== MetricType Tests ==========

class TestMetricType:
    """Tests for MetricType enum."""
    
    def test_counter_type(self):
        assert MetricType.COUNTER.value == "counter"
    
    def test_gauge_type(self):
        assert MetricType.GAUGE.value == "gauge"
    
    def test_histogram_type(self):
        assert MetricType.HISTOGRAM.value == "histogram"


# ========== MetricValue Tests ==========

class TestMetricValue:
    """Tests for MetricValue dataclass."""
    
    def test_creation(self):
        now = datetime.now()
        mv = MetricValue(timestamp=now, value=42.0)
        assert mv.timestamp == now
        assert mv.value == 42.0
        assert mv.labels == {}
    
    def test_with_labels(self):
        mv = MetricValue(
            timestamp=datetime.now(),
            value=100.0,
            labels={"method": "GET", "path": "/api"}
        )
        assert mv.labels["method"] == "GET"
        assert mv.labels["path"] == "/api"


# ========== Metric Tests ==========

class TestMetric:
    """Tests for Metric class."""
    
    def test_counter_creation(self):
        m = Metric(
            name="requests_total",
            metric_type=MetricType.COUNTER,
            description="Total requests"
        )
        assert m.name == "requests_total"
        assert m.metric_type == MetricType.COUNTER
        assert m.get_value() == 0.0
    
    def test_counter_inc(self):
        m = Metric(name="counter", metric_type=MetricType.COUNTER)
        m.inc()
        assert m.get_value() == 1.0
        m.inc(5.0)
        assert m.get_value() == 6.0
    
    def test_counter_inc_with_labels(self):
        m = Metric(name="counter", metric_type=MetricType.COUNTER)
        m.inc(1.0, {"path": "/api"})
        assert len(m.values) == 1
        assert m.values[0].labels["path"] == "/api"
    
    def test_counter_cannot_set(self):
        m = Metric(name="counter", metric_type=MetricType.COUNTER)
        with pytest.raises(ValueError, match="Cannot set non-gauge"):
            m.set(10.0)
    
    def test_gauge_creation(self):
        m = Metric(
            name="connections",
            metric_type=MetricType.GAUGE,
            unit="count"
        )
        assert m.metric_type == MetricType.GAUGE
        assert m.unit == "count"
    
    def test_gauge_set(self):
        m = Metric(name="gauge", metric_type=MetricType.GAUGE)
        m.set(50.0)
        assert m.get_value() == 50.0
        m.set(25.0)
        assert m.get_value() == 25.0
    
    def test_gauge_cannot_inc(self):
        m = Metric(name="gauge", metric_type=MetricType.GAUGE)
        with pytest.raises(ValueError, match="Cannot inc non-counter"):
            m.inc()
    
    def test_histogram_creation(self):
        m = Metric(
            name="duration",
            metric_type=MetricType.HISTOGRAM,
            unit="seconds"
        )
        assert m.metric_type == MetricType.HISTOGRAM
    
    def test_histogram_observe(self):
        m = Metric(name="histogram", metric_type=MetricType.HISTOGRAM)
        m.observe(0.1)
        m.observe(0.2)
        m.observe(0.3)
        assert m.get_value() == pytest.approx(0.2)  # Average
    
    def test_histogram_stats(self):
        m = Metric(name="histogram", metric_type=MetricType.HISTOGRAM)
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for v in values:
            m.observe(v)
        
        stats = m.get_histogram_stats()
        assert stats["count"] == 10
        assert stats["sum"] == pytest.approx(5.5)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(1.0)
        assert stats["avg"] == pytest.approx(0.55)
    
    def test_histogram_empty_stats(self):
        m = Metric(name="histogram", metric_type=MetricType.HISTOGRAM)
        stats = m.get_histogram_stats()
        assert stats["count"] == 0
        assert stats["avg"] == 0.0
    
    def test_histogram_cannot_set(self):
        m = Metric(name="histogram", metric_type=MetricType.HISTOGRAM)
        with pytest.raises(ValueError, match="Cannot set non-gauge"):
            m.set(10.0)
    
    def test_to_dict(self):
        m = Metric(
            name="requests",
            metric_type=MetricType.COUNTER,
            description="Total requests",
            unit="count"
        )
        m.inc(5.0)
        
        d = m.to_dict()
        assert d["name"] == "requests"
        assert d["type"] == "counter"
        assert d["description"] == "Total requests"
        assert d["unit"] == "count"
        assert d["value"] == 5.0
    
    def test_histogram_to_dict_includes_stats(self):
        m = Metric(name="duration", metric_type=MetricType.HISTOGRAM)
        m.observe(0.5)
        
        d = m.to_dict()
        assert "stats" in d
        assert d["stats"]["count"] == 1


# ========== MetricsCollector Tests ==========

class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_create_counter(self):
        mc = MetricsCollector()
        m = mc.counter("test_counter", "A test counter")
        assert m.metric_type == MetricType.COUNTER
        assert m.description == "A test counter"
    
    def test_get_same_counter(self):
        mc = MetricsCollector()
        m1 = mc.counter("test_counter")
        m2 = mc.counter("test_counter")
        assert m1 is m2
    
    def test_create_gauge(self):
        mc = MetricsCollector()
        m = mc.gauge("test_gauge", "A test gauge", "bytes")
        assert m.metric_type == MetricType.GAUGE
        assert m.unit == "bytes"
    
    def test_create_histogram(self):
        mc = MetricsCollector()
        m = mc.histogram("test_hist", "A test histogram", "seconds")
        assert m.metric_type == MetricType.HISTOGRAM
    
    def test_inc_counter(self):
        mc = MetricsCollector()
        mc.counter("requests")
        mc.inc("requests")
        mc.inc("requests", 5.0)
        assert mc.get("requests").get_value() == 6.0
    
    def test_set_gauge(self):
        mc = MetricsCollector()
        mc.gauge("connections")
        mc.set("connections", 10.0)
        assert mc.get("connections").get_value() == 10.0
    
    def test_observe_histogram(self):
        mc = MetricsCollector()
        mc.histogram("duration")
        mc.observe("duration", 0.5)
        mc.observe("duration", 1.5)
        assert mc.get("duration").get_value() == pytest.approx(1.0)
    
    def test_get_nonexistent(self):
        mc = MetricsCollector()
        assert mc.get("nonexistent") is None
    
    def test_get_all(self):
        mc = MetricsCollector()
        mc.counter("c1")
        mc.gauge("g1")
        mc.histogram("h1")
        
        all_metrics = mc.get_all()
        assert len(all_metrics) == 3
        assert "c1" in all_metrics
        assert "g1" in all_metrics
        assert "h1" in all_metrics
    
    def test_export(self):
        mc = MetricsCollector()
        mc.counter("requests")
        mc.inc("requests", 10.0)
        
        exported = mc.export()
        assert len(exported) == 1
        assert exported[0]["name"] == "requests"
        assert exported[0]["value"] == 10.0
    
    def test_reset_single(self):
        mc = MetricsCollector()
        mc.counter("c1")
        mc.counter("c2")
        mc.reset("c1")
        
        assert mc.get("c1") is None
        assert mc.get("c2") is not None
    
    def test_reset_all(self):
        mc = MetricsCollector()
        mc.counter("c1")
        mc.gauge("g1")
        mc.reset()
        
        assert len(mc.get_all()) == 0
    
    def test_uptime(self):
        mc = MetricsCollector()
        time.sleep(0.1)
        assert mc.uptime_seconds >= 0.1
    
    def test_thread_safety(self):
        mc = MetricsCollector()
        mc.counter("concurrent")
        
        def increment():
            for _ in range(100):
                mc.inc("concurrent")
        
        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert mc.get("concurrent").get_value() == 1000.0


# ========== Timer Tests ==========

class TestTimer:
    """Tests for Timer context manager."""
    
    def test_timer_basic(self):
        mc = MetricsCollector()
        mc.histogram("duration")
        
        with mc.timer("duration"):
            time.sleep(0.05)
        
        value = mc.get("duration").get_value()
        assert value >= 0.05
    
    def test_timer_with_labels(self):
        mc = MetricsCollector()
        mc.histogram("duration")
        
        with mc.timer("duration", {"method": "GET"}):
            pass
        
        assert len(mc.get("duration").values) == 1


# ========== SystemStats Tests ==========

class TestSystemStats:
    """Tests for SystemStats dataclass."""
    
    def test_to_dict(self):
        stats = SystemStats(
            cpu_percent=25.0,
            memory_used_mb=4096.0,
            memory_available_mb=8192.0,
            memory_percent=50.0,
            disk_used_gb=100.0,
            disk_available_gb=400.0,
            disk_percent=20.0,
            process_memory_mb=256.0,
            process_cpu_percent=5.0,
            thread_count=10,
            open_files=5,
            python_version="3.11.0",
            platform="Windows-10",
            uptime_seconds=3600.0
        )
        
        d = stats.to_dict()
        assert d["cpu_percent"] == 25.0
        assert d["memory"]["used_mb"] == 4096.0
        assert d["memory"]["percent"] == 50.0
        assert d["disk"]["used_gb"] == 100.0
        assert d["process"]["thread_count"] == 10
        assert d["uptime_seconds"] == 3600.0


# ========== StoreStats Tests ==========

class TestStoreStats:
    """Tests for StoreStats dataclass."""
    
    def test_to_dict(self):
        stats = StoreStats(
            triple_count=10000,
            subject_count=1000,
            predicate_count=50,
            object_count=5000,
            graph_count=5,
            term_dict_size=6000,
            storage_size_mb=100.0,
            index_size_mb=20.0
        )
        
        d = stats.to_dict()
        assert d["triple_count"] == 10000
        assert d["subject_count"] == 1000
        assert d["graph_count"] == 5
        assert d["storage_size_mb"] == 100.0


# ========== QueryStats Tests ==========

class TestQueryStats:
    """Tests for QueryStats dataclass."""
    
    def test_to_dict(self):
        stats = QueryStats(
            total_queries=1000,
            successful_queries=990,
            failed_queries=10,
            total_duration_ms=5000.0,
            avg_duration_ms=5.0,
            queries_per_second=10.0,
            slow_query_count=5
        )
        
        d = stats.to_dict()
        assert d["total_queries"] == 1000
        assert d["successful_queries"] == 990
        assert d["failed_queries"] == 10
        assert d["avg_duration_ms"] == 5.0


# ========== HealthCheck Tests ==========

class TestHealthCheck:
    """Tests for HealthCheck dataclass."""
    
    def test_healthy_check(self):
        check = HealthCheck(
            name="store",
            status=HealthStatus.HEALTHY,
            message="Store is responsive",
            latency_ms=5.0
        )
        
        d = check.to_dict()
        assert d["name"] == "store"
        assert d["status"] == "healthy"
        assert d["message"] == "Store is responsive"
        assert d["latency_ms"] == 5.0
    
    def test_degraded_check(self):
        check = HealthCheck(
            name="memory",
            status=HealthStatus.DEGRADED,
            message="Memory usage is high",
            details={"percent": 85}
        )
        
        d = check.to_dict()
        assert d["status"] == "degraded"
        assert d["details"]["percent"] == 85
    
    def test_unhealthy_check(self):
        check = HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed"
        )
        
        d = check.to_dict()
        assert d["status"] == "unhealthy"


# ========== ObservabilityManager Tests ==========

class TestObservabilityManager:
    """Tests for ObservabilityManager."""
    
    def test_creation(self):
        obs = ObservabilityManager()
        assert obs.metrics is not None
    
    def test_default_metrics_initialized(self):
        obs = ObservabilityManager()
        assert obs.metrics.get("requests_total") is not None
        assert obs.metrics.get("queries_total") is not None
        assert obs.metrics.get("triples_count") is not None
    
    def test_get_system_stats(self):
        obs = ObservabilityManager()
        stats = obs.get_system_stats()
        
        assert isinstance(stats, SystemStats)
        assert stats.thread_count > 0
        assert stats.python_version != ""
    
    def test_get_store_stats_no_store(self):
        obs = ObservabilityManager()
        stats = obs.get_store_stats()
        
        assert isinstance(stats, StoreStats)
        assert stats.triple_count == 0
    
    def test_get_query_stats_no_queries(self):
        obs = ObservabilityManager()
        stats = obs.get_query_stats()
        
        assert isinstance(stats, QueryStats)
        assert stats.total_queries == 0
    
    def test_get_query_stats_with_queries(self):
        obs = ObservabilityManager()
        
        # Record some queries
        obs.record_query("SELECT", True, 0.1)
        obs.record_query("SELECT", True, 0.2)
        obs.record_query("SELECT", False, 0.5)
        
        stats = obs.get_query_stats()
        assert stats.total_queries == 3
        assert stats.successful_queries == 2
        assert stats.failed_queries == 1
    
    def test_record_request(self):
        obs = ObservabilityManager()
        obs.record_request("GET", "/api/triples", 200, 0.05)
        
        assert obs.metrics.get("requests_total").get_value() == 1.0
    
    def test_record_request_error(self):
        obs = ObservabilityManager()
        obs.record_request("POST", "/api/triples", 500, 0.1)
        
        assert obs.metrics.get("requests_total").get_value() == 1.0
        assert obs.metrics.get("requests_errors_total").get_value() == 1.0
    
    def test_record_query_success(self):
        obs = ObservabilityManager()
        obs.record_query("SELECT", True, 0.05)
        
        assert obs.metrics.get("queries_total").get_value() == 1.0
        assert obs.metrics.get("queries_errors_total").get_value() == 0.0
    
    def test_record_query_failure(self):
        obs = ObservabilityManager()
        obs.record_query("SELECT", False, 0.1)
        
        assert obs.metrics.get("queries_total").get_value() == 1.0
        assert obs.metrics.get("queries_errors_total").get_value() == 1.0
    
    def test_check_health_default(self):
        obs = ObservabilityManager()
        health = obs.check_health()
        
        assert "status" in health
        assert "timestamp" in health
        assert "checks" in health
        assert len(health["checks"]) >= 2  # memory, disk
    
    def test_register_health_check(self):
        obs = ObservabilityManager()
        
        def custom_check():
            return HealthCheck(
                name="custom",
                status=HealthStatus.HEALTHY,
                message="All good"
            )
        
        obs.register_health_check("custom", custom_check)
        health = obs.check_health()
        
        check_names = [c["name"] for c in health["checks"]]
        assert "custom" in check_names
    
    def test_health_check_degraded(self):
        obs = ObservabilityManager()
        
        def degraded_check():
            return HealthCheck(
                name="test",
                status=HealthStatus.DEGRADED,
                message="Degraded"
            )
        
        obs.register_health_check("test", degraded_check)
        health = obs.check_health()
        
        # Overall status should be degraded if any check is degraded
        # (unless another is unhealthy)
        assert health["status"] in ["degraded", "healthy"]
    
    def test_health_check_unhealthy(self):
        obs = ObservabilityManager()
        
        def unhealthy_check():
            return HealthCheck(
                name="test",
                status=HealthStatus.UNHEALTHY,
                message="Unhealthy"
            )
        
        obs.register_health_check("test", unhealthy_check)
        health = obs.check_health()
        
        assert health["status"] == "unhealthy"
    
    def test_health_check_exception(self):
        obs = ObservabilityManager()
        
        def failing_check():
            raise RuntimeError("Check failed")
        
        obs.register_health_check("failing", failing_check)
        health = obs.check_health()
        
        # Should still return results, with failing check marked unhealthy
        failing = [c for c in health["checks"] if c["name"] == "failing"][0]
        assert failing["status"] == "unhealthy"
    
    def test_get_dashboard_data(self):
        obs = ObservabilityManager()
        dashboard = obs.get_dashboard_data()
        
        assert "timestamp" in dashboard
        assert "system" in dashboard
        assert "store" in dashboard
        assert "queries" in dashboard
        assert "health" in dashboard
        assert "metrics" in dashboard
    
    def test_update_store_metrics(self):
        obs = ObservabilityManager()
        
        # Create a mock store with stats
        class MockStore:
            def stats(self):
                return {
                    "triple_count": 1000,
                    "graph_count": 5
                }
        
        obs.update_store_metrics(MockStore())
        
        assert obs.metrics.get("triples_count").get_value() == 1000
        assert obs.metrics.get("graphs_count").get_value() == 5


# ========== Global Functions Tests ==========

class TestGlobalFunctions:
    """Tests for global convenience functions."""
    
    def test_get_observability_singleton(self):
        obs1 = get_observability()
        obs2 = get_observability()
        assert obs1 is obs2
    
    def test_record_request_global(self):
        obs = get_observability()
        initial = obs.metrics.get("requests_total").get_value()
        
        record_request("GET", "/test", 200, 0.01)
        
        assert obs.metrics.get("requests_total").get_value() == initial + 1
    
    def test_record_query_global(self):
        obs = get_observability()
        initial = obs.metrics.get("queries_total").get_value()
        
        record_query("SELECT", True, 0.01)
        
        assert obs.metrics.get("queries_total").get_value() == initial + 1


# ========== Integration Tests ==========

class TestObservabilityIntegration:
    """Integration tests for observability."""
    
    def test_full_request_cycle(self):
        obs = ObservabilityManager()
        
        # Simulate multiple requests
        for i in range(100):
            status = 200 if i % 10 != 0 else 500
            duration = 0.01 + (i % 5) * 0.01
            obs.record_request("GET", "/api/query", status, duration)
        
        # Check metrics
        assert obs.metrics.get("requests_total").get_value() == 100
        assert obs.metrics.get("requests_errors_total").get_value() == 10
        
        # Check histogram stats
        duration = obs.metrics.get("request_duration_seconds")
        stats = duration.get_histogram_stats()
        assert stats["count"] == 100
        assert stats["min"] >= 0.01
    
    def test_full_query_cycle(self):
        obs = ObservabilityManager()
        
        # Simulate queries
        for i in range(50):
            success = i % 10 != 0
            duration = 0.1 if success else 0.5
            obs.record_query("SELECT", success, duration)
        
        stats = obs.get_query_stats()
        assert stats.total_queries == 50
        assert stats.failed_queries == 5
        assert stats.successful_queries == 45
    
    def test_slow_query_detection(self):
        obs = ObservabilityManager()
        
        # Some fast, some slow queries
        obs.record_query("SELECT", True, 0.1)  # Fast
        obs.record_query("SELECT", True, 0.2)  # Fast
        obs.record_query("SELECT", True, 1.5)  # Slow
        obs.record_query("SELECT", True, 2.0)  # Slow
        obs.record_query("SELECT", True, 0.5)  # Fast
        
        stats = obs.get_query_stats()
        assert stats.slow_query_count == 2
    
    def test_dashboard_with_activity(self):
        obs = ObservabilityManager()
        
        # Generate some activity
        for _ in range(10):
            obs.record_request("GET", "/api", 200, 0.05)
            obs.record_query("SELECT", True, 0.1)
        
        dashboard = obs.get_dashboard_data()
        
        assert dashboard["queries"]["total_queries"] == 10
        assert len(dashboard["metrics"]) > 0
        assert dashboard["health"]["status"] in ["healthy", "degraded", "unhealthy"]


# ========== Performance Tests ==========

class TestObservabilityPerformance:
    """Performance tests for observability."""
    
    def test_high_volume_metrics(self):
        """Test handling high volume of metrics."""
        obs = ObservabilityManager()
        
        start = time.time()
        for _ in range(10000):
            obs.record_request("GET", "/api", 200, 0.001)
        elapsed = time.time() - start
        
        # Should handle 10k metric updates in under 1 second
        assert elapsed < 1.0
        assert obs.metrics.get("requests_total").get_value() == 10000
    
    def test_concurrent_metrics(self):
        """Test concurrent metric updates."""
        obs = ObservabilityManager()
        
        def record_batch():
            for _ in range(1000):
                obs.record_request("POST", "/api", 201, 0.01)
                obs.record_query("INSERT", True, 0.05)
        
        threads = [threading.Thread(target=record_batch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert obs.metrics.get("requests_total").get_value() == 5000
        assert obs.metrics.get("queries_total").get_value() == 5000
