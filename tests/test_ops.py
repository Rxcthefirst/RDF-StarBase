"""Tests for the enterprise operations module."""

import pytest
import json
import time
from datetime import datetime

from rdf_starbase.storage.ops import (
    HealthStatus,
    CheckType,
    HealthCheckResult,
    HealthReport,
    HealthChecker,
    MetricType,
    MetricDefinition,
    MetricsCollector,
    SpanContext,
    Span,
    Tracer,
    OperationsManager,
    create_health_checker,
    create_metrics_collector,
    create_tracer,
    create_operations_manager,
)


# =============================================================================
# HealthCheckResult Tests
# =============================================================================

class TestHealthCheckResult:
    """Tests for HealthCheckResult."""
    
    def test_create_result(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connection OK",
        )
        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
    
    def test_to_dict(self):
        """Test serialization."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.DEGRADED,
            duration_ms=15.5,
            details={"connections": 5},
        )
        
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "degraded"
        assert data["duration_ms"] == 15.5
        assert data["details"]["connections"] == 5


# =============================================================================
# HealthReport Tests
# =============================================================================

class TestHealthReport:
    """Tests for HealthReport."""
    
    def test_create_report(self):
        """Test creating a health report."""
        checks = [
            HealthCheckResult("db", HealthStatus.HEALTHY),
            HealthCheckResult("cache", HealthStatus.HEALTHY),
        ]
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            checks=checks,
            version="1.0.0",
        )
        
        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.checks) == 2
    
    def test_to_dict(self):
        """Test report serialization."""
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            checks=[],
            version="1.0.0",
            uptime_seconds=3600,
        )
        
        data = report.to_dict()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] == 3600
    
    def test_to_json(self):
        """Test JSON serialization."""
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            checks=[],
        )
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] == "healthy"


# =============================================================================
# HealthChecker Tests
# =============================================================================

class TestHealthChecker:
    """Tests for HealthChecker."""
    
    def test_create_checker(self):
        """Test creating health checker."""
        checker = HealthChecker(version="1.0.0")
        assert checker.version == "1.0.0"
        assert not checker._is_ready
    
    def test_uptime(self):
        """Test uptime calculation."""
        checker = HealthChecker()
        time.sleep(0.1)
        assert checker.uptime_seconds >= 0.1
    
    def test_register_check(self):
        """Test registering a health check."""
        checker = HealthChecker()
        
        def my_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY)
        
        checker.register_check("test", my_check)
        result = checker.run_check("test")
        
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
    
    def test_run_check_not_found(self):
        """Test running non-existent check."""
        checker = HealthChecker()
        result = checker.run_check("nonexistent")
        
        assert result.status == HealthStatus.UNKNOWN
    
    def test_run_check_exception(self):
        """Test check that throws exception."""
        checker = HealthChecker()
        
        def failing_check():
            raise RuntimeError("Check failed")
        
        checker.register_check("failing", failing_check)
        result = checker.run_check("failing")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
    
    def test_run_all_checks(self):
        """Test running all checks."""
        checker = HealthChecker()
        
        checker.register_check(
            "healthy",
            lambda: HealthCheckResult("healthy", HealthStatus.HEALTHY)
        )
        checker.register_check(
            "degraded",
            lambda: HealthCheckResult("degraded", HealthStatus.DEGRADED)
        )
        
        report = checker.run_all_checks()
        
        assert len(report.checks) == 2
        assert report.overall_status == HealthStatus.DEGRADED
    
    def test_run_checks_by_type(self):
        """Test running checks filtered by type."""
        checker = HealthChecker()
        
        checker.register_check(
            "liveness",
            lambda: HealthCheckResult("live", HealthStatus.HEALTHY),
            CheckType.LIVENESS,
        )
        checker.register_check(
            "readiness",
            lambda: HealthCheckResult("ready", HealthStatus.HEALTHY),
            CheckType.READINESS,
        )
        
        liveness = checker.run_all_checks(CheckType.LIVENESS)
        assert len(liveness.checks) == 1
        assert liveness.checks[0].name == "live"
    
    def test_liveness_check(self):
        """Test liveness probe."""
        checker = HealthChecker()
        report = checker.liveness_check()
        
        assert report.overall_status == HealthStatus.HEALTHY
    
    def test_readiness_check_not_ready(self):
        """Test readiness probe when not ready."""
        checker = HealthChecker()
        report = checker.readiness_check()
        
        assert report.overall_status == HealthStatus.UNHEALTHY
    
    def test_readiness_check_ready(self):
        """Test readiness probe when ready."""
        checker = HealthChecker()
        checker.set_ready(True)
        
        report = checker.readiness_check()
        assert report.overall_status == HealthStatus.HEALTHY
    
    def test_startup_check(self):
        """Test startup probe."""
        checker = HealthChecker()
        
        report1 = checker.startup_check()
        assert report1.overall_status == HealthStatus.UNHEALTHY
        
        checker.set_started(True)
        report2 = checker.startup_check()
        assert report2.overall_status == HealthStatus.HEALTHY
    
    def test_unregister_check(self):
        """Test unregistering a check."""
        checker = HealthChecker()
        checker.register_check(
            "test",
            lambda: HealthCheckResult("test", HealthStatus.HEALTHY)
        )
        
        checker.unregister_check("test")
        result = checker.run_check("test")
        
        assert result.status == HealthStatus.UNKNOWN


# =============================================================================
# MetricsCollector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_create_collector(self):
        """Test creating metrics collector."""
        collector = MetricsCollector(prefix="test")
        assert collector.prefix == "test"
    
    def test_register_counter(self):
        """Test registering a counter."""
        collector = MetricsCollector()
        collector.register_counter("requests", "Total requests")
        
        assert "rdfstarbase_requests" in collector._metrics
    
    def test_increment_counter(self):
        """Test incrementing a counter."""
        collector = MetricsCollector()
        collector.register_counter("requests", "Total requests")
        
        collector.inc_counter("requests")
        collector.inc_counter("requests", 5)
        
        data = collector.export_json()
        values = data["metrics"]["rdfstarbase_requests"]["values"]
        assert values[0]["value"] == 6
    
    def test_counter_with_labels(self):
        """Test counter with labels."""
        collector = MetricsCollector()
        collector.register_counter("requests", "Requests", labels=["method"])
        
        collector.inc_counter("requests", labels={"method": "GET"})
        collector.inc_counter("requests", labels={"method": "POST"})
        collector.inc_counter("requests", labels={"method": "GET"})
        
        data = collector.export_json()
        values = data["metrics"]["rdfstarbase_requests"]["values"]
        
        get_count = next(v["value"] for v in values if ("method", "GET") in v["labels"].items())
        post_count = next(v["value"] for v in values if ("method", "POST") in v["labels"].items())
        
        assert get_count == 2
        assert post_count == 1
    
    def test_set_gauge(self):
        """Test setting a gauge."""
        collector = MetricsCollector()
        collector.register_gauge("connections", "Active connections")
        
        collector.set_gauge("connections", 10)
        collector.set_gauge("connections", 15)
        
        data = collector.export_json()
        values = data["metrics"]["rdfstarbase_connections"]["values"]
        assert values[0]["value"] == 15
    
    def test_inc_dec_gauge(self):
        """Test incrementing/decrementing a gauge."""
        collector = MetricsCollector()
        collector.register_gauge("active", "Active items")
        
        collector.set_gauge("active", 10)
        collector.inc_gauge("active", 3)
        collector.dec_gauge("active", 2)
        
        data = collector.export_json()
        values = data["metrics"]["rdfstarbase_active"]["values"]
        assert values[0]["value"] == 11
    
    def test_observe_histogram(self):
        """Test histogram observations."""
        collector = MetricsCollector()
        collector.register_histogram("duration", "Request duration")
        
        collector.observe_histogram("duration", 0.1)
        collector.observe_histogram("duration", 0.5)
        collector.observe_histogram("duration", 1.0)
        
        data = collector.export_json()
        hist = data["metrics"]["rdfstarbase_duration"]["values"][0]
        
        assert hist["count"] == 3
        assert hist["sum"] == 1.6
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        collector = MetricsCollector()
        collector.register_histogram("operation", "Operation duration")
        
        with collector.timer("operation"):
            time.sleep(0.05)
        
        data = collector.export_json()
        hist = data["metrics"]["rdfstarbase_operation"]["values"][0]
        
        assert hist["count"] == 1
        assert hist["sum"] >= 0.05
    
    def test_export_prometheus(self):
        """Test Prometheus exposition format."""
        collector = MetricsCollector(prefix="test")
        collector.register_counter("requests", "Total requests")
        collector.register_gauge("connections", "Active connections")
        
        collector.inc_counter("requests", 10)
        collector.set_gauge("connections", 5)
        
        output = collector.export_prometheus()
        
        assert "# TYPE test_requests counter" in output
        assert "test_requests 10" in output
        assert "# TYPE test_connections gauge" in output
        assert "test_connections 5" in output
    
    def test_export_json(self):
        """Test JSON export."""
        collector = MetricsCollector()
        collector.register_counter("test", "Test counter")
        collector.inc_counter("test")
        
        data = collector.export_json()
        
        assert "timestamp" in data
        assert "metrics" in data
        assert "rdfstarbase_test" in data["metrics"]
    
    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.register_counter("test", "Test")
        collector.inc_counter("test", 100)
        
        collector.reset()
        
        data = collector.export_json()
        assert len(data["metrics"]) == 0


# =============================================================================
# Tracer Tests
# =============================================================================

class TestTracer:
    """Tests for distributed tracing."""
    
    def test_create_tracer(self):
        """Test creating a tracer."""
        tracer = Tracer(service_name="test-service")
        assert tracer.service_name == "test-service"
    
    def test_start_span(self):
        """Test starting a span."""
        tracer = Tracer("test")
        span = tracer.start_span("operation")
        
        assert span.name == "operation"
        assert span.context.trace_id is not None
        assert span.context.span_id is not None
    
    def test_span_attributes(self):
        """Test setting span attributes."""
        tracer = Tracer("test")
        span = tracer.start_span("operation")
        
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"
    
    def test_span_events(self):
        """Test adding span events."""
        tracer = Tracer("test")
        span = tracer.start_span("operation")
        
        span.add_event("checkpoint", {"step": 1})
        
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
    
    def test_span_status(self):
        """Test setting span status."""
        tracer = Tracer("test")
        span = tracer.start_span("operation")
        
        span.set_status("ERROR", "Something failed")
        
        assert span.status == "ERROR"
        assert span.attributes["status_message"] == "Something failed"
    
    def test_end_span(self):
        """Test ending a span."""
        tracer = Tracer("test")
        span = tracer.start_span("operation")
        time.sleep(0.05)
        tracer.end_span(span)
        
        assert span.end_time is not None
        assert span.duration_ms >= 50
    
    def test_span_context_manager(self):
        """Test span context manager."""
        tracer = Tracer("test")
        
        with tracer.span("operation") as span:
            span.set_attribute("key", "value")
            time.sleep(0.01)
        
        assert span.end_time is not None
        assert span.attributes["key"] == "value"
    
    def test_span_exception_handling(self):
        """Test span captures exceptions."""
        tracer = Tracer("test")
        
        with pytest.raises(ValueError):
            with tracer.span("operation") as span:
                raise ValueError("Test error")
        
        assert span.status == "ERROR"
    
    def test_nested_spans(self):
        """Test nested spans inherit trace ID."""
        tracer = Tracer("test")
        
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                assert child.context.trace_id == parent.context.trace_id
                assert child.context.parent_span_id == parent.context.span_id
    
    def test_current_span(self):
        """Test getting current span."""
        # Create fresh tracer to avoid state from previous tests
        tracer = Tracer("test")
        # Clear any existing thread-local state
        tracer._local.span_stack = []
        
        assert tracer.current_span is None
        
        with tracer.span("operation") as span:
            assert tracer.current_span == span
        
        assert tracer.current_span is None
    
    def test_sample_rate(self):
        """Test sampling."""
        tracer = Tracer("test", sample_rate=0.0)
        span = tracer.start_span("operation")
        
        # Span should be created but not sampled
        assert span.context.trace_flags == 0
    
    def test_get_completed_spans(self):
        """Test retrieving completed spans."""
        tracer = Tracer("test")
        
        with tracer.span("op1"):
            pass
        with tracer.span("op2"):
            pass
        
        spans = tracer.get_completed_spans()
        assert len(spans) == 2
    
    def test_export_otlp(self):
        """Test OTLP export format."""
        tracer = Tracer("test-service")
        
        with tracer.span("operation"):
            pass
        
        data = tracer.export_otlp()
        
        assert "resourceSpans" in data
        assert len(data["resourceSpans"]) == 1
        
        resource = data["resourceSpans"][0]["resource"]
        assert resource["attributes"][0]["value"]["stringValue"] == "test-service"
    
    def test_clear(self):
        """Test clearing recorded spans."""
        tracer = Tracer("test")
        
        with tracer.span("operation"):
            pass
        
        tracer.clear()
        assert len(tracer.get_completed_spans()) == 0


# =============================================================================
# OperationsManager Tests
# =============================================================================

class TestOperationsManager:
    """Tests for unified OperationsManager."""
    
    def test_create_manager(self):
        """Test creating operations manager."""
        manager = OperationsManager(
            service_name="rdfstarbase",
            version="1.0.0",
        )
        
        assert manager.service_name == "rdfstarbase"
        assert manager.health is not None
        assert manager.metrics is not None
        assert manager.tracer is not None
    
    def test_record_request(self):
        """Test recording HTTP request."""
        manager = OperationsManager()
        
        manager.record_request("GET", "/query", 200, 0.1)
        
        data = manager.metrics.export_json()
        assert "rdfstarbase_requests_total" in data["metrics"]
    
    def test_record_query(self):
        """Test recording SPARQL query."""
        manager = OperationsManager()
        
        manager.record_query("SELECT", 0.5)
        
        data = manager.metrics.export_json()
        assert "rdfstarbase_queries_total" in data["metrics"]
    
    def test_update_storage_stats(self):
        """Test updating storage statistics."""
        manager = OperationsManager()
        
        manager.update_storage_stats("repo1", 1000, 1024 * 1024)
        
        data = manager.metrics.export_json()
        assert "rdfstarbase_triples_total" in data["metrics"]
        assert "rdfstarbase_storage_bytes" in data["metrics"]
    
    def test_get_health_endpoints(self):
        """Test getting health endpoint handlers."""
        manager = OperationsManager()
        
        endpoints = manager.get_health_endpoints()
        
        assert "/health" in endpoints
        assert "/health/live" in endpoints
        assert "/health/ready" in endpoints
        assert "/health/startup" in endpoints
    
    def test_get_metrics_endpoint_prometheus(self):
        """Test getting Prometheus metrics."""
        manager = OperationsManager()
        manager.record_request("GET", "/test", 200, 0.1)
        
        output = manager.get_metrics_endpoint("prometheus")
        assert "rdfstarbase_requests_total" in output
    
    def test_get_metrics_endpoint_json(self):
        """Test getting JSON metrics."""
        manager = OperationsManager()
        manager.record_request("GET", "/test", 200, 0.1)
        
        output = manager.get_metrics_endpoint("json")
        data = json.loads(output)
        
        assert "metrics" in data


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_health_checker(self):
        """Test create_health_checker."""
        checker = create_health_checker(version="1.0.0")
        assert checker.version == "1.0.0"
    
    def test_create_metrics_collector(self):
        """Test create_metrics_collector."""
        collector = create_metrics_collector(prefix="myapp")
        assert collector.prefix == "myapp"
    
    def test_create_tracer(self):
        """Test create_tracer."""
        tracer = create_tracer("my-service", sample_rate=0.5)
        assert tracer.service_name == "my-service"
        assert tracer.sample_rate == 0.5
    
    def test_create_operations_manager(self):
        """Test create_operations_manager."""
        manager = create_operations_manager(
            service_name="test",
            version="1.0.0",
        )
        assert manager.service_name == "test"
        assert manager.version == "1.0.0"


# =============================================================================
# Integration Tests
# =============================================================================

class TestOpsIntegration:
    """Integration tests for operations module."""
    
    def test_full_observability_workflow(self):
        """Test complete observability workflow."""
        manager = OperationsManager(
            service_name="rdfstarbase",
            version="1.0.0",
        )
        
        # Register custom health checks
        manager.health.register_check(
            "database",
            lambda: HealthCheckResult("database", HealthStatus.HEALTHY),
            CheckType.READINESS,
        )
        
        # Mark service as ready
        manager.health.set_ready(True)
        manager.health.set_started(True)
        
        # Record some activity
        with manager.tracer.span("process_request") as span:
            span.set_attribute("endpoint", "/query")
            
            # Simulate work
            manager.record_request("POST", "/query", 200, 0.15)
            manager.record_query("SELECT", 0.1)
        
        # Update storage stats
        manager.update_storage_stats("main", 50000, 10 * 1024 * 1024)
        
        # Check health
        health = manager.health.readiness_check()
        assert health.overall_status == HealthStatus.HEALTHY
        
        # Get metrics
        metrics = manager.get_metrics_endpoint("json")
        data = json.loads(metrics)
        assert len(data["metrics"]) > 0
        
        # Get traces
        spans = manager.tracer.get_completed_spans()
        assert len(spans) >= 1
    
    def test_health_check_aggregation(self):
        """Test health check status aggregation."""
        manager = OperationsManager()
        manager.health.set_ready(True)
        
        # All healthy
        manager.health.register_check(
            "db",
            lambda: HealthCheckResult("db", HealthStatus.HEALTHY),
        )
        manager.health.register_check(
            "cache",
            lambda: HealthCheckResult("cache", HealthStatus.HEALTHY),
        )
        
        report = manager.health.run_all_checks()
        assert report.overall_status == HealthStatus.HEALTHY
        
        # One degraded
        manager.health.register_check(
            "external",
            lambda: HealthCheckResult("external", HealthStatus.DEGRADED),
        )
        
        report = manager.health.run_all_checks()
        assert report.overall_status == HealthStatus.DEGRADED
        
        # One unhealthy
        manager.health.register_check(
            "critical",
            lambda: HealthCheckResult("critical", HealthStatus.UNHEALTHY),
        )
        
        report = manager.health.run_all_checks()
        assert report.overall_status == HealthStatus.UNHEALTHY
