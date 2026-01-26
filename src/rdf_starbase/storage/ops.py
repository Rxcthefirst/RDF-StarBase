"""
Enterprise Operations module for RDF-StarBase.

Provides production-ready operations capabilities:
- Health checks for orchestration
- Prometheus metrics endpoint
- OpenTelemetry distributed tracing
- Readiness and liveness probes
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = auto()   # Is the process alive?
    READINESS = auto()  # Is the service ready to accept traffic?
    STARTUP = auto()    # Has the service completed startup?


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize health check result."""
        return {
            "name": self.name,
            "status": self.status.name.lower(),
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Aggregated health report."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.now)
    version: Optional[str] = None
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize health report."""
        return {
            "status": self.overall_status.name.lower(),
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HealthChecker:
    """
    Health check system for orchestration integration.
    
    Supports:
    - Liveness probes (is the process alive?)
    - Readiness probes (can it accept traffic?)
    - Startup probes (has initialization completed?)
    - Custom health checks
    """
    
    def __init__(self, version: Optional[str] = None):
        self.version = version
        self._start_time = datetime.now()
        self._checks: Dict[str, Tuple[Callable[[], HealthCheckResult], CheckType]] = {}
        self._is_ready = False
        self._is_started = False
        self._lock = threading.Lock()
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        check_type: CheckType = CheckType.READINESS,
    ):
        """Register a health check function."""
        with self._lock:
            self._checks[name] = (check_func, check_type)
    
    def unregister_check(self, name: str):
        """Remove a health check."""
        with self._lock:
            self._checks.pop(name, None)
    
    def set_ready(self, ready: bool = True):
        """Set readiness status."""
        self._is_ready = ready
    
    def set_started(self, started: bool = True):
        """Set startup completion status."""
        self._is_started = started
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check by name."""
        with self._lock:
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check '{name}' not found",
                )
            check_func, _ = self._checks[name]
        
        start = time.time()
        try:
            result = check_func()
            result.duration_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
    
    def run_all_checks(
        self,
        check_type: Optional[CheckType] = None,
    ) -> HealthReport:
        """Run all health checks of a specific type."""
        with self._lock:
            checks_to_run = [
                (name, func) for name, (func, ct) in self._checks.items()
                if check_type is None or ct == check_type
            ]
        
        results = []
        for name, func in checks_to_run:
            start = time.time()
            try:
                result = func()
                result.duration_ms = (time.time() - start) * 1000
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    duration_ms=(time.time() - start) * 1000,
                ))
        
        # Determine overall status
        if not results:
            overall = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return HealthReport(
            overall_status=overall,
            checks=results,
            version=self.version,
            uptime_seconds=self.uptime_seconds,
        )
    
    def liveness_check(self) -> HealthReport:
        """Check if the process is alive (for /health/live)."""
        # Liveness is basic: if we can respond, we're alive
        return HealthReport(
            overall_status=HealthStatus.HEALTHY,
            checks=[HealthCheckResult(
                name="process",
                status=HealthStatus.HEALTHY,
                message="Process is running",
            )],
            version=self.version,
            uptime_seconds=self.uptime_seconds,
        )
    
    def readiness_check(self) -> HealthReport:
        """Check if service is ready to accept traffic (for /health/ready)."""
        if not self._is_ready:
            return HealthReport(
                overall_status=HealthStatus.UNHEALTHY,
                checks=[HealthCheckResult(
                    name="readiness",
                    status=HealthStatus.UNHEALTHY,
                    message="Service not ready",
                )],
                version=self.version,
                uptime_seconds=self.uptime_seconds,
            )
        
        return self.run_all_checks(CheckType.READINESS)
    
    def startup_check(self) -> HealthReport:
        """Check if startup is complete (for /health/startup)."""
        status = HealthStatus.HEALTHY if self._is_started else HealthStatus.UNHEALTHY
        return HealthReport(
            overall_status=status,
            checks=[HealthCheckResult(
                name="startup",
                status=status,
                message="Startup complete" if self._is_started else "Starting up",
            )],
            version=self.version,
            uptime_seconds=self.uptime_seconds,
        )


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    help_text: str
    labels: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.
    
    Supports:
    - Counter (monotonically increasing)
    - Gauge (can go up and down)
    - Histogram (bucketed observations)
    - Summary (quantile observations)
    """
    
    # Default histogram buckets (in seconds)
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    
    def __init__(self, prefix: str = "rdfstarbase"):
        self.prefix = prefix
        self._metrics: Dict[str, MetricDefinition] = {}
        self._counters: Dict[str, Dict[Tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[Tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[Tuple, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._histogram_buckets: Dict[str, Tuple[float, ...]] = {}
        self._lock = threading.Lock()
    
    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        return f"{self.prefix}_{name}"
    
    def register_counter(
        self,
        name: str,
        help_text: str,
        labels: Optional[List[str]] = None,
    ):
        """Register a counter metric."""
        full_name = self._full_name(name)
        self._metrics[full_name] = MetricDefinition(
            name=full_name,
            metric_type=MetricType.COUNTER,
            help_text=help_text,
            labels=labels or [],
        )
    
    def register_gauge(
        self,
        name: str,
        help_text: str,
        labels: Optional[List[str]] = None,
    ):
        """Register a gauge metric."""
        full_name = self._full_name(name)
        self._metrics[full_name] = MetricDefinition(
            name=full_name,
            metric_type=MetricType.GAUGE,
            help_text=help_text,
            labels=labels or [],
        )
    
    def register_histogram(
        self,
        name: str,
        help_text: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        """Register a histogram metric."""
        full_name = self._full_name(name)
        self._metrics[full_name] = MetricDefinition(
            name=full_name,
            metric_type=MetricType.HISTOGRAM,
            help_text=help_text,
            labels=labels or [],
        )
        self._histogram_buckets[full_name] = buckets or self.DEFAULT_BUCKETS
    
    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter."""
        full_name = self._full_name(name)
        label_tuple = tuple(sorted((labels or {}).items()))
        
        with self._lock:
            self._counters[full_name][label_tuple] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge value."""
        full_name = self._full_name(name)
        label_tuple = tuple(sorted((labels or {}).items()))
        
        with self._lock:
            self._gauges[full_name][label_tuple] = value
    
    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a gauge."""
        full_name = self._full_name(name)
        label_tuple = tuple(sorted((labels or {}).items()))
        
        with self._lock:
            self._gauges[full_name][label_tuple] += value
    
    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Decrement a gauge."""
        self.inc_gauge(name, -value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a histogram observation."""
        full_name = self._full_name(name)
        label_tuple = tuple(sorted((labels or {}).items()))
        
        with self._lock:
            self._histograms[full_name][label_tuple].append(value)
    
    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager to time operations and record as histogram."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.observe_histogram(name, duration, labels)
    
    def _format_labels(self, label_tuple: Tuple) -> str:
        """Format labels for Prometheus exposition."""
        if not label_tuple:
            return ""
        labels = ",".join(f'{k}="{v}"' for k, v in label_tuple)
        return "{" + labels + "}"
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        
        with self._lock:
            # Export counters
            for name, values in self._counters.items():
                if name in self._metrics:
                    defn = self._metrics[name]
                    lines.append(f"# HELP {name} {defn.help_text}")
                    lines.append(f"# TYPE {name} counter")
                
                for labels, value in values.items():
                    label_str = self._format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")
            
            # Export gauges
            for name, values in self._gauges.items():
                if name in self._metrics:
                    defn = self._metrics[name]
                    lines.append(f"# HELP {name} {defn.help_text}")
                    lines.append(f"# TYPE {name} gauge")
                
                for labels, value in values.items():
                    label_str = self._format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")
            
            # Export histograms
            for name, values in self._histograms.items():
                if name in self._metrics:
                    defn = self._metrics[name]
                    lines.append(f"# HELP {name} {defn.help_text}")
                    lines.append(f"# TYPE {name} histogram")
                
                buckets = self._histogram_buckets.get(name, self.DEFAULT_BUCKETS)
                
                for labels, observations in values.items():
                    label_str = self._format_labels(labels)
                    
                    # Calculate bucket counts
                    sorted_obs = sorted(observations)
                    count = len(sorted_obs)
                    total = sum(sorted_obs)
                    
                    for bucket in buckets:
                        bucket_count = sum(1 for o in sorted_obs if o <= bucket)
                        if labels:
                            bucket_labels = dict(labels)
                            bucket_labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(bucket_labels.items()))
                            lines.append(f'{name}_bucket{{{bucket_labels_str},le="{bucket}"}} {bucket_count}')
                        else:
                            lines.append(f'{name}_bucket{{le="{bucket}"}} {bucket_count}')
                    
                    # +Inf bucket
                    if labels:
                        bucket_labels = dict(labels)
                        bucket_labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(bucket_labels.items()))
                        lines.append(f'{name}_bucket{{{bucket_labels_str},le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    
                    lines.append(f"{name}_sum{label_str} {total}")
                    lines.append(f"{name}_count{label_str} {count}")
        
        return "\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "prefix": self.prefix,
            "metrics": {},
        }
        
        with self._lock:
            for name, values in self._counters.items():
                result["metrics"][name] = {
                    "type": "counter",
                    "values": [
                        {"labels": dict(labels), "value": value}
                        for labels, value in values.items()
                    ],
                }
            
            for name, values in self._gauges.items():
                result["metrics"][name] = {
                    "type": "gauge",
                    "values": [
                        {"labels": dict(labels), "value": value}
                        for labels, value in values.items()
                    ],
                }
            
            for name, values in self._histograms.items():
                result["metrics"][name] = {
                    "type": "histogram",
                    "values": [
                        {
                            "labels": dict(labels),
                            "count": len(obs),
                            "sum": sum(obs),
                            "observations": obs,
                        }
                        for labels, obs in values.items()
                    ],
                }
        
        return result
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


@dataclass
class SpanContext:
    """OpenTelemetry-compatible span context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # Sampled
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
        }


@dataclass
class Span:
    """OpenTelemetry-compatible span for distributed tracing."""
    name: str
    context: SpanContext
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })
    
    def set_status(self, status: str, message: Optional[str] = None):
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status_message"] = message
    
    def end(self):
        """End the span."""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize span for export."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """
    OpenTelemetry-compatible tracer for distributed tracing.
    
    Supports:
    - Span creation and management
    - Context propagation
    - Span export (OTLP-compatible format)
    """
    
    _local = threading.local()
    
    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0,
        max_spans: int = 10000,
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.max_spans = max_spans
        
        self._spans: List[Span] = []
        self._lock = threading.Lock()
    
    def _generate_id(self, length: int = 16) -> str:
        """Generate a random hex ID."""
        import random
        return "".join(random.choices("0123456789abcdef", k=length * 2))
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random
        return random.random() < self.sample_rate
    
    @property
    def current_span(self) -> Optional[Span]:
        """Get the current span from context."""
        stack = getattr(self._local, "span_stack", [])
        return stack[-1] if stack else None
    
    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        if not self._should_sample():
            # Return a no-op span for unsampled traces
            context = SpanContext(
                trace_id=self._generate_id(16),
                span_id=self._generate_id(8),
                trace_flags=0,
            )
            return Span(name=name, context=context)
        
        # Determine parent context
        if parent is None:
            parent = self.current_span
        
        if parent:
            context = SpanContext(
                trace_id=parent.context.trace_id,
                span_id=self._generate_id(8),
                parent_span_id=parent.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=self._generate_id(16),
                span_id=self._generate_id(8),
            )
        
        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )
        
        # Add service name
        span.set_attribute("service.name", self.service_name)
        
        # Push to span stack
        if not hasattr(self._local, "span_stack"):
            self._local.span_stack = []
        self._local.span_stack.append(span)
        
        return span
    
    def end_span(self, span: Span):
        """End a span and record it."""
        span.end()
        
        # Pop from stack
        stack = getattr(self._local, "span_stack", [])
        if stack and stack[-1] == span:
            stack.pop()
        
        # Record completed span
        if span.context.trace_flags > 0:
            with self._lock:
                self._spans.append(span)
                
                # Trim if over limit
                if len(self._spans) > self.max_spans:
                    self._spans = self._spans[-self.max_spans:]
    
    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for span lifecycle."""
        s = self.start_span(name, attributes=attributes)
        try:
            yield s
        except Exception as e:
            s.set_status("ERROR", str(e))
            raise
        finally:
            self.end_span(s)
    
    def get_completed_spans(self) -> List[Span]:
        """Get all completed spans."""
        with self._lock:
            return list(self._spans)
    
    def export_otlp(self) -> Dict[str, Any]:
        """Export spans in OTLP-compatible format."""
        with self._lock:
            spans = list(self._spans)
        
        return {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}},
                    ],
                },
                "scopeSpans": [{
                    "scope": {"name": "rdfstarbase"},
                    "spans": [s.to_dict() for s in spans],
                }],
            }],
        }
    
    def clear(self):
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()


class OperationsManager:
    """
    Unified operations management for enterprise deployments.
    
    Combines:
    - Health checks
    - Metrics collection
    - Distributed tracing
    """
    
    def __init__(
        self,
        service_name: str = "rdfstarbase",
        version: Optional[str] = None,
    ):
        self.service_name = service_name
        self.version = version
        
        self.health = HealthChecker(version=version)
        self.metrics = MetricsCollector(prefix=service_name.replace("-", "_"))
        self.tracer = Tracer(service_name=service_name)
        
        # Register default metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register standard metrics."""
        # Request metrics
        self.metrics.register_counter(
            "requests_total",
            "Total number of requests",
            labels=["method", "endpoint", "status"],
        )
        
        self.metrics.register_histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            labels=["method", "endpoint"],
        )
        
        # Query metrics
        self.metrics.register_counter(
            "queries_total",
            "Total number of SPARQL queries",
            labels=["type"],
        )
        
        self.metrics.register_histogram(
            "query_duration_seconds",
            "Query duration in seconds",
            labels=["type"],
        )
        
        # Storage metrics
        self.metrics.register_gauge(
            "triples_total",
            "Total number of triples",
            labels=["repository"],
        )
        
        self.metrics.register_gauge(
            "storage_bytes",
            "Storage usage in bytes",
            labels=["repository"],
        )
        
        # System metrics
        self.metrics.register_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
        )
        
        self.metrics.register_gauge(
            "active_connections",
            "Number of active connections",
        )
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
    ):
        """Record an HTTP request."""
        labels = {"method": method, "endpoint": endpoint, "status": str(status)}
        self.metrics.inc_counter("requests_total", labels=labels)
        
        duration_labels = {"method": method, "endpoint": endpoint}
        self.metrics.observe_histogram("request_duration_seconds", duration, duration_labels)
    
    def record_query(self, query_type: str, duration: float):
        """Record a SPARQL query."""
        self.metrics.inc_counter("queries_total", labels={"type": query_type})
        self.metrics.observe_histogram("query_duration_seconds", duration, {"type": query_type})
    
    def update_storage_stats(self, repository: str, triples: int, bytes_used: int):
        """Update storage statistics."""
        self.metrics.set_gauge("triples_total", triples, {"repository": repository})
        self.metrics.set_gauge("storage_bytes", bytes_used, {"repository": repository})
    
    def get_health_endpoints(self) -> Dict[str, Callable]:
        """Get health check endpoint handlers."""
        return {
            "/health": lambda: self.health.run_all_checks(),
            "/health/live": lambda: self.health.liveness_check(),
            "/health/ready": lambda: self.health.readiness_check(),
            "/health/startup": lambda: self.health.startup_check(),
        }
    
    def get_metrics_endpoint(self, format: str = "prometheus") -> str:
        """Get metrics in requested format."""
        if format == "json":
            return json.dumps(self.metrics.export_json(), indent=2)
        return self.metrics.export_prometheus()


# Convenience functions

def create_health_checker(version: Optional[str] = None) -> HealthChecker:
    """Create a health checker."""
    return HealthChecker(version=version)


def create_metrics_collector(prefix: str = "rdfstarbase") -> MetricsCollector:
    """Create a metrics collector."""
    return MetricsCollector(prefix=prefix)


def create_tracer(
    service_name: str,
    sample_rate: float = 1.0,
) -> Tracer:
    """Create a tracer."""
    return Tracer(service_name=service_name, sample_rate=sample_rate)


def create_operations_manager(
    service_name: str = "rdfstarbase",
    version: Optional[str] = None,
) -> OperationsManager:
    """Create a unified operations manager."""
    return OperationsManager(service_name=service_name, version=version)
