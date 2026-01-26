"""
Observability Layer for RDF-StarBase.

Provides comprehensive monitoring and metrics:
- Metrics collection (counters, gauges, histograms)
- Stats API for runtime statistics
- Health checks
- Performance tracking
- Admin dashboard data
"""
from __future__ import annotations

import logging
import os
import platform
import psutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values


@dataclass
class MetricValue:
    """A single metric observation."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """A metric with its metadata and values."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    values: List[MetricValue] = field(default_factory=list)
    
    # For counters
    _counter_value: float = 0.0
    
    # For gauges
    _gauge_value: float = 0.0
    
    # For histograms
    _histogram_values: List[float] = field(default_factory=list)
    _histogram_buckets: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter."""
        if self.metric_type != MetricType.COUNTER:
            raise ValueError(f"Cannot inc non-counter metric: {self.name}")
        self._counter_value += value
        self.values.append(MetricValue(
            timestamp=datetime.now(),
            value=self._counter_value,
            labels=labels or {}
        ))
    
    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        if self.metric_type != MetricType.GAUGE:
            raise ValueError(f"Cannot set non-gauge metric: {self.name}")
        self._gauge_value = value
        self.values.append(MetricValue(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        ))
    
    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a histogram value."""
        if self.metric_type != MetricType.HISTOGRAM:
            raise ValueError(f"Cannot observe non-histogram metric: {self.name}")
        self._histogram_values.append(value)
        self.values.append(MetricValue(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        ))
    
    def get_value(self) -> float:
        """Get the current value."""
        if self.metric_type == MetricType.COUNTER:
            return self._counter_value
        elif self.metric_type == MetricType.GAUGE:
            return self._gauge_value
        else:
            return sum(self._histogram_values) / len(self._histogram_values) if self._histogram_values else 0.0
    
    def get_histogram_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if self.metric_type != MetricType.HISTOGRAM:
            return {}
        
        if not self._histogram_values:
            return {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0
            }
        
        sorted_values = sorted(self._histogram_values)
        count = len(sorted_values)
        
        def percentile(p: float) -> float:
            idx = int(p * count / 100)
            return sorted_values[min(idx, count - 1)]
        
        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": percentile(50),
            "p90": percentile(90),
            "p99": percentile(99)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metric to dictionary."""
        result = {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "value": self.get_value()
        }
        
        if self.metric_type == MetricType.HISTOGRAM:
            result["stats"] = self.get_histogram_stats()
        
        return result


class MetricsCollector:
    """
    Collects and manages metrics.
    
    Thread-safe metric collection with support for:
    - Counters: request_count, error_count
    - Gauges: active_connections, memory_usage
    - Histograms: query_duration, response_size
    
    Usage:
        collector = MetricsCollector()
        
        # Counter
        collector.counter("requests_total", "Total requests")
        collector.inc("requests_total")
        
        # Gauge
        collector.gauge("connections_active", "Active connections")
        collector.set("connections_active", 5)
        
        # Histogram
        collector.histogram("query_duration_seconds", "Query duration")
        with collector.timer("query_duration_seconds"):
            execute_query()
    """
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        self._start_time = datetime.now()
    
    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = ""
    ) -> Metric:
        """Create or get a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    description=description,
                    unit=unit
                )
            return self._metrics[name]
    
    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = ""
    ) -> Metric:
        """Create or get a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    description=description,
                    unit=unit
                )
            return self._metrics[name]
    
    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = ""
    ) -> Metric:
        """Create or get a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.HISTOGRAM,
                    description=description,
                    unit=unit
                )
            return self._metrics[name]
    
    def inc(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].inc(value, labels)
    
    def set(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].set(value, labels)
    
    def observe(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a histogram value."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].observe(value, labels)
    
    def timer(self, name: str, labels: Dict[str, str] = None) -> "Timer":
        """Create a timer context manager for histogram metrics."""
        return Timer(self, name, labels)
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def get_all(self) -> Dict[str, Metric]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)
    
    def export(self) -> List[Dict[str, Any]]:
        """Export all metrics as dictionaries."""
        with self._lock:
            return [m.to_dict() for m in self._metrics.values()]
    
    def reset(self, name: str = None) -> None:
        """Reset metric(s)."""
        with self._lock:
            if name:
                if name in self._metrics:
                    del self._metrics[name]
            else:
                self._metrics.clear()
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.observe(self.name, duration, self.labels)


@dataclass
class SystemStats:
    """System-level statistics."""
    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_available_gb: float
    disk_percent: float
    process_memory_mb: float
    process_cpu_percent: float
    thread_count: int
    open_files: int
    python_version: str
    platform: str
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory": {
                "used_mb": self.memory_used_mb,
                "available_mb": self.memory_available_mb,
                "percent": self.memory_percent
            },
            "disk": {
                "used_gb": self.disk_used_gb,
                "available_gb": self.disk_available_gb,
                "percent": self.disk_percent
            },
            "process": {
                "memory_mb": self.process_memory_mb,
                "cpu_percent": self.process_cpu_percent,
                "thread_count": self.thread_count,
                "open_files": self.open_files
            },
            "python_version": self.python_version,
            "platform": self.platform,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class StoreStats:
    """Store-level statistics."""
    triple_count: int
    subject_count: int
    predicate_count: int
    object_count: int
    graph_count: int
    term_dict_size: int
    storage_size_mb: float
    index_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple_count": self.triple_count,
            "subject_count": self.subject_count,
            "predicate_count": self.predicate_count,
            "object_count": self.object_count,
            "graph_count": self.graph_count,
            "term_dict_size": self.term_dict_size,
            "storage_size_mb": self.storage_size_mb,
            "index_size_mb": self.index_size_mb
        }


@dataclass
class QueryStats:
    """Query execution statistics."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_duration_ms: float
    avg_duration_ms: float
    queries_per_second: float
    slow_query_count: int  # Queries > 1 second
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "queries_per_second": self.queries_per_second,
            "slow_query_count": self.slow_query_count
        }


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details
        }


class ObservabilityManager:
    """
    Central observability manager.
    
    Provides:
    - Metrics collection
    - System stats
    - Store stats
    - Health checks
    - Admin dashboard data
    
    Usage:
        obs = ObservabilityManager()
        
        # Get system stats
        stats = obs.get_system_stats()
        
        # Run health checks
        health = obs.check_health()
        
        # Get admin dashboard data
        dashboard = obs.get_dashboard_data()
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self._start_time = datetime.now()
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self) -> None:
        """Initialize default metrics."""
        # Request metrics
        self.metrics.counter("requests_total", "Total HTTP requests")
        self.metrics.counter("requests_errors_total", "Total HTTP errors")
        self.metrics.histogram("request_duration_seconds", "Request duration", "seconds")
        
        # Query metrics
        self.metrics.counter("queries_total", "Total SPARQL queries")
        self.metrics.counter("queries_errors_total", "Total SPARQL query errors")
        self.metrics.histogram("query_duration_seconds", "Query duration", "seconds")
        
        # Store metrics
        self.metrics.gauge("triples_count", "Number of triples")
        self.metrics.gauge("terms_count", "Number of unique terms")
        self.metrics.gauge("graphs_count", "Number of named graphs")
        
        # Connection metrics
        self.metrics.gauge("connections_active", "Active connections")
        self.metrics.gauge("pool_size", "Connection pool size")
    
    def get_system_stats(self) -> SystemStats:
        """Get current system statistics."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process = psutil.Process()
            
            return SystemStats(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_used_mb=mem.used / (1024 * 1024),
                memory_available_mb=mem.available / (1024 * 1024),
                memory_percent=mem.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_available_gb=disk.free / (1024 * 1024 * 1024),
                disk_percent=disk.percent,
                process_memory_mb=process.memory_info().rss / (1024 * 1024),
                process_cpu_percent=process.cpu_percent(),
                thread_count=threading.active_count(),
                open_files=len(process.open_files()) if hasattr(process, 'open_files') else 0,
                python_version=platform.python_version(),
                platform=platform.platform(),
                uptime_seconds=self.metrics.uptime_seconds
            )
        except Exception as e:
            logger.warning(f"Error getting system stats: {e}")
            return SystemStats(
                cpu_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                memory_percent=0.0,
                disk_used_gb=0.0,
                disk_available_gb=0.0,
                disk_percent=0.0,
                process_memory_mb=0.0,
                process_cpu_percent=0.0,
                thread_count=threading.active_count(),
                open_files=0,
                python_version=platform.python_version(),
                platform=platform.platform(),
                uptime_seconds=self.metrics.uptime_seconds
            )
    
    def get_store_stats(self, store: Any = None) -> StoreStats:
        """Get store statistics."""
        if store is None:
            return StoreStats(
                triple_count=0,
                subject_count=0,
                predicate_count=0,
                object_count=0,
                graph_count=0,
                term_dict_size=0,
                storage_size_mb=0.0,
                index_size_mb=0.0
            )
        
        try:
            # Try to get stats from store
            stats = store.stats() if hasattr(store, 'stats') else {}
            
            return StoreStats(
                triple_count=stats.get('triple_count', len(store._df) if hasattr(store, '_df') else 0),
                subject_count=stats.get('subject_count', 0),
                predicate_count=stats.get('predicate_count', 0),
                object_count=stats.get('object_count', 0),
                graph_count=stats.get('graph_count', 0),
                term_dict_size=stats.get('term_dict_size', 0),
                storage_size_mb=stats.get('storage_size_mb', 0.0),
                index_size_mb=stats.get('index_size_mb', 0.0)
            )
        except Exception as e:
            logger.warning(f"Error getting store stats: {e}")
            return StoreStats(
                triple_count=0,
                subject_count=0,
                predicate_count=0,
                object_count=0,
                graph_count=0,
                term_dict_size=0,
                storage_size_mb=0.0,
                index_size_mb=0.0
            )
    
    def get_query_stats(self) -> QueryStats:
        """Get query execution statistics."""
        total = self.metrics.get("queries_total")
        errors = self.metrics.get("queries_errors_total")
        duration = self.metrics.get("query_duration_seconds")
        
        total_count = int(total.get_value()) if total else 0
        error_count = int(errors.get_value()) if errors else 0
        
        histogram_stats = duration.get_histogram_stats() if duration else {}
        total_duration_ms = histogram_stats.get("sum", 0.0) * 1000
        avg_duration_ms = histogram_stats.get("avg", 0.0) * 1000
        
        # Calculate queries per second
        uptime = self.metrics.uptime_seconds
        qps = total_count / uptime if uptime > 0 else 0.0
        
        # Count slow queries (> 1 second)
        slow_count = 0
        if duration and duration._histogram_values:
            slow_count = sum(1 for v in duration._histogram_values if v > 1.0)
        
        return QueryStats(
            total_queries=total_count,
            successful_queries=total_count - error_count,
            failed_queries=error_count,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration_ms,
            queries_per_second=qps,
            slow_query_count=slow_count
        )
    
    def register_health_check(
        self,
        name: str,
        check: Callable[[], HealthCheck]
    ) -> None:
        """Register a health check function."""
        self._health_checks[name] = check
    
    def check_health(self, store: Any = None) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns overall status and individual check results.
        """
        checks = []
        overall_status = HealthStatus.HEALTHY
        
        # Default store health check
        try:
            start = time.time()
            if store:
                # Try a simple query
                store.stats() if hasattr(store, 'stats') else None
            latency = (time.time() - start) * 1000
            
            checks.append(HealthCheck(
                name="store",
                status=HealthStatus.HEALTHY,
                message="Store is responsive",
                latency_ms=latency
            ))
        except Exception as e:
            checks.append(HealthCheck(
                name="store",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            ))
            overall_status = HealthStatus.UNHEALTHY
        
        # Memory check
        try:
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                checks.append(HealthCheck(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Memory usage is {mem.percent}%",
                    details={"percent": mem.percent}
                ))
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            else:
                checks.append(HealthCheck(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory usage is {mem.percent}%",
                    details={"percent": mem.percent}
                ))
        except Exception as e:
            checks.append(HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            ))
        
        # Disk check
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                checks.append(HealthCheck(
                    name="disk",
                    status=HealthStatus.DEGRADED,
                    message=f"Disk usage is {disk.percent}%",
                    details={"percent": disk.percent}
                ))
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            else:
                checks.append(HealthCheck(
                    name="disk",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk usage is {disk.percent}%",
                    details={"percent": disk.percent}
                ))
        except Exception as e:
            checks.append(HealthCheck(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            ))
        
        # Run registered health checks
        for name, check_fn in self._health_checks.items():
            try:
                result = check_fn()
                checks.append(result)
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            except Exception as e:
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                ))
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.metrics.uptime_seconds,
            "checks": [c.to_dict() for c in checks]
        }
    
    def get_dashboard_data(self, store: Any = None) -> Dict[str, Any]:
        """
        Get data for admin dashboard.
        
        Combines system stats, store stats, metrics, and health.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_stats().to_dict(),
            "store": self.get_store_stats(store).to_dict(),
            "queries": self.get_query_stats().to_dict(),
            "health": self.check_health(store),
            "metrics": self.metrics.export()
        }
    
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float
    ) -> None:
        """Record an HTTP request."""
        labels = {"method": method, "path": path}
        self.metrics.inc("requests_total", labels=labels)
        
        if status_code >= 400:
            self.metrics.inc("requests_errors_total", labels=labels)
        
        self.metrics.observe("request_duration_seconds", duration_seconds, labels=labels)
    
    def record_query(
        self,
        query_type: str,
        success: bool,
        duration_seconds: float
    ) -> None:
        """Record a SPARQL query execution."""
        labels = {"type": query_type}
        self.metrics.inc("queries_total", labels=labels)
        
        if not success:
            self.metrics.inc("queries_errors_total", labels=labels)
        
        self.metrics.observe("query_duration_seconds", duration_seconds, labels=labels)
    
    def update_store_metrics(self, store: Any) -> None:
        """Update store-related gauges."""
        stats = self.get_store_stats(store)
        self.metrics.set("triples_count", stats.triple_count)
        self.metrics.set("terms_count", stats.term_dict_size)
        self.metrics.set("graphs_count", stats.graph_count)


# Global instance for convenience
_default_manager: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get the default observability manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ObservabilityManager()
    return _default_manager


def record_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Record an HTTP request using the default manager."""
    get_observability().record_request(method, path, status_code, duration)


def record_query(query_type: str, success: bool, duration: float) -> None:
    """Record a SPARQL query using the default manager."""
    get_observability().record_query(query_type, success, duration)
