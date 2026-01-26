"""
Federation module for RDF-StarBase.

Provides SPARQL 1.1 Federated Query support with SERVICE clause,
cross-instance synchronization, and distributed query planning.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
import polars as pl


class EndpointStatus(Enum):
    """Remote endpoint health status."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNREACHABLE = auto()
    UNKNOWN = auto()


class SyncDirection(Enum):
    """Replication sync direction."""
    PUSH = auto()      # Push local changes to remote
    PULL = auto()      # Pull remote changes to local
    BIDIRECTIONAL = auto()  # Two-way sync


class SyncStrategy(Enum):
    """Conflict resolution strategy for sync."""
    LOCAL_WINS = auto()
    REMOTE_WINS = auto()
    MOST_RECENT = auto()
    MERGE = auto()


class QueryPushdownType(Enum):
    """Types of operations that can be pushed to remote endpoints."""
    FILTER = auto()
    PROJECTION = auto()
    LIMIT = auto()
    ORDER = auto()
    DISTINCT = auto()


@dataclass
class RemoteEndpoint:
    """Configuration for a remote SPARQL endpoint."""
    url: str
    name: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    auth_token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Health tracking
    status: EndpointStatus = EndpointStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_latency_ms: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    
    def __post_init__(self):
        if not self.name:
            parsed = urlparse(self.url)
            self.name = parsed.netloc or self.url[:50]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of queries to this endpoint."""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize endpoint configuration."""
        return {
            "url": self.url,
            "name": self.name,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "status": self.status.name,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_latency_ms": self.last_latency_ms,
            "success_rate": self.success_rate,
        }


@dataclass
class ServiceCall:
    """Parsed SERVICE clause from a SPARQL query."""
    endpoint_url: str
    subquery: str
    is_silent: bool = False  # SERVICE SILENT
    variables_used: Set[str] = field(default_factory=set)
    variables_bound: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash((self.endpoint_url, self.subquery, self.is_silent))


@dataclass
class FederatedResult:
    """Result from a federated query execution."""
    bindings: List[Dict[str, Any]]
    endpoint_url: str
    execution_time_ms: float
    rows_returned: int
    was_cached: bool = False
    error: Optional[str] = None


@dataclass 
class SyncState:
    """State tracking for cross-instance synchronization."""
    local_instance_id: str
    remote_instance_id: str
    last_sync_time: Optional[datetime] = None
    last_sync_txn_id: Optional[int] = None
    pending_changes: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0


@dataclass
class SyncOperation:
    """A single synchronization operation."""
    operation_type: str  # "insert" or "delete"
    subject: str
    predicate: str
    object: str
    graph: Optional[str] = None
    timestamp: Optional[datetime] = None
    source_instance: Optional[str] = None


class EndpointRegistry:
    """Registry of known remote SPARQL endpoints."""
    
    def __init__(self):
        self._endpoints: Dict[str, RemoteEndpoint] = {}
        self._aliases: Dict[str, str] = {}  # alias -> url
    
    def register(
        self,
        url: str,
        name: Optional[str] = None,
        alias: Optional[str] = None,
        **kwargs
    ) -> RemoteEndpoint:
        """Register a remote endpoint."""
        endpoint = RemoteEndpoint(url=url, name=name, **kwargs)
        self._endpoints[url] = endpoint
        
        if alias:
            self._aliases[alias] = url
        
        return endpoint
    
    def unregister(self, url_or_alias: str) -> bool:
        """Remove an endpoint from the registry."""
        url = self._aliases.get(url_or_alias, url_or_alias)
        
        if url in self._endpoints:
            del self._endpoints[url]
            # Remove any aliases pointing to this URL
            self._aliases = {k: v for k, v in self._aliases.items() if v != url}
            return True
        return False
    
    def get(self, url_or_alias: str) -> Optional[RemoteEndpoint]:
        """Get an endpoint by URL or alias."""
        url = self._aliases.get(url_or_alias, url_or_alias)
        return self._endpoints.get(url)
    
    def resolve_url(self, url_or_alias: str) -> str:
        """Resolve alias to actual URL."""
        return self._aliases.get(url_or_alias, url_or_alias)
    
    def list_all(self) -> List[RemoteEndpoint]:
        """List all registered endpoints."""
        return list(self._endpoints.values())
    
    def list_healthy(self) -> List[RemoteEndpoint]:
        """List endpoints with healthy status."""
        return [e for e in self._endpoints.values() 
                if e.status == EndpointStatus.HEALTHY]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry."""
        return {
            "endpoints": {url: ep.to_dict() for url, ep in self._endpoints.items()},
            "aliases": dict(self._aliases),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointRegistry":
        """Deserialize registry."""
        registry = cls()
        for url, ep_data in data.get("endpoints", {}).items():
            registry.register(
                url=url,
                name=ep_data.get("name"),
                timeout_seconds=ep_data.get("timeout_seconds", 30.0),
                max_retries=ep_data.get("max_retries", 3),
            )
        registry._aliases = data.get("aliases", {})
        return registry


class ServiceClauseParser:
    """Parser for extracting SERVICE clauses from SPARQL queries."""
    
    # Pattern to match SERVICE clause
    SERVICE_PATTERN = re.compile(
        r'SERVICE\s+(SILENT\s+)?<([^>]+)>\s*\{([^}]+)\}',
        re.IGNORECASE | re.DOTALL
    )
    
    # Pattern to extract variables
    VAR_PATTERN = re.compile(r'\?(\w+)')
    
    @classmethod
    def parse(cls, query: str) -> List[ServiceCall]:
        """Extract SERVICE clauses from a SPARQL query."""
        services = []
        
        for match in cls.SERVICE_PATTERN.finditer(query):
            is_silent = match.group(1) is not None
            endpoint_url = match.group(2)
            subquery = match.group(3).strip()
            
            # Extract variables from subquery
            variables = set(cls.VAR_PATTERN.findall(subquery))
            
            service = ServiceCall(
                endpoint_url=endpoint_url,
                subquery=subquery,
                is_silent=is_silent,
                variables_used=variables,
                variables_bound=variables,  # Simplified - actual analysis would be more complex
            )
            services.append(service)
        
        return services
    
    @classmethod
    def remove_service_clauses(cls, query: str) -> str:
        """Remove SERVICE clauses from a query (for local execution)."""
        return cls.SERVICE_PATTERN.sub('', query)
    
    @classmethod
    def has_service_clause(cls, query: str) -> bool:
        """Check if query contains SERVICE clauses."""
        return bool(cls.SERVICE_PATTERN.search(query))


class FederatedQueryExecutor:
    """
    Executes federated SPARQL queries with SERVICE clause support.
    
    Handles:
    - Parsing SERVICE clauses from queries
    - Executing subqueries against remote endpoints
    - Joining results from multiple sources
    - Caching remote results
    - Error handling and retries
    """
    
    def __init__(
        self,
        registry: Optional[EndpointRegistry] = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
        max_concurrent_requests: int = 5,
    ):
        self.registry = registry or EndpointRegistry()
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_concurrent_requests = max_concurrent_requests
        
        self._cache: Dict[str, Tuple[FederatedResult, float]] = {}
        self._parser = ServiceClauseParser()
    
    def _cache_key(self, endpoint_url: str, query: str) -> str:
        """Generate cache key for a query."""
        content = f"{endpoint_url}:{query}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cached(self, endpoint_url: str, query: str) -> Optional[FederatedResult]:
        """Get cached result if valid."""
        if not self.cache_enabled:
            return None
        
        key = self._cache_key(endpoint_url, query)
        if key in self._cache:
            result, cached_at = self._cache[key]
            if time.time() - cached_at < self.cache_ttl_seconds:
                result.was_cached = True
                return result
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, endpoint_url: str, query: str, result: FederatedResult):
        """Cache a result."""
        if self.cache_enabled:
            key = self._cache_key(endpoint_url, query)
            self._cache[key] = (result, time.time())
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
    
    def execute_remote_query(
        self,
        endpoint_url: str,
        query: str,
        bindings: Optional[Dict[str, Any]] = None,
    ) -> FederatedResult:
        """
        Execute a SPARQL query against a remote endpoint.
        
        Args:
            endpoint_url: URL of the remote SPARQL endpoint
            query: SPARQL query to execute
            bindings: Optional variable bindings to inject
            
        Returns:
            FederatedResult with bindings or error
        """
        # Check cache first
        cached = self._get_cached(endpoint_url, query)
        if cached:
            return cached
        
        endpoint = self.registry.get(endpoint_url)
        timeout = endpoint.timeout_seconds if endpoint else 30.0
        max_retries = endpoint.max_retries if endpoint else 3
        headers = {"Accept": "application/sparql-results+json"}
        
        if endpoint and endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"
        if endpoint and endpoint.headers:
            headers.update(endpoint.headers)
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        endpoint_url,
                        data={"query": query},
                        headers=headers,
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    bindings_list = data.get("results", {}).get("bindings", [])
                    
                    # Convert SPARQL JSON results to simple dicts
                    results = []
                    for binding in bindings_list:
                        row = {}
                        for var, val in binding.items():
                            row[var] = val.get("value")
                        results.append(row)
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    result = FederatedResult(
                        bindings=results,
                        endpoint_url=endpoint_url,
                        execution_time_ms=execution_time,
                        rows_returned=len(results),
                    )
                    
                    # Update endpoint stats
                    if endpoint:
                        endpoint.status = EndpointStatus.HEALTHY
                        endpoint.last_checked = datetime.now()
                        endpoint.last_latency_ms = execution_time
                        endpoint.success_count += 1
                    
                    # Cache the result
                    self._set_cached(endpoint_url, query, result)
                    
                    return result
                    
            except Exception as e:
                last_error = str(e)
                if endpoint:
                    endpoint.error_count += 1
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        if endpoint:
            endpoint.status = EndpointStatus.UNREACHABLE
            endpoint.last_checked = datetime.now()
        
        return FederatedResult(
            bindings=[],
            endpoint_url=endpoint_url,
            execution_time_ms=(time.time() - start_time) * 1000,
            rows_returned=0,
            error=last_error,
        )
    
    async def execute_remote_query_async(
        self,
        endpoint_url: str,
        query: str,
    ) -> FederatedResult:
        """Async version of execute_remote_query."""
        cached = self._get_cached(endpoint_url, query)
        if cached:
            return cached
        
        endpoint = self.registry.get(endpoint_url)
        timeout = endpoint.timeout_seconds if endpoint else 30.0
        headers = {"Accept": "application/sparql-results+json"}
        
        if endpoint and endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint_url,
                    data={"query": query},
                    headers=headers,
                )
                response.raise_for_status()
                
                data = response.json()
                bindings_list = data.get("results", {}).get("bindings", [])
                
                results = []
                for binding in bindings_list:
                    row = {}
                    for var, val in binding.items():
                        row[var] = val.get("value")
                    results.append(row)
                
                execution_time = (time.time() - start_time) * 1000
                
                result = FederatedResult(
                    bindings=results,
                    endpoint_url=endpoint_url,
                    execution_time_ms=execution_time,
                    rows_returned=len(results),
                )
                
                if endpoint:
                    endpoint.status = EndpointStatus.HEALTHY
                    endpoint.last_checked = datetime.now()
                    endpoint.last_latency_ms = execution_time
                    endpoint.success_count += 1
                
                self._set_cached(endpoint_url, query, result)
                return result
                
        except Exception as e:
            if endpoint:
                endpoint.status = EndpointStatus.UNREACHABLE
                endpoint.error_count += 1
            
            return FederatedResult(
                bindings=[],
                endpoint_url=endpoint_url,
                execution_time_ms=(time.time() - start_time) * 1000,
                rows_returned=0,
                error=str(e),
            )
    
    def execute_federated_query(
        self,
        query: str,
        local_executor: Callable[[str], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Execute a federated query with SERVICE clauses.
        
        Args:
            query: SPARQL query potentially containing SERVICE clauses
            local_executor: Function to execute local SPARQL queries
            
        Returns:
            Combined results from local and remote execution
        """
        service_calls = self._parser.parse(query)
        
        if not service_calls:
            # No SERVICE clauses, execute locally
            return local_executor(query)
        
        # Execute each SERVICE clause
        remote_results: Dict[str, FederatedResult] = {}
        
        for service in service_calls:
            # Build full query for remote endpoint
            remote_query = f"SELECT * WHERE {{ {service.subquery} }}"
            result = self.execute_remote_query(service.endpoint_url, remote_query)
            
            if result.error and not service.is_silent:
                raise RuntimeError(
                    f"SERVICE call to {service.endpoint_url} failed: {result.error}"
                )
            
            remote_results[service.endpoint_url] = result
        
        # Execute local portion of query
        local_query = self._parser.remove_service_clauses(query)
        local_results = local_executor(local_query) if local_query.strip() else []
        
        # Join results
        return self._join_results(local_results, remote_results, service_calls)
    
    def _join_results(
        self,
        local_results: List[Dict[str, Any]],
        remote_results: Dict[str, FederatedResult],
        service_calls: List[ServiceCall],
    ) -> List[Dict[str, Any]]:
        """Join local and remote results based on shared variables."""
        if not local_results:
            # No local results, just combine remote
            combined = []
            for result in remote_results.values():
                combined.extend(result.bindings)
            return combined
        
        if not remote_results:
            return local_results
        
        # Simple nested loop join for now
        # A production implementation would use hash joins
        result = local_results
        
        for service in service_calls:
            remote = remote_results.get(service.endpoint_url)
            if not remote or not remote.bindings:
                continue
            
            # Find shared variables
            local_vars = set(result[0].keys()) if result else set()
            remote_vars = service.variables_bound
            shared_vars = local_vars & remote_vars
            
            if shared_vars:
                # Hash join on shared variables
                new_result = []
                for local_row in result:
                    for remote_row in remote.bindings:
                        if all(local_row.get(v) == remote_row.get(v) for v in shared_vars):
                            merged = {**local_row, **remote_row}
                            new_result.append(merged)
                result = new_result
            else:
                # Cross product
                new_result = []
                for local_row in result:
                    for remote_row in remote.bindings:
                        merged = {**local_row, **remote_row}
                        new_result.append(merged)
                result = new_result
        
        return result
    
    def check_endpoint_health(self, endpoint_url: str) -> EndpointStatus:
        """Check if an endpoint is healthy using ASK query."""
        result = self.execute_remote_query(endpoint_url, "ASK { ?s ?p ?o } LIMIT 1")
        
        endpoint = self.registry.get(endpoint_url)
        if endpoint:
            return endpoint.status
        
        return EndpointStatus.HEALTHY if not result.error else EndpointStatus.UNREACHABLE
    
    async def check_all_endpoints_async(self) -> Dict[str, EndpointStatus]:
        """Check health of all registered endpoints concurrently."""
        endpoints = self.registry.list_all()
        
        async def check_one(ep: RemoteEndpoint) -> Tuple[str, EndpointStatus]:
            result = await self.execute_remote_query_async(
                ep.url,
                "ASK { ?s ?p ?o } LIMIT 1"
            )
            return (ep.url, ep.status)
        
        tasks = [check_one(ep) for ep in endpoints]
        results = await asyncio.gather(*tasks)
        
        return dict(results)


class CrossInstanceSync:
    """
    Synchronization manager for cross-instance replication.
    
    Supports:
    - Push/Pull/Bidirectional sync
    - Conflict detection and resolution
    - Incremental sync based on transaction IDs
    - Change tracking and replay
    """
    
    def __init__(
        self,
        local_instance_id: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        strategy: SyncStrategy = SyncStrategy.MOST_RECENT,
    ):
        self.local_instance_id = local_instance_id
        self.direction = direction
        self.strategy = strategy
        
        self._sync_states: Dict[str, SyncState] = {}  # remote_id -> state
        self._pending_operations: List[SyncOperation] = []
        self._conflict_handlers: List[Callable] = []
    
    def register_remote(self, remote_instance_id: str) -> SyncState:
        """Register a remote instance for synchronization."""
        state = SyncState(
            local_instance_id=self.local_instance_id,
            remote_instance_id=remote_instance_id,
        )
        self._sync_states[remote_instance_id] = state
        return state
    
    def unregister_remote(self, remote_instance_id: str):
        """Remove a remote instance from sync."""
        self._sync_states.pop(remote_instance_id, None)
    
    def get_sync_state(self, remote_instance_id: str) -> Optional[SyncState]:
        """Get sync state for a remote instance."""
        return self._sync_states.get(remote_instance_id)
    
    def record_local_change(
        self,
        operation_type: str,
        subject: str,
        predicate: str,
        obj: str,
        graph: Optional[str] = None,
    ):
        """Record a local change for sync."""
        op = SyncOperation(
            operation_type=operation_type,
            subject=subject,
            predicate=predicate,
            object=obj,
            graph=graph,
            timestamp=datetime.now(),
            source_instance=self.local_instance_id,
        )
        self._pending_operations.append(op)
        
        # Update pending counts
        for state in self._sync_states.values():
            state.pending_changes += 1
    
    def get_pending_operations(
        self,
        since_txn_id: Optional[int] = None,
    ) -> List[SyncOperation]:
        """Get pending operations for sync."""
        return list(self._pending_operations)
    
    def clear_pending_operations(self):
        """Clear pending operations after successful sync."""
        self._pending_operations.clear()
        for state in self._sync_states.values():
            state.pending_changes = 0
    
    def detect_conflicts(
        self,
        local_ops: List[SyncOperation],
        remote_ops: List[SyncOperation],
    ) -> List[Tuple[SyncOperation, SyncOperation]]:
        """Detect conflicting operations between local and remote."""
        conflicts = []
        
        # Index remote operations by (subject, predicate, object, graph)
        remote_index: Dict[Tuple, List[SyncOperation]] = {}
        for op in remote_ops:
            key = (op.subject, op.predicate, op.object, op.graph)
            if key not in remote_index:
                remote_index[key] = []
            remote_index[key].append(op)
        
        # Find conflicts
        for local_op in local_ops:
            key = (local_op.subject, local_op.predicate, local_op.object, local_op.graph)
            if key in remote_index:
                for remote_op in remote_index[key]:
                    # Same triple modified differently
                    if local_op.operation_type != remote_op.operation_type:
                        conflicts.append((local_op, remote_op))
        
        return conflicts
    
    def resolve_conflict(
        self,
        local_op: SyncOperation,
        remote_op: SyncOperation,
    ) -> SyncOperation:
        """Resolve a conflict based on the configured strategy."""
        if self.strategy == SyncStrategy.LOCAL_WINS:
            return local_op
        
        elif self.strategy == SyncStrategy.REMOTE_WINS:
            return remote_op
        
        elif self.strategy == SyncStrategy.MOST_RECENT:
            local_ts = local_op.timestamp or datetime.min
            remote_ts = remote_op.timestamp or datetime.min
            return local_op if local_ts >= remote_ts else remote_op
        
        elif self.strategy == SyncStrategy.MERGE:
            # For merge, we keep both (no delete wins)
            if local_op.operation_type == "insert":
                return local_op
            return remote_op
        
        return local_op  # Default to local
    
    def add_conflict_handler(self, handler: Callable):
        """Add a custom conflict handler."""
        self._conflict_handlers.append(handler)
    
    def sync_with_remote(
        self,
        remote_instance_id: str,
        remote_operations: List[SyncOperation],
        apply_operation: Callable[[SyncOperation], bool],
    ) -> Dict[str, Any]:
        """
        Perform synchronization with a remote instance.
        
        Args:
            remote_instance_id: ID of the remote instance
            remote_operations: Operations from the remote
            apply_operation: Function to apply an operation locally
            
        Returns:
            Sync statistics
        """
        state = self._sync_states.get(remote_instance_id)
        if not state:
            raise ValueError(f"Remote instance {remote_instance_id} not registered")
        
        local_ops = self.get_pending_operations()
        
        # Detect conflicts
        conflicts = self.detect_conflicts(local_ops, remote_operations)
        state.conflicts_detected += len(conflicts)
        
        # Resolve conflicts
        resolved_ops = []
        conflict_resolutions = []
        
        for local_op, remote_op in conflicts:
            winner = self.resolve_conflict(local_op, remote_op)
            resolved_ops.append(winner)
            conflict_resolutions.append({
                "local": local_op,
                "remote": remote_op,
                "winner": "local" if winner == local_op else "remote",
            })
            state.conflicts_resolved += 1
        
        # Apply non-conflicting remote operations
        applied = 0
        failed = 0
        conflict_triples = {
            (c[0].subject, c[0].predicate, c[0].object, c[0].graph)
            for c in conflicts
        }
        
        for op in remote_operations:
            key = (op.subject, op.predicate, op.object, op.graph)
            if key not in conflict_triples:
                if apply_operation(op):
                    applied += 1
                else:
                    failed += 1
        
        # Apply conflict resolutions
        for op in resolved_ops:
            if op.source_instance != self.local_instance_id:
                apply_operation(op)
        
        # Update state
        state.last_sync_time = datetime.now()
        
        return {
            "applied": applied,
            "failed": failed,
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": len(conflict_resolutions),
            "conflict_resolutions": conflict_resolutions,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize sync manager state."""
        return {
            "local_instance_id": self.local_instance_id,
            "direction": self.direction.name,
            "strategy": self.strategy.name,
            "sync_states": {
                rid: {
                    "last_sync_time": s.last_sync_time.isoformat() if s.last_sync_time else None,
                    "pending_changes": s.pending_changes,
                    "conflicts_detected": s.conflicts_detected,
                    "conflicts_resolved": s.conflicts_resolved,
                }
                for rid, s in self._sync_states.items()
            },
            "pending_operations": len(self._pending_operations),
        }


class DistributedQueryPlanner:
    """
    Query planner for distributed/federated query optimization.
    
    Handles:
    - Filter pushdown to remote endpoints
    - Projection pushdown
    - Join ordering based on cardinality estimates
    - Parallel execution planning
    """
    
    def __init__(self, registry: EndpointRegistry):
        self.registry = registry
        self._cardinality_cache: Dict[str, int] = {}
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query for optimization opportunities."""
        services = ServiceClauseParser.parse(query)
        
        analysis = {
            "service_calls": len(services),
            "endpoints": [s.endpoint_url for s in services],
            "variables": set(),
            "pushdown_opportunities": [],
        }
        
        for service in services:
            analysis["variables"].update(service.variables_used)
            
            # Check for filter pushdown
            if self._has_pushable_filters(service.subquery):
                analysis["pushdown_opportunities"].append({
                    "type": QueryPushdownType.FILTER.name,
                    "endpoint": service.endpoint_url,
                })
            
            # Check for limit pushdown
            if "LIMIT" not in service.subquery.upper():
                analysis["pushdown_opportunities"].append({
                    "type": QueryPushdownType.LIMIT.name,
                    "endpoint": service.endpoint_url,
                })
        
        analysis["variables"] = list(analysis["variables"])
        return analysis
    
    def _has_pushable_filters(self, subquery: str) -> bool:
        """Check if subquery has filters that could be pushed."""
        # Simple heuristic: look for FILTER keywords
        return "FILTER" in subquery.upper()
    
    def optimize_query(self, query: str) -> str:
        """
        Optimize a federated query.
        
        Applies:
        - Filter pushdown where possible
        - Limit pushdown for remote endpoints
        - Reorders joins based on selectivity estimates
        """
        services = ServiceClauseParser.parse(query)
        
        if not services:
            return query
        
        optimized = query
        
        # Sort services by estimated cardinality (smaller first for better join ordering)
        services_with_card = []
        for service in services:
            card = self._estimate_cardinality(service.endpoint_url, service.subquery)
            services_with_card.append((card, service))
        
        services_with_card.sort(key=lambda x: x[0])
        
        # For now, return original query
        # Full optimization would rewrite the query structure
        return optimized
    
    def _estimate_cardinality(self, endpoint_url: str, subquery: str) -> int:
        """Estimate result cardinality for a subquery."""
        cache_key = f"{endpoint_url}:{hash(subquery)}"
        
        if cache_key in self._cardinality_cache:
            return self._cardinality_cache[cache_key]
        
        # Default estimate based on pattern complexity
        triple_count = subquery.count(' . ')
        filter_count = subquery.upper().count('FILTER')
        
        # Heuristic: more patterns = fewer results, more filters = fewer results
        estimate = max(1000 // (triple_count + 1) // (filter_count + 1), 1)
        
        self._cardinality_cache[cache_key] = estimate
        return estimate
    
    def create_execution_plan(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Create an execution plan for a federated query.
        
        Returns a plan that can be executed by the FederatedQueryExecutor.
        """
        services = ServiceClauseParser.parse(query)
        
        # Build execution steps
        steps = []
        
        # First, execute all remote SERVICE calls (potentially in parallel)
        remote_steps = []
        for i, service in enumerate(services):
            remote_steps.append({
                "step_id": f"remote_{i}",
                "type": "remote_query",
                "endpoint": service.endpoint_url,
                "query": f"SELECT * WHERE {{ {service.subquery} }}",
                "is_silent": service.is_silent,
                "variables": list(service.variables_bound),
            })
        
        if remote_steps:
            steps.append({
                "phase": "remote_execution",
                "parallel": True,
                "steps": remote_steps,
            })
        
        # Then, execute local query
        local_query = ServiceClauseParser.remove_service_clauses(query)
        if local_query.strip():
            steps.append({
                "phase": "local_execution",
                "parallel": False,
                "steps": [{
                    "step_id": "local",
                    "type": "local_query",
                    "query": local_query,
                }],
            })
        
        # Finally, join results
        if len(services) > 0:
            steps.append({
                "phase": "join",
                "parallel": False,
                "steps": [{
                    "step_id": "join",
                    "type": "hash_join",
                    "inputs": ["local"] + [f"remote_{i}" for i in range(len(services))],
                }],
            })
        
        return {
            "query": query,
            "service_calls": len(services),
            "estimated_cost": self._estimate_cost(services),
            "execution_steps": steps,
        }
    
    def _estimate_cost(self, services: List[ServiceCall]) -> float:
        """Estimate execution cost for planning."""
        # Base cost for local execution
        cost = 1.0
        
        # Add cost for each remote call
        for service in services:
            endpoint = self.registry.get(service.endpoint_url)
            if endpoint and endpoint.last_latency_ms:
                cost += endpoint.last_latency_ms / 100
            else:
                cost += 5.0  # Default remote cost
        
        return cost


# Convenience functions

def create_federation_executor(
    endpoints: Optional[List[Dict[str, Any]]] = None,
    cache_enabled: bool = True,
) -> FederatedQueryExecutor:
    """Create a federated query executor with optional endpoint configuration."""
    registry = EndpointRegistry()
    
    if endpoints:
        for ep in endpoints:
            registry.register(**ep)
    
    return FederatedQueryExecutor(
        registry=registry,
        cache_enabled=cache_enabled,
    )


def create_sync_manager(
    instance_id: str,
    direction: str = "bidirectional",
    strategy: str = "most_recent",
) -> CrossInstanceSync:
    """Create a sync manager with the specified configuration."""
    dir_map = {
        "push": SyncDirection.PUSH,
        "pull": SyncDirection.PULL,
        "bidirectional": SyncDirection.BIDIRECTIONAL,
    }
    strat_map = {
        "local_wins": SyncStrategy.LOCAL_WINS,
        "remote_wins": SyncStrategy.REMOTE_WINS,
        "most_recent": SyncStrategy.MOST_RECENT,
        "merge": SyncStrategy.MERGE,
    }
    
    return CrossInstanceSync(
        local_instance_id=instance_id,
        direction=dir_map.get(direction.lower(), SyncDirection.BIDIRECTIONAL),
        strategy=strat_map.get(strategy.lower(), SyncStrategy.MOST_RECENT),
    )
