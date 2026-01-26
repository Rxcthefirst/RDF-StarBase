"""
Slow Query Log and Result Export for RDF-StarBase.

Provides:
- Slow query logging with EXPLAIN plans
- Query result export (CSV, JSON, Parquet)
- Cursor-based pagination
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Slow Query Log
# ============================================================================

@dataclass
class SlowQueryEntry:
    """A slow query log entry."""
    query_id: str
    query: str
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    repository: str = ""
    user: Optional[str] = None
    result_count: int = 0
    explain_plan: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "repository": self.repository,
            "user": self.user,
            "result_count": self.result_count,
            "explain_plan": self.explain_plan,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlowQueryEntry":
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            duration_seconds=data["duration_seconds"],
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            repository=data.get("repository", ""),
            user=data.get("user"),
            result_count=data.get("result_count", 0),
            explain_plan=data.get("explain_plan"),
            parameters=data.get("parameters", {})
        )


class SlowQueryLog:
    """
    Logs queries that exceed a configurable threshold.
    
    Features:
    - Configurable threshold
    - EXPLAIN plan capture
    - Log rotation
    - Query analysis
    """
    
    def __init__(
        self,
        threshold_seconds: float = 1.0,
        max_entries: int = 1000,
        log_file: Optional[Path] = None,
        capture_explain: bool = True
    ):
        self.threshold_seconds = threshold_seconds
        self.max_entries = max_entries
        self.log_file = Path(log_file) if log_file else None
        self.capture_explain = capture_explain
        
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = threading.RLock()
        self._explain_provider: Optional[Callable[[str], str]] = None
        
        if self.log_file:
            self._load_entries()
    
    def _load_entries(self) -> None:
        """Load entries from log file."""
        if not self.log_file or not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry_data in data.get("entries", []):
                    self._entries.append(SlowQueryEntry.from_dict(entry_data))
        except Exception as e:
            logger.warning(f"Error loading slow query log: {e}")
    
    def _save_entries(self) -> None:
        """Save entries to log file."""
        if not self.log_file:
            return
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [e.to_dict() for e in self._entries],
                "threshold_seconds": self.threshold_seconds,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving slow query log: {e}")
    
    def set_explain_provider(
        self,
        provider: Callable[[str], str]
    ) -> None:
        """Set a function to generate EXPLAIN plans."""
        self._explain_provider = provider
    
    def log_query(
        self,
        query: str,
        duration_seconds: float,
        started_at: datetime,
        repository: str = "",
        user: Optional[str] = None,
        result_count: int = 0,
        parameters: Dict[str, Any] = None
    ) -> Optional[SlowQueryEntry]:
        """
        Log a query if it exceeds the threshold.
        
        Returns the entry if logged, None otherwise.
        """
        if duration_seconds < self.threshold_seconds:
            return None
        
        import uuid
        query_id = str(uuid.uuid4())[:8]
        
        # Capture EXPLAIN plan if provider is set
        explain_plan = None
        if self.capture_explain and self._explain_provider:
            try:
                explain_plan = self._explain_provider(query)
            except Exception as e:
                logger.warning(f"Error capturing EXPLAIN: {e}")
        
        entry = SlowQueryEntry(
            query_id=query_id,
            query=query,
            duration_seconds=duration_seconds,
            started_at=started_at,
            finished_at=started_at + __import__('datetime').timedelta(seconds=duration_seconds),
            repository=repository,
            user=user,
            result_count=result_count,
            explain_plan=explain_plan,
            parameters=parameters or {}
        )
        
        with self._lock:
            self._entries.append(entry)
            self._save_entries()
        
        logger.warning(
            f"Slow query [{query_id}]: {duration_seconds:.2f}s - {query[:100]}..."
        )
        
        return entry
    
    def get_entries(
        self,
        limit: int = 100,
        min_duration: Optional[float] = None,
        repository: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[SlowQueryEntry]:
        """Get slow query entries with optional filtering."""
        with self._lock:
            entries = list(self._entries)
        
        # Apply filters
        if min_duration is not None:
            entries = [e for e in entries if e.duration_seconds >= min_duration]
        
        if repository:
            entries = [e for e in entries if e.repository == repository]
        
        if since:
            entries = [e for e in entries if e.started_at >= since]
        
        # Sort by duration descending
        entries.sort(key=lambda e: e.duration_seconds, reverse=True)
        
        return entries[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about slow queries."""
        with self._lock:
            entries = list(self._entries)
        
        if not entries:
            return {
                "count": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "threshold": self.threshold_seconds
            }
        
        durations = [e.duration_seconds for e in entries]
        
        return {
            "count": len(entries),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_duration": sum(durations),
            "threshold": self.threshold_seconds,
            "repositories": list(set(e.repository for e in entries if e.repository))
        }
    
    def clear(self) -> int:
        """Clear the log. Returns number of entries cleared."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._save_entries()
            return count
    
    def set_threshold(self, seconds: float) -> None:
        """Update the slow query threshold."""
        self.threshold_seconds = seconds


# ============================================================================
# Query Timer Context Manager
# ============================================================================

class QueryTimer:
    """Context manager for timing queries with automatic slow query logging."""
    
    def __init__(
        self,
        slow_log: SlowQueryLog,
        query: str,
        repository: str = "",
        user: Optional[str] = None
    ):
        self.slow_log = slow_log
        self.query = query
        self.repository = repository
        self.user = user
        self.start_time = None
        self.result_count = 0
    
    def __enter__(self) -> "QueryTimer":
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        self.slow_log.log_query(
            query=self.query,
            duration_seconds=duration,
            started_at=self.start_time,
            repository=self.repository,
            user=self.user,
            result_count=self.result_count
        )
    
    def set_result_count(self, count: int) -> None:
        """Set the result count for logging."""
        self.result_count = count


# ============================================================================
# Result Export
# ============================================================================

class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    PARQUET = "parquet"
    TSV = "tsv"


@dataclass
class ExportOptions:
    """Options for result export."""
    format: ExportFormat = ExportFormat.CSV
    include_headers: bool = True
    pretty_print: bool = False
    encoding: str = "utf-8"
    delimiter: str = ","
    null_value: str = ""
    datetime_format: str = "%Y-%m-%dT%H:%M:%S"
    max_string_length: Optional[int] = None


class ResultExporter:
    """
    Exports query results to various formats.
    
    Supports:
    - CSV with headers
    - JSON (array or object)
    - JSON Lines (streaming)
    - Parquet (requires pyarrow)
    - TSV (tab-separated)
    """
    
    def __init__(self, options: Optional[ExportOptions] = None):
        self.options = options or ExportOptions()
    
    def export(
        self,
        results: List[Dict[str, Any]],
        format: Optional[ExportFormat] = None
    ) -> bytes:
        """
        Export results to bytes.
        
        Args:
            results: List of result dictionaries (bindings)
            format: Override format (uses options.format if None)
        
        Returns:
            Exported data as bytes
        """
        fmt = format or self.options.format
        
        if fmt == ExportFormat.CSV:
            return self._export_csv(results)
        elif fmt == ExportFormat.TSV:
            return self._export_tsv(results)
        elif fmt == ExportFormat.JSON:
            return self._export_json(results)
        elif fmt == ExportFormat.JSONL:
            return self._export_jsonl(results)
        elif fmt == ExportFormat.PARQUET:
            return self._export_parquet(results)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    
    def export_to_file(
        self,
        results: List[Dict[str, Any]],
        path: Path,
        format: Optional[ExportFormat] = None
    ) -> int:
        """
        Export results to a file.
        
        Returns number of rows exported.
        """
        data = self.export(results, format)
        
        mode = "wb"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, mode) as f:
            f.write(data)
        
        return len(results)
    
    def _get_columns(self, results: List[Dict[str, Any]]) -> List[str]:
        """Get ordered list of all columns."""
        columns = []
        seen = set()
        
        for row in results:
            for key in row.keys():
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
        
        return columns
    
    def _format_value(self, value: Any) -> str:
        """Format a value for export."""
        if value is None:
            return self.options.null_value
        
        if isinstance(value, datetime):
            return value.strftime(self.options.datetime_format)
        
        result = str(value)
        
        if self.options.max_string_length and len(result) > self.options.max_string_length:
            result = result[:self.options.max_string_length] + "..."
        
        return result
    
    def _export_csv(self, results: List[Dict[str, Any]]) -> bytes:
        """Export to CSV format."""
        output = io.StringIO()
        columns = self._get_columns(results)
        
        writer = csv.writer(
            output,
            delimiter=self.options.delimiter,
            quoting=csv.QUOTE_MINIMAL
        )
        
        if self.options.include_headers:
            writer.writerow(columns)
        
        for row in results:
            writer.writerow([
                self._format_value(row.get(col))
                for col in columns
            ])
        
        return output.getvalue().encode(self.options.encoding)
    
    def _export_tsv(self, results: List[Dict[str, Any]]) -> bytes:
        """Export to TSV format."""
        original_delimiter = self.options.delimiter
        self.options.delimiter = "\t"
        try:
            return self._export_csv(results)
        finally:
            self.options.delimiter = original_delimiter
    
    def _export_json(self, results: List[Dict[str, Any]]) -> bytes:
        """Export to JSON format."""
        # Convert datetime objects
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.strftime(self.options.datetime_format)
            return str(obj)
        
        indent = 2 if self.options.pretty_print else None
        
        data = json.dumps(
            results,
            default=serialize,
            indent=indent,
            ensure_ascii=False
        )
        
        return data.encode(self.options.encoding)
    
    def _export_jsonl(self, results: List[Dict[str, Any]]) -> bytes:
        """Export to JSON Lines format."""
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.strftime(self.options.datetime_format)
            return str(obj)
        
        lines = []
        for row in results:
            lines.append(json.dumps(row, default=serialize, ensure_ascii=False))
        
        return "\n".join(lines).encode(self.options.encoding)
    
    def _export_parquet(self, results: List[Dict[str, Any]]) -> bytes:
        """Export to Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("Parquet export requires pyarrow: pip install pyarrow")
        
        if not results:
            # Empty table
            return b""
        
        columns = self._get_columns(results)
        
        # Build column data
        column_data = {col: [] for col in columns}
        for row in results:
            for col in columns:
                column_data[col].append(row.get(col))
        
        # Create PyArrow table
        table = pa.table(column_data)
        
        # Write to bytes
        output = io.BytesIO()
        pq.write_table(table, output)
        return output.getvalue()


# ============================================================================
# Cursor-based Pagination
# ============================================================================

@dataclass
class CursorPage:
    """A page of results with cursor information."""
    results: List[Dict[str, Any]]
    cursor: Optional[str] = None
    next_cursor: Optional[str] = None
    has_more: bool = False
    total_count: Optional[int] = None
    page_size: int = 0


class CursorPaginator:
    """
    Cursor-based pagination for query results.
    
    More efficient than OFFSET/LIMIT for large datasets.
    Uses opaque cursors that encode position.
    """
    
    def __init__(
        self,
        default_page_size: int = 100,
        max_page_size: int = 1000
    ):
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
        self._cursors: Dict[str, Any] = {}
    
    def paginate(
        self,
        results: Iterator[Dict[str, Any]],
        page_size: Optional[int] = None,
        cursor: Optional[str] = None,
        order_by: Optional[str] = None
    ) -> CursorPage:
        """
        Paginate results using cursor.
        
        Args:
            results: Iterator of result rows
            page_size: Number of results per page
            cursor: Cursor from previous page
            order_by: Column to order by for stable pagination
        
        Returns:
            CursorPage with results and next cursor
        """
        size = min(page_size or self.default_page_size, self.max_page_size)
        
        # Decode cursor to get starting position
        start_pos = 0
        if cursor:
            start_pos = self._decode_cursor(cursor)
        
        # Collect results
        page_results = []
        current_pos = 0
        has_more = False
        
        for row in results:
            if current_pos < start_pos:
                current_pos += 1
                continue
            
            if len(page_results) >= size:
                has_more = True
                break
            
            page_results.append(row)
            current_pos += 1
        
        # Create next cursor
        next_cursor = None
        if has_more:
            next_cursor = self._encode_cursor(current_pos)
        
        return CursorPage(
            results=page_results,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=has_more,
            page_size=len(page_results)
        )
    
    def paginate_list(
        self,
        results: List[Dict[str, Any]],
        page_size: Optional[int] = None,
        cursor: Optional[str] = None
    ) -> CursorPage:
        """Paginate a list of results."""
        size = min(page_size or self.default_page_size, self.max_page_size)
        
        start_pos = 0
        if cursor:
            start_pos = self._decode_cursor(cursor)
        
        end_pos = start_pos + size
        page_results = results[start_pos:end_pos]
        
        has_more = end_pos < len(results)
        next_cursor = self._encode_cursor(end_pos) if has_more else None
        
        return CursorPage(
            results=page_results,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=has_more,
            total_count=len(results),
            page_size=len(page_results)
        )
    
    def _encode_cursor(self, position: int) -> str:
        """Encode position as cursor string."""
        import base64
        data = f"pos:{position}".encode()
        return base64.urlsafe_b64encode(data).decode()
    
    def _decode_cursor(self, cursor: str) -> int:
        """Decode cursor string to position."""
        try:
            import base64
            data = base64.urlsafe_b64decode(cursor.encode()).decode()
            if data.startswith("pos:"):
                return int(data[4:])
        except:
            pass
        return 0


# ============================================================================
# Convenience Functions
# ============================================================================

def create_slow_log(
    threshold_seconds: float = 1.0,
    log_file: Optional[Path] = None
) -> SlowQueryLog:
    """Create a slow query log."""
    return SlowQueryLog(threshold_seconds=threshold_seconds, log_file=log_file)


def export_results(
    results: List[Dict[str, Any]],
    format: str = "csv"
) -> bytes:
    """Export results to the specified format."""
    fmt = ExportFormat(format.lower())
    exporter = ResultExporter(ExportOptions(format=fmt))
    return exporter.export(results)


def paginate_results(
    results: List[Dict[str, Any]],
    page_size: int = 100,
    cursor: Optional[str] = None
) -> CursorPage:
    """Paginate a list of results."""
    paginator = CursorPaginator(default_page_size=page_size)
    return paginator.paginate_list(results, page_size, cursor)
