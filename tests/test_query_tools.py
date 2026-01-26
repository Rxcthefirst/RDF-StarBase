"""Tests for slow query log, result export, and pagination."""
import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from rdf_starbase.storage.query_tools import (
    SlowQueryEntry,
    SlowQueryLog,
    QueryTimer,
    ExportFormat,
    ExportOptions,
    ResultExporter,
    CursorPage,
    CursorPaginator,
    create_slow_log,
    export_results,
    paginate_results,
)


# ========== SlowQueryEntry Tests ==========

class TestSlowQueryEntry:
    def test_creation(self):
        entry = SlowQueryEntry(
            query_id="q1",
            query="SELECT * WHERE { ?s ?p ?o }",
            duration_seconds=2.5,
            started_at=datetime.now(),
            finished_at=datetime.now()
        )
        assert entry.query_id == "q1"
        assert entry.duration_seconds == 2.5
    
    def test_to_dict(self):
        entry = SlowQueryEntry(
            query_id="q1",
            query="SELECT ?s WHERE { ?s ?p ?o }",
            duration_seconds=1.5,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            repository="test-repo",
            result_count=100
        )
        d = entry.to_dict()
        assert d["query_id"] == "q1"
        assert d["duration_seconds"] == 1.5
        assert d["result_count"] == 100
    
    def test_from_dict(self):
        data = {
            "query_id": "q2",
            "query": "SELECT * WHERE { ?s ?p ?o }",
            "duration_seconds": 3.0,
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "explain_plan": "Scan all triples"
        }
        entry = SlowQueryEntry.from_dict(data)
        assert entry.query_id == "q2"
        assert entry.explain_plan == "Scan all triples"


# ========== SlowQueryLog Tests ==========

class TestSlowQueryLog:
    def test_creation(self):
        log = SlowQueryLog(threshold_seconds=2.0)
        assert log.threshold_seconds == 2.0
    
    def test_log_fast_query(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        result = log.log_query(
            query="SELECT * WHERE { ?s ?p ?o }",
            duration_seconds=0.5,
            started_at=datetime.now()
        )
        assert result is None
    
    def test_log_slow_query(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        result = log.log_query(
            query="SELECT * WHERE { ?s ?p ?o }",
            duration_seconds=1.5,
            started_at=datetime.now(),
            repository="test",
            result_count=1000
        )
        assert result is not None
        assert result.duration_seconds == 1.5
    
    def test_get_entries(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.log_query("q1", 1.5, datetime.now())
        log.log_query("q2", 2.5, datetime.now())
        log.log_query("q3", 3.5, datetime.now())
        
        entries = log.get_entries()
        assert len(entries) == 3
        # Sorted by duration descending
        assert entries[0].duration_seconds == 3.5
    
    def test_filter_by_duration(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.log_query("q1", 1.5, datetime.now())
        log.log_query("q2", 3.0, datetime.now())
        
        entries = log.get_entries(min_duration=2.0)
        assert len(entries) == 1
        assert entries[0].duration_seconds == 3.0
    
    def test_filter_by_repository(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.log_query("q1", 1.5, datetime.now(), repository="repo1")
        log.log_query("q2", 2.0, datetime.now(), repository="repo2")
        
        entries = log.get_entries(repository="repo1")
        assert len(entries) == 1
    
    def test_get_statistics(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.log_query("q1", 1.0, datetime.now())
        log.log_query("q2", 2.0, datetime.now())
        log.log_query("q3", 3.0, datetime.now())
        
        stats = log.get_statistics()
        assert stats["count"] == 3
        assert stats["avg_duration"] == 2.0
        assert stats["max_duration"] == 3.0
    
    def test_empty_statistics(self):
        log = SlowQueryLog()
        stats = log.get_statistics()
        assert stats["count"] == 0
    
    def test_clear(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.log_query("q1", 2.0, datetime.now())
        log.log_query("q2", 2.0, datetime.now())
        
        cleared = log.clear()
        assert cleared == 2
        assert len(log.get_entries()) == 0
    
    def test_set_threshold(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        log.set_threshold(0.5)
        assert log.threshold_seconds == 0.5
    
    def test_max_entries(self):
        log = SlowQueryLog(threshold_seconds=0.0, max_entries=5)
        for i in range(10):
            log.log_query(f"q{i}", 1.0, datetime.now())
        
        entries = log.get_entries(limit=10)
        assert len(entries) == 5
    
    def test_explain_provider(self):
        log = SlowQueryLog(threshold_seconds=1.0, capture_explain=True)
        log.set_explain_provider(lambda q: f"EXPLAIN: {q[:20]}")
        
        result = log.log_query("SELECT * WHERE { ?s ?p ?o }", 2.0, datetime.now())
        assert result.explain_plan.startswith("EXPLAIN:")
    
    def test_persistence(self, tmp_path):
        log_file = tmp_path / "slow_queries.json"
        
        log = SlowQueryLog(threshold_seconds=1.0, log_file=log_file)
        log.log_query("q1", 2.0, datetime.now())
        
        # Create new log - should load from file
        log2 = SlowQueryLog(threshold_seconds=1.0, log_file=log_file)
        assert len(log2.get_entries()) == 1


# ========== QueryTimer Tests ==========

class TestQueryTimer:
    def test_timer_fast_query(self):
        log = SlowQueryLog(threshold_seconds=1.0)
        
        with QueryTimer(log, "SELECT * WHERE { ?s ?p ?o }") as timer:
            pass  # Fast query
        
        assert len(log.get_entries()) == 0
    
    def test_timer_slow_query(self):
        log = SlowQueryLog(threshold_seconds=0.05)
        
        with QueryTimer(log, "SELECT * WHERE { ?s ?p ?o }") as timer:
            time.sleep(0.1)
        
        entries = log.get_entries()
        assert len(entries) == 1
    
    def test_timer_with_result_count(self):
        log = SlowQueryLog(threshold_seconds=0.0)
        
        with QueryTimer(log, "query") as timer:
            timer.set_result_count(100)
        
        entries = log.get_entries()
        assert entries[0].result_count == 100


# ========== ExportFormat Tests ==========

class TestExportFormat:
    def test_formats(self):
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.PARQUET.value == "parquet"


# ========== ExportOptions Tests ==========

class TestExportOptions:
    def test_defaults(self):
        options = ExportOptions()
        assert options.format == ExportFormat.CSV
        assert options.include_headers is True
    
    def test_custom_options(self):
        options = ExportOptions(
            format=ExportFormat.JSON,
            pretty_print=True,
            max_string_length=100
        )
        assert options.pretty_print is True
        assert options.max_string_length == 100


# ========== ResultExporter Tests ==========

class TestResultExporter:
    @pytest.fixture
    def sample_results(self):
        return [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "Chicago"}
        ]
    
    def test_export_csv(self, sample_results):
        exporter = ResultExporter()
        data = exporter.export(sample_results, ExportFormat.CSV)
        
        assert b"name,age,city" in data
        assert b"Alice" in data
        assert b"Bob" in data
    
    def test_export_csv_no_headers(self, sample_results):
        exporter = ResultExporter(ExportOptions(include_headers=False))
        data = exporter.export(sample_results, ExportFormat.CSV)
        
        lines = data.decode().strip().split("\n")
        assert len(lines) == 3  # No header row
    
    def test_export_tsv(self, sample_results):
        exporter = ResultExporter()
        data = exporter.export(sample_results, ExportFormat.TSV)
        
        assert b"\t" in data
    
    def test_export_json(self, sample_results):
        exporter = ResultExporter()
        data = exporter.export(sample_results, ExportFormat.JSON)
        
        parsed = json.loads(data.decode())
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Alice"
    
    def test_export_json_pretty(self, sample_results):
        exporter = ResultExporter(ExportOptions(pretty_print=True))
        data = exporter.export(sample_results, ExportFormat.JSON)
        
        assert b"\n" in data  # Pretty printed has newlines
    
    def test_export_jsonl(self, sample_results):
        exporter = ResultExporter()
        data = exporter.export(sample_results, ExportFormat.JSONL)
        
        lines = data.decode().strip().split("\n")
        assert len(lines) == 3
        
        # Each line is valid JSON
        for line in lines:
            json.loads(line)
    
    def test_export_empty(self):
        exporter = ResultExporter()
        data = exporter.export([], ExportFormat.JSON)
        
        parsed = json.loads(data.decode())
        assert parsed == []
    
    def test_export_to_file(self, sample_results, tmp_path):
        exporter = ResultExporter()
        path = tmp_path / "results.csv"
        
        count = exporter.export_to_file(sample_results, path)
        assert count == 3
        assert path.exists()
    
    def test_format_datetime(self):
        results = [{"timestamp": datetime(2024, 1, 15, 10, 30, 0)}]
        exporter = ResultExporter()
        data = exporter.export(results, ExportFormat.JSON)
        
        parsed = json.loads(data.decode())
        assert "2024-01-15" in parsed[0]["timestamp"]
    
    def test_null_values(self):
        results = [{"name": "Alice", "value": None}]
        exporter = ResultExporter(ExportOptions(null_value="NULL"))
        data = exporter.export(results, ExportFormat.CSV)
        
        assert b"NULL" in data
    
    def test_max_string_length(self):
        results = [{"text": "A" * 1000}]
        exporter = ResultExporter(ExportOptions(max_string_length=10))
        data = exporter.export(results, ExportFormat.CSV)
        
        assert b"AAAAAAAAAA..." in data


# ========== CursorPage Tests ==========

class TestCursorPage:
    def test_creation(self):
        page = CursorPage(
            results=[{"a": 1}, {"a": 2}],
            cursor="abc",
            next_cursor="def",
            has_more=True,
            page_size=2
        )
        assert len(page.results) == 2
        assert page.has_more is True


# ========== CursorPaginator Tests ==========

class TestCursorPaginator:
    @pytest.fixture
    def sample_data(self):
        return [{"id": i} for i in range(100)]
    
    def test_first_page(self, sample_data):
        paginator = CursorPaginator(default_page_size=10)
        page = paginator.paginate_list(sample_data)
        
        assert len(page.results) == 10
        assert page.has_more is True
        assert page.next_cursor is not None
        assert page.total_count == 100
    
    def test_with_cursor(self, sample_data):
        paginator = CursorPaginator(default_page_size=10)
        
        # Get first page
        page1 = paginator.paginate_list(sample_data)
        
        # Get second page
        page2 = paginator.paginate_list(sample_data, cursor=page1.next_cursor)
        
        assert page2.results[0]["id"] == 10
    
    def test_last_page(self, sample_data):
        paginator = CursorPaginator(default_page_size=50)
        
        page1 = paginator.paginate_list(sample_data)
        page2 = paginator.paginate_list(sample_data, cursor=page1.next_cursor)
        
        assert page2.has_more is False
        assert page2.next_cursor is None
    
    def test_custom_page_size(self, sample_data):
        paginator = CursorPaginator()
        page = paginator.paginate_list(sample_data, page_size=25)
        
        assert len(page.results) == 25
    
    def test_max_page_size(self, sample_data):
        paginator = CursorPaginator(max_page_size=20)
        page = paginator.paginate_list(sample_data, page_size=50)
        
        assert len(page.results) == 20  # Capped at max
    
    def test_iterator_pagination(self):
        def generate_results():
            for i in range(50):
                yield {"id": i}
        
        paginator = CursorPaginator(default_page_size=10)
        page = paginator.paginate(generate_results())
        
        assert len(page.results) == 10
        assert page.has_more is True
    
    def test_empty_results(self):
        paginator = CursorPaginator()
        page = paginator.paginate_list([])
        
        assert len(page.results) == 0
        assert page.has_more is False


# ========== Convenience Functions Tests ==========

class TestConvenienceFunctions:
    def test_create_slow_log(self):
        log = create_slow_log(threshold_seconds=2.0)
        assert isinstance(log, SlowQueryLog)
        assert log.threshold_seconds == 2.0
    
    def test_export_results(self):
        results = [{"a": 1}, {"b": 2}]
        data = export_results(results, "json")
        
        parsed = json.loads(data.decode())
        assert len(parsed) == 2
    
    def test_paginate_results(self):
        results = [{"id": i} for i in range(50)]
        page = paginate_results(results, page_size=10)
        
        assert len(page.results) == 10
        assert page.has_more is True
