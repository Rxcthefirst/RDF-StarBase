"""Tests for memory budget enforcement module."""

import gc
import pickle
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from rdf_starbase.storage.memory_budget import (
    BudgetedStore,
    MemoryBudget,
    MemoryGuard,
    MemoryPressure,
    MemoryStats,
    SpillRecord,
    TrackedObject,
    configure_budget,
    enforce,
    estimate_size,
    get_budget,
    track,
    untrack,
)


class TestMemoryPressure:
    """Tests for memory pressure levels."""
    
    def test_pressure_levels(self):
        """Test that pressure levels are properly ordered."""
        assert MemoryPressure.LOW.value < MemoryPressure.MODERATE.value
        assert MemoryPressure.MODERATE.value < MemoryPressure.HIGH.value
        assert MemoryPressure.HIGH.value < MemoryPressure.CRITICAL.value


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""
    
    def test_usage_ratio(self):
        """Test usage ratio calculation."""
        stats = MemoryStats(
            budget_bytes=1000,
            used_bytes=250,
            available_bytes=750,
            pressure=MemoryPressure.LOW,
            tracked_objects=5,
            spilled_objects=0,
        )
        assert stats.usage_ratio == 0.25
    
    def test_usage_percent(self):
        """Test usage percent calculation."""
        stats = MemoryStats(
            budget_bytes=1000,
            used_bytes=750,
            available_bytes=250,
            pressure=MemoryPressure.HIGH,
            tracked_objects=10,
            spilled_objects=2,
        )
        assert stats.usage_percent == 75.0
    
    def test_zero_budget(self):
        """Test handling of zero budget."""
        stats = MemoryStats(
            budget_bytes=0,
            used_bytes=0,
            available_bytes=0,
            pressure=MemoryPressure.LOW,
            tracked_objects=0,
            spilled_objects=0,
        )
        assert stats.usage_ratio == 0.0


class TestEstimateSize:
    """Tests for size estimation function."""
    
    def test_estimate_polars_dataframe(self):
        """Test size estimation of Polars DataFrame."""
        df = pl.DataFrame({
            "a": range(1000),
            "b": ["hello"] * 1000,
        })
        size = estimate_size(df)
        assert size > 0
        # Should be close to DataFrame's own estimate
        assert size == df.estimated_size()
    
    def test_estimate_dict(self):
        """Test size estimation of dictionary."""
        d = {"key": "value", "num": 123, "nested": {"a": 1}}
        size = estimate_size(d)
        assert size > 0
    
    def test_estimate_list(self):
        """Test size estimation of list."""
        lst = [1, 2, 3, "hello", {"key": "value"}]
        size = estimate_size(lst)
        assert size > 0
    
    def test_estimate_tuple(self):
        """Test size estimation of tuple."""
        t = (1, 2, 3, "hello")
        size = estimate_size(t)
        assert size > 0
    
    def test_estimate_set(self):
        """Test size estimation of set."""
        s = {1, 2, 3, 4, 5}
        size = estimate_size(s)
        assert size > 0
    
    def test_estimate_string(self):
        """Test size estimation of string."""
        s = "Hello, World!" * 100
        size = estimate_size(s)
        assert size > 0


class TestMemoryBudget:
    """Tests for MemoryBudget class."""
    
    @pytest.fixture
    def budget(self, tmp_path):
        """Create a memory budget for testing."""
        return MemoryBudget(
            max_bytes=10000,
            spill_dir=tmp_path / "spill",
            spill_threshold=0.8,
            critical_threshold=0.95,
        )
    
    def test_init(self, budget):
        """Test budget initialization."""
        assert budget.max_bytes == 10000
        assert budget.used_bytes == 0
        assert budget.available_bytes == 10000
    
    def test_track_object(self, budget):
        """Test tracking an object."""
        obj = [1, 2, 3]
        budget.track("test_obj", obj, size_bytes=100)
        
        assert budget.used_bytes == 100
        assert "test_obj" in budget._tracked
    
    def test_track_auto_size(self, budget):
        """Test tracking with automatic size estimation."""
        obj = {"key": "value"}
        budget.track("test_obj", obj)
        
        assert budget.used_bytes > 0
    
    def test_track_replace(self, budget):
        """Test replacing a tracked object."""
        budget.track("obj", [1, 2, 3], size_bytes=100)
        budget.track("obj", [4, 5, 6], size_bytes=200)
        
        assert budget.used_bytes == 200
    
    def test_untrack(self, budget):
        """Test untracking an object."""
        obj = [1, 2, 3]
        budget.track("obj", obj, size_bytes=100)
        
        result = budget.untrack("obj")
        assert result is obj
        assert budget.used_bytes == 0
    
    def test_untrack_missing(self, budget):
        """Test untracking non-existent object."""
        result = budget.untrack("missing")
        assert result is None
    
    def test_get_tracked(self, budget):
        """Test getting a tracked object."""
        obj = {"data": 123}
        budget.track("obj", obj, size_bytes=100)
        
        result = budget.get("obj")
        assert result is obj
    
    def test_get_updates_access_time(self, budget):
        """Test that get updates access time."""
        obj = [1, 2, 3]
        budget.track("obj", obj, size_bytes=100)
        
        old_time = budget._tracked["obj"].last_access
        time.sleep(0.01)
        budget.get("obj")
        new_time = budget._tracked["obj"].last_access
        
        assert new_time > old_time
    
    def test_get_missing(self, budget):
        """Test getting non-existent object."""
        result = budget.get("missing")
        assert result is None
    
    def test_update_size(self, budget):
        """Test updating tracked size."""
        budget.track("obj", [1, 2, 3], size_bytes=100)
        budget.update_size("obj", 500)
        
        assert budget.used_bytes == 500
    
    def test_pressure_levels(self, budget):
        """Test memory pressure calculation."""
        # Low pressure
        assert budget.get_pressure() == MemoryPressure.LOW
        
        # Moderate (50-75%)
        budget.track("obj1", [1], size_bytes=6000)
        assert budget.get_pressure() == MemoryPressure.MODERATE
        
        # High (75-90%)
        budget.update_size("obj1", 8000)
        assert budget.get_pressure() == MemoryPressure.HIGH
        
        # Critical (>90%)
        budget.update_size("obj1", 9500)
        assert budget.get_pressure() == MemoryPressure.CRITICAL
    
    def test_should_spill(self, budget):
        """Test spill threshold detection."""
        assert not budget.should_spill()
        
        budget.track("obj", [1], size_bytes=8500)  # 85% > 80%
        assert budget.should_spill()
    
    def test_is_critical(self, budget):
        """Test critical threshold detection."""
        assert not budget.is_critical()
        
        budget.track("obj", [1], size_bytes=9600)  # 96% > 95%
        assert budget.is_critical()
    
    def test_get_stats(self, budget):
        """Test getting statistics."""
        budget.track("obj", [1, 2, 3], size_bytes=500)
        
        stats = budget.get_stats()
        assert stats.budget_bytes == 10000
        assert stats.used_bytes == 500
        assert stats.available_bytes == 9500
        assert stats.tracked_objects == 1
        assert stats.spilled_objects == 0
    
    def test_pin_object(self, budget):
        """Test pinning an object."""
        budget.track("obj", [1], size_bytes=100)
        
        assert not budget._tracked["obj"].pinned
        budget.pin("obj")
        assert budget._tracked["obj"].pinned
    
    def test_unpin_object(self, budget):
        """Test unpinning an object."""
        budget.track("obj", [1], size_bytes=100, pinned=True)
        
        assert budget._tracked["obj"].pinned
        budget.unpin("obj")
        assert not budget._tracked["obj"].pinned
    
    def test_clear(self, budget, tmp_path):
        """Test clearing all tracked objects."""
        budget.track("obj1", [1], size_bytes=100)
        budget.track("obj2", [2], size_bytes=200)
        
        budget.clear()
        
        assert budget.used_bytes == 0
        assert len(budget._tracked) == 0


class TestSpilling:
    """Tests for spill-to-disk functionality."""
    
    @pytest.fixture
    def budget(self, tmp_path):
        """Create a memory budget for testing."""
        return MemoryBudget(
            max_bytes=1000,
            spill_dir=tmp_path / "spill",
        )
    
    def test_spill_object(self, budget):
        """Test spilling an object to disk."""
        obj = {"data": list(range(100))}
        budget.track("obj", obj, size_bytes=500)
        
        path = budget.spill("obj")
        
        assert path is not None
        assert path.exists()
        assert budget.used_bytes == 0
        assert "obj" in budget._spilled
        assert "obj" not in budget._tracked
    
    def test_spill_dataframe(self, budget):
        """Test spilling a Polars DataFrame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        budget.track("df", df, size_bytes=500)
        
        path = budget.spill("df")
        
        assert path is not None
        assert path.exists()
    
    def test_spill_pinned_fails(self, budget):
        """Test that pinned objects can't be spilled."""
        budget.track("obj", [1, 2, 3], size_bytes=100, pinned=True)
        
        path = budget.spill("obj")
        
        assert path is None
        assert budget.used_bytes == 100  # Still tracked
    
    def test_spill_missing(self, budget):
        """Test spilling non-existent object."""
        path = budget.spill("missing")
        assert path is None
    
    def test_get_spill_candidates(self, budget):
        """Test LRU-based spill candidate selection."""
        budget.track("old", [1], size_bytes=100)
        time.sleep(0.01)
        budget.track("newer", [2], size_bytes=100)
        time.sleep(0.01)
        budget.track("newest", [3], size_bytes=100)
        
        candidates = budget.get_spill_candidates(2)
        
        assert len(candidates) == 2
        assert candidates[0] == "old"  # Oldest first
        assert candidates[1] == "newer"
    
    def test_spill_candidates_skip_pinned(self, budget):
        """Test that spill candidates skip pinned objects."""
        budget.track("old_pinned", [1], size_bytes=100, pinned=True)
        budget.track("newer", [2], size_bytes=100)
        
        candidates = budget.get_spill_candidates(2)
        
        assert "old_pinned" not in candidates
        assert "newer" in candidates
    
    def test_spill_lru(self, budget):
        """Test LRU-based spilling."""
        budget.track("obj1", [1], size_bytes=400)
        time.sleep(0.01)
        budget.track("obj2", [2], size_bytes=400)
        time.sleep(0.01)
        budget.track("obj3", [3], size_bytes=200)
        # Total: 1000 bytes (100%)
        
        freed = budget.spill_lru(target_bytes=400)
        
        assert freed >= 400
        assert budget.used_bytes <= 600
    
    def test_enforce_budget(self, budget):
        """Test automatic budget enforcement."""
        # Fill to 90% (above 80% threshold)
        budget.track("obj1", [1], size_bytes=500)
        budget.track("obj2", [2], size_bytes=400)
        
        freed = budget.enforce_budget()
        
        # Should have spilled to get under 80%
        assert budget.usage_ratio <= 0.8


class TestRecovery:
    """Tests for recovering spilled objects."""
    
    @pytest.fixture
    def budget(self, tmp_path):
        """Create a memory budget for testing."""
        return MemoryBudget(
            max_bytes=1000,
            spill_dir=tmp_path / "spill",
        )
    
    def test_recover_dict(self, budget):
        """Test recovering a spilled dict."""
        original = {"key": "value", "num": 123}
        budget.track("obj", original, size_bytes=100)
        budget.spill("obj")
        
        recovered = budget.recover("obj")
        
        assert recovered == original
        assert "obj" in budget._tracked
        assert "obj" not in budget._spilled
    
    def test_recover_dataframe(self, budget):
        """Test recovering a spilled DataFrame."""
        original = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        budget.track("df", original, size_bytes=100)
        budget.spill("df")
        
        recovered = budget.recover("df")
        
        assert recovered.equals(original)
    
    def test_get_with_recovery(self, budget):
        """Test that get() recovers spilled objects."""
        original = {"data": 123}
        budget.track("obj", original, size_bytes=100)
        budget.spill("obj")
        
        result = budget.get("obj", recover=True)
        
        assert result == original
        assert "obj" in budget._tracked
    
    def test_get_without_recovery(self, budget):
        """Test get() without recovery."""
        budget.track("obj", [1, 2, 3], size_bytes=100)
        budget.spill("obj")
        
        result = budget.get("obj", recover=False)
        
        assert result is None
    
    def test_recover_missing(self, budget):
        """Test recovering non-existent object."""
        result = budget.recover("missing")
        assert result is None


class TestMemoryGuard:
    """Tests for MemoryGuard context manager."""
    
    def test_guard_basic(self, tmp_path):
        """Test basic memory guard usage."""
        budget = MemoryBudget(max_bytes=1000, spill_dir=tmp_path)
        
        with MemoryGuard(budget, "temp", lambda: [1, 2, 3]) as data:
            assert data == [1, 2, 3]
            assert "temp" in budget._tracked
        
        # Should be untracked after context
        assert "temp" not in budget._tracked
    
    def test_guard_pinned(self, tmp_path):
        """Test memory guard with pinned object."""
        budget = MemoryBudget(max_bytes=1000, spill_dir=tmp_path)
        
        with MemoryGuard(budget, "pinned", lambda: [1], pinned=True) as data:
            assert budget._tracked["pinned"].pinned


class TestBudgetedStore:
    """Tests for BudgetedStore wrapper."""
    
    def test_budgeted_store(self, tmp_path):
        """Test budgeted store wrapper."""
        budget = MemoryBudget(max_bytes=1000, spill_dir=tmp_path)
        store = MagicMock()
        
        budgeted = BudgetedStore(budget, store)
        
        assert budgeted.budget is budget
        assert budgeted.store is store
    
    def test_check_budget_auto_spill(self, tmp_path):
        """Test automatic budget checking."""
        budget = MemoryBudget(max_bytes=1000, spill_dir=tmp_path)
        store = MagicMock()
        budgeted = BudgetedStore(budget, store, auto_spill=True)
        
        # Fill budget above threshold
        budget.track("obj1", [1], size_bytes=500)
        budget.track("obj2", [2], size_bytes=400)
        
        freed = budgeted.check_budget()
        
        assert freed >= 0
    
    def test_get_stats(self, tmp_path):
        """Test getting stats from budgeted store."""
        budget = MemoryBudget(max_bytes=1000, spill_dir=tmp_path)
        store = MagicMock()
        budgeted = BudgetedStore(budget, store)
        
        stats = budgeted.get_stats()
        
        assert isinstance(stats, MemoryStats)


class TestGlobalBudget:
    """Tests for global budget functions."""
    
    def test_configure_and_get(self, tmp_path):
        """Test configuring and getting global budget."""
        budget = configure_budget(5000, spill_dir=tmp_path)
        
        assert budget is not None
        assert get_budget() is budget
        assert budget.max_bytes == 5000
    
    def test_track_global(self, tmp_path):
        """Test tracking with global budget."""
        configure_budget(5000, spill_dir=tmp_path)
        
        obj = [1, 2, 3]
        track("global_obj", obj, size_bytes=100)
        
        result = get_budget().get("global_obj")
        assert result is obj
    
    def test_untrack_global(self, tmp_path):
        """Test untracking from global budget."""
        configure_budget(5000, spill_dir=tmp_path)
        
        track("obj", [1, 2, 3], size_bytes=100)
        result = untrack("obj")
        
        assert result == [1, 2, 3]
    
    def test_enforce_global(self, tmp_path):
        """Test enforcing global budget."""
        configure_budget(1000, spill_dir=tmp_path)
        budget = get_budget()
        
        # Fill above threshold
        track("obj1", [1], size_bytes=500)
        track("obj2", [2], size_bytes=400)
        
        freed = enforce()
        
        assert freed >= 0


class TestConcurrency:
    """Tests for thread-safety."""
    
    def test_concurrent_tracking(self, tmp_path):
        """Test concurrent object tracking."""
        budget = MemoryBudget(max_bytes=100000, spill_dir=tmp_path)
        errors = []
        
        def track_objects(start_id):
            try:
                for i in range(100):
                    budget.track(f"obj_{start_id}_{i}", [i], size_bytes=10)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=track_objects, args=(t,))
            for t in range(4)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(budget._tracked) == 400  # 4 threads * 100 objects
    
    def test_concurrent_spilling(self, tmp_path):
        """Test concurrent spill operations."""
        budget = MemoryBudget(max_bytes=10000, spill_dir=tmp_path)
        
        # Track some objects
        for i in range(20):
            budget.track(f"obj_{i}", [i] * 10, size_bytes=100)
        
        errors = []
        
        def spill_objects():
            try:
                for _ in range(5):
                    candidates = budget.get_spill_candidates(1)
                    if candidates:
                        budget.spill(candidates[0])
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=spill_objects) for _ in range(4)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestCustomCallbacks:
    """Tests for custom spill/recovery callbacks."""
    
    def test_custom_spill_callback(self, tmp_path):
        """Test custom spill callback."""
        spill_calls = []
        
        def custom_spill(object_id, path):
            spill_calls.append((object_id, path))
            with open(path, "w") as f:
                f.write("custom")
        
        def custom_recover(object_id, path):
            with open(path, "r") as f:
                return f.read()
        
        budget = MemoryBudget(
            max_bytes=1000,
            spill_dir=tmp_path,
            spill_callback=custom_spill,
            recovery_callback=custom_recover,
        )
        
        budget.track("obj", "test_data", size_bytes=100)
        budget.spill("obj")
        
        assert len(spill_calls) == 1
        assert spill_calls[0][0] == "obj"
    
    def test_custom_recovery_callback(self, tmp_path):
        """Test custom recovery callback."""
        def custom_spill(object_id, path):
            with open(path, "w") as f:
                f.write(f"data:{object_id}")
        
        def custom_recover(object_id, path):
            with open(path, "r") as f:
                return f.read()
        
        budget = MemoryBudget(
            max_bytes=1000,
            spill_dir=tmp_path,
            spill_callback=custom_spill,
            recovery_callback=custom_recover,
        )
        
        budget.track("myobj", "original", size_bytes=100)
        budget.spill("myobj")
        
        recovered = budget.recover("myobj")
        
        assert recovered == "data:myobj"


class TestCleanup:
    """Tests for cleanup functionality."""
    
    def test_cleanup_spills(self, tmp_path):
        """Test cleanup of orphaned spill files."""
        spill_dir = tmp_path / "spill"
        spill_dir.mkdir()
        
        # Create orphaned spill files
        (spill_dir / "orphan1.spill").write_text("test")
        (spill_dir / "orphan2.spill").write_text("test")
        
        budget = MemoryBudget(max_bytes=1000, spill_dir=spill_dir)
        
        # Track and spill a real object
        budget.track("real", [1, 2, 3], size_bytes=100)
        budget.spill("real")
        
        removed = budget.cleanup_spills()
        
        assert removed == 2  # Orphaned files removed
        assert (spill_dir / "real.spill").exists()  # Real spill kept
