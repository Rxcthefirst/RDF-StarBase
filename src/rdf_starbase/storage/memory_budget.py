"""
Memory Budget Enforcement Module

Provides memory tracking, spill-to-disk triggers, and recovery mechanisms
to ensure the RDF store stays within configured memory limits.
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import polars as pl


# Type variable for generic tracked objects
T = TypeVar("T")


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = auto()       # < 50% of budget
    MODERATE = auto()  # 50-75% of budget
    HIGH = auto()      # 75-90% of budget
    CRITICAL = auto()  # > 90% of budget


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    budget_bytes: int
    used_bytes: int
    available_bytes: int
    pressure: MemoryPressure
    tracked_objects: int
    spilled_objects: int
    last_gc_time: Optional[float] = None
    last_spill_time: Optional[float] = None
    
    @property
    def usage_ratio(self) -> float:
        """Return memory usage as ratio of budget."""
        if self.budget_bytes == 0:
            return 0.0
        return self.used_bytes / self.budget_bytes
    
    @property
    def usage_percent(self) -> float:
        """Return memory usage as percentage."""
        return self.usage_ratio * 100


@dataclass 
class SpillRecord:
    """Record of a spilled object."""
    object_id: str
    spill_path: Path
    size_bytes: int
    spill_time: float
    object_type: str


@dataclass
class TrackedObject:
    """A memory-tracked object."""
    object_id: str
    obj: Any
    size_bytes: int
    last_access: float
    access_count: int = 0
    pinned: bool = False  # Pinned objects won't be spilled


class MemoryBudget:
    """
    Memory budget enforcement with spill-to-disk capabilities.
    
    Features:
    - Configurable memory budget
    - Automatic tracking of registered objects
    - LRU-based spilling when budget exceeded
    - Async spill callbacks
    - Memory pressure monitoring
    
    Example:
        budget = MemoryBudget(max_bytes=1_000_000_000)  # 1GB
        
        # Track a DataFrame
        budget.track("facts", df, estimate_size(df))
        
        # Check if we need to spill
        if budget.should_spill():
            budget.spill_lru()
    """
    
    def __init__(
        self,
        max_bytes: int = 1_000_000_000,  # 1GB default
        spill_dir: Optional[Path] = None,
        spill_threshold: float = 0.8,     # Start spilling at 80%
        critical_threshold: float = 0.95,  # Emergency action at 95%
        gc_on_spill: bool = True,
        spill_callback: Optional[Callable[[str, Path], None]] = None,
        recovery_callback: Optional[Callable[[str, Path], Any]] = None,
    ):
        """
        Initialize memory budget.
        
        Args:
            max_bytes: Maximum memory budget in bytes
            spill_dir: Directory for spilled objects
            spill_threshold: Ratio at which to start spilling
            critical_threshold: Ratio at which to take emergency action
            gc_on_spill: Whether to run GC after spilling
            spill_callback: Custom callback for spilling objects
            recovery_callback: Custom callback for recovering objects
        """
        self._max_bytes = max_bytes
        self._spill_dir = spill_dir or Path("./spill")
        self._spill_threshold = spill_threshold
        self._critical_threshold = critical_threshold
        self._gc_on_spill = gc_on_spill
        self._spill_callback = spill_callback
        self._recovery_callback = recovery_callback
        
        # Tracking state
        self._tracked: Dict[str, TrackedObject] = {}
        self._spilled: Dict[str, SpillRecord] = {}
        self._used_bytes = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._last_gc_time: Optional[float] = None
        self._last_spill_time: Optional[float] = None
        self._total_spills = 0
        self._total_recoveries = 0
        
        # Ensure spill directory exists
        self._spill_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_bytes(self) -> int:
        """Return maximum memory budget."""
        return self._max_bytes
    
    @property 
    def used_bytes(self) -> int:
        """Return currently used bytes."""
        with self._lock:
            return self._used_bytes
    
    @property
    def available_bytes(self) -> int:
        """Return available bytes."""
        return max(0, self._max_bytes - self._used_bytes)
    
    @property
    def usage_ratio(self) -> float:
        """Return current usage ratio."""
        if self._max_bytes == 0:
            return 0.0
        return self._used_bytes / self._max_bytes
    
    def get_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        ratio = self.usage_ratio
        if ratio < 0.5:
            return MemoryPressure.LOW
        elif ratio < 0.75:
            return MemoryPressure.MODERATE
        elif ratio < 0.9:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        with self._lock:
            return MemoryStats(
                budget_bytes=self._max_bytes,
                used_bytes=self._used_bytes,
                available_bytes=self.available_bytes,
                pressure=self.get_pressure(),
                tracked_objects=len(self._tracked),
                spilled_objects=len(self._spilled),
                last_gc_time=self._last_gc_time,
                last_spill_time=self._last_spill_time,
            )
    
    def track(
        self,
        object_id: str,
        obj: Any,
        size_bytes: Optional[int] = None,
        pinned: bool = False,
    ) -> None:
        """
        Track an object for memory management.
        
        Args:
            object_id: Unique identifier for the object
            obj: The object to track
            size_bytes: Size in bytes (estimated if not provided)
            pinned: If True, object won't be spilled
        """
        if size_bytes is None:
            size_bytes = estimate_size(obj)
        
        with self._lock:
            # Remove old tracking if exists
            if object_id in self._tracked:
                old = self._tracked[object_id]
                self._used_bytes -= old.size_bytes
            
            # Track new object
            tracked = TrackedObject(
                object_id=object_id,
                obj=obj,
                size_bytes=size_bytes,
                last_access=time.time(),
                pinned=pinned,
            )
            self._tracked[object_id] = tracked
            self._used_bytes += size_bytes
    
    def untrack(self, object_id: str) -> Optional[Any]:
        """
        Stop tracking an object.
        
        Args:
            object_id: ID of object to untrack
            
        Returns:
            The object if it was tracked, None otherwise
        """
        with self._lock:
            if object_id in self._tracked:
                tracked = self._tracked.pop(object_id)
                self._used_bytes -= tracked.size_bytes
                return tracked.obj
            return None
    
    def get(self, object_id: str, recover: bool = True) -> Optional[Any]:
        """
        Get a tracked object, recovering from spill if needed.
        
        Args:
            object_id: ID of object to get
            recover: If True, recover from spill if needed
            
        Returns:
            The object if found, None otherwise
        """
        with self._lock:
            # Check in-memory first
            if object_id in self._tracked:
                tracked = self._tracked[object_id]
                tracked.last_access = time.time()
                tracked.access_count += 1
                return tracked.obj
            
            # Check if spilled
            if recover and object_id in self._spilled:
                return self.recover(object_id)
            
            return None
    
    def update_size(self, object_id: str, new_size: int) -> None:
        """Update the tracked size of an object."""
        with self._lock:
            if object_id in self._tracked:
                old = self._tracked[object_id]
                self._used_bytes -= old.size_bytes
                old.size_bytes = new_size
                self._used_bytes += new_size
    
    def should_spill(self) -> bool:
        """Check if we should start spilling."""
        return self.usage_ratio >= self._spill_threshold
    
    def is_critical(self) -> bool:
        """Check if we're at critical memory pressure."""
        return self.usage_ratio >= self._critical_threshold
    
    def get_spill_candidates(self, count: int = 1) -> List[str]:
        """
        Get objects to spill based on LRU policy.
        
        Args:
            count: Number of candidates to return
            
        Returns:
            List of object IDs to spill
        """
        with self._lock:
            # Filter out pinned objects
            candidates = [
                (oid, tracked)
                for oid, tracked in self._tracked.items()
                if not tracked.pinned
            ]
            
            # Sort by last access (oldest first)
            candidates.sort(key=lambda x: x[1].last_access)
            
            return [oid for oid, _ in candidates[:count]]
    
    def spill(self, object_id: str) -> Optional[Path]:
        """
        Spill an object to disk.
        
        Args:
            object_id: ID of object to spill
            
        Returns:
            Path where object was spilled, or None if failed
        """
        with self._lock:
            if object_id not in self._tracked:
                return None
            
            tracked = self._tracked[object_id]
            if tracked.pinned:
                return None
            
            # Determine spill path
            spill_path = self._spill_dir / f"{object_id}.spill"
            
            try:
                # Use custom callback or default serialization
                if self._spill_callback:
                    self._spill_callback(object_id, spill_path)
                else:
                    self._default_spill(tracked.obj, spill_path)
                
                # Record spill
                record = SpillRecord(
                    object_id=object_id,
                    spill_path=spill_path,
                    size_bytes=tracked.size_bytes,
                    spill_time=time.time(),
                    object_type=type(tracked.obj).__name__,
                )
                self._spilled[object_id] = record
                
                # Remove from tracked
                self._used_bytes -= tracked.size_bytes
                del self._tracked[object_id]
                
                self._last_spill_time = time.time()
                self._total_spills += 1
                
                # Optionally run GC
                if self._gc_on_spill:
                    gc.collect()
                    self._last_gc_time = time.time()
                
                return spill_path
                
            except Exception:
                # Spill failed, object stays in memory
                return None
    
    def _default_spill(self, obj: Any, path: Path) -> None:
        """Default spill implementation using pickle."""
        # Handle Polars DataFrames specially
        if isinstance(obj, pl.DataFrame):
            obj.write_parquet(str(path))
        else:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    
    def recover(self, object_id: str) -> Optional[Any]:
        """
        Recover a spilled object.
        
        Args:
            object_id: ID of object to recover
            
        Returns:
            The recovered object, or None if failed
        """
        with self._lock:
            if object_id not in self._spilled:
                return None
            
            record = self._spilled[object_id]
            
            try:
                # Use custom callback or default deserialization
                if self._recovery_callback:
                    obj = self._recovery_callback(object_id, record.spill_path)
                else:
                    obj = self._default_recover(record)
                
                # Track recovered object
                self.track(object_id, obj, record.size_bytes)
                
                # Remove spill record and file
                del self._spilled[object_id]
                if record.spill_path.exists():
                    record.spill_path.unlink()
                
                self._total_recoveries += 1
                
                return obj
                
            except Exception:
                return None
    
    def _default_recover(self, record: SpillRecord) -> Any:
        """Default recovery implementation."""
        path = record.spill_path
        
        # Handle Parquet files (for DataFrames)
        if path.suffix == ".spill" and record.object_type == "DataFrame":
            return pl.read_parquet(str(path))
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def spill_lru(self, target_bytes: Optional[int] = None) -> int:
        """
        Spill objects using LRU until under budget or target.
        
        Args:
            target_bytes: Target bytes to free (or spill until under threshold)
            
        Returns:
            Number of bytes freed
        """
        freed = 0
        
        if target_bytes is None:
            # Free until under spill threshold
            target = self._max_bytes * self._spill_threshold
            target_bytes = max(0, self._used_bytes - int(target))
        
        while freed < target_bytes:
            candidates = self.get_spill_candidates(1)
            if not candidates:
                break
            
            object_id = candidates[0]
            with self._lock:
                if object_id in self._tracked:
                    size = self._tracked[object_id].size_bytes
                    if self.spill(object_id):
                        freed += size
                    else:
                        break
        
        return freed
    
    def enforce_budget(self) -> int:
        """
        Enforce memory budget by spilling as needed.
        
        Returns:
            Number of bytes freed
        """
        if not self.should_spill():
            return 0
        
        return self.spill_lru()
    
    def pin(self, object_id: str) -> bool:
        """Pin an object so it won't be spilled."""
        with self._lock:
            if object_id in self._tracked:
                self._tracked[object_id].pinned = True
                return True
            return False
    
    def unpin(self, object_id: str) -> bool:
        """Unpin an object so it can be spilled."""
        with self._lock:
            if object_id in self._tracked:
                self._tracked[object_id].pinned = False
                return True
            return False
    
    def clear(self) -> None:
        """Clear all tracked objects and spill files."""
        with self._lock:
            self._tracked.clear()
            self._used_bytes = 0
            
            # Remove spill files
            for record in self._spilled.values():
                if record.spill_path.exists():
                    record.spill_path.unlink()
            self._spilled.clear()
    
    def cleanup_spills(self) -> int:
        """Remove orphaned spill files. Returns count of files removed."""
        removed = 0
        if self._spill_dir.exists():
            for path in self._spill_dir.glob("*.spill"):
                object_id = path.stem
                if object_id not in self._spilled:
                    path.unlink()
                    removed += 1
        return removed


def estimate_size(obj: Any) -> int:
    """
    Estimate the memory size of an object in bytes.
    
    Works with:
    - Polars DataFrames
    - NumPy arrays
    - Python built-in types
    - General objects (via sys.getsizeof)
    """
    # Polars DataFrame
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size()
    
    # Check for NumPy array
    if hasattr(obj, "nbytes"):
        return obj.nbytes
    
    # Dictionaries - estimate recursively
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            estimate_size(k) + estimate_size(v) 
            for k, v in obj.items()
        )
    
    # Lists - estimate recursively
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(estimate_size(item) for item in obj)
    
    # Sets - estimate recursively  
    if isinstance(obj, (set, frozenset)):
        return sys.getsizeof(obj) + sum(estimate_size(item) for item in obj)
    
    # Default: use sys.getsizeof
    return sys.getsizeof(obj)


class MemoryGuard:
    """
    Context manager for memory-guarded operations.
    
    Example:
        budget = MemoryBudget(max_bytes=1_000_000)
        
        with MemoryGuard(budget, "temp_data", lambda: large_computation()):
            # Work with result
            pass
        # Object is automatically untracked after context
    """
    
    def __init__(
        self,
        budget: MemoryBudget,
        object_id: str,
        factory: Callable[[], Any],
        pinned: bool = False,
    ):
        self._budget = budget
        self._object_id = object_id
        self._factory = factory
        self._pinned = pinned
        self._obj: Optional[Any] = None
    
    def __enter__(self) -> Any:
        # Create and track object
        self._obj = self._factory()
        self._budget.track(self._object_id, self._obj, pinned=self._pinned)
        return self._obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Untrack object
        self._budget.untrack(self._object_id)
        self._obj = None
        return False


class BudgetedStore:
    """
    Wrapper that adds memory budget enforcement to a store.
    
    This is a higher-level abstraction for integrating memory budget
    with an RDF store.
    """
    
    def __init__(
        self,
        budget: MemoryBudget,
        store: Any,
        auto_spill: bool = True,
    ):
        self._budget = budget
        self._store = store
        self._auto_spill = auto_spill
    
    @property
    def budget(self) -> MemoryBudget:
        """Return the memory budget."""
        return self._budget
    
    @property
    def store(self) -> Any:
        """Return the underlying store."""
        return self._store
    
    def check_budget(self) -> int:
        """Check and enforce budget, return bytes freed."""
        if self._auto_spill and self._budget.should_spill():
            return self._budget.enforce_budget()
        return 0
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self._budget.get_stats()


# Module-level convenience functions

_global_budget: Optional[MemoryBudget] = None


def configure_budget(max_bytes: int, **kwargs) -> MemoryBudget:
    """Configure the global memory budget."""
    global _global_budget
    _global_budget = MemoryBudget(max_bytes=max_bytes, **kwargs)
    return _global_budget


def get_budget() -> Optional[MemoryBudget]:
    """Get the global memory budget."""
    return _global_budget


def track(object_id: str, obj: Any, **kwargs) -> None:
    """Track an object with the global budget."""
    if _global_budget:
        _global_budget.track(object_id, obj, **kwargs)


def untrack(object_id: str) -> Optional[Any]:
    """Untrack an object from the global budget."""
    if _global_budget:
        return _global_budget.untrack(object_id)
    return None


def enforce() -> int:
    """Enforce the global memory budget."""
    if _global_budget:
        return _global_budget.enforce_budget()
    return 0
