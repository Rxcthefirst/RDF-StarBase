"""
B-tree index for fast point and range lookups.

Provides O(log n) lookups for high-cardinality columns like subject and object.
Uses sorted arrays with binary search for simplicity and memory efficiency.
"""

from __future__ import annotations

import bisect
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

if TYPE_CHECKING:
    import polars as pl


@dataclass
class IndexEntry:
    """An entry in the index mapping a key to row positions."""
    key: int
    positions: list[int] = field(default_factory=list)


@dataclass
class IndexStats:
    """Statistics for an index."""
    column_name: str
    num_keys: int
    num_entries: int
    memory_bytes: int
    height: int = 1  # For B-tree compatibility (we use flat sorted array)


class SortedIndex:
    """
    A sorted index for fast lookups on a specific column.
    
    Uses binary search on a sorted key array for O(log n) lookups.
    Each key maps to a list of row positions containing that value.
    
    Example:
        idx = SortedIndex("subject")
        idx.build(df)
        
        # Point lookup
        positions = idx.lookup(term_id)
        
        # Range lookup
        positions = idx.range_lookup(min_id, max_id)
    """
    
    def __init__(self, column_name: str):
        """
        Initialize sorted index.
        
        Args:
            column_name: Name of column to index (e.g., "subject", "object")
        """
        self.column_name = column_name
        
        # Sorted key array for binary search
        self._keys: list[int] = []
        
        # Parallel array of row positions
        self._positions: list[list[int]] = []
        
        # For quick existence check
        self._key_set: set[int] = set()
        
        # Stats
        self._num_entries = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def build(self, df: "pl.DataFrame") -> None:
        """
        Build index from a DataFrame.
        
        Args:
            df: DataFrame containing the column to index
        """
        import polars as pl
        
        with self._lock:
            self._keys.clear()
            self._positions.clear()
            self._key_set.clear()
            self._num_entries = 0
            
            if self.column_name not in df.columns:
                return
            
            # Group by column value and collect row indices
            column = df[self.column_name]
            
            # Build temporary dict
            key_to_positions: dict[int, list[int]] = {}
            
            for row_idx, value in enumerate(column.to_list()):
                if value not in key_to_positions:
                    key_to_positions[value] = []
                key_to_positions[value].append(row_idx)
            
            # Sort keys and build parallel arrays
            sorted_keys = sorted(key_to_positions.keys())
            
            self._keys = sorted_keys
            self._positions = [key_to_positions[k] for k in sorted_keys]
            self._key_set = set(sorted_keys)
            self._num_entries = len(df)
    
    def lookup(self, key: int) -> list[int]:
        """
        Look up row positions for a specific key.
        
        Args:
            key: Value to look up
            
        Returns:
            List of row positions (0-indexed)
        """
        with self._lock:
            if key not in self._key_set:
                return []
            
            # Binary search for key
            idx = bisect.bisect_left(self._keys, key)
            if idx < len(self._keys) and self._keys[idx] == key:
                return self._positions[idx].copy()
            
            return []
    
    def range_lookup(
        self, 
        min_key: Optional[int] = None, 
        max_key: Optional[int] = None
    ) -> list[int]:
        """
        Look up row positions in a key range.
        
        Args:
            min_key: Minimum key (inclusive), None for no lower bound
            max_key: Maximum key (inclusive), None for no upper bound
            
        Returns:
            List of row positions (0-indexed)
        """
        with self._lock:
            if not self._keys:
                return []
            
            # Find start index
            if min_key is None:
                start_idx = 0
            else:
                start_idx = bisect.bisect_left(self._keys, min_key)
            
            # Find end index
            if max_key is None:
                end_idx = len(self._keys)
            else:
                end_idx = bisect.bisect_right(self._keys, max_key)
            
            # Collect all positions in range
            result: list[int] = []
            for idx in range(start_idx, end_idx):
                result.extend(self._positions[idx])
            
            return result
    
    def contains(self, key: int) -> bool:
        """Check if key exists in index."""
        with self._lock:
            return key in self._key_set
    
    def add(self, key: int, position: int) -> None:
        """
        Add a single entry to the index.
        
        For bulk operations, prefer build() which is more efficient.
        
        Args:
            key: Key value
            position: Row position
        """
        with self._lock:
            if key in self._key_set:
                # Find existing position list
                idx = bisect.bisect_left(self._keys, key)
                self._positions[idx].append(position)
            else:
                # Insert new key maintaining sorted order
                idx = bisect.bisect_left(self._keys, key)
                self._keys.insert(idx, key)
                self._positions.insert(idx, [position])
                self._key_set.add(key)
            
            self._num_entries += 1
    
    def remove(self, key: int, position: Optional[int] = None) -> None:
        """
        Remove entries from the index.
        
        Args:
            key: Key value
            position: Specific position to remove, or None to remove all
        """
        with self._lock:
            if key not in self._key_set:
                return
            
            idx = bisect.bisect_left(self._keys, key)
            if idx >= len(self._keys) or self._keys[idx] != key:
                return
            
            if position is None:
                # Remove entire key
                self._num_entries -= len(self._positions[idx])
                del self._keys[idx]
                del self._positions[idx]
                self._key_set.remove(key)
            else:
                # Remove specific position
                if position in self._positions[idx]:
                    self._positions[idx].remove(position)
                    self._num_entries -= 1
                    
                    # Remove key if no positions left
                    if not self._positions[idx]:
                        del self._keys[idx]
                        del self._positions[idx]
                        self._key_set.remove(key)
    
    def stats(self) -> IndexStats:
        """Get index statistics."""
        with self._lock:
            # Estimate memory usage
            key_bytes = len(self._keys) * 8  # 8 bytes per int
            pos_bytes = sum(len(p) * 8 for p in self._positions)
            set_bytes = len(self._key_set) * 8
            
            return IndexStats(
                column_name=self.column_name,
                num_keys=len(self._keys),
                num_entries=self._num_entries,
                memory_bytes=key_bytes + pos_bytes + set_bytes,
            )
    
    def save(self, path: Path) -> None:
        """Save index to file."""
        with self._lock:
            data = {
                "column_name": self.column_name,
                "keys": self._keys,
                "positions": self._positions,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "SortedIndex":
        """Load index from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        idx = cls(data["column_name"])
        idx._keys = data["keys"]
        idx._positions = data["positions"]
        idx._key_set = set(idx._keys)
        idx._num_entries = sum(len(p) for p in idx._positions)
        
        return idx
    
    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self._keys.clear()
            self._positions.clear()
            self._key_set.clear()
            self._num_entries = 0


class IndexManager:
    """
    Manages multiple indexes on a DataFrame.
    
    Example:
        manager = IndexManager()
        manager.create_index("subject")
        manager.create_index("object")
        manager.build_all(df)
        
        # Use indexes for lookups
        positions = manager.lookup("subject", term_id)
    """
    
    def __init__(self):
        """Initialize index manager."""
        self._indexes: dict[str, SortedIndex] = {}
        self._lock = threading.RLock()
    
    def create_index(self, column_name: str) -> SortedIndex:
        """
        Create an index on a column.
        
        Args:
            column_name: Column to index
            
        Returns:
            The created index
        """
        with self._lock:
            if column_name not in self._indexes:
                self._indexes[column_name] = SortedIndex(column_name)
            return self._indexes[column_name]
    
    def get_index(self, column_name: str) -> Optional[SortedIndex]:
        """Get an existing index."""
        with self._lock:
            return self._indexes.get(column_name)
    
    def has_index(self, column_name: str) -> bool:
        """Check if column is indexed."""
        with self._lock:
            return column_name in self._indexes
    
    def drop_index(self, column_name: str) -> bool:
        """
        Drop an index.
        
        Returns:
            True if index existed and was dropped
        """
        with self._lock:
            if column_name in self._indexes:
                del self._indexes[column_name]
                return True
            return False
    
    def build_all(self, df: "pl.DataFrame") -> None:
        """Build all indexes from DataFrame."""
        with self._lock:
            for idx in self._indexes.values():
                idx.build(df)
    
    def lookup(self, column_name: str, key: int) -> Optional[list[int]]:
        """
        Look up positions using an index.
        
        Returns:
            List of positions, or None if column not indexed
        """
        with self._lock:
            idx = self._indexes.get(column_name)
            if idx is None:
                return None
            return idx.lookup(key)
    
    def range_lookup(
        self,
        column_name: str,
        min_key: Optional[int] = None,
        max_key: Optional[int] = None,
    ) -> Optional[list[int]]:
        """
        Range lookup using an index.
        
        Returns:
            List of positions, or None if column not indexed
        """
        with self._lock:
            idx = self._indexes.get(column_name)
            if idx is None:
                return None
            return idx.range_lookup(min_key, max_key)
    
    def list_indexes(self) -> list[str]:
        """List all indexed columns."""
        with self._lock:
            return list(self._indexes.keys())
    
    def stats(self) -> dict[str, IndexStats]:
        """Get stats for all indexes."""
        with self._lock:
            return {name: idx.stats() for name, idx in self._indexes.items()}
    
    def save(self, directory: Path) -> None:
        """Save all indexes to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            for name, idx in self._indexes.items():
                idx.save(directory / f"index_{name}.pkl")
            
            # Save metadata
            metadata = {
                "indexes": list(self._indexes.keys()),
            }
            with open(directory / "index_meta.json", "w") as f:
                import json
                json.dump(metadata, f)
    
    @classmethod
    def load(cls, directory: Path) -> "IndexManager":
        """Load indexes from a directory."""
        import json
        
        directory = Path(directory)
        manager = cls()
        
        meta_file = directory / "index_meta.json"
        if not meta_file.exists():
            return manager
        
        with open(meta_file) as f:
            metadata = json.load(f)
        
        for name in metadata.get("indexes", []):
            idx_file = directory / f"index_{name}.pkl"
            if idx_file.exists():
                manager._indexes[name] = SortedIndex.load(idx_file)
        
        return manager
    
    def clear_all(self) -> None:
        """Clear all indexes."""
        with self._lock:
            for idx in self._indexes.values():
                idx.clear()


def indexed_filter(
    df: "pl.DataFrame",
    index: SortedIndex,
    key: int,
) -> "pl.DataFrame":
    """
    Filter DataFrame using an index.
    
    More efficient than full table scan when selectivity is low.
    
    Args:
        df: DataFrame to filter
        index: Index on the filter column
        key: Key value to filter by
        
    Returns:
        Filtered DataFrame
    """
    positions = index.lookup(key)
    if not positions:
        return df.head(0)  # Empty with same schema
    
    return df[positions]


def estimate_selectivity(index: SortedIndex, key: int) -> float:
    """
    Estimate selectivity of a lookup.
    
    Returns fraction of rows that match the key (0.0 to 1.0).
    """
    stats = index.stats()
    if stats.num_entries == 0:
        return 0.0
    
    positions = index.lookup(key)
    return len(positions) / stats.num_entries


def should_use_index(
    index: SortedIndex, 
    total_rows: int,
    threshold: float = 0.1,
) -> bool:
    """
    Decide whether using the index is beneficial.
    
    For very high selectivity (many rows match), a full scan may be faster.
    
    Args:
        index: The index to consider
        total_rows: Total rows in the table
        threshold: Use index if selectivity is below this (default 10%)
        
    Returns:
        True if index should be used
    """
    stats = index.stats()
    
    # If index covers few keys, likely beneficial
    if stats.num_keys < total_rows * threshold:
        return True
    
    # Average entries per key
    if stats.num_keys > 0:
        avg_per_key = stats.num_entries / stats.num_keys
        # If each key maps to few entries, use index
        return avg_per_key < total_rows * threshold
    
    return True
