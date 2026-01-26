"""
Predicate-based partitioning for large RDF datasets.

Partitions facts by predicate to improve query performance on large
datasets where most queries filter by predicate.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

import polars as pl

if TYPE_CHECKING:
    from rdf_starbase.storage.terms import TermDict, TermId


@dataclass
class PartitionInfo:
    """Information about a partition."""
    
    predicate_id: int
    predicate_iri: str
    fact_count: int
    size_bytes: int
    file_path: Optional[str] = None
    in_memory: bool = True
    last_accessed: float = 0.0


@dataclass
class PartitionStats:
    """Statistics for partitioned storage."""
    
    total_partitions: int = 0
    total_facts: int = 0
    in_memory_partitions: int = 0
    on_disk_partitions: int = 0
    total_memory_bytes: int = 0
    hot_partitions: list[str] = field(default_factory=list)
    cold_partitions: list[str] = field(default_factory=list)


class PredicatePartitioner:
    """
    Partitions RDF facts by predicate for improved query performance.
    
    Benefits:
    - Queries filtered by predicate only scan relevant partition
    - Memory efficiency: cold partitions can be spilled to disk
    - Parallelism: independent partitions can be queried concurrently
    
    Example:
        partitioner = PredicatePartitioner(term_dict)
        
        # Add facts (automatically partitioned)
        partitioner.add_facts([(g, s, p, o), ...])
        
        # Query specific predicate (fast - single partition)
        df = partitioner.get_partition(predicate_id)
        
        # Query across predicates (scans multiple partitions)
        df = partitioner.get_all_facts()
    """
    
    def __init__(
        self,
        term_dict: "TermDict",
        storage_dir: Optional[Path] = None,
        max_memory_mb: float = 1024.0,
        partition_threshold: int = 10000,
        hot_partition_limit: int = 50,
    ):
        """
        Initialize partitioner.
        
        Args:
            term_dict: Term dictionary for resolving predicate IRIs
            storage_dir: Directory for spilling partitions to disk
            max_memory_mb: Maximum memory before spilling cold partitions
            partition_threshold: Minimum facts before creating partition
            hot_partition_limit: Maximum in-memory partitions
        """
        self._term_dict = term_dict
        self._storage_dir = storage_dir
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._partition_threshold = partition_threshold
        self._hot_partition_limit = hot_partition_limit
        
        # Partitions: predicate_id -> DataFrame
        self._partitions: dict[int, pl.DataFrame] = {}
        
        # Default partition for predicates below threshold
        self._default_partition: pl.DataFrame = pl.DataFrame({
            "graph": pl.Series([], dtype=pl.UInt32),
            "subject": pl.Series([], dtype=pl.UInt32),
            "predicate": pl.Series([], dtype=pl.UInt32),
            "object": pl.Series([], dtype=pl.UInt32),
        })
        
        # Partition metadata
        self._partition_info: dict[int, PartitionInfo] = {}
        self._predicate_counts: dict[int, int] = {}
        
        # Access tracking for LRU eviction
        self._access_order: list[int] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Storage directory setup
        if storage_dir:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)
    
    def add_facts(
        self, 
        facts: list[tuple[int, int, int, int]],
    ) -> None:
        """
        Add facts to partitions.
        
        Args:
            facts: List of (graph, subject, predicate, object) tuples
        """
        if not facts:
            return
        
        with self._lock:
            # Group facts by predicate
            by_predicate: dict[int, list[tuple[int, int, int, int]]] = {}
            for g, s, p, o in facts:
                if p not in by_predicate:
                    by_predicate[p] = []
                by_predicate[p].append((g, s, p, o))
            
            # Add to appropriate partitions
            for pred_id, pred_facts in by_predicate.items():
                self._add_to_partition(pred_id, pred_facts)
            
            # Check memory and spill if needed
            self._check_memory_pressure()
    
    def _add_to_partition(
        self, 
        predicate_id: int, 
        facts: list[tuple[int, int, int, int]]
    ) -> None:
        """Add facts to a specific partition."""
        # Update count
        current_count = self._predicate_counts.get(predicate_id, 0)
        new_count = current_count + len(facts)
        self._predicate_counts[predicate_id] = new_count
        
        # Create DataFrame for new facts
        new_df = pl.DataFrame({
            "graph": [f[0] for f in facts],
            "subject": [f[1] for f in facts],
            "predicate": [f[2] for f in facts],
            "object": [f[3] for f in facts],
        }).cast({
            "graph": pl.UInt32,
            "subject": pl.UInt32,
            "predicate": pl.UInt32,
            "object": pl.UInt32,
        })
        
        # Check if predicate should have its own partition
        if new_count >= self._partition_threshold:
            # Ensure partition exists
            if predicate_id not in self._partitions:
                # Promote from default partition
                self._promote_to_partition(predicate_id)
            
            # Append to partition
            if predicate_id in self._partitions:
                self._partitions[predicate_id] = pl.concat([
                    self._partitions[predicate_id],
                    new_df
                ])
            else:
                self._partitions[predicate_id] = new_df
            
            # Update access order
            self._touch_partition(predicate_id)
            
            # Update info
            self._update_partition_info(predicate_id)
        else:
            # Add to default partition
            self._default_partition = pl.concat([
                self._default_partition,
                new_df
            ])
    
    def _promote_to_partition(self, predicate_id: int) -> None:
        """Promote predicate from default partition to its own partition."""
        if len(self._default_partition) == 0:
            return
        
        # Extract facts for this predicate from default partition
        mask = self._default_partition["predicate"] == predicate_id
        predicate_facts = self._default_partition.filter(mask)
        
        if len(predicate_facts) > 0:
            self._partitions[predicate_id] = predicate_facts
            # Remove from default partition
            self._default_partition = self._default_partition.filter(~mask)
    
    def _touch_partition(self, predicate_id: int) -> None:
        """Update access order for LRU tracking."""
        import time
        
        if predicate_id in self._access_order:
            self._access_order.remove(predicate_id)
        self._access_order.append(predicate_id)
        
        if predicate_id in self._partition_info:
            self._partition_info[predicate_id].last_accessed = time.time()
    
    def _update_partition_info(self, predicate_id: int) -> None:
        """Update partition metadata."""
        import time
        
        if predicate_id not in self._partitions:
            return
        
        df = self._partitions[predicate_id]
        predicate_iri = self._term_dict.get_lex(predicate_id) or ""
        
        self._partition_info[predicate_id] = PartitionInfo(
            predicate_id=predicate_id,
            predicate_iri=predicate_iri,
            fact_count=len(df),
            size_bytes=df.estimated_size(),
            in_memory=True,
            last_accessed=time.time(),
        )
    
    def _check_memory_pressure(self) -> None:
        """Check memory usage and spill cold partitions if needed."""
        if not self._storage_dir:
            return
        
        current_memory = self._estimate_memory()
        
        while (
            current_memory > self._max_memory_bytes 
            and len(self._access_order) > 1
        ):
            # Spill coldest partition
            coldest = self._access_order[0]
            self._spill_partition(coldest)
            current_memory = self._estimate_memory()
    
    def _estimate_memory(self) -> int:
        """Estimate current memory usage."""
        total = self._default_partition.estimated_size()
        for df in self._partitions.values():
            total += df.estimated_size()
        return total
    
    def _spill_partition(self, predicate_id: int) -> None:
        """Spill a partition to disk."""
        if predicate_id not in self._partitions:
            return
        
        if not self._storage_dir:
            return
        
        df = self._partitions[predicate_id]
        file_path = self._storage_dir / f"partition_{predicate_id}.parquet"
        
        df.write_parquet(file_path)
        
        # Update info
        if predicate_id in self._partition_info:
            self._partition_info[predicate_id].file_path = str(file_path)
            self._partition_info[predicate_id].in_memory = False
        
        # Remove from memory
        del self._partitions[predicate_id]
        
        # Keep in access order but mark as spilled
        if predicate_id in self._access_order:
            self._access_order.remove(predicate_id)
    
    def _load_partition(self, predicate_id: int) -> Optional[pl.DataFrame]:
        """Load a spilled partition from disk."""
        if predicate_id in self._partitions:
            return self._partitions[predicate_id]
        
        if predicate_id not in self._partition_info:
            return None
        
        info = self._partition_info[predicate_id]
        if not info.file_path or not Path(info.file_path).exists():
            return None
        
        # Load from disk
        df = pl.read_parquet(info.file_path)
        
        # Check if we need to spill something else
        if len(self._partitions) >= self._hot_partition_limit:
            self._check_memory_pressure()
        
        # Put back in memory
        self._partitions[predicate_id] = df
        info.in_memory = True
        self._touch_partition(predicate_id)
        
        return df
    
    def get_partition(self, predicate_id: int) -> pl.DataFrame:
        """
        Get facts for a specific predicate.
        
        Args:
            predicate_id: Predicate term ID
            
        Returns:
            DataFrame of facts for this predicate
        """
        with self._lock:
            self._touch_partition(predicate_id)
            
            # Check dedicated partition
            if predicate_id in self._partitions:
                return self._partitions[predicate_id]
            
            # Check if spilled to disk
            if predicate_id in self._partition_info:
                df = self._load_partition(predicate_id)
                if df is not None:
                    return df
            
            # Check default partition
            if len(self._default_partition) > 0:
                mask = self._default_partition["predicate"] == predicate_id
                return self._default_partition.filter(mask)
            
            # Return empty DataFrame
            return pl.DataFrame({
                "graph": pl.Series([], dtype=pl.UInt32),
                "subject": pl.Series([], dtype=pl.UInt32),
                "predicate": pl.Series([], dtype=pl.UInt32),
                "object": pl.Series([], dtype=pl.UInt32),
            })
    
    def get_all_facts(self) -> pl.DataFrame:
        """
        Get all facts across all partitions.
        
        Returns:
            DataFrame of all facts
        """
        with self._lock:
            frames = [self._default_partition]
            
            # In-memory partitions
            frames.extend(self._partitions.values())
            
            # Load spilled partitions
            for pred_id, info in self._partition_info.items():
                if not info.in_memory and info.file_path:
                    df = self._load_partition(pred_id)
                    if df is not None:
                        frames.append(df)
            
            if not frames:
                return pl.DataFrame({
                    "graph": pl.Series([], dtype=pl.UInt32),
                    "subject": pl.Series([], dtype=pl.UInt32),
                    "predicate": pl.Series([], dtype=pl.UInt32),
                    "object": pl.Series([], dtype=pl.UInt32),
                })
            
            return pl.concat(frames)
    
    def query(
        self,
        predicate_id: Optional[int] = None,
        subject_id: Optional[int] = None,
        object_id: Optional[int] = None,
        graph_id: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Query facts with optional filters.
        
        If predicate_id is provided, only that partition is scanned.
        Otherwise all partitions are scanned.
        """
        with self._lock:
            if predicate_id is not None:
                df = self.get_partition(predicate_id)
            else:
                df = self.get_all_facts()
            
            # Apply filters
            if subject_id is not None:
                df = df.filter(pl.col("subject") == subject_id)
            if object_id is not None:
                df = df.filter(pl.col("object") == object_id)
            if graph_id is not None:
                df = df.filter(pl.col("graph") == graph_id)
            
            return df
    
    def stats(self) -> PartitionStats:
        """Get partition statistics."""
        with self._lock:
            in_memory = len(self._partitions)
            on_disk = sum(
                1 for info in self._partition_info.values() 
                if not info.in_memory
            )
            
            total_facts = len(self._default_partition)
            for df in self._partitions.values():
                total_facts += len(df)
            for info in self._partition_info.values():
                if not info.in_memory:
                    total_facts += info.fact_count
            
            memory_bytes = self._estimate_memory()
            
            # Hot partitions (recently accessed)
            hot = [
                self._partition_info[p].predicate_iri
                for p in self._access_order[-10:]
                if p in self._partition_info
            ]
            
            # Cold partitions (on disk)
            cold = [
                info.predicate_iri
                for info in self._partition_info.values()
                if not info.in_memory
            ]
            
            return PartitionStats(
                total_partitions=len(self._partition_info) + 1,  # +1 for default
                total_facts=total_facts,
                in_memory_partitions=in_memory + 1,  # +1 for default
                on_disk_partitions=on_disk,
                total_memory_bytes=memory_bytes,
                hot_partitions=hot,
                cold_partitions=cold,
            )
    
    def list_partitions(self) -> list[PartitionInfo]:
        """List all partition info."""
        with self._lock:
            return list(self._partition_info.values())
    
    def save(self, directory: Path) -> None:
        """Save all partitions to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save default partition
            if len(self._default_partition) > 0:
                self._default_partition.write_parquet(
                    directory / "partition_default.parquet"
                )
            
            # Save in-memory partitions
            for pred_id, df in self._partitions.items():
                df.write_parquet(directory / f"partition_{pred_id}.parquet")
            
            # Copy already-spilled partitions
            for pred_id, info in self._partition_info.items():
                if info.file_path and not info.in_memory:
                    src = Path(info.file_path)
                    dst = directory / f"partition_{pred_id}.parquet"
                    if src != dst and src.exists():
                        import shutil
                        shutil.copy(src, dst)
            
            # Save metadata
            import json
            metadata = {
                "predicate_counts": self._predicate_counts,
                "partition_info": {
                    str(k): {
                        "predicate_id": v.predicate_id,
                        "predicate_iri": v.predicate_iri,
                        "fact_count": v.fact_count,
                    }
                    for k, v in self._partition_info.items()
                }
            }
            with open(directory / "partition_meta.json", "w") as f:
                json.dump(metadata, f)
    
    @classmethod
    def load(
        cls, 
        directory: Path, 
        term_dict: "TermDict",
        **kwargs
    ) -> "PredicatePartitioner":
        """Load partitions from a directory."""
        import json
        
        directory = Path(directory)
        
        partitioner = cls(term_dict, **kwargs)
        
        # Load metadata
        meta_file = directory / "partition_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
            partitioner._predicate_counts = {
                int(k): v for k, v in metadata.get("predicate_counts", {}).items()
            }
        
        # Load default partition
        default_file = directory / "partition_default.parquet"
        if default_file.exists():
            partitioner._default_partition = pl.read_parquet(default_file)
        
        # Load predicate partitions
        for file in directory.glob("partition_*.parquet"):
            if file.name == "partition_default.parquet":
                continue
            
            pred_id = int(file.stem.split("_")[1])
            df = pl.read_parquet(file)
            partitioner._partitions[pred_id] = df
            partitioner._update_partition_info(pred_id)
        
        return partitioner
    
    def count(self) -> int:
        """Get total fact count."""
        with self._lock:
            total = len(self._default_partition)
            for df in self._partitions.values():
                total += len(df)
            for info in self._partition_info.values():
                if not info.in_memory:
                    total += info.fact_count
            return total
    
    def clear(self) -> None:
        """Clear all partitions."""
        with self._lock:
            self._partitions.clear()
            self._default_partition = pl.DataFrame({
                "graph": pl.Series([], dtype=pl.UInt32),
                "subject": pl.Series([], dtype=pl.UInt32),
                "predicate": pl.Series([], dtype=pl.UInt32),
                "object": pl.Series([], dtype=pl.UInt32),
            })
            self._partition_info.clear()
            self._predicate_counts.clear()
            self._access_order.clear()
