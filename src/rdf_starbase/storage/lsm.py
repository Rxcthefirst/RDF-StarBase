"""
Predicate-Partitioned Parquet Storage.

Implements LSM-style base+delta storage with predicate partitioning
for high-performance RDF★ queries.

Key design decisions (from storage-spec.md):
- Primary partition key: predicate (p)
- Optional secondary partition key: graph (g)
- Base dataset: compacted partitions
- Delta dataset: append-only write batches
- Compaction with deduplication on (g,s,p,o)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import shutil

import polars as pl

from rdf_starbase.storage.terms import TermId, TermDict
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID


# =============================================================================
# Partition Metadata
# =============================================================================

@dataclass
class PartitionStats:
    """Statistics for a partition (predicate or predicate+graph)."""
    predicate_id: TermId
    graph_id: Optional[TermId]
    row_count: int
    min_subject: Optional[TermId] = None
    max_subject: Optional[TermId] = None
    distinct_subjects: int = 0
    file_size_bytes: int = 0
    last_compacted: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "predicate_id": self.predicate_id,
            "graph_id": self.graph_id,
            "row_count": self.row_count,
            "min_subject": self.min_subject,
            "max_subject": self.max_subject,
            "distinct_subjects": self.distinct_subjects,
            "file_size_bytes": self.file_size_bytes,
            "last_compacted": self.last_compacted.isoformat() if self.last_compacted else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PartitionStats":
        return cls(
            predicate_id=d["predicate_id"],
            graph_id=d.get("graph_id"),
            row_count=d["row_count"],
            min_subject=d.get("min_subject"),
            max_subject=d.get("max_subject"),
            distinct_subjects=d.get("distinct_subjects", 0),
            file_size_bytes=d.get("file_size_bytes", 0),
            last_compacted=datetime.fromisoformat(d["last_compacted"]) if d.get("last_compacted") else None,
        )


# =============================================================================
# LSM Storage Manager
# =============================================================================

class LSMStorage:
    """
    LSM-style storage manager with predicate partitioning.
    
    Directory structure:
    - data/
      - term_dict/
        - term_dict.parquet
        - term_hash.parquet
      - qt_dict/
        - qt_dict.parquet
      - facts/
        - base/
          - p=<pid>/
            - part-0.parquet
            - part-1.parquet
        - delta/
          - p=<pid>/
            - batch-<txn>.parquet
      - meta/
        - schema_version.json
        - partitions.json
    """
    
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(
        self,
        path: Path,
        term_dict: Optional[TermDict] = None,
        qt_dict: Optional[QtDict] = None,
        fact_store: Optional[FactStore] = None,
    ):
        """
        Initialize the LSM storage manager.
        
        Args:
            path: Root directory for storage
            term_dict: Existing TermDict (or create new)
            qt_dict: Existing QtDict (or create new)
            fact_store: Existing FactStore (or create new)
        """
        self.path = Path(path)
        
        # Initialize or use existing components
        if term_dict is None:
            term_dict = TermDict()
        self.term_dict = term_dict
        
        if qt_dict is None:
            qt_dict = QtDict(term_dict)
        self.qt_dict = qt_dict
        
        if fact_store is None:
            fact_store = FactStore(term_dict, qt_dict)
        self.fact_store = fact_store
        
        # Partition statistics
        self._partition_stats: Dict[TermId, PartitionStats] = {}
        
        # Delta buffer (facts not yet written to Parquet)
        self._delta_buffer: List[pl.DataFrame] = []
        self._delta_txn_count = 0
        
        # Configuration
        self.delta_flush_threshold = 10000  # Flush delta when this many rows accumulated
        self.compaction_size_threshold = 100000  # Compact when partition exceeds this
    
    # =========================================================================
    # Directory Structure
    # =========================================================================
    
    @property
    def _term_dict_path(self) -> Path:
        return self.path / "data" / "term_dict"
    
    @property
    def _qt_dict_path(self) -> Path:
        return self.path / "data" / "qt_dict"
    
    @property
    def _facts_base_path(self) -> Path:
        return self.path / "data" / "facts" / "base"
    
    @property
    def _facts_delta_path(self) -> Path:
        return self.path / "data" / "facts" / "delta"
    
    @property
    def _meta_path(self) -> Path:
        return self.path / "data" / "meta"
    
    def _partition_base_path(self, p: TermId) -> Path:
        return self._facts_base_path / f"p={p}"
    
    def _partition_delta_path(self, p: TermId) -> Path:
        return self._facts_delta_path / f"p={p}"
    
    # =========================================================================
    # Initialization and Persistence
    # =========================================================================
    
    def initialize(self):
        """Create directory structure for a new storage."""
        self._term_dict_path.mkdir(parents=True, exist_ok=True)
        self._qt_dict_path.mkdir(parents=True, exist_ok=True)
        self._facts_base_path.mkdir(parents=True, exist_ok=True)
        self._facts_delta_path.mkdir(parents=True, exist_ok=True)
        self._meta_path.mkdir(parents=True, exist_ok=True)
        
        # Write schema version
        self._save_schema_version()
    
    def _save_schema_version(self):
        """Save schema version metadata."""
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "encoding": "tagged_ids",
            "quoting_scope": "graph_agnostic",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._meta_path / "schema_version.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def _save_partition_stats(self):
        """Save partition statistics."""
        stats_list = [s.to_dict() for s in self._partition_stats.values()]
        with open(self._meta_path / "partitions.json", "w") as f:
            json.dump(stats_list, f, indent=2)
    
    def _load_partition_stats(self):
        """Load partition statistics."""
        stats_file = self._meta_path / "partitions.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats_list = json.load(f)
            self._partition_stats = {
                s["predicate_id"]: PartitionStats.from_dict(s)
                for s in stats_list
            }
    
    def save(self):
        """
        Persist all data to disk.
        
        Saves:
        - Term dictionary
        - Quoted triple dictionary
        - Flushes delta buffer to Parquet
        - Partition statistics
        """
        self.initialize()
        
        # Save dictionaries
        self.term_dict.save(self._term_dict_path)
        self.qt_dict.save(self._qt_dict_path)
        
        # Flush any buffered deltas
        self._flush_delta_buffer()
        
        # Save partition stats
        self._save_partition_stats()
    
    @classmethod
    def load(cls, path: Path) -> "LSMStorage":
        """
        Load storage from disk.
        
        Reconstructs:
        - Term dictionary
        - Quoted triple dictionary
        - Partition statistics
        - In-memory fact store from base + delta
        """
        path = Path(path)
        
        # Load term dictionary
        term_dict = TermDict.load(path / "data" / "term_dict")
        
        # Load qt dictionary
        qt_dict = QtDict.load(path / "data" / "qt_dict", term_dict)
        
        # Create storage instance
        instance = cls(path, term_dict, qt_dict)
        
        # Load partition stats
        instance._load_partition_stats()
        
        # Reconstruct fact store from base + delta
        instance._reconstruct_fact_store()
        
        return instance
    
    def _reconstruct_fact_store(self):
        """Reconstruct in-memory fact store from Parquet files."""
        dfs = []
        
        # Load base partitions
        if self._facts_base_path.exists():
            for p_dir in self._facts_base_path.iterdir():
                if p_dir.is_dir() and p_dir.name.startswith("p="):
                    for parquet_file in p_dir.glob("*.parquet"):
                        dfs.append(pl.read_parquet(parquet_file))
        
        # Load delta partitions
        if self._facts_delta_path.exists():
            for p_dir in self._facts_delta_path.iterdir():
                if p_dir.is_dir() and p_dir.name.startswith("p="):
                    for parquet_file in p_dir.glob("*.parquet"):
                        dfs.append(pl.read_parquet(parquet_file))
        
        if dfs:
            self.fact_store._df = pl.concat(dfs, how="vertical")
            # Update txn counter
            if len(self.fact_store._df) > 0:
                max_txn = self.fact_store._df["txn"].max()
                if max_txn is not None:
                    self.fact_store._next_txn = int(max_txn) + 1
    
    # =========================================================================
    # Write Path (Batch Ingestion)
    # =========================================================================
    
    def add_facts_batch(
        self,
        facts: List[tuple],  # (g, s, p, o)
        source: Optional[TermId] = None,
        confidence: float = 1.0,
        process: Optional[TermId] = None,
    ) -> int:
        """
        Add a batch of facts with shared provenance.
        
        Facts are added to the in-memory store and buffered for delta writes.
        """
        txn = self.fact_store.add_facts_batch(
            facts,
            source=source,
            confidence=confidence,
            process=process,
        )
        
        # Buffer for delta write
        self._delta_txn_count += 1
        
        # Check if we should flush
        if len(self.fact_store) > self.delta_flush_threshold:
            self._flush_delta_buffer()
        
        return txn
    
    def _flush_delta_buffer(self):
        """Write buffered facts to delta Parquet files."""
        if len(self.fact_store) == 0:
            return
        
        df = self.fact_store.to_dataframe()
        
        # Group by predicate
        predicates = df.select("p").unique()["p"].to_list()
        
        for p in predicates:
            p_df = df.filter(pl.col("p") == p)
            
            # Create partition directory
            delta_dir = self._partition_delta_path(p)
            delta_dir.mkdir(parents=True, exist_ok=True)
            
            # Write with txn-based filename
            txn = p_df["txn"].max()
            delta_file = delta_dir / f"batch-{txn}.parquet"
            p_df.write_parquet(delta_file)
    
    # =========================================================================
    # Compaction
    # =========================================================================
    
    def compact_partition(self, predicate_id: TermId):
        """
        Compact a single partition.
        
        Algorithm (from storage-spec.md §7.4):
        1. Read base files for partition
        2. Read delta files for partition
        3. Concatenate → stable sort by (s, o, txn)
        4. Groupby (g, s, p, o) selecting latest row
        5. Write new base part file(s)
        6. Delete compacted delta files
        7. Update partition stats
        """
        base_dir = self._partition_base_path(predicate_id)
        delta_dir = self._partition_delta_path(predicate_id)
        
        dfs = []
        delta_files = []
        
        # Read base files
        if base_dir.exists():
            for f in base_dir.glob("*.parquet"):
                dfs.append(pl.read_parquet(f))
        
        # Read delta files
        if delta_dir.exists():
            for f in delta_dir.glob("*.parquet"):
                dfs.append(pl.read_parquet(f))
                delta_files.append(f)
        
        if not dfs:
            return
        
        # Concatenate
        combined = pl.concat(dfs, how="vertical")
        
        # Sort by (s, o, txn)
        combined = combined.sort(["s", "o", "txn"])
        
        # Deduplicate: keep latest (max txn) for each (g, s, p, o)
        # If deleted, apply last-write-wins
        compacted = combined.group_by(["g", "s", "p", "o"]).agg([
            pl.col("flags").last(),
            pl.col("txn").last(),
            pl.col("t_added").last(),
            pl.col("source").last(),
            pl.col("confidence").last(),
            pl.col("process").last(),
        ])
        
        # Remove tombstones (deleted facts) during compaction
        compacted = compacted.filter(
            (pl.col("flags") & int(FactFlags.DELETED)) == 0
        )
        
        # Write new base file
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove old base files
        for f in base_dir.glob("*.parquet"):
            f.unlink()
        
        # Write compacted data
        if len(compacted) > 0:
            compacted.write_parquet(base_dir / "part-0.parquet")
        
        # Delete compacted delta files
        for f in delta_files:
            f.unlink()
        
        # Update stats
        self._partition_stats[predicate_id] = PartitionStats(
            predicate_id=predicate_id,
            graph_id=None,
            row_count=len(compacted),
            min_subject=compacted["s"].min() if len(compacted) > 0 else None,
            max_subject=compacted["s"].max() if len(compacted) > 0 else None,
            distinct_subjects=compacted.select("s").n_unique() if len(compacted) > 0 else 0,
            file_size_bytes=(base_dir / "part-0.parquet").stat().st_size if len(compacted) > 0 else 0,
            last_compacted=datetime.now(timezone.utc),
        )
    
    def compact_all(self):
        """Compact all partitions."""
        predicates = set()
        
        # Collect predicates from base
        if self._facts_base_path.exists():
            for p_dir in self._facts_base_path.iterdir():
                if p_dir.is_dir() and p_dir.name.startswith("p="):
                    p_id = int(p_dir.name.split("=")[1])
                    predicates.add(p_id)
        
        # Collect predicates from delta
        if self._facts_delta_path.exists():
            for p_dir in self._facts_delta_path.iterdir():
                if p_dir.is_dir() and p_dir.name.startswith("p="):
                    p_id = int(p_dir.name.split("=")[1])
                    predicates.add(p_id)
        
        for p_id in predicates:
            self.compact_partition(p_id)
        
        self._save_partition_stats()
    
    # =========================================================================
    # Query Primitives
    # =========================================================================
    
    def scan_partition(
        self,
        predicate_id: TermId,
        include_deleted: bool = False,
    ) -> pl.DataFrame:
        """
        Scan a specific predicate partition.
        
        Reads from both base and delta, returning combined results.
        This enables partition pruning for predicate-selective queries.
        """
        dfs = []
        
        # Read base
        base_dir = self._partition_base_path(predicate_id)
        if base_dir.exists():
            for f in base_dir.glob("*.parquet"):
                dfs.append(pl.read_parquet(f))
        
        # Read delta
        delta_dir = self._partition_delta_path(predicate_id)
        if delta_dir.exists():
            for f in delta_dir.glob("*.parquet"):
                dfs.append(pl.read_parquet(f))
        
        if not dfs:
            return self.fact_store._create_empty_dataframe()
        
        result = pl.concat(dfs, how="vertical")
        
        if not include_deleted:
            result = result.filter(
                (pl.col("flags") & int(FactFlags.DELETED)) == 0
            )
        
        return result
    
    def get_partition_stats(self, predicate_id: TermId) -> Optional[PartitionStats]:
        """Get statistics for a partition."""
        return self._partition_stats.get(predicate_id)
    
    def list_partitions(self) -> List[TermId]:
        """List all partition predicate IDs."""
        return list(self._partition_stats.keys())
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def stats(self) -> dict:
        """Return storage statistics."""
        total_base_size = 0
        total_delta_size = 0
        
        if self._facts_base_path.exists():
            for f in self._facts_base_path.rglob("*.parquet"):
                total_base_size += f.stat().st_size
        
        if self._facts_delta_path.exists():
            for f in self._facts_delta_path.rglob("*.parquet"):
                total_delta_size += f.stat().st_size
        
        return {
            "term_dict": self.term_dict.stats(),
            "qt_dict": self.qt_dict.stats(),
            "fact_store": self.fact_store.stats(),
            "partitions": len(self._partition_stats),
            "base_size_bytes": total_base_size,
            "delta_size_bytes": total_delta_size,
            "total_size_bytes": total_base_size + total_delta_size,
        }
