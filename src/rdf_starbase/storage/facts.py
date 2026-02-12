"""
Integer-Based Facts Store.

Implements the `facts` table with dictionary-encoded integer columns.
No string terms inside facts - everything is ID-based for maximum performance.

Key design decisions (from storage-spec.md):
- All columns are integer IDs (g, s, p, o are TermIds)
- RDF★ metadata triples stored by setting s/o to QtId
- Batch-first ingestion with monotonic txn IDs
- Flags bitset for asserted/inferred/deleted states
- Predicate-partitioned storage layout
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntFlag
from typing import Optional, List, Tuple, Any
from pathlib import Path
from uuid import UUID, uuid4
import struct

import polars as pl

from rdf_starbase.storage.indexing import IndexManager, SortedIndex, indexed_filter
from rdf_starbase.storage.terms import (
    TermId,
    TermKind,
    TermDict,
    Term,
    get_term_kind,
    is_quoted_triple,
)
from rdf_starbase.storage.quoted_triples import QtDict, QtId


# =============================================================================
# Fact Flags
# =============================================================================

class FactFlags(IntFlag):
    """
    Bitset flags for fact state.
    
    Stored in the `flags` column (u16).
    """
    NONE = 0
    ASSERTED = 1 << 0      # Explicitly asserted
    INFERRED = 1 << 1      # Derived by inference
    DELETED = 1 << 2       # Tombstone (soft delete)
    METADATA = 1 << 3      # This is a metadata triple (s or o is QtId)


# =============================================================================
# Default Graph
# =============================================================================

# The default graph is represented as TermId 0
DEFAULT_GRAPH_ID: TermId = 0


# =============================================================================
# Fact Store
# =============================================================================

class FactStore:
    """
    Integer-based facts store.
    
    Stores quads as (g, s, p, o) where all components are TermIds.
    Supports RDF★ by allowing QtIds as subjects or objects.
    
    Schema matches storage-spec.md §3.4:
    - g: u64 (GraphId, DEFAULT_GRAPH_ID for default)
    - s: u64 (TermId, may be QtId)
    - p: u64 (TermId)
    - o: u64 (TermId, may be QtId)
    - flags: u16 (FactFlags bitset)
    - txn: u64 (transaction/commit ID)
    - t_added: u64 (timestamp, microseconds since epoch)
    
    Also stores provenance columns for backward compatibility:
    - source: u64 (TermId for source IRI/literal)
    - confidence: f64
    - process: u64 (TermId for process IRI)
    """
    
    def __init__(self, term_dict: TermDict, qt_dict: QtDict):
        """
        Initialize the fact store.
        
        Args:
            term_dict: The TermDict for term interning
            qt_dict: The QtDict for quoted triple interning
        """
        self._term_dict = term_dict
        self._qt_dict = qt_dict
        
        # Transaction counter
        self._next_txn: int = 0
        
        # Facts DataFrame with integer columns
        self._df = self._create_empty_dataframe()
        
        # Batch buffer for deferred concat (None = immediate mode)
        self._batch_buffer: Optional[list] = None
        
        # Indexes for fast lookups (s, p, o columns)
        self._index_manager = IndexManager()
        self._index_manager.create_index("s")
        self._index_manager.create_index("p")
        self._index_manager.create_index("o")
        self._indexes_stale = True
        
        # Pre-intern the default graph marker
        self._default_graph_id = DEFAULT_GRAPH_ID
    
    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty facts DataFrame with the correct schema."""
        return pl.DataFrame({
            "g": pl.Series([], dtype=pl.UInt64),
            "s": pl.Series([], dtype=pl.UInt64),
            "p": pl.Series([], dtype=pl.UInt64),
            "o": pl.Series([], dtype=pl.UInt64),
            "flags": pl.Series([], dtype=pl.UInt16),
            "txn": pl.Series([], dtype=pl.UInt64),
            "t_added": pl.Series([], dtype=pl.UInt64),
            # Provenance columns (for backward compatibility)
            "source": pl.Series([], dtype=pl.UInt64),
            "confidence": pl.Series([], dtype=pl.Float64),
            "process": pl.Series([], dtype=pl.UInt64),
        })
    
    def _allocate_txn(self) -> int:
        """Allocate the next transaction ID."""
        txn = self._next_txn
        self._next_txn += 1
        return txn
    
    def add_fact(
        self,
        s: TermId,
        p: TermId,
        o: TermId,
        g: TermId = DEFAULT_GRAPH_ID,
        flags: FactFlags = FactFlags.ASSERTED,
        source: Optional[TermId] = None,
        confidence: float = 1.0,
        process: Optional[TermId] = None,
        t_added: Optional[int] = None,
    ) -> int:
        """
        Add a single fact to the store.
        
        Args:
            s: Subject TermId (may be QtId for metadata triples)
            p: Predicate TermId
            o: Object TermId (may be QtId for metadata triples)
            g: Graph TermId (DEFAULT_GRAPH_ID for default graph)
            flags: Fact flags (ASSERTED by default)
            source: Source TermId for provenance
            confidence: Confidence score (0.0 to 1.0)
            process: Process TermId for provenance
            t_added: Timestamp in microseconds since epoch (default: now)
            
        Returns:
            Transaction ID
        """
        txn = self._allocate_txn()
        if t_added is None:
            t_added = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        # Auto-detect metadata triples
        if is_quoted_triple(s) or is_quoted_triple(o):
            flags = flags | FactFlags.METADATA
        
        new_row = pl.DataFrame({
            "g": [g],
            "s": [s],
            "p": [p],
            "o": [o],
            "flags": [int(flags)],
            "txn": [txn],
            "t_added": [t_added],
            "source": [source if source is not None else 0],
            "confidence": [confidence],
            "process": [process if process is not None else 0],
        }).cast({
            "g": pl.UInt64,
            "s": pl.UInt64,
            "p": pl.UInt64,
            "o": pl.UInt64,
            "flags": pl.UInt16,
            "txn": pl.UInt64,
            "t_added": pl.UInt64,
            "source": pl.UInt64,
            "confidence": pl.Float64,
            "process": pl.UInt64,
        })
        
        self._df = pl.concat([self._df, new_row], how="vertical")
        return txn
    
    def add_facts_batch(
        self,
        facts: List[Tuple[TermId, TermId, TermId, TermId]],  # (g, s, p, o)
        flags: FactFlags = FactFlags.ASSERTED,
        source: Optional[TermId] = None,
        confidence: float = 1.0,
        process: Optional[TermId] = None,
    ) -> int:
        """
        Add a batch of facts with shared provenance.
        
        This is the recommended ingestion path for performance.
        
        Args:
            facts: List of (g, s, p, o) tuples
            flags: Shared flags for all facts
            source: Shared source TermId
            confidence: Shared confidence score
            process: Shared process TermId
            
        Returns:
            Transaction ID for the batch
        """
        if not facts:
            return self._allocate_txn()
        
        txn = self._allocate_txn()
        t_added = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        rows = []
        for g, s, p, o in facts:
            fact_flags = flags
            if is_quoted_triple(s) or is_quoted_triple(o):
                fact_flags = fact_flags | FactFlags.METADATA
            
            rows.append({
                "g": g,
                "s": s,
                "p": p,
                "o": o,
                "flags": int(fact_flags),
                "txn": txn,
                "t_added": t_added,
                "source": source if source is not None else 0,
                "confidence": confidence,
                "process": process if process is not None else 0,
            })
        
        new_df = pl.DataFrame(rows).cast({
            "g": pl.UInt64,
            "s": pl.UInt64,
            "p": pl.UInt64,
            "o": pl.UInt64,
            "flags": pl.UInt16,
            "txn": pl.UInt64,
            "t_added": pl.UInt64,
            "source": pl.UInt64,
            "confidence": pl.Float64,
            "process": pl.UInt64,
        })
        
        if self._batch_buffer is not None:
            self._batch_buffer.append(new_df)
        else:
            self._df = pl.concat([self._df, new_df], how="vertical")
        return txn
    
    def add_facts_columnar(
        self,
        g_col: List[TermId],
        s_col: List[TermId],
        p_col: List[TermId],
        o_col: List[TermId],
        flags: FactFlags = FactFlags.ASSERTED,
        source: Optional[TermId] = None,
        confidence: float = 1.0,
        process: Optional[TermId] = None,
    ) -> int:
        """
        Add facts from pre-built column lists (TRUE vectorized path).
        
        This is the FASTEST ingestion method. Build your column data
        separately, then pass it here for a single DataFrame creation.
        
        Args:
            g_col: List of graph TermIds
            s_col: List of subject TermIds
            p_col: List of predicate TermIds
            o_col: List of object TermIds
            flags: Shared flags for all facts
            source: Shared source TermId
            confidence: Shared confidence score
            process: Shared process TermId
            
        Returns:
            Transaction ID
        """
        n = len(s_col)
        if n == 0:
            return self._allocate_txn()
        
        txn = self._allocate_txn()
        t_added = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        # Build DataFrame directly from columns (no Python loop for repeated values)
        new_df = pl.DataFrame({
            "g": pl.Series(g_col, dtype=pl.UInt64),
            "s": pl.Series(s_col, dtype=pl.UInt64),
            "p": pl.Series(p_col, dtype=pl.UInt64),
            "o": pl.Series(o_col, dtype=pl.UInt64),
        }).with_columns([
            pl.lit(int(flags)).cast(pl.UInt16).alias("flags"),
            pl.lit(txn).cast(pl.UInt64).alias("txn"),
            pl.lit(t_added).cast(pl.UInt64).alias("t_added"),
            pl.lit(source if source else 0).cast(pl.UInt64).alias("source"),
            pl.lit(confidence).alias("confidence"),
            pl.lit(process if process else 0).cast(pl.UInt64).alias("process"),
        ])
        
        if self._batch_buffer is not None:
            self._batch_buffer.append(new_df)
        else:
            self._df = pl.concat([self._df, new_df], how="vertical")
        return txn

    def begin_batch(self) -> None:
        """Begin batched insertion mode. DataFrames are buffered and concatenated once on flush."""
        self._batch_buffer = []

    def flush_batch(self) -> int:
        """Flush all buffered DataFrames in a single concat (avoids O(n²) repeated copies)."""
        if self._batch_buffer is None:
            return 0
        buf = self._batch_buffer
        self._batch_buffer = None
        if not buf:
            return 0
        combined = pl.concat(buf, how="vertical")
        self._df = pl.concat([self._df, combined], how="vertical")
        self._indexes_stale = True
        return len(combined)

    def ensure_indexes(self) -> None:
        """Rebuild indexes if stale. Called lazily before first indexed query."""
        if self._indexes_stale and len(self._df) > 0:
            self._index_manager.build_all(self._df)
            self._indexes_stale = False

    def lookup_by_index(self, column: str, key: int) -> Optional[pl.DataFrame]:
        """Use index for O(log n) point lookup. Returns filtered DataFrame or None if no index."""
        self.ensure_indexes()
        idx = self._index_manager.get_index(column)
        if idx is None:
            return None
        positions = idx.lookup(key)
        if not positions:
            return self._df.head(0)  # empty with same schema
        return self._df[positions]

    def add_facts_with_provenance(
        self,
        facts: List[Tuple[TermId, TermId, TermId, TermId, Optional[TermId], float, Optional[TermId]]],
        flags: FactFlags = FactFlags.ASSERTED,
    ) -> int:
        """
        Add facts with per-fact provenance (confidence, source, process).
        
        This is the recommended path for ingesting data with provenance metadata
        stored in native columns rather than as separate RDF triples.
        
        Args:
            facts: List of (g, s, p, o, source, confidence, process) tuples
                   - source: TermId for data source (or None)
                   - confidence: Float confidence score (0.0 to 1.0)
                   - process: TermId for generating process (or None)
            flags: Base flags for all facts
            
        Returns:
            Transaction ID for the batch
        """
        if not facts:
            return self._allocate_txn()
        
        txn = self._allocate_txn()
        t_added = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        rows = []
        for g, s, p, o, source, confidence, process in facts:
            fact_flags = flags
            if is_quoted_triple(s) or is_quoted_triple(o):
                fact_flags = fact_flags | FactFlags.METADATA
            
            rows.append({
                "g": g,
                "s": s,
                "p": p,
                "o": o,
                "flags": int(fact_flags),
                "txn": txn,
                "t_added": t_added,
                "source": source if source is not None else 0,
                "confidence": confidence,
                "process": process if process is not None else 0,
            })
        
        new_df = pl.DataFrame(rows).cast({
            "g": pl.UInt64,
            "s": pl.UInt64,
            "p": pl.UInt64,
            "o": pl.UInt64,
            "flags": pl.UInt16,
            "txn": pl.UInt64,
            "t_added": pl.UInt64,
            "source": pl.UInt64,
            "confidence": pl.Float64,
            "process": pl.UInt64,
        })
        
        self._df = pl.concat([self._df, new_df], how="vertical")
        return txn
    
    def scan_by_confidence(
        self,
        min_confidence: float,
        max_confidence: Optional[float] = None,
        include_metadata: bool = True,
    ) -> pl.DataFrame:
        """
        Scan facts by confidence threshold using native column.
        
        This is O(n) scan but uses vectorized Polars filtering - 
        no string parsing or joins required.
        
        Args:
            min_confidence: Minimum confidence (exclusive)
            max_confidence: Maximum confidence (inclusive, optional)
            include_metadata: Whether to include metadata facts
            
        Returns:
            DataFrame with all columns for matching facts
        """
        df = self._df.lazy()
        
        # Filter by confidence
        df = df.filter(pl.col("confidence") > min_confidence)
        if max_confidence is not None:
            df = df.filter(pl.col("confidence") <= max_confidence)
        
        # Exclude deleted
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        # Optionally filter out metadata facts
        if not include_metadata:
            df = df.filter((pl.col("flags") & int(FactFlags.METADATA)) == 0)
        
        return df.collect()
    
    def scan_by_source(
        self,
        source: TermId,
        include_metadata: bool = True,
    ) -> pl.DataFrame:
        """
        Scan facts by source using native column.
        
        Args:
            source: Source TermId to filter by
            include_metadata: Whether to include metadata facts
            
        Returns:
            DataFrame with all columns for matching facts
        """
        df = self._df.lazy()
        df = df.filter(pl.col("source") == source)
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        if not include_metadata:
            df = df.filter((pl.col("flags") & int(FactFlags.METADATA)) == 0)
        
        return df.collect()
    
    # =========================================================================
    # Query Primitives (storage-spec.md §8)
    # =========================================================================
    
    def scan_facts(
        self,
        p: Optional[TermId] = None,
        g: Optional[TermId] = None,
        include_deleted: bool = False,
    ) -> pl.DataFrame:
        """
        Scan facts with optional predicate and graph filters.
        
        This is the primary scan primitive for query execution.
        When predicate is specified, this enables partition pruning.
        
        Args:
            p: Optional predicate filter
            g: Optional graph filter
            include_deleted: Whether to include deleted facts
            
        Returns:
            DataFrame with columns: g, s, p, o, flags, txn, t_added, source, confidence, process
        """
        df = self._df.lazy()
        
        if p is not None:
            df = df.filter(pl.col("p") == p)
        
        if g is not None:
            df = df.filter(pl.col("g") == g)
        
        if not include_deleted:
            df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    def scan_facts_by_s(
        self,
        s: TermId,
        p: Optional[TermId] = None,
        g: Optional[TermId] = None,
    ) -> pl.DataFrame:
        """
        Scan facts by subject with optional predicate and graph filters.
        
        Useful for "show me all facts about entity X" queries.
        """
        df = self._df.lazy().filter(pl.col("s") == s)
        
        if p is not None:
            df = df.filter(pl.col("p") == p)
        
        if g is not None:
            df = df.filter(pl.col("g") == g)
        
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    def scan_facts_by_o(
        self,
        o: TermId,
        p: Optional[TermId] = None,
        g: Optional[TermId] = None,
    ) -> pl.DataFrame:
        """
        Scan facts by object with optional predicate and graph filters.
        
        Useful for reverse lookups (inbound edges).
        """
        df = self._df.lazy().filter(pl.col("o") == o)
        
        if p is not None:
            df = df.filter(pl.col("p") == p)
        
        if g is not None:
            df = df.filter(pl.col("g") == g)
        
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    def scan_metadata_facts(
        self,
        qt_id: Optional[QtId] = None,
        p: Optional[TermId] = None,
    ) -> pl.DataFrame:
        """
        Scan facts where subject is a quoted triple (metadata facts).
        
        This is the key primitive for RDF★ metadata queries.
        
        Args:
            qt_id: Optional specific quoted triple to filter by
            p: Optional predicate filter (e.g., prov:wasDerivedFrom)
        """
        df = self._df.lazy().filter(
            (pl.col("flags") & int(FactFlags.METADATA)) != 0
        )
        
        if qt_id is not None:
            df = df.filter(pl.col("s") == qt_id)
        
        if p is not None:
            df = df.filter(pl.col("p") == p)
        
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    def scan_facts_at_time(
        self,
        as_of_time: datetime,
        p: Optional[TermId] = None,
        g: Optional[TermId] = None,
        s: Optional[TermId] = None,
    ) -> pl.DataFrame:
        """
        Time-travel query: return facts as they existed at a specific point in time.
        
        This is a key capability for compliance and auditing:
        - "What did we believe to be true on 2024-01-15?"
        - "What assertions existed before the data refresh?"
        
        Args:
            as_of_time: The point in time to query
            p: Optional predicate filter
            g: Optional graph filter
            s: Optional subject filter
            
        Returns:
            DataFrame with facts that existed at the specified time
        """
        # Convert datetime to microseconds timestamp
        as_of_ts = int(as_of_time.timestamp() * 1_000_000)
        
        df = self._df.lazy()
        
        # Only include facts added before the specified time
        df = df.filter(pl.col("t_added") <= as_of_ts)
        
        # Apply optional filters
        if p is not None:
            df = df.filter(pl.col("p") == p)
        if g is not None:
            df = df.filter(pl.col("g") == g)
        if s is not None:
            df = df.filter(pl.col("s") == s)
        
        # For time-travel, we need to show the state at that time
        # If a fact was deleted after as_of_time, it should still show
        # This implementation shows all facts added by that time
        # (For full versioning, we'd need to track delete timestamps too)
        df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    def scan_facts_by_txn_range(
        self,
        start_txn: int,
        end_txn: Optional[int] = None,
        include_deleted: bool = False,
    ) -> pl.DataFrame:
        """
        Scan facts by transaction ID range.
        
        Useful for:
        - Incremental sync: "give me all changes since txn 1000"
        - Change data capture
        - Debugging specific ingestion batches
        
        Args:
            start_txn: Start transaction ID (inclusive)
            end_txn: End transaction ID (inclusive, optional)
            include_deleted: Whether to include deleted facts
            
        Returns:
            DataFrame with facts in the specified transaction range
        """
        df = self._df.lazy().filter(pl.col("txn") >= start_txn)
        
        if end_txn is not None:
            df = df.filter(pl.col("txn") <= end_txn)
        
        if not include_deleted:
            df = df.filter((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        
        return df.collect()
    
    # =========================================================================
    # RDF★ Expansion Joins
    # =========================================================================
    
    def expand_qt_metadata(
        self,
        metadata_predicate: TermId,
    ) -> pl.DataFrame:
        """
        Critical RDF★ expansion join.
        
        Given a metadata predicate (e.g., prov:wasDerivedFrom), finds all
        quoted triples with that metadata and expands them to (s, p, o).
        
        Returns DataFrame with: qt_id, base_s, base_p, base_o, metadata_o
        
        This implements the join pattern from storage-spec.md §8:
        1. scan_facts(p=metadata_predicate) → qt_id, metadata_o
        2. lookup_qt(qt_ids) → qt_id, s, p, o
        3. join → base triple + metadata value
        """
        # Step 1: Get all facts with the metadata predicate where subject is a qt
        df1 = self._df.lazy().filter(
            (pl.col("p") == metadata_predicate) &
            ((pl.col("flags") & int(FactFlags.METADATA)) != 0) &
            ((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        ).select([
            pl.col("s").alias("qt_id"),
            pl.col("o").alias("metadata_o"),
            pl.col("confidence"),
            pl.col("source"),
        ]).collect()
        
        if df1.is_empty():
            return pl.DataFrame({
                "qt_id": pl.Series([], dtype=pl.UInt64),
                "base_s": pl.Series([], dtype=pl.UInt64),
                "base_p": pl.Series([], dtype=pl.UInt64),
                "base_o": pl.Series([], dtype=pl.UInt64),
                "metadata_o": pl.Series([], dtype=pl.UInt64),
                "confidence": pl.Series([], dtype=pl.Float64),
                "source": pl.Series([], dtype=pl.UInt64),
            })
        
        # Step 2: Expand qt_ids to base triples
        qt_ids = df1["qt_id"].to_list()
        df2 = self._qt_dict.expand_to_dataframe(qt_ids)
        
        if df2.is_empty():
            return pl.DataFrame({
                "qt_id": pl.Series([], dtype=pl.UInt64),
                "base_s": pl.Series([], dtype=pl.UInt64),
                "base_p": pl.Series([], dtype=pl.UInt64),
                "base_o": pl.Series([], dtype=pl.UInt64),
                "metadata_o": pl.Series([], dtype=pl.UInt64),
                "confidence": pl.Series([], dtype=pl.Float64),
                "source": pl.Series([], dtype=pl.UInt64),
            })
        
        # Rename columns for join
        df2 = df2.rename({
            "s": "base_s",
            "p": "base_p",
            "o": "base_o",
        })
        
        # Step 3: Join
        return df1.join(df2, on="qt_id", how="inner")
    
    def expand_metadata_df(self, metadata_df: pl.DataFrame) -> pl.DataFrame:
        """
        Expand a DataFrame of metadata facts.
        
        Takes a DataFrame that has at minimum an 's' column containing qt_ids,
        and expands each qt_id to its (base_s, base_p, base_o) components.
        
        This is useful when you've already filtered metadata facts and
        want to expand them.
        
        Args:
            metadata_df: DataFrame with 's' column containing qt_ids
            
        Returns:
            DataFrame with original columns plus base_s, base_p, base_o
        """
        if metadata_df.is_empty():
            return metadata_df.with_columns([
                pl.lit(0).cast(pl.UInt64).alias("base_s"),
                pl.lit(0).cast(pl.UInt64).alias("base_p"),
                pl.lit(0).cast(pl.UInt64).alias("base_o"),
            ]).filter(pl.lit(False))  # Empty with correct schema
        
        # Get qt_ids from subject column
        qt_ids = metadata_df["s"].to_list()
        
        # Expand using qt_dict
        qt_df = self._qt_dict.expand_to_dataframe(qt_ids)
        
        if qt_df.is_empty():
            return metadata_df.with_columns([
                pl.lit(0).cast(pl.UInt64).alias("base_s"),
                pl.lit(0).cast(pl.UInt64).alias("base_p"),
                pl.lit(0).cast(pl.UInt64).alias("base_o"),
            ]).filter(pl.lit(False))
        
        # Rename for clarity
        qt_df = qt_df.rename({
            "qt_id": "s",  # Match the join key
            "s": "base_s",
            "p": "base_p", 
            "o": "base_o",
        })
        
        # Join on s (the qt_id)
        return metadata_df.join(qt_df, on="s", how="inner")
    
    # =========================================================================
    # Soft Delete and Deprecation
    # =========================================================================
    
    def mark_deleted(
        self,
        s: Optional[TermId] = None,
        p: Optional[TermId] = None,
        o: Optional[TermId] = None,
        g: Optional[TermId] = None,
    ) -> int:
        """
        Soft-delete facts matching the given pattern.
        
        Returns the number of facts marked as deleted.
        """
        mask = pl.lit(True)
        
        if s is not None:
            mask = mask & (pl.col("s") == s)
        if p is not None:
            mask = mask & (pl.col("p") == p)
        if o is not None:
            mask = mask & (pl.col("o") == o)
        if g is not None:
            mask = mask & (pl.col("g") == g)
        
        before_count = self._df.filter(
            mask & ((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        ).height
        
        self._df = self._df.with_columns([
            pl.when(mask)
            .then(pl.col("flags") | int(FactFlags.DELETED))
            .otherwise(pl.col("flags"))
            .alias("flags")
        ])
        
        return before_count
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def to_dataframe(self) -> pl.DataFrame:
        """Return the facts as a DataFrame."""
        return self._df
    
    def save(self, path: Path):
        """Save facts to a Parquet file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(path / "facts.parquet")
        
        # Save transaction counter
        with open(path / "facts_meta.txt", "w") as f:
            f.write(f"next_txn={self._next_txn}\n")
    
    @classmethod
    def load(cls, path: Path, term_dict: TermDict, qt_dict: QtDict) -> "FactStore":
        """Load facts from a Parquet file."""
        path = Path(path)
        
        instance = cls(term_dict, qt_dict)
        instance._df = pl.read_parquet(path / "facts.parquet")
        
        # Load transaction counter
        meta_file = path / "facts_meta.txt"
        if meta_file.exists():
            with open(meta_file) as f:
                for line in f:
                    if line.startswith("next_txn="):
                        instance._next_txn = int(line.split("=")[1].strip())
        
        return instance
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def __len__(self) -> int:
        """Return the total number of facts (including deleted)."""
        return len(self._df)
    
    def count_active(self) -> int:
        """Return the number of non-deleted facts."""
        return self._df.filter(
            (pl.col("flags") & int(FactFlags.DELETED)) == 0
        ).height
    
    def count_metadata(self) -> int:
        """Return the number of metadata facts."""
        return self._df.filter(
            ((pl.col("flags") & int(FactFlags.METADATA)) != 0) &
            ((pl.col("flags") & int(FactFlags.DELETED)) == 0)
        ).height
    
    def stats(self) -> dict:
        """Return statistics about the fact store."""
        active = self.count_active()
        metadata = self.count_metadata()
        
        return {
            "total_facts": len(self),
            "active_facts": active,
            "deleted_facts": len(self) - active,
            "metadata_facts": metadata,
            "base_facts": active - metadata,
            "next_txn": self._next_txn,
            "unique_predicates": self._df.select("p").n_unique(),
            "unique_subjects": self._df.select("s").n_unique(),
        }
