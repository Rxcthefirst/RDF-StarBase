"""
Quoted Triple Dictionary.

Implements the qt_dict catalog for RDF★ quoted triples.
Quoted triples are first-class terms that can appear as subjects or objects.

Key design decisions (from storage-spec.md):
- qt_id is a TermId with QUOTED_TRIPLE kind
- Graph-agnostic quoting: key is (s,p,o) only (simpler, lower cardinality)
- Hash-based interning for fast bulk dedupe
- Stores qt_hash for fast rebuild at startup
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import struct
import hashlib

import polars as pl

from rdf_starbase.storage.terms import (
    TermId,
    TermKind,
    TermDict,
    make_term_id,
    get_term_payload,
)


# Type alias for quoted triple identifiers
QtId = TermId  # QtId is a TermId with kind=QUOTED_TRIPLE


@dataclass(frozen=True, slots=True)
class QuotedTriple:
    """
    Internal representation of a quoted triple.
    
    All components are TermIds (already dictionary-encoded).
    """
    s: TermId
    p: TermId
    o: TermId
    
    def to_tuple(self) -> Tuple[TermId, TermId, TermId]:
        """Return as tuple for hashing."""
        return (self.s, self.p, self.o)
    
    def compute_hash(self) -> int:
        """Compute 128-bit hash for bulk deduplication."""
        data = struct.pack('>QQQ', self.s, self.p, self.o)
        h = hashlib.md5(data).digest()
        return int.from_bytes(h, 'big')


class QtDict:
    """
    Quoted Triple Dictionary.
    
    Catalogs quoted triples and assigns them stable QtIds (which are TermIds).
    Supports O(1) lookup by (s,p,o) tuple and by qt_id.
    
    Relationship with TermDict:
    - QtDict allocates from the QUOTED_TRIPLE ID space
    - The qt_id can be used as a subject or object in facts
    - TermDict handles IRIs, literals, and bnodes; QtDict handles quoted triples
    """
    
    def __init__(self, term_dict: TermDict):
        """
        Initialize the quoted triple dictionary.
        
        Args:
            term_dict: The TermDict to coordinate ID allocation with
        """
        self._term_dict = term_dict
        
        # Hash -> QtId (for interning)
        self._hash_to_id: dict[int, QtId] = {}
        
        # QtId -> QuotedTriple (for expansion)
        self._id_to_qt: dict[QtId, QuotedTriple] = {}
        
        # Statistics
        self._collision_count = 0
    
    def _allocate_id(self) -> QtId:
        """Allocate the next QtId."""
        # Use TermDict's allocation to keep ID spaces coordinated
        payload = self._term_dict._next_payload[TermKind.QUOTED_TRIPLE]
        self._term_dict._next_payload[TermKind.QUOTED_TRIPLE] = payload + 1
        return make_term_id(TermKind.QUOTED_TRIPLE, payload)
    
    def get_or_create(self, s: TermId, p: TermId, o: TermId) -> QtId:
        """
        Intern a quoted triple, returning its QtId.
        
        If the quoted triple already exists, returns the existing ID.
        Otherwise, allocates a new ID and stores the triple.
        
        Args:
            s: Subject TermId
            p: Predicate TermId
            o: Object TermId
            
        Returns:
            QtId for the quoted triple
        """
        qt = QuotedTriple(s, p, o)
        qt_hash = qt.compute_hash()
        
        if qt_hash in self._hash_to_id:
            existing_id = self._hash_to_id[qt_hash]
            # Verify it's actually the same triple (hash collision check)
            if self._id_to_qt[existing_id] == qt:
                return existing_id
            # Hash collision
            self._collision_count += 1
        
        # Allocate new ID
        qt_id = self._allocate_id()
        self._hash_to_id[qt_hash] = qt_id
        self._id_to_qt[qt_id] = qt
        
        return qt_id
    
    def get_or_create_batch(
        self, 
        triples: list[Tuple[TermId, TermId, TermId]]
    ) -> list[QtId]:
        """
        Bulk intern a batch of quoted triples.
        
        Optimized for ingestion performance.
        """
        return [self.get_or_create(s, p, o) for s, p, o in triples]
    
    def lookup(self, qt_id: QtId) -> Optional[QuotedTriple]:
        """
        Expand a QtId to its (s,p,o) components.
        
        This is the critical operation for RDF★ expansion joins.
        """
        return self._id_to_qt.get(qt_id)
    
    def lookup_batch(self, qt_ids: list[QtId]) -> list[Optional[QuotedTriple]]:
        """
        Bulk expand QtIds to their components.
        
        Returns QuotedTriple objects (or None for unknown IDs).
        """
        return [self._id_to_qt.get(qt_id) for qt_id in qt_ids]
    
    def expand_to_dataframe(self, qt_ids: list[QtId]) -> pl.DataFrame:
        """
        Expand a list of QtIds to a DataFrame with columns: qt_id, s, p, o.
        
        This is the storage primitive for RDF★ expansion joins
        (storage-spec.md §8: lookup_qt).
        """
        rows = []
        for qt_id in qt_ids:
            qt = self._id_to_qt.get(qt_id)
            if qt is not None:
                rows.append({
                    "qt_id": qt_id,
                    "s": qt.s,
                    "p": qt.p,
                    "o": qt.o,
                })
        
        if not rows:
            return pl.DataFrame({
                "qt_id": pl.Series([], dtype=pl.UInt64),
                "s": pl.Series([], dtype=pl.UInt64),
                "p": pl.Series([], dtype=pl.UInt64),
                "o": pl.Series([], dtype=pl.UInt64),
            })
        
        return pl.DataFrame(rows).cast({
            "qt_id": pl.UInt64,
            "s": pl.UInt64,
            "p": pl.UInt64,
            "o": pl.UInt64,
        })
    
    def get_id(self, s: TermId, p: TermId, o: TermId) -> Optional[QtId]:
        """Get the QtId for a triple if it exists, without creating it."""
        qt = QuotedTriple(s, p, o)
        qt_hash = qt.compute_hash()
        if qt_hash not in self._hash_to_id:
            return None
        existing_id = self._hash_to_id[qt_hash]
        if self._id_to_qt[existing_id] == qt:
            return existing_id
        return None
    
    def contains(self, s: TermId, p: TermId, o: TermId) -> bool:
        """Check if a quoted triple is already interned."""
        return self.get_id(s, p, o) is not None
    
    def __len__(self) -> int:
        """Return the total number of quoted triples."""
        return len(self._id_to_qt)
    
    @property
    def collision_count(self) -> int:
        """Return the number of hash collisions encountered."""
        return self._collision_count
    
    # =========================================================================
    # Persistence (Parquet)
    # =========================================================================
    
    def to_dataframe(self) -> pl.DataFrame:
        """
        Export the quoted triple dictionary to a Polars DataFrame.
        
        Schema matches storage-spec.md §3.3:
        - qt_id: u64
        - s: u64
        - p: u64
        - o: u64
        - qt_hash: stored as two u64 columns (hash_high, hash_low)
        """
        if not self._id_to_qt:
            return pl.DataFrame({
                "qt_id": pl.Series([], dtype=pl.UInt64),
                "s": pl.Series([], dtype=pl.UInt64),
                "p": pl.Series([], dtype=pl.UInt64),
                "o": pl.Series([], dtype=pl.UInt64),
                "hash_high": pl.Series([], dtype=pl.UInt64),
                "hash_low": pl.Series([], dtype=pl.UInt64),
            })
        
        rows = []
        for qt_id, qt in self._id_to_qt.items():
            qt_hash = qt.compute_hash()
            rows.append({
                "qt_id": qt_id,
                "s": qt.s,
                "p": qt.p,
                "o": qt.o,
                "hash_high": qt_hash >> 64,
                "hash_low": qt_hash & ((1 << 64) - 1),
            })
        
        return pl.DataFrame(rows).cast({
            "qt_id": pl.UInt64,
            "s": pl.UInt64,
            "p": pl.UInt64,
            "o": pl.UInt64,
            "hash_high": pl.UInt64,
            "hash_low": pl.UInt64,
        })
    
    def save(self, path: Path):
        """Save the quoted triple dictionary to a Parquet file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().write_parquet(path / "qt_dict.parquet")
    
    @classmethod
    def load(cls, path: Path, term_dict: TermDict) -> "QtDict":
        """Load a quoted triple dictionary from a Parquet file."""
        path = Path(path)
        
        instance = cls(term_dict)
        
        df = pl.read_parquet(path / "qt_dict.parquet")
        for row in df.iter_rows(named=True):
            qt_id = row["qt_id"]
            qt = QuotedTriple(row["s"], row["p"], row["o"])
            qt_hash = (row["hash_high"] << 64) | row["hash_low"]
            
            instance._id_to_qt[qt_id] = qt
            instance._hash_to_id[qt_hash] = qt_id
            
            # Update sequence counter in TermDict
            payload = get_term_payload(qt_id)
            if payload >= term_dict._next_payload[TermKind.QUOTED_TRIPLE]:
                term_dict._next_payload[TermKind.QUOTED_TRIPLE] = payload + 1
        
        return instance
    
    def stats(self) -> dict:
        """Return statistics about the quoted triple dictionary."""
        return {
            "total_quoted_triples": len(self),
            "hash_collisions": self._collision_count,
        }
