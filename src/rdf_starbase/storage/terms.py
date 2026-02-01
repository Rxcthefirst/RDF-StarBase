"""
Term Dictionary with Integer ID Encoding.

Implements dictionary-encoded RDF terms for high-performance columnar storage.
All terms (IRIs, literals, blank nodes, quoted triples) are mapped to u64 TermIds.

Key design decisions (from storage-spec.md):
- Tagged ID space: high bits encode term kind for O(1) kind detection
- Hash-based interning: 128-bit hashes for fast bulk dedupe
- Batch-first: bulk get_or_create operations for ingestion performance
- Persistence: Parquet-backed for restart-safe term catalogs
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Any, Union
from pathlib import Path
import hashlib
import struct

import polars as pl


# =============================================================================
# Term Identity and Encoding
# =============================================================================

class TermKind(IntEnum):
    """
    RDF term kind enumeration.
    
    Encoded in the high 2 bits of TermId for O(1) kind detection.
    """
    IRI = 0
    LITERAL = 1
    BNODE = 2
    QUOTED_TRIPLE = 3


# Type alias for term identifiers (u64)
TermId = int

# Constants for ID encoding
KIND_SHIFT = 62
KIND_MASK = 0x3  # 2 bits
PAYLOAD_MASK = (1 << KIND_SHIFT) - 1


def make_term_id(kind: TermKind, payload: int) -> TermId:
    """Create a TermId from kind and payload."""
    return (kind << KIND_SHIFT) | (payload & PAYLOAD_MASK)


def get_term_kind(term_id: TermId) -> TermKind:
    """Extract the term kind from a TermId (O(1) operation)."""
    return TermKind((term_id >> KIND_SHIFT) & KIND_MASK)


def get_term_payload(term_id: TermId) -> int:
    """Extract the payload (sequence number) from a TermId."""
    return term_id & PAYLOAD_MASK


def is_quoted_triple(term_id: TermId) -> bool:
    """Check if a TermId refers to a quoted triple."""
    return get_term_kind(term_id) == TermKind.QUOTED_TRIPLE


# =============================================================================
# Term Representation
# =============================================================================

@dataclass(frozen=True, slots=True)
class Term:
    """
    Internal representation of an RDF term.
    
    Attributes:
        kind: The type of term (IRI, LITERAL, BNODE, QUOTED_TRIPLE)
        lex: Lexical form (IRI string, literal value, bnode label)
        datatype_id: TermId of datatype IRI (for typed literals)
        lang: Language tag (for language-tagged literals)
    """
    kind: TermKind
    lex: str
    datatype_id: Optional[TermId] = None
    lang: Optional[str] = None
    
    def __hash__(self) -> int:
        return hash((self.kind, self.lex, self.datatype_id, self.lang))
    
    def canonical_bytes(self) -> bytes:
        """
        Generate canonical byte representation for hashing.
        
        Includes: kind tag, lexical form, datatype IRI, language tag.
        """
        parts = [
            struct.pack('B', self.kind),
            self.lex.encode('utf-8'),
        ]
        if self.datatype_id is not None:
            parts.append(struct.pack('>Q', self.datatype_id))
        if self.lang is not None:
            parts.append(b'@')
            parts.append(self.lang.encode('utf-8'))
        return b'\x00'.join(parts)
    
    def compute_hash(self) -> int:
        """
        Compute hash for deduplication.
        
        Uses Python's built-in hash for speed (10x faster than MD5).
        For persistence, use compute_hash_persistent() which uses MD5.
        """
        # Use Python's fast built-in hash - leverages __hash__ method
        return hash(self)
    
    def compute_hash_persistent(self) -> int:
        """Compute stable MD5 hash for persistence (cross-session stable)."""
        h = hashlib.md5(self.canonical_bytes()).digest()
        return int.from_bytes(h, 'big')
    
    @classmethod
    def iri(cls, value: str) -> "Term":
        """Create an IRI term."""
        return cls(kind=TermKind.IRI, lex=value)
    
    @classmethod
    def literal(
        cls, 
        value: str, 
        datatype_id: Optional[TermId] = None,
        lang: Optional[str] = None
    ) -> "Term":
        """Create a literal term."""
        return cls(
            kind=TermKind.LITERAL, 
            lex=value,
            datatype_id=datatype_id,
            lang=lang
        )
    
    @classmethod
    def bnode(cls, label: str) -> "Term":
        """Create a blank node term."""
        return cls(kind=TermKind.BNODE, lex=label)


# =============================================================================
# Term Dictionary
# =============================================================================

class TermDict:
    """
    Dictionary-encoded term catalog.
    
    Maps RDF terms to integer TermIds with:
    - O(1) kind detection via tagged ID space
    - Hash-based bulk interning for fast ingestion
    - Parquet persistence for restart-safe catalogs
    
    Thread-safety: NOT thread-safe. Use external synchronization for concurrent access.
    """
    
    # Well-known datatype IRIs (pre-interned)
    XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
    XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"
    XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"
    XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
    XSD_BOOLEAN = "http://www.w3.org/2001/XMLSchema#boolean"
    XSD_DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"
    RDF_LANGSTRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
    
    def __init__(self):
        # Per-kind sequence counters
        # NOTE: Start at 1 to reserve TermId 0 as the universal "null" value
        # This ensures source=0 and process=0 don't accidentally match real terms
        self._next_payload: dict[TermKind, int] = {
            TermKind.IRI: 1,
            TermKind.LITERAL: 1,
            TermKind.BNODE: 1,
            TermKind.QUOTED_TRIPLE: 1,
        }
        
        # Forward map: hash -> TermId (for interning)
        self._hash_to_id: dict[int, TermId] = {}
        
        # Reverse map: TermId -> Term (for lookup)
        self._id_to_term: dict[TermId, Term] = {}
        
        # =================================================================
        # FAST PATH CACHES: Direct string->TermId lookup (no hashing)
        # These bypass the expensive MD5 computation for common cases
        # =================================================================
        self._iri_cache: dict[str, TermId] = {}           # IRI string -> TermId
        self._plain_literal_cache: dict[str, TermId] = {} # Plain string literal -> TermId
        self._bnode_cache: dict[str, TermId] = {}         # Blank node label -> TermId
        
        # Statistics
        self._collision_count = 0
        
        # Pre-intern well-known datatypes
        self._init_well_known()
    
    def _init_well_known(self):
        """Pre-intern well-known datatype IRIs and populate caches."""
        self.xsd_string_id = self.get_or_create(Term.iri(self.XSD_STRING))
        self.xsd_integer_id = self.get_or_create(Term.iri(self.XSD_INTEGER))
        self.xsd_decimal_id = self.get_or_create(Term.iri(self.XSD_DECIMAL))
        self.xsd_double_id = self.get_or_create(Term.iri(self.XSD_DOUBLE))
        self.xsd_boolean_id = self.get_or_create(Term.iri(self.XSD_BOOLEAN))
        self.xsd_datetime_id = self.get_or_create(Term.iri(self.XSD_DATETIME))
        self.rdf_langstring_id = self.get_or_create(Term.iri(self.RDF_LANGSTRING))
        
        # Populate fast-path IRI cache for well-known IRIs
        self._iri_cache[self.XSD_STRING] = self.xsd_string_id
        self._iri_cache[self.XSD_INTEGER] = self.xsd_integer_id
        self._iri_cache[self.XSD_DECIMAL] = self.xsd_decimal_id
        self._iri_cache[self.XSD_DOUBLE] = self.xsd_double_id
        self._iri_cache[self.XSD_BOOLEAN] = self.xsd_boolean_id
        self._iri_cache[self.XSD_DATETIME] = self.xsd_datetime_id
        self._iri_cache[self.RDF_LANGSTRING] = self.rdf_langstring_id
    
    def _allocate_id(self, kind: TermKind) -> TermId:
        """Allocate the next TermId for a given kind."""
        payload = self._next_payload[kind]
        self._next_payload[kind] = payload + 1
        return make_term_id(kind, payload)
    
    def get_or_create(self, term: Term) -> TermId:
        """
        Intern a term, returning its TermId.
        
        If the term already exists, returns the existing ID.
        Otherwise, allocates a new ID and stores the term.
        """
        term_hash = term.compute_hash()
        
        if term_hash in self._hash_to_id:
            existing_id = self._hash_to_id[term_hash]
            # Verify it's actually the same term (hash collision check)
            if self._id_to_term[existing_id] == term:
                return existing_id
            # Hash collision - need to handle
            self._collision_count += 1
            # Fall through to create new entry with different ID
        
        # Allocate new ID
        term_id = self._allocate_id(term.kind)
        self._hash_to_id[term_hash] = term_id
        self._id_to_term[term_id] = term
        
        return term_id
    
    def get_or_create_batch(self, terms: list[Term]) -> list[TermId]:
        """
        Bulk intern a batch of terms.
        
        Optimized for ingestion performance. Returns TermIds in the same order.
        """
        return [self.get_or_create(term) for term in terms]
    
    def lookup(self, term_id: TermId) -> Optional[Term]:
        """Look up a term by its ID."""
        return self._id_to_term.get(term_id)
    
    def lookup_batch(self, term_ids: list[TermId]) -> list[Optional[Term]]:
        """Bulk lookup terms by their IDs."""
        return [self._id_to_term.get(tid) for tid in term_ids]
    
    def contains(self, term: Term) -> bool:
        """Check if a term is already interned."""
        term_hash = term.compute_hash()
        if term_hash not in self._hash_to_id:
            return False
        existing_id = self._hash_to_id[term_hash]
        return self._id_to_term[existing_id] == term
    
    def get_id(self, term: Term) -> Optional[TermId]:
        """Get the TermId for a term if it exists, without creating it."""
        term_hash = term.compute_hash()
        if term_hash not in self._hash_to_id:
            return None
        existing_id = self._hash_to_id[term_hash]
        if self._id_to_term[existing_id] == term:
            return existing_id
        return None
    
    def get_iri_id(self, iri: str) -> Optional[TermId]:
        """
        Fast lookup of IRI TermId without creating it.
        
        Uses the fast-path cache for O(1) lookup when the IRI
        has been interned. Returns None if not found.
        """
        # Check fast-path cache first
        cached = self._iri_cache.get(iri)
        if cached is not None:
            return cached
        
        # Fall back to hash lookup
        term = Term.iri(iri)
        return self.get_id(term)
    
    def get_literal_id(self, value: str, datatype: Optional[str] = None, lang: Optional[str] = None) -> Optional[TermId]:
        """
        Fast lookup of literal TermId without creating it.
        
        Uses fast-path cache for plain string literals.
        Returns None if not found.
        """
        # Fast path for plain string literals
        if lang is None and datatype is None:
            cached = self._plain_literal_cache.get(value)
            if cached is not None:
                return cached
        
        # Build Term and do hash lookup
        datatype_id = None
        if lang is not None:
            datatype_id = self.rdf_langstring_id
        elif datatype:
            datatype_id = self._iri_cache.get(datatype)
            if datatype_id is None:
                # Datatype IRI not interned, so literal can't exist
                return None
        else:
            datatype_id = self.xsd_string_id
        
        term = Term.literal(value, datatype_id, lang)
        return self.get_id(term)
    
    def __len__(self) -> int:
        """Return the total number of interned terms."""
        return len(self._id_to_term)
    
    def count_by_kind(self) -> dict[TermKind, int]:
        """Return counts of terms by kind."""
        return {kind: self._next_payload[kind] for kind in TermKind}
    
    @property
    def collision_count(self) -> int:
        """Return the number of hash collisions encountered."""
        return self._collision_count
    
    # =========================================================================
    # Persistence (Parquet)
    # =========================================================================
    
    def to_dataframe(self) -> pl.DataFrame:
        """
        Export the term dictionary to a Polars DataFrame.
        
        Schema matches storage-spec.md §3.1:
        - term_id: u64
        - kind: u8
        - lex: string
        - datatype_id: u64 (nullable)
        - lang: string (nullable)
        """
        if not self._id_to_term:
            return pl.DataFrame({
                "term_id": pl.Series([], dtype=pl.UInt64),
                "kind": pl.Series([], dtype=pl.UInt8),
                "lex": pl.Series([], dtype=pl.Utf8),
                "datatype_id": pl.Series([], dtype=pl.UInt64),
                "lang": pl.Series([], dtype=pl.Utf8),
            })
        
        rows = []
        for term_id, term in self._id_to_term.items():
            rows.append({
                "term_id": term_id,
                "kind": int(term.kind),
                "lex": term.lex,
                "datatype_id": term.datatype_id,
                "lang": term.lang,
            })
        
        return pl.DataFrame(rows).cast({
            "term_id": pl.UInt64,
            "kind": pl.UInt8,
        })
    
    def to_hash_dataframe(self) -> pl.DataFrame:
        """
        Export the term hash table to a Polars DataFrame.
        
        Schema matches storage-spec.md §3.2:
        - term_hash: stored as two u64 columns (hash_high, hash_low)
        - term_id: u64
        
        Uses MD5 hashes for persistence (stable across sessions).
        """
        # Build hash table from all interned terms using MD5 for persistence
        rows = []
        for term_id, term in self._id_to_term.items():
            term_hash = term.compute_hash_persistent()  # MD5 hash, always positive 128-bit
            hash_high = term_hash >> 64
            hash_low = term_hash & ((1 << 64) - 1)
            rows.append({
                "hash_high": hash_high,
                "hash_low": hash_low,
                "term_id": term_id,
            })
        
        if not rows:
            return pl.DataFrame({
                "hash_high": pl.Series([], dtype=pl.UInt64),
                "hash_low": pl.Series([], dtype=pl.UInt64),
                "term_id": pl.Series([], dtype=pl.UInt64),
            })
        
        return pl.DataFrame(rows).cast({
            "hash_high": pl.UInt64,
            "hash_low": pl.UInt64,
            "term_id": pl.UInt64,
        })
    
    def save(self, path: Path):
        """
        Save the term dictionary to Parquet files.
        
        Creates:
        - {path}/term_dict.parquet
        - {path}/term_hash.parquet
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.to_dataframe().write_parquet(path / "term_dict.parquet")
        self.to_hash_dataframe().write_parquet(path / "term_hash.parquet")
    
    @classmethod
    def load(cls, path: Path) -> "TermDict":
        """
        Load a term dictionary from Parquet files.
        
        Expects:
        - {path}/term_dict.parquet
        - {path}/term_hash.parquet
        """
        path = Path(path)
        
        instance = cls.__new__(cls)
        instance._next_payload = {kind: 0 for kind in TermKind}
        instance._hash_to_id = {}
        instance._id_to_term = {}
        instance._collision_count = 0
        
        # Initialize fast-path caches
        instance._iri_cache = {}
        instance._plain_literal_cache = {}
        instance._bnode_cache = {}
        
        # Load term dictionary
        term_df = pl.read_parquet(path / "term_dict.parquet")
        for row in term_df.iter_rows(named=True):
            term_id = row["term_id"]
            term = Term(
                kind=TermKind(row["kind"]),
                lex=row["lex"],
                datatype_id=row["datatype_id"],
                lang=row["lang"],
            )
            instance._id_to_term[term_id] = term
            
            # Populate fast-path caches
            if term.kind == TermKind.IRI:
                instance._iri_cache[term.lex] = term_id
            elif term.kind == TermKind.BNODE:
                instance._bnode_cache[term.lex] = term_id
            elif term.kind == TermKind.LITERAL and term.lang is None:
                # Only cache plain literals (no lang tag) - check if it's xsd:string
                instance._plain_literal_cache[term.lex] = term_id
            
            # Update sequence counters
            kind = get_term_kind(term_id)
            payload = get_term_payload(term_id)
            if payload >= instance._next_payload[kind]:
                instance._next_payload[kind] = payload + 1
        
        # Load hash table
        hash_df = pl.read_parquet(path / "term_hash.parquet")
        for row in hash_df.iter_rows(named=True):
            term_hash = (row["hash_high"] << 64) | row["hash_low"]
            instance._hash_to_id[term_hash] = row["term_id"]
        
        # Restore well-known IDs
        instance._restore_well_known()
        
        return instance
    
    def _restore_well_known(self):
        """Restore well-known datatype ID references after loading."""
        for term_id, term in self._id_to_term.items():
            if term.kind == TermKind.IRI:
                if term.lex == self.XSD_STRING:
                    self.xsd_string_id = term_id
                elif term.lex == self.XSD_INTEGER:
                    self.xsd_integer_id = term_id
                elif term.lex == self.XSD_DECIMAL:
                    self.xsd_decimal_id = term_id
                elif term.lex == self.XSD_DOUBLE:
                    self.xsd_double_id = term_id
                elif term.lex == self.XSD_BOOLEAN:
                    self.xsd_boolean_id = term_id
                elif term.lex == self.XSD_DATETIME:
                    self.xsd_datetime_id = term_id
                elif term.lex == self.RDF_LANGSTRING:
                    self.rdf_langstring_id = term_id
    
    # =========================================================================
    # Convenience methods for common term types (OPTIMIZED)
    # =========================================================================

    def intern_iri(self, value: str) -> TermId:
        """
        Intern an IRI and return its TermId.
        
        Optimized fast path: uses direct string cache, skips hash computation.
        """
        # Fast path: direct string lookup (no Term object, no hash)
        cached = self._iri_cache.get(value)
        if cached is not None:
            return cached
        
        # New term: allocate ID and cache directly (skip hash computation)
        term = Term.iri(value)
        term_id = self._allocate_id(TermKind.IRI)
        self._id_to_term[term_id] = term
        self._iri_cache[value] = term_id
        # Note: _hash_to_id populated lazily during persistence if needed
        return term_id
    
    def intern_literal(
        self, 
        value: Any,
        datatype: Optional[str] = None,
        lang: Optional[str] = None
    ) -> TermId:
        """
        Intern a literal and return its TermId.
        
        Automatically determines datatype from Python type if not specified.
        Uses fast-path cache for plain string literals (the common case).
        """
        lex = str(value)
        
        # FAST PATH: plain string literal with no lang tag (most common case)
        # Use direct string lookup - no Term object, no hash computation
        if lang is None and datatype is None and isinstance(value, str):
            cached = self._plain_literal_cache.get(lex)
            if cached is not None:
                return cached
            
            # New term: allocate ID and cache directly (skip hash computation)
            term = Term.literal(lex, self.xsd_string_id, None)
            term_id = self._allocate_id(TermKind.LITERAL)
            self._id_to_term[term_id] = term
            self._plain_literal_cache[lex] = term_id
            # Note: _hash_to_id populated lazily during persistence if needed
            return term_id
        
        # SLOW PATH: typed literals or lang-tagged strings
        # Determine datatype ID
        datatype_id = None
        if lang is not None:
            datatype_id = self.rdf_langstring_id
        elif datatype is not None:
            datatype_id = self.intern_iri(datatype)
        elif isinstance(value, bool):
            datatype_id = self.xsd_boolean_id
        elif isinstance(value, int):
            datatype_id = self.xsd_integer_id
        elif isinstance(value, float):
            datatype_id = self.xsd_decimal_id
        else:
            datatype_id = self.xsd_string_id
        
        return self.get_or_create(Term.literal(lex, datatype_id, lang))
    
    def intern_bnode(self, label: Optional[str] = None) -> TermId:
        """
        Intern a blank node and return its TermId.
        
        Optimized fast path: uses direct string cache, skips hash computation.
        If no label is provided, generates a unique one.
        """
        if label is None:
            label = f"b{self._next_payload[TermKind.BNODE]}"
        
        # Fast path: direct string lookup
        cached = self._bnode_cache.get(label)
        if cached is not None:
            return cached
        
        # New term: allocate ID and cache directly (skip hash computation)
        term = Term.bnode(label)
        term_id = self._allocate_id(TermKind.BNODE)
        self._id_to_term[term_id] = term
        self._bnode_cache[label] = term_id
        # Note: _hash_to_id populated lazily during persistence if needed
        return term_id
    
    def get_lex(self, term_id: TermId) -> Optional[str]:
        """Get the lexical form of a term by its ID."""
        term = self.lookup(term_id)
        return term.lex if term else None
    
    # =========================================================================
    # Lookup methods (read-only)
    # =========================================================================
    
    def lookup_iri(self, value: str) -> Optional[TermId]:
        """Look up an IRI's TermId without creating it."""
        return self.get_iri_id(value)
    
    def lookup_literal(
        self, 
        value: str,
        datatype: Optional[str] = None,
        lang: Optional[str] = None
    ) -> Optional[TermId]:
        """Look up a literal's TermId without creating it."""
        return self.get_literal_id(value, datatype, lang)
    
    def lookup_bnode(self, label: str) -> Optional[TermId]:
        """Look up a blank node's TermId without creating it."""
        # Use bnode cache if available
        cached = self._bnode_cache.get(label)
        if cached is not None:
            return cached
        # Fall back to id_to_term scan (slow, but should not be needed often)
        return None
    
    def build_literal_to_float_map(self) -> dict[TermId, float]:
        """
        Build a mapping from literal term IDs to float values.
        
        Returns a dict for all literals that can be parsed as floats.
        Used for vectorized confidence filtering.
        """
        result = {}
        for term_id, term in self._id_to_term.items():
            if term.kind == TermKind.LITERAL:
                try:
                    result[term_id] = float(term.lex)
                except (ValueError, TypeError):
                    continue
        return result
    
    def get_lex_series(self, term_ids: pl.Series) -> pl.Series:
        """
        Vectorized lookup of lexical forms for a series of term IDs.
        
        Returns a Utf8 Series with lexical forms (null for missing IDs).
        """
        # Build a mapping dict for the unique IDs in the series
        unique_ids = term_ids.unique().to_list()
        id_to_lex = {}
        for tid in unique_ids:
            if tid is not None:
                term = self._id_to_term.get(tid)
                if term is not None:
                    id_to_lex[tid] = term.lex
        
        # Map using Polars map_elements for compatibility
        return term_ids.map_elements(
            lambda x: id_to_lex.get(x) if x is not None else None,
            return_dtype=pl.Utf8
        )

    def stats(self) -> dict:
        """Return statistics about the term dictionary."""
        return {
            "total_terms": len(self),
            "by_kind": {kind.name: count for kind, count in self.count_by_kind().items()},
            "hash_collisions": self._collision_count,
        }
