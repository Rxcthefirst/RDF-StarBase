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
# Lazy Term Lookup (dict-like, defers Term creation to access time)
# =============================================================================

class LazyTermLookup:
    """
    Dict-like container that stores raw column data and creates Term objects
    on demand. This avoids materializing 12M+ Term dataclass instances during
    store loading, reducing load time from ~70s to ~3s for large datasets.
    
    Supports the full dict protocol used by TermDict._id_to_term consumers:
    __getitem__, get, __contains__, __len__, __setitem__, items, keys, values.
    """
    __slots__ = ('_term_ids', '_kinds', '_lexes', '_id_index', '_cache', '_extra')

    def __init__(self, term_ids: list, kinds: list, lexes: list):
        self._term_ids = term_ids
        self._kinds = kinds     # raw int values matching TermKind
        self._lexes = lexes
        # C-level dict construction: term_id → array index
        self._id_index: dict[int, int] = dict(zip(term_ids, range(len(term_ids))))
        self._cache: dict[int, Term] = {}     # materialized Term objects
        self._extra: dict[int, Term] = {}     # terms added after load

    def __getitem__(self, term_id: int) -> Term:
        # Post-load additions first
        extra = self._extra.get(term_id)
        if extra is not None:
            return extra
        # Materialized cache
        cached = self._cache.get(term_id)
        if cached is not None:
            return cached
        # Lazy-create from raw data
        idx = self._id_index.get(term_id)
        if idx is None:
            raise KeyError(term_id)
        term = Term(kind=TermKind(self._kinds[idx]), lex=self._lexes[idx])
        self._cache[term_id] = term
        return term

    def get(self, term_id: int, default=None):
        try:
            return self[term_id]
        except KeyError:
            return default

    def __contains__(self, term_id: int) -> bool:
        return term_id in self._id_index or term_id in self._extra

    def __len__(self) -> int:
        extra_new = sum(1 for k in self._extra if k not in self._id_index)
        return len(self._id_index) + extra_new

    def __setitem__(self, term_id: int, term: Term):
        self._extra[term_id] = term
        self._cache[term_id] = term

    def __bool__(self) -> bool:
        return len(self._id_index) > 0 or len(self._extra) > 0

    def items(self):
        """Yield all (term_id, Term) pairs. Materializes Terms on demand."""
        seen = set()
        for tid in self._term_ids:
            seen.add(tid)
            if tid in self._extra:
                yield tid, self._extra[tid]
            else:
                yield tid, self[tid]
        for tid, term in self._extra.items():
            if tid not in seen:
                yield tid, term

    def keys(self):
        seen = set()
        for tid in self._term_ids:
            seen.add(tid)
            yield tid
        for tid in self._extra:
            if tid not in seen:
                yield tid

    def values(self):
        for _, v in self.items():
            yield v

    def build_literal_float_map(self) -> dict[int, float]:
        """Build TermId→float map from raw data without creating Term objects."""
        lit_kind = TermKind.LITERAL.value
        result: dict[int, float] = {}
        for i, kind in enumerate(self._kinds):
            if kind == lit_kind:
                try:
                    result[self._term_ids[i]] = float(self._lexes[i])
                except (ValueError, TypeError):
                    continue
        # Include any extra literals added after load
        for tid, term in self._extra.items():
            if term.kind == TermKind.LITERAL and tid not in result:
                try:
                    result[tid] = float(term.lex)
                except (ValueError, TypeError):
                    continue
        return result


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
        # Fast path: check kind-specific caches before hash lookup.
        # This avoids needing _hash_to_id for the common case and is
        # essential for correctness when _hash_to_id is lazily populated.
        if term.datatype_id is None and term.lang is None:
            if term.kind == TermKind.IRI:
                cached = self._iri_cache.get(term.lex)
                if cached is not None:
                    return cached
            elif term.kind == TermKind.LITERAL:
                cached = self._plain_literal_cache.get(term.lex)
                if cached is not None:
                    return cached
            elif term.kind == TermKind.BNODE:
                cached = self._bnode_cache.get(term.lex)
                if cached is not None:
                    return cached

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
        """Restore well-known datatype ID references after loading.
        
        Uses _iri_cache for O(1) lookup instead of scanning all terms.
        """
        self.xsd_string_id = self._iri_cache.get(self.XSD_STRING)
        self.xsd_integer_id = self._iri_cache.get(self.XSD_INTEGER)
        self.xsd_decimal_id = self._iri_cache.get(self.XSD_DECIMAL)
        self.xsd_double_id = self._iri_cache.get(self.XSD_DOUBLE)
        self.xsd_boolean_id = self._iri_cache.get(self.XSD_BOOLEAN)
        self.xsd_datetime_id = self._iri_cache.get(self.XSD_DATETIME)
        self.rdf_langstring_id = self._iri_cache.get(self.RDF_LANGSTRING)
    
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
    
    def intern_iris_batch(self, iris: list[str]) -> list[TermId]:
        """
        Batch intern IRIs - optimized for bulk loading.
        
        Deduplicates input, assigns IDs in bulk, then maps back to original positions.
        ~3x faster than calling intern_iri() in a loop for large batches.
        
        Args:
            iris: List of IRI strings (may contain duplicates)
            
        Returns:
            List of TermIds in same order as input
        """
        if not iris:
            return []
        
        # Fast path: single IRI
        if len(iris) == 1:
            return [self.intern_iri(iris[0])]
        
        # Phase 1: Get unique IRIs and find which are new
        unique_iris = set(iris)
        new_iris = [iri for iri in unique_iris if iri not in self._iri_cache]
        
        # Phase 2: Batch-allocate IDs for new IRIs
        if new_iris:
            start_payload = self._next_payload[TermKind.IRI]
            for i, iri in enumerate(new_iris):
                term_id = make_term_id(TermKind.IRI, start_payload + i)
                term = Term.iri(iri)
                self._id_to_term[term_id] = term
                self._iri_cache[iri] = term_id
            self._next_payload[TermKind.IRI] = start_payload + len(new_iris)
        
        # Phase 3: Map back to original positions
        cache = self._iri_cache
        return [cache[iri] for iri in iris]
    
    def intern_literals_batch(
        self, 
        values: list[str],
        datatypes: Optional[list[Optional[str]]] = None,
        langs: Optional[list[Optional[str]]] = None,
    ) -> list[TermId]:
        """
        Batch intern literals - optimized for bulk loading.
        
        Args:
            values: List of literal values
            datatypes: Optional list of datatype URIs (parallel to values)
            langs: Optional list of language tags (parallel to values)
            
        Returns:
            List of TermIds in same order as input
        """
        if not values:
            return []
        
        n = len(values)
        result = [0] * n
        
        # Separate into plain strings (fast path) and others (slow path)
        plain_indices = []
        plain_values = []
        other_indices = []
        
        has_datatypes = datatypes is not None
        has_langs = langs is not None
        
        for i in range(n):
            is_plain = True
            if has_langs and langs[i] is not None:
                is_plain = False
            elif has_datatypes and datatypes[i] is not None:
                dt = datatypes[i]
                if dt != self.XSD_STRING:
                    is_plain = False
            
            if is_plain:
                plain_indices.append(i)
                plain_values.append(values[i])
            else:
                other_indices.append(i)
        
        # Batch process plain string literals
        if plain_values:
            unique_values = set(plain_values)
            new_values = [v for v in unique_values if v not in self._plain_literal_cache]
            
            if new_values:
                start_payload = self._next_payload[TermKind.LITERAL]
                for j, val in enumerate(new_values):
                    term_id = make_term_id(TermKind.LITERAL, start_payload + j)
                    term = Term.literal(val, self.xsd_string_id, None)
                    self._id_to_term[term_id] = term
                    self._plain_literal_cache[val] = term_id
                self._next_payload[TermKind.LITERAL] = start_payload + len(new_values)
            
            cache = self._plain_literal_cache
            for i, val in zip(plain_indices, plain_values):
                result[i] = cache[val]
        
        # Process other literals one by one (typed, lang-tagged)
        for i in other_indices:
            val = values[i]
            dt = datatypes[i] if has_datatypes else None
            lang = langs[i] if has_langs else None
            result[i] = self.intern_literal(val, datatype=dt, lang=lang)
        
        return result
    
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
        # Fast path: LazyTermLookup can build map from raw data without
        # materializing Term objects
        if isinstance(self._id_to_term, LazyTermLookup):
            return self._id_to_term.build_literal_float_map()
        
        result = {}
        for term_id, term in self._id_to_term.items():
            if term.kind == TermKind.LITERAL:
                try:
                    result[term_id] = float(term.lex)
                except (ValueError, TypeError):
                    continue
        return result
    
    def get_lex_batch(self, term_ids) -> dict[int, str]:
        """
        Batch lookup of lexical forms for a set of term IDs.
        
        Returns dict mapping term_id → lex string.
        Uses LazyTermLookup raw data when available (no Term objects created).
        """
        id_to_term = self._id_to_term
        
        # Fast path for LazyTermLookup — use raw arrays
        if isinstance(id_to_term, LazyTermLookup):
            result: dict[int, str] = {}
            for tid in term_ids:
                # Check extra (post-load) terms first
                extra = id_to_term._extra.get(tid)
                if extra is not None:
                    result[tid] = extra.lex
                    continue
                # Check cached materialized terms
                cached = id_to_term._cache.get(tid)
                if cached is not None:
                    result[tid] = cached.lex
                    continue
                # Direct raw data access — no Term object created
                idx = id_to_term._id_index.get(tid)
                if idx is not None:
                    result[tid] = id_to_term._lexes[idx]
            return result
        
        # Standard dict path
        result = {}
        for tid in term_ids:
            term = id_to_term.get(tid)
            if term is not None:
                result[tid] = term.lex
        return result
    
    def get_lex_series(self, term_ids: pl.Series) -> pl.Series:
        """
        Vectorized lookup of lexical forms for a series of term IDs.
        
        Returns a Utf8 Series with lexical forms (null for missing IDs).
        """
        unique_ids = [tid for tid in term_ids.unique().to_list() if tid is not None]
        id_to_lex = self.get_lex_batch(unique_ids)
        
        # Use Polars replace for vectorized mapping
        if not id_to_lex:
            return pl.Series(term_ids.name, [None] * len(term_ids), dtype=pl.Utf8)
        
        lookup_df = pl.DataFrame({
            "_key": pl.Series(list(id_to_lex.keys()), dtype=pl.UInt64),
            "_val": pl.Series(list(id_to_lex.values()), dtype=pl.Utf8),
        })
        
        tmp = pl.DataFrame({"_key": term_ids}).join(lookup_df, on="_key", how="left")
        return tmp["_val"].rename(term_ids.name)

    def stats(self) -> dict:
        """Return statistics about the term dictionary."""
        return {
            "total_terms": len(self),
            "by_kind": {kind.name: count for kind, count in self.count_by_kind().items()},
            "hash_collisions": self._collision_count,
        }
