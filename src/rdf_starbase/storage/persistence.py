"""
Persistence layer for RDF-StarBase storage.

Provides save/load functionality for the dictionary-encoded storage layer:
- TermDict: Term catalog (term_id, kind, lex)
- FactStore: Facts table (g, s, p, o, provenance)
- QtDict: Quoted triples table (qt_id, s_id, p_id, o_id)

Uses Parquet format for efficient, columnar storage with good compression.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import os

import polars as pl

from rdf_starbase.storage.terms import TermDict, Term, TermId, TermKind, make_term_id
from rdf_starbase.storage.quoted_triples import QtDict, QuotedTriple
from rdf_starbase.storage.facts import FactStore


class StoragePersistence:
    """
    Handles save/load operations for the storage layer.
    
    File layout:
        base_path/
            terms.parquet     - TermDict catalog
            facts.parquet     - FactStore facts
            quoted.parquet    - QtDict quoted triples
            metadata.parquet  - Counters and metadata
    """
    
    TERMS_FILE = "terms.parquet"
    FACTS_FILE = "facts.parquet"
    QUOTED_FILE = "quoted.parquet"
    METADATA_FILE = "metadata.parquet"
    
    def __init__(self, base_path: str | Path):
        """
        Initialize persistence with a base directory path.
        
        Args:
            base_path: Directory where storage files will be saved/loaded
        """
        self.base_path = Path(base_path)
    
    def save(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict
    ) -> None:
        """
        Save all storage components to disk.
        
        Args:
            term_dict: The term dictionary to save
            fact_store: The fact store to save
            qt_dict: The quoted triple dictionary to save
        """
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Save term dictionary
        self._save_terms(term_dict)
        
        # Save facts
        self._save_facts(fact_store)
        
        # Save quoted triples
        self._save_quoted(qt_dict)
        
        # Save metadata (counters, etc.)
        self._save_metadata(term_dict, fact_store, qt_dict)
    
    def load(self) -> tuple[TermDict, FactStore, QtDict]:
        """
        Load all storage components from disk.
        
        Args:
            memory_map: If True, use memory-mapped loading for the facts table.
                       This reduces memory usage for large datasets by lazily
                       loading data from disk as needed. Default False.
        
        Returns:
            Tuple of (TermDict, FactStore, QtDict)
            
        Raises:
            FileNotFoundError: If the storage directory doesn't exist
        """
        if not self.base_path.exists():
            raise FileNotFoundError(f"Storage directory not found: {self.base_path}")
        
        # Load term dictionary first (needed by others)
        term_dict = self._load_terms()
        
        # Load quoted triples (needed by fact_store)
        qt_dict = self._load_quoted(term_dict)
        
        # Load facts
        fact_store = self._load_facts(term_dict, qt_dict)
        
        # Restore metadata
        self._load_metadata(term_dict, fact_store, qt_dict)
        
        return term_dict, fact_store, qt_dict
    
    def load_streaming(self) -> tuple[TermDict, FactStore, QtDict]:
        """
        Load storage with memory-mapped Parquet for large datasets.
        
        This method uses Polars scan_parquet() for the facts table, which
        memory-maps the file and only loads data as needed. This is ideal
        for datasets larger than available RAM.
        
        Returns:
            Tuple of (TermDict, FactStore, QtDict)
            
        Raises:
            FileNotFoundError: If the storage directory doesn't exist
        """
        if not self.base_path.exists():
            raise FileNotFoundError(f"Storage directory not found: {self.base_path}")
        
        # Terms and quoted must be fully loaded (used for lookups)
        term_dict = self._load_terms()
        qt_dict = self._load_quoted(term_dict)
        
        # Load facts with memory-mapping
        fact_store = self._load_facts_streaming(term_dict, qt_dict)
        
        # Restore metadata
        self._load_metadata(term_dict, fact_store, qt_dict)
        
        return term_dict, fact_store, qt_dict
    
    def exists(self) -> bool:
        """Check if a saved storage exists at the base path."""
        return (
            self.base_path.exists() and
            (self.base_path / self.TERMS_FILE).exists()
        )
    
    def _save_terms(self, term_dict: TermDict) -> None:
        """Save term dictionary to Parquet."""
        # Build DataFrame from term_dict internal state
        term_ids = []
        kinds = []
        lexes = []
        
        for term_id, term in term_dict._id_to_term.items():
            term_ids.append(term_id)
            kinds.append(term.kind.value)
            lexes.append(term.lex)
        
        df = pl.DataFrame({
            "term_id": pl.Series(term_ids, dtype=pl.UInt64),
            "kind": pl.Series(kinds, dtype=pl.UInt8),
            "lex": pl.Series(lexes, dtype=pl.Utf8),
        })
        
        df.write_parquet(self.base_path / self.TERMS_FILE)
    
    def _load_terms(self) -> TermDict:
        """Load term dictionary from Parquet."""
        df = pl.read_parquet(self.base_path / self.TERMS_FILE)
        
        term_dict = TermDict.__new__(TermDict)
        term_dict._next_payload = {
            TermKind.IRI: 0,
            TermKind.LITERAL: 0,
            TermKind.BNODE: 0,
            TermKind.QUOTED_TRIPLE: 0,
        }
        term_dict._hash_to_id = {}
        term_dict._id_to_term = {}
        term_dict._collision_count = 0
        
        # Initialize fast-path caches (added for performance)
        term_dict._iri_cache = {}
        term_dict._plain_literal_cache = {}
        term_dict._bnode_cache = {}
        
        # Restore terms
        for row in df.iter_rows(named=True):
            term_id = row["term_id"]
            kind = TermKind(row["kind"])
            lex = row["lex"]
            
            term = Term(kind=kind, lex=lex)
            term_dict._id_to_term[term_id] = term
            term_dict._hash_to_id[term.compute_hash()] = term_id
            
            # Populate fast-path caches
            if kind == TermKind.IRI:
                term_dict._iri_cache[lex] = term_id
            elif kind == TermKind.BNODE:
                term_dict._bnode_cache[lex] = term_id
            elif kind == TermKind.LITERAL:
                term_dict._plain_literal_cache[lex] = term_id
        
        return term_dict
    
    def _save_facts(self, fact_store: FactStore) -> None:
        """Save fact store to Parquet."""
        fact_store._df.write_parquet(self.base_path / self.FACTS_FILE)
    
    def _load_facts(self, term_dict: TermDict, qt_dict: QtDict) -> FactStore:
        """Load fact store from Parquet."""
        fact_store = FactStore.__new__(FactStore)
        fact_store._term_dict = term_dict
        fact_store._qt_dict = qt_dict
        fact_store._next_txn = 0
        fact_store._default_graph_id = 0
        
        facts_path = self.base_path / self.FACTS_FILE
        if facts_path.exists():
            fact_store._df = pl.read_parquet(facts_path)
        else:
            fact_store._df = fact_store._create_empty_dataframe()
        
        return fact_store
    
    def _load_facts_streaming(self, term_dict: TermDict, qt_dict: QtDict) -> FactStore:
        """
        Load fact store with memory-mapped Parquet (lazy/streaming).
        
        Uses scan_parquet() which memory-maps the file and defers loading
        until data is actually accessed. The LazyFrame is collected into
        a DataFrame but Polars optimizes memory usage for large files.
        """
        fact_store = FactStore.__new__(FactStore)
        fact_store._term_dict = term_dict
        fact_store._qt_dict = qt_dict
        fact_store._next_txn = 0
        fact_store._default_graph_id = 0
        
        facts_path = self.base_path / self.FACTS_FILE
        if facts_path.exists():
            # Use scan_parquet for memory-mapped lazy loading
            # memory_map=True tells Polars to use mmap for the file
            lazy_df = pl.scan_parquet(facts_path, memory_map=True)
            # Collect immediately but Polars will use streaming internally
            # for files larger than available memory
            fact_store._df = lazy_df.collect(streaming=True)
        else:
            fact_store._df = fact_store._create_empty_dataframe()
        
        return fact_store
    
    def _save_quoted(self, qt_dict: QtDict) -> None:
        """Save quoted triple dictionary to Parquet."""
        qt_ids = []
        s_ids = []
        p_ids = []
        o_ids = []
        
        for qt_id, qt in qt_dict._id_to_qt.items():
            qt_ids.append(qt_id)
            s_ids.append(qt.s)
            p_ids.append(qt.p)
            o_ids.append(qt.o)
        
        df = pl.DataFrame({
            "qt_id": pl.Series(qt_ids, dtype=pl.UInt64),
            "s": pl.Series(s_ids, dtype=pl.UInt64),
            "p": pl.Series(p_ids, dtype=pl.UInt64),
            "o": pl.Series(o_ids, dtype=pl.UInt64),
        })
        
        df.write_parquet(self.base_path / self.QUOTED_FILE)
    
    def _load_quoted(self, term_dict: TermDict) -> QtDict:
        """Load quoted triple dictionary from Parquet."""
        qt_dict = QtDict.__new__(QtDict)
        qt_dict._term_dict = term_dict
        qt_dict._hash_to_id = {}
        qt_dict._id_to_qt = {}
        qt_dict._collision_count = 0
        
        quoted_path = self.base_path / self.QUOTED_FILE
        if quoted_path.exists():
            df = pl.read_parquet(quoted_path)
            
            for row in df.iter_rows(named=True):
                qt_id = row["qt_id"]
                qt = QuotedTriple(row["s"], row["p"], row["o"])
                qt_dict._id_to_qt[qt_id] = qt
                qt_dict._hash_to_id[hash(qt)] = qt_id
        
        return qt_dict
    
    def _save_metadata(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict
    ) -> None:
        """Save counters and metadata to Parquet."""
        # Store counter values for each kind
        df = pl.DataFrame({
            "key": [
                "next_iri", "next_literal", "next_bnode", "next_qt", "next_txn"
            ],
            "value": [
                term_dict._next_payload[TermKind.IRI],
                term_dict._next_payload[TermKind.LITERAL],
                term_dict._next_payload[TermKind.BNODE],
                term_dict._next_payload[TermKind.QUOTED_TRIPLE],
                fact_store._next_txn,
            ],
        })
        
        df.write_parquet(self.base_path / self.METADATA_FILE)
    
    def _load_metadata(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict
    ) -> None:
        """Restore counters and metadata from Parquet."""
        metadata_path = self.base_path / self.METADATA_FILE
        if not metadata_path.exists():
            # Infer counters from loaded data
            self._infer_counters(term_dict, fact_store)
            return
        
        df = pl.read_parquet(metadata_path)
        
        # Build a lookup dict
        meta = dict(zip(df["key"].to_list(), df["value"].to_list()))
        
        term_dict._next_payload[TermKind.IRI] = meta.get("next_iri", 0)
        term_dict._next_payload[TermKind.LITERAL] = meta.get("next_literal", 0)
        term_dict._next_payload[TermKind.BNODE] = meta.get("next_bnode", 0)
        term_dict._next_payload[TermKind.QUOTED_TRIPLE] = meta.get("next_qt", 0)
        fact_store._next_txn = meta.get("next_txn", 0)
        
        # Re-initialize well-known IDs
        term_dict._init_well_known()
    
    def _infer_counters(
        self,
        term_dict: TermDict,
        fact_store: FactStore
    ) -> None:
        """Infer counter values from loaded data."""
        # Find max payload for each kind
        for term_id, term in term_dict._id_to_term.items():
            kind = term.kind
            payload = term_id & 0x00FFFFFFFFFFFFFF  # Extract payload
            if payload >= term_dict._next_payload[kind]:
                term_dict._next_payload[kind] = payload + 1
        
        # Infer next_txn from facts
        if len(fact_store._df) > 0 and "txn" in fact_store._df.columns:
            max_txn = fact_store._df["txn"].max()
            if max_txn is not None:
                fact_store._next_txn = max_txn + 1
        
        # Re-initialize well-known IDs
        term_dict._init_well_known()


def save_storage(
    base_path: str | Path,
    term_dict: TermDict,
    fact_store: FactStore,
    qt_dict: QtDict
) -> None:
    """
    Convenience function to save storage to disk.
    
    Args:
        base_path: Directory path for storage files
        term_dict: Term dictionary to save
        fact_store: Fact store to save
        qt_dict: Quoted triple dictionary to save
    """
    persistence = StoragePersistence(base_path)
    persistence.save(term_dict, fact_store, qt_dict)


def load_storage(base_path: str | Path) -> tuple[TermDict, FactStore, QtDict]:
    """
    Convenience function to load storage from disk.
    
    Args:
        base_path: Directory path containing storage files
        
    Returns:
        Tuple of (TermDict, FactStore, QtDict)
    """
    persistence = StoragePersistence(base_path)
    return persistence.load()
