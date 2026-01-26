"""
Persistence layer for RDF-StarBase storage.

Provides save/load functionality for the dictionary-encoded storage layer:
- TermDict: Term catalog (term_id, kind, lex)
- FactStore: Facts table (g, s, p, o, provenance)
- QtDict: Quoted triples table (qt_id, s_id, p_id, o_id)

Uses Parquet format for efficient, columnar storage with good compression.

Incremental Persistence (IncrementalPersistence class):
- Delta files for new data since last compaction
- Manifest tracks segments and versions
- Compaction merges deltas into base when threshold exceeded
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import os
import json
import glob

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


# =============================================================================
# Incremental Persistence
# =============================================================================

class IncrementalPersistence:
    """
    Incremental persistence with delta files and compaction.
    
    Instead of rewriting the entire dataset on each save, this class:
    1. Tracks which rows have been persisted (by txn ID)
    2. Writes only new rows to delta files
    3. Compacts deltas into base when threshold is exceeded
    
    File layout:
        base_path/
            manifest.json         - Tracks segments and last persisted state
            base/
                terms.parquet     - Compacted terms
                facts.parquet     - Compacted facts  
                quoted.parquet    - Compacted quoted triples
                metadata.parquet  - Counters
            deltas/
                delta_0001_terms.parquet
                delta_0001_facts.parquet
                delta_0001_quoted.parquet
                delta_0002_terms.parquet
                ...
    
    Manifest structure:
        {
            "version": 1,
            "last_compacted_txn": 1000,
            "last_compacted_term_count": 5000,
            "last_compacted_qt_count": 100,
            "delta_count": 3,
            "compaction_threshold": 10,
            "created_at": "2026-01-25T12:00:00Z",
            "last_save_at": "2026-01-25T14:00:00Z"
        }
    """
    
    MANIFEST_FILE = "manifest.json"
    BASE_DIR = "base"
    DELTAS_DIR = "deltas"
    DEFAULT_COMPACTION_THRESHOLD = 10  # Compact after this many deltas
    
    def __init__(self, base_path: str | Path, compaction_threshold: int = None):
        """
        Initialize incremental persistence.
        
        Args:
            base_path: Directory for storage files
            compaction_threshold: Number of deltas before auto-compaction
        """
        self.base_path = Path(base_path)
        self.compaction_threshold = compaction_threshold or self.DEFAULT_COMPACTION_THRESHOLD
        self._manifest = None
    
    def _load_manifest(self) -> dict:
        """Load or create manifest."""
        manifest_path = self.base_path / self.MANIFEST_FILE
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return {
            "version": 1,
            "last_compacted_txn": -1,
            "last_compacted_term_count": 0,
            "last_compacted_qt_count": 0,
            "delta_count": 0,
            "compaction_threshold": self.compaction_threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_save_at": None,
        }
    
    def _save_manifest(self, manifest: dict) -> None:
        """Save manifest to disk."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.base_path / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def save(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict,
        force_full: bool = False
    ) -> dict:
        """
        Save storage incrementally.
        
        Only writes data that has changed since the last save/compaction.
        
        Args:
            term_dict: Term dictionary to save
            fact_store: Fact store to save
            qt_dict: Quoted triple dictionary to save
            force_full: If True, force a full compaction write
            
        Returns:
            Dict with save statistics (delta_rows, was_compacted, etc.)
        """
        self.base_path.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        
        # Check if this is first save or force full
        base_dir = self.base_path / self.BASE_DIR
        if force_full or not base_dir.exists():
            return self._save_full(term_dict, fact_store, qt_dict, manifest)
        
        # Calculate what's new since last compaction
        last_txn = manifest["last_compacted_txn"]
        last_term_count = manifest["last_compacted_term_count"]
        last_qt_count = manifest["last_compacted_qt_count"]
        
        # Get new facts (txn > last_compacted_txn)
        if last_txn >= 0:
            new_facts_df = fact_store._df.filter(pl.col("txn") > last_txn)
        else:
            new_facts_df = fact_store._df
        
        # Get new terms (count > last_compacted_term_count)
        current_term_count = len(term_dict._id_to_term)
        new_term_count = current_term_count - last_term_count
        
        # Get new quoted triples
        current_qt_count = len(qt_dict._id_to_qt)
        new_qt_count = current_qt_count - last_qt_count
        
        # If nothing new, skip save
        if len(new_facts_df) == 0 and new_term_count == 0 and new_qt_count == 0:
            return {
                "status": "no_changes",
                "delta_facts": 0,
                "delta_terms": 0,
                "delta_quoted": 0,
                "was_compacted": False,
            }
        
        # Write delta files
        delta_num = manifest["delta_count"] + 1
        deltas_dir = self.base_path / self.DELTAS_DIR
        deltas_dir.mkdir(parents=True, exist_ok=True)
        
        delta_prefix = f"delta_{delta_num:04d}"
        
        # Save delta facts
        if len(new_facts_df) > 0:
            new_facts_df.write_parquet(deltas_dir / f"{delta_prefix}_facts.parquet")
        
        # Save delta terms (only new ones)
        if new_term_count > 0:
            new_terms = self._get_new_terms(term_dict, last_term_count)
            if len(new_terms) > 0:
                new_terms.write_parquet(deltas_dir / f"{delta_prefix}_terms.parquet")
        
        # Save delta quoted triples (only new ones)
        if new_qt_count > 0:
            new_qt = self._get_new_quoted(qt_dict, last_qt_count)
            if len(new_qt) > 0:
                new_qt.write_parquet(deltas_dir / f"{delta_prefix}_quoted.parquet")
        
        # Update manifest with current state
        # This ensures the next save only writes truly new data
        max_txn = -1
        if len(fact_store._df) > 0 and "txn" in fact_store._df.columns:
            max_txn_val = fact_store._df["txn"].max()
            if max_txn_val is not None:
                max_txn = max_txn_val
        
        manifest["delta_count"] = delta_num
        manifest["last_compacted_txn"] = max_txn
        manifest["last_compacted_term_count"] = len(term_dict._id_to_term)
        manifest["last_compacted_qt_count"] = len(qt_dict._id_to_qt)
        manifest["last_save_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)
        
        stats = {
            "status": "delta_saved",
            "delta_num": delta_num,
            "delta_facts": len(new_facts_df),
            "delta_terms": new_term_count,
            "delta_quoted": new_qt_count,
            "was_compacted": False,
        }
        
        # Auto-compact if threshold exceeded
        if delta_num >= manifest["compaction_threshold"]:
            self.compact(term_dict, fact_store, qt_dict)
            stats["was_compacted"] = True
            stats["status"] = "compacted"
        
        return stats
    
    def _get_new_terms(self, term_dict: TermDict, last_count: int) -> pl.DataFrame:
        """Get terms added after last_count."""
        # Terms are stored by ID, we need to extract the newest ones
        # Since term IDs are not sequential by insertion order, we need
        # to track this differently. For now, extract all and filter.
        all_terms = []
        for term_id, term in term_dict._id_to_term.items():
            all_terms.append({
                "term_id": term_id,
                "kind": term.kind.value,
                "lex": term.lex,
            })
        
        if len(all_terms) <= last_count:
            return pl.DataFrame({
                "term_id": pl.Series([], dtype=pl.UInt64),
                "kind": pl.Series([], dtype=pl.UInt8),
                "lex": pl.Series([], dtype=pl.Utf8),
            })
        
        # Take only new terms (assuming insertion order in dict)
        new_terms = all_terms[last_count:]
        return pl.DataFrame({
            "term_id": pl.Series([t["term_id"] for t in new_terms], dtype=pl.UInt64),
            "kind": pl.Series([t["kind"] for t in new_terms], dtype=pl.UInt8),
            "lex": pl.Series([t["lex"] for t in new_terms], dtype=pl.Utf8),
        })
    
    def _get_new_quoted(self, qt_dict: QtDict, last_count: int) -> pl.DataFrame:
        """Get quoted triples added after last_count."""
        all_qt = []
        for qt_id, qt in qt_dict._id_to_qt.items():
            all_qt.append({
                "qt_id": qt_id,
                "s": qt.s,
                "p": qt.p,
                "o": qt.o,
            })
        
        if len(all_qt) <= last_count:
            return pl.DataFrame({
                "qt_id": pl.Series([], dtype=pl.UInt64),
                "s": pl.Series([], dtype=pl.UInt64),
                "p": pl.Series([], dtype=pl.UInt64),
                "o": pl.Series([], dtype=pl.UInt64),
            })
        
        new_qt = all_qt[last_count:]
        return pl.DataFrame({
            "qt_id": pl.Series([q["qt_id"] for q in new_qt], dtype=pl.UInt64),
            "s": pl.Series([q["s"] for q in new_qt], dtype=pl.UInt64),
            "p": pl.Series([q["p"] for q in new_qt], dtype=pl.UInt64),
            "o": pl.Series([q["o"] for q in new_qt], dtype=pl.UInt64),
        })
    
    def _save_full(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict,
        manifest: dict
    ) -> dict:
        """Perform a full save (first save or forced compaction)."""
        base_dir = self.base_path / self.BASE_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Use StoragePersistence for the actual write
        base_persistence = StoragePersistence(base_dir)
        base_persistence.save(term_dict, fact_store, qt_dict)
        
        # Update manifest
        max_txn = -1
        if len(fact_store._df) > 0 and "txn" in fact_store._df.columns:
            max_txn_val = fact_store._df["txn"].max()
            if max_txn_val is not None:
                max_txn = max_txn_val
        
        manifest["last_compacted_txn"] = max_txn
        manifest["last_compacted_term_count"] = len(term_dict._id_to_term)
        manifest["last_compacted_qt_count"] = len(qt_dict._id_to_qt)
        manifest["delta_count"] = 0
        manifest["last_save_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)
        
        # Clean up any old deltas
        deltas_dir = self.base_path / self.DELTAS_DIR
        if deltas_dir.exists():
            for f in deltas_dir.glob("delta_*.parquet"):
                f.unlink()
        
        return {
            "status": "full_save",
            "total_facts": len(fact_store._df),
            "total_terms": len(term_dict._id_to_term),
            "total_quoted": len(qt_dict._id_to_qt),
            "was_compacted": True,
        }
    
    def compact(
        self,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict
    ) -> dict:
        """
        Compact all deltas into base.
        
        Merges all delta files into the base files and removes deltas.
        """
        manifest = self._load_manifest()
        return self._save_full(term_dict, fact_store, qt_dict, manifest)
    
    def load(self) -> tuple[TermDict, FactStore, QtDict]:
        """
        Load storage, merging base + all deltas.
        
        After loading, updates the manifest to reflect the current state
        so subsequent saves only write truly new data.
        
        Returns:
            Tuple of (TermDict, FactStore, QtDict)
        """
        if not self.base_path.exists():
            raise FileNotFoundError(f"Storage directory not found: {self.base_path}")
        
        manifest = self._load_manifest()
        base_dir = self.base_path / self.BASE_DIR
        
        # Load base using StoragePersistence
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        
        base_persistence = StoragePersistence(base_dir)
        term_dict, fact_store, qt_dict = base_persistence.load()
        
        # Apply deltas in order
        deltas_dir = self.base_path / self.DELTAS_DIR
        if deltas_dir.exists():
            delta_nums = self._get_delta_numbers(deltas_dir)
            for delta_num in sorted(delta_nums):
                self._apply_delta(delta_num, deltas_dir, term_dict, fact_store, qt_dict)
        
        # Update fact_store._next_txn to be max(txn) + 1
        # This ensures new facts get unique txn IDs
        max_txn = -1
        if len(fact_store._df) > 0 and "txn" in fact_store._df.columns:
            max_txn_val = fact_store._df["txn"].max()
            if max_txn_val is not None:
                max_txn = max_txn_val
                fact_store._next_txn = max_txn + 1
        
        # Update manifest to reflect current loaded state
        # This ensures subsequent saves only write truly new data
        manifest["last_compacted_txn"] = max_txn
        manifest["last_compacted_term_count"] = len(term_dict._id_to_term)
        manifest["last_compacted_qt_count"] = len(qt_dict._id_to_qt)
        self._save_manifest(manifest)
        
        return term_dict, fact_store, qt_dict
    
    def _get_delta_numbers(self, deltas_dir: Path) -> set:
        """Get set of delta numbers from delta files."""
        delta_nums = set()
        for f in deltas_dir.glob("delta_*_*.parquet"):
            # Parse delta_0001_facts.parquet -> 1
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    delta_nums.add(int(parts[1]))
                except ValueError:
                    pass
        return delta_nums
    
    def _apply_delta(
        self,
        delta_num: int,
        deltas_dir: Path,
        term_dict: TermDict,
        fact_store: FactStore,
        qt_dict: QtDict
    ) -> None:
        """Apply a single delta to the in-memory stores."""
        prefix = f"delta_{delta_num:04d}"
        
        # Apply terms delta
        terms_file = deltas_dir / f"{prefix}_terms.parquet"
        if terms_file.exists():
            terms_df = pl.read_parquet(terms_file)
            for row in terms_df.iter_rows(named=True):
                term_id = row["term_id"]
                kind = TermKind(row["kind"])
                lex = row["lex"]
                term = Term(kind=kind, lex=lex)
                term_dict._id_to_term[term_id] = term
                term_dict._hash_to_id[term.compute_hash()] = term_id
                # Update caches
                if kind == TermKind.IRI:
                    term_dict._iri_cache[lex] = term_id
                elif kind == TermKind.BNODE:
                    term_dict._bnode_cache[lex] = term_id
                elif kind == TermKind.LITERAL:
                    term_dict._plain_literal_cache[lex] = term_id
        
        # Apply quoted triples delta
        quoted_file = deltas_dir / f"{prefix}_quoted.parquet"
        if quoted_file.exists():
            qt_df = pl.read_parquet(quoted_file)
            for row in qt_df.iter_rows(named=True):
                qt_id = row["qt_id"]
                qt = QuotedTriple(row["s"], row["p"], row["o"])
                qt_dict._id_to_qt[qt_id] = qt
                qt_dict._hash_to_id[hash(qt)] = qt_id
        
        # Apply facts delta
        facts_file = deltas_dir / f"{prefix}_facts.parquet"
        if facts_file.exists():
            facts_df = pl.read_parquet(facts_file)
            if len(facts_df) > 0:
                fact_store._df = pl.concat([fact_store._df, facts_df], how="vertical")
    
    def exists(self) -> bool:
        """Check if incremental storage exists."""
        return (
            self.base_path.exists() and
            (self.base_path / self.MANIFEST_FILE).exists()
        )
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        if not self.exists():
            return {"exists": False}
        
        manifest = self._load_manifest()
        deltas_dir = self.base_path / self.DELTAS_DIR
        
        delta_files = []
        if deltas_dir.exists():
            delta_files = list(deltas_dir.glob("delta_*.parquet"))
        
        return {
            "exists": True,
            "delta_count": manifest["delta_count"],
            "last_compacted_txn": manifest["last_compacted_txn"],
            "last_compacted_term_count": manifest["last_compacted_term_count"],
            "compaction_threshold": manifest["compaction_threshold"],
            "delta_files": len(delta_files),
            "created_at": manifest.get("created_at"),
            "last_save_at": manifest.get("last_save_at"),
        }


def save_incremental(
    base_path: str | Path,
    term_dict: TermDict,
    fact_store: FactStore,
    qt_dict: QtDict,
    force_full: bool = False
) -> dict:
    """
    Convenience function for incremental save.
    
    Args:
        base_path: Directory path for storage files
        term_dict: Term dictionary to save
        fact_store: Fact store to save
        qt_dict: Quoted triple dictionary to save
        force_full: Force full compaction write
        
    Returns:
        Save statistics dict
    """
    persistence = IncrementalPersistence(base_path)
    return persistence.save(term_dict, fact_store, qt_dict, force_full=force_full)


def load_incremental(base_path: str | Path) -> tuple[TermDict, FactStore, QtDict]:
    """
    Convenience function to load from incremental storage.
    
    Args:
        base_path: Directory path containing storage files
        
    Returns:
        Tuple of (TermDict, FactStore, QtDict)
    """
    persistence = IncrementalPersistence(base_path)
    return persistence.load()
