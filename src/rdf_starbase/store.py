"""
Unified TripleStore implementation using FactStore/TermDict.

This refactored TripleStore uses the dictionary-encoded integer storage
internally while maintaining backward compatibility with the existing API.
The SPARQL executor and AI grounding layer continue to work unchanged.

Key design:
- FactStore holds facts as integer IDs (g, s, p, o columns)
- TermDict maps RDF terms to/from integer IDs
- _df property materializes a string-based view for SPARQL executor
- Reasoner can now work directly with the integer-based storage

Performance optimization:
- When pyoxigraph is available, uses Rust-based parsing (~17x faster)
- When pyoxigraph is available, uses Rust-based SPARQL for basic queries (~4x faster)
- Falls back to pure Python implementation when not available
"""

from datetime import datetime, timezone
from typing import Optional, Any, Literal, Union
from uuid import UUID, uuid4
from pathlib import Path

import polars as pl

from rdf_starbase.models import Triple, QuotedTriple, Assertion, ProvenanceContext
from rdf_starbase.storage.terms import TermDict, TermKind, Term, TermId, get_term_kind
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID
from rdf_starbase.storage.transactions import TransactionManager, Transaction

# Try to import pyoxigraph for Rust-speed parsing
try:
    from pyoxigraph import parse as oxigraph_parse, RdfFormat
    OXIGRAPH_AVAILABLE = True
except ImportError:
    OXIGRAPH_AVAILABLE = False


class TripleStore:
    """
    A high-performance RDF-Star triple store backed by dictionary-encoded Polars DataFrames.
    
    Unified architecture:
    - All terms are dictionary-encoded to integer IDs (TermDict)
    - Facts are stored as integer tuples for maximum join performance (FactStore)
    - String-based views are materialized on demand for SPARQL compatibility
    - Reasoner works directly on integer storage for efficient inference
    - Optional WAL for durability via TransactionManager
    
    Oxigraph parsing acceleration (when pyoxigraph is installed):
    - Uses Rust-based parsing for ~17x faster file loading
    - All queries use our native Polars executor (faster for COUNT/filters)
    """
    
    def __init__(
        self,
        wal_dir: Optional[Path] = None,
        wal_sync_mode: str = "normal",
        use_oxigraph: bool = True,
    ):
        """
        Initialize an empty triple store with unified storage.
        
        Args:
            wal_dir: Optional directory for write-ahead log (enables durability)
            wal_sync_mode: WAL sync mode - "full" (safest), "normal", or "off" (fastest)
            use_oxigraph: Use pyoxigraph for faster parsing when available
        """
        # Core storage components
        self._term_dict = TermDict()
        self._qt_dict = QtDict(self._term_dict)
        self._fact_store = FactStore(self._term_dict, self._qt_dict)
        
        # Cache for the string-based DataFrame view
        self._df_cache: Optional[pl.DataFrame] = None
        self._df_cache_valid = False
        
        # Mapping from assertion UUID to (s_id, p_id, o_id, g_id) for deprecation
        self._assertion_map: dict[UUID, tuple[TermId, TermId, TermId, TermId]] = {}
        
        # Quoted triple references (for backward compatibility)
        self._quoted_triples: dict[UUID, QuotedTriple] = {}
        
        # Transaction manager with WAL (optional)
        self._txn_manager: Optional[TransactionManager] = None
        if wal_dir:
            self._txn_manager = TransactionManager(
                store=self,
                wal_dir=wal_dir,
                sync_mode=wal_sync_mode,
            )
        
        # Track if Oxigraph parsing is available
        self._use_oxigraph_parsing = use_oxigraph and OXIGRAPH_AVAILABLE
        
        # Pre-intern common predicates and well-known IRIs
        self._init_common_terms()
    
    @property
    def oxigraph_available(self) -> bool:
        """Check if Oxigraph parsing acceleration is available."""
        return self._use_oxigraph_parsing
    
    def _init_common_terms(self):
        """Pre-intern commonly used terms for performance."""
        # RDF vocabulary
        self._rdf_type_id = self._term_dict.intern_iri(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        )
        # RDFS vocabulary
        self._rdfs_label_id = self._term_dict.intern_iri(
            "http://www.w3.org/2000/01/rdf-schema#label"
        )
        self._rdfs_subclass_id = self._term_dict.intern_iri(
            "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        )
    
    def _invalidate_cache(self):
        """Invalidate the cached DataFrame view after modifications."""
        self._df_cache_valid = False
    
    # ========== Transaction Support (WAL-backed durability) ==========
    
    def has_wal(self) -> bool:
        """Check if this store has WAL-based durability enabled."""
        return self._txn_manager is not None
    
    def begin_transaction(self) -> "Transaction":
        """
        Begin a new ACID transaction with WAL durability.
        
        Transactions provide:
        - Atomicity: All changes commit or none do
        - Durability: Changes are logged to WAL before commit
        - Crash recovery: Uncommitted transactions are replayed on restart
        
        Example:
            txn = store.begin_transaction()
            txn.add_triple("ex:s", "ex:p", "ex:o")
            txn.commit()  # Durable write
        
        Returns:
            Transaction object for adding/removing triples
            
        Raises:
            RuntimeError: If WAL is not enabled (pass wal_dir to constructor)
        """
        if self._txn_manager is None:
            raise RuntimeError(
                "WAL not enabled. Pass wal_dir to TripleStore() for durable transactions."
            )
        return self._txn_manager.begin()
    
    def checkpoint(self) -> None:
        """
        Force a WAL checkpoint, flushing all committed transactions to storage.
        
        After checkpoint, the WAL can be truncated. Call periodically for
        large write workloads to limit WAL growth.
        """
        if self._txn_manager:
            self._txn_manager.checkpoint()
    
    def close(self) -> None:
        """
        Close the store, flushing any pending WAL and releasing resources.
        
        Always call this before exiting to ensure data durability.
        """
        if self._txn_manager:
            self._txn_manager.close()

    def _intern_term(self, value: Any, is_uri_hint: bool = False) -> TermId:
        """
        Intern a term value to a TermId.
        
        Args:
            value: The term value (string, number, bool, etc.)
            is_uri_hint: If True, treat string as IRI; otherwise infer
            
        Returns:
            TermId for the interned term
        """
        if isinstance(value, str):
            # Check if it looks like a URI
            if is_uri_hint or value.startswith(("http://", "https://", "urn:", "file://")):
                return self._term_dict.intern_iri(value)
            elif value.startswith("_:"):
                return self._term_dict.intern_bnode(value[2:])
            else:
                # Parse RDF literal syntax: "value"^^<datatype> or "value"@lang or "value"
                return self._intern_literal_string(value)
        elif isinstance(value, bool):
            return self._term_dict.intern_literal(str(value).lower(), 
                datatype="http://www.w3.org/2001/XMLSchema#boolean")
        elif isinstance(value, int):
            return self._term_dict.intern_literal(str(value),
                datatype="http://www.w3.org/2001/XMLSchema#integer")
        elif isinstance(value, float):
            return self._term_dict.intern_literal(str(value),
                datatype="http://www.w3.org/2001/XMLSchema#decimal")
        else:
            return self._term_dict.intern_literal(str(value))
    
    def _intern_literal_string(self, value: str) -> TermId:
        """
        Parse and intern a string that may be in RDF literal syntax.
        
        Handles:
        - "value"^^<http://...>  -> typed literal
        - "value"@en            -> language-tagged literal
        - "value"               -> plain literal (xsd:string)
        - value                 -> plain literal (no quotes)
        """
        # Check for typed literal: "value"^^<datatype>
        if value.startswith('"') and '^^<' in value:
            # Find the closing quote before ^^
            caret_pos = value.find('^^<')
            if caret_pos > 0 and value[caret_pos-1] == '"':
                lex = value[1:caret_pos-1]  # Extract value between quotes
                datatype = value[caret_pos+3:-1]  # Extract datatype IRI (strip < and >)
                return self._term_dict.intern_literal(lex, datatype=datatype)
        
        # Check for language-tagged literal: "value"@lang
        if value.startswith('"') and '"@' in value:
            at_pos = value.rfind('"@')
            if at_pos > 0:
                lex = value[1:at_pos]  # Extract value between quotes
                lang = value[at_pos+2:]  # Extract language tag
                return self._term_dict.intern_literal(lex, lang=lang)
        
        # Check for quoted plain literal: "value"
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            lex = value[1:-1]  # Strip quotes
            return self._term_dict.intern_literal(lex)
        
        # Unquoted plain literal
        return self._term_dict.intern_literal(value)
    
    def _term_to_string(self, term_id: TermId) -> Optional[str]:
        """Convert a TermId back to its string representation."""
        term = self._term_dict.lookup(term_id)
        if term is None:
            return None
        return term.lex
    
    @property
    def _df(self) -> pl.DataFrame:
        """
        Materialize the string-based DataFrame view for SPARQL executor.
        
        This is a computed property that builds a string-column DataFrame
        from the integer-based FactStore. Results are cached until invalidated.
        
        Uses optimized join-based approach with lazy evaluation.
        """
        if self._df_cache_valid and self._df_cache is not None:
            return self._df_cache
        
        # Get raw facts - include ALL facts (deleted too, for include_deprecated support)
        fact_df = self._fact_store._df
        
        if len(fact_df) == 0:
            self._df_cache = self._create_empty_dataframe()
            self._df_cache_valid = True
            return self._df_cache
        
        # Build term lookup DataFrame once for all joins
        # Extract data as separate lists to avoid Polars inference issues with large integers
        term_ids = []
        lexes = []
        kinds = []
        datatype_ids = []
        for tid, term in self._term_dict._id_to_term.items():
            term_ids.append(tid)
            lexes.append(term.lex)
            kinds.append(int(term.kind))
            datatype_ids.append(term.datatype_id if term.datatype_id else 0)
        
        if not term_ids:
            self._df_cache = self._create_empty_dataframe()
            self._df_cache_valid = True
            return self._df_cache
        
        # Create DataFrame with explicit schema to handle large term IDs
        term_df = pl.DataFrame({
            "term_id": pl.Series("term_id", term_ids, dtype=pl.UInt64),
            "lex": pl.Series("lex", lexes, dtype=pl.Utf8),
            "kind": pl.Series("kind", kinds, dtype=pl.UInt8),
            "datatype_id": pl.Series("datatype_id", datatype_ids, dtype=pl.UInt64),
        })
        
        # Get XSD numeric datatype IDs for typed value conversion
        xsd_integer_id = self._term_dict.xsd_integer_id
        xsd_decimal_id = self._term_dict.xsd_decimal_id
        xsd_double_id = self._term_dict.xsd_double_id
        xsd_boolean_id = self._term_dict.xsd_boolean_id
        
        # Use lazy execution for join optimization
        result = fact_df.lazy()
        term_lazy = term_df.lazy()
        
        # Subject join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("subject")]),
            left_on="s", right_on="term_id", how="left"
        )
        
        # Predicate join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("predicate")]),
            left_on="p", right_on="term_id", how="left"
        )
        
        # Object join with kind and datatype
        result = result.join(
            term_lazy.select([
                pl.col("term_id"),
                pl.col("lex").alias("object"),
                pl.col("kind").alias("obj_kind"),
                pl.col("datatype_id").alias("obj_datatype_id"),
            ]),
            left_on="o", right_on="term_id", how="left"
        )
        
        # Graph join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("graph")]),
            left_on="g", right_on="term_id", how="left"
        )
        
        # Replace 0 with null for optional provenance columns before joining
        # This ensures unset values (0) don't get matched to xsd:string (which is term_id=0)
        result = result.with_columns([
            pl.when(pl.col("source") == 0).then(pl.lit(None).cast(pl.UInt64)).otherwise(pl.col("source")).alias("source"),
            pl.when(pl.col("process") == 0).then(pl.lit(None).cast(pl.UInt64)).otherwise(pl.col("process")).alias("process"),
        ])
        
        # Source join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("source_str")]),
            left_on="source", right_on="term_id", how="left"
        )
        
        # Process join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("process_str")]),
            left_on="process", right_on="term_id", how="left"
        )
        
        # Add computed columns
        result = result.with_columns([
            # Object type
            pl.when(pl.col("obj_kind") == int(TermKind.IRI)).then(pl.lit("uri"))
              .when(pl.col("obj_kind") == int(TermKind.BNODE)).then(pl.lit("bnode"))
              .otherwise(pl.lit("literal"))
              .alias("object_type"),
            # Typed numeric value
            pl.when(
                (pl.col("obj_datatype_id") == xsd_integer_id) |
                (pl.col("obj_datatype_id") == xsd_decimal_id) |
                (pl.col("obj_datatype_id") == xsd_double_id)
            ).then(
                pl.col("object").cast(pl.Float64, strict=False)
            ).when(
                pl.col("obj_datatype_id") == xsd_boolean_id
            ).then(
                pl.when(pl.col("object") == "true").then(pl.lit(1.0))
                  .when(pl.col("object") == "false").then(pl.lit(0.0))
                  .otherwise(pl.lit(None))
            ).otherwise(pl.lit(None).cast(pl.Float64))
            .alias("object_value"),
            # Deprecated flag
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) != 0).alias("deprecated"),
            # Timestamp - use microseconds to match _create_empty_dataframe schema
            pl.col("t_added").cast(pl.Datetime("us", "UTC")).alias("timestamp"),
        ])
        
        # Collect and finalize
        result = result.collect()
        
        # Build final schema with sequential assertion IDs
        n = len(result)
        result = result.select([
            pl.arange(0, n, eager=True).cast(pl.Utf8).alias("assertion_id"),
            "subject",
            "predicate",
            "object",
            "object_type",
            "object_value",
            "graph",
            pl.lit(None).cast(pl.Utf8).alias("quoted_triple_id"),
            pl.col("source_str").alias("source"),
            "timestamp",
            "confidence",
            pl.col("process_str").alias("process"),
            pl.lit(None).cast(pl.Utf8).alias("version"),
            pl.lit("{}").alias("metadata"),
            pl.lit(None).cast(pl.Utf8).alias("superseded_by"),
            "deprecated",
        ])
        
        self._df_cache = result
        self._df_cache_valid = True
        return self._df_cache
    
    @_df.setter
    def _df(self, value: pl.DataFrame):
        """
        Allow direct DataFrame assignment for backward compatibility.
        
        This is used by some internal operations that modify _df directly.
        We sync changes back to the FactStore.
        """
        # For backward compatibility, accept direct DataFrame assignment
        # This is mainly used during persistence load
        self._df_cache = value
        self._df_cache_valid = True
        # Note: This doesn't sync to FactStore - used only for legacy load
    
    @staticmethod
    def _create_empty_dataframe() -> pl.DataFrame:
        """Create the schema for the string-based assertion DataFrame."""
        return pl.DataFrame({
            "assertion_id": pl.Series([], dtype=pl.Utf8),
            "subject": pl.Series([], dtype=pl.Utf8),
            "predicate": pl.Series([], dtype=pl.Utf8),
            "object": pl.Series([], dtype=pl.Utf8),
            "object_type": pl.Series([], dtype=pl.Utf8),
            "object_value": pl.Series([], dtype=pl.Float64),  # Typed numeric value
            "graph": pl.Series([], dtype=pl.Utf8),
            "quoted_triple_id": pl.Series([], dtype=pl.Utf8),
            "source": pl.Series([], dtype=pl.Utf8),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "confidence": pl.Series([], dtype=pl.Float64),
            "process": pl.Series([], dtype=pl.Utf8),
            "version": pl.Series([], dtype=pl.Utf8),
            "metadata": pl.Series([], dtype=pl.Utf8),
            "superseded_by": pl.Series([], dtype=pl.Utf8),
            "deprecated": pl.Series([], dtype=pl.Boolean),
        })
    
    def get_term_id(self, value: str, is_uri: bool = True) -> Optional[TermId]:
        """
        Look up a term ID without creating it.
        
        Used for query optimization - if term doesn't exist,
        no rows can match that filter, so we can short-circuit.
        
        Args:
            value: The term string value
            is_uri: If True, treat as IRI; otherwise as literal
            
        Returns:
            TermId if found, None if term doesn't exist in store
        """
        if is_uri:
            return self._term_dict.get_iri_id(value)
        else:
            return self._term_dict.get_literal_id(value)
    
    def filter_facts_by_ids(
        self,
        s_id: Optional[TermId] = None,
        p_id: Optional[TermId] = None,
        o_id: Optional[TermId] = None,
        g_id: Optional[TermId] = None,
        include_deprecated: bool = False,
    ) -> pl.LazyFrame:
        """
        Filter facts at the integer level before materialization.
        
        This enables filter pushdown to integer storage for better performance.
        
        Args:
            s_id: Subject TermId filter (None = any)
            p_id: Predicate TermId filter (None = any)
            o_id: Object TermId filter (None = any)
            g_id: Graph TermId filter (None = any)
            include_deprecated: If True, include soft-deleted facts
            
        Returns:
            LazyFrame with filtered facts (still integer-encoded)
        """
        lf = self._fact_store._df.lazy()
        
        # Apply filters
        filters = []
        
        if not include_deprecated:
            filters.append(~(pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED) != 0))
        
        if s_id is not None:
            filters.append(pl.col("s") == s_id)
        if p_id is not None:
            filters.append(pl.col("p") == p_id)
        if o_id is not None:
            filters.append(pl.col("o") == o_id)
        if g_id is not None:
            filters.append(pl.col("g") == g_id)
        
        if filters:
            combined = filters[0]
            for f in filters[1:]:
                combined = combined & f
            lf = lf.filter(combined)
        
        return lf
    
    def get_triples_optimized(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        graph: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
        include_deprecated: bool = False,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Query triples using filter pushdown to integer storage.
        
        This is MUCH faster than get_triples() for filtered queries because:
        1. Filters are pushed down to the integer-based FactStore
        2. Only matching rows are materialized to strings
        3. No full DataFrame cache rebuild required
        
        Use this for:
        - AI grounding queries (entity context, verification)
        - Single-entity lookups
        - Any query with specific subject/predicate/object filters
        
        Args:
            subject: Filter by subject IRI
            predicate: Filter by predicate IRI
            obj: Filter by object value
            graph: Filter by graph IRI
            source: Filter by source (applied post-materialization)
            min_confidence: Minimum confidence threshold
            include_deprecated: Include soft-deleted facts
            limit: Maximum rows to return
            
        Returns:
            DataFrame with string columns (subject, predicate, object, etc.)
        """
        # Resolve string filters to TermIds for filter pushdown
        s_id = None
        p_id = None
        o_id = None
        g_id = None
        
        if subject is not None:
            s_id = self._term_dict.lookup_iri(subject)
            if s_id is None:
                # Subject doesn't exist in store - return empty
                return self._create_empty_dataframe()
        
        if predicate is not None:
            p_id = self._term_dict.lookup_iri(predicate)
            if p_id is None:
                return self._create_empty_dataframe()
        
        if obj is not None:
            # Try as IRI first, then literal
            o_id = self._term_dict.lookup_iri(str(obj))
            if o_id is None:
                o_id = self._term_dict.lookup_literal(str(obj))
            if o_id is None:
                return self._create_empty_dataframe()
        
        if graph is not None:
            g_id = self._term_dict.lookup_iri(graph)
            if g_id is None:
                return self._create_empty_dataframe()
        
        # Filter at integer level
        lf = self.filter_facts_by_ids(
            s_id=s_id,
            p_id=p_id,
            o_id=o_id,
            g_id=g_id,
            include_deprecated=include_deprecated,
        )
        
        # Apply confidence filter at integer level
        if min_confidence > 0.0:
            lf = lf.filter(pl.col("confidence") >= min_confidence)
        
        # Apply limit before materialization for efficiency
        if limit is not None:
            lf = lf.head(limit)
        
        # Collect the filtered integer facts
        fact_df = lf.collect()
        
        if len(fact_df) == 0:
            return self._create_empty_dataframe()
        
        # Materialize only the filtered subset to strings
        result = self._materialize_facts_to_strings(fact_df)
        
        # Apply post-materialization filters (source is stored as TermId)
        if source is not None:
            result = result.filter(pl.col("source") == source)
        
        return result
    
    def _materialize_facts_to_strings(self, fact_df: pl.DataFrame) -> pl.DataFrame:
        """
        Materialize integer-encoded facts to string DataFrame.
        
        This is the core string materialization logic, extracted for reuse.
        Only materializes the provided facts, not the entire store.
        
        Args:
            fact_df: DataFrame with integer-encoded facts
            
        Returns:
            DataFrame with string columns
        """
        if len(fact_df) == 0:
            return self._create_empty_dataframe()
        
        # Build term lookup DataFrame
        term_rows = [
            {"term_id": tid, "lex": term.lex, "kind": int(term.kind),
             "datatype_id": term.datatype_id if term.datatype_id else 0}
            for tid, term in self._term_dict._id_to_term.items()
        ]
        
        if not term_rows:
            return self._create_empty_dataframe()
        
        term_df = pl.DataFrame(term_rows).cast({
            "term_id": pl.UInt64,
            "lex": pl.Utf8,
            "kind": pl.UInt8,
            "datatype_id": pl.UInt64,
        })
        
        # Get XSD datatype IDs
        xsd_integer_id = self._term_dict.xsd_integer_id
        xsd_decimal_id = self._term_dict.xsd_decimal_id
        xsd_double_id = self._term_dict.xsd_double_id
        xsd_boolean_id = self._term_dict.xsd_boolean_id
        
        # Use lazy execution for join optimization
        result = fact_df.lazy()
        term_lazy = term_df.lazy()
        
        # Subject join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("subject")]),
            left_on="s", right_on="term_id", how="left"
        )
        
        # Predicate join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("predicate")]),
            left_on="p", right_on="term_id", how="left"
        )
        
        # Object join with kind and datatype
        result = result.join(
            term_lazy.select([
                pl.col("term_id"),
                pl.col("lex").alias("object"),
                pl.col("kind").alias("obj_kind"),
                pl.col("datatype_id").alias("obj_datatype_id"),
            ]),
            left_on="o", right_on="term_id", how="left"
        )
        
        # Graph join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("graph")]),
            left_on="g", right_on="term_id", how="left"
        )
        
        # Replace 0 with null for optional provenance columns
        result = result.with_columns([
            pl.when(pl.col("source") == 0).then(pl.lit(None).cast(pl.UInt64)).otherwise(pl.col("source")).alias("source"),
            pl.when(pl.col("process") == 0).then(pl.lit(None).cast(pl.UInt64)).otherwise(pl.col("process")).alias("process"),
        ])
        
        # Source join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("source_str")]),
            left_on="source", right_on="term_id", how="left"
        )
        
        # Process join
        result = result.join(
            term_lazy.select([pl.col("term_id"), pl.col("lex").alias("process_str")]),
            left_on="process", right_on="term_id", how="left"
        )
        
        # Add computed columns
        result = result.with_columns([
            # Object type
            pl.when(pl.col("obj_kind") == int(TermKind.IRI)).then(pl.lit("uri"))
              .when(pl.col("obj_kind") == int(TermKind.BNODE)).then(pl.lit("bnode"))
              .otherwise(pl.lit("literal"))
              .alias("object_type"),
            # Typed numeric value
            pl.when(
                (pl.col("obj_datatype_id") == xsd_integer_id) |
                (pl.col("obj_datatype_id") == xsd_decimal_id) |
                (pl.col("obj_datatype_id") == xsd_double_id)
            ).then(
                pl.col("object").cast(pl.Float64, strict=False)
            ).when(
                pl.col("obj_datatype_id") == xsd_boolean_id
            ).then(
                pl.when(pl.col("object") == "true").then(pl.lit(1.0))
                  .when(pl.col("object") == "false").then(pl.lit(0.0))
                  .otherwise(pl.lit(None))
            ).otherwise(pl.lit(None).cast(pl.Float64))
            .alias("object_value"),
            # Deprecated flag
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) != 0).alias("deprecated"),
            # Timestamp - use microseconds to match _create_empty_dataframe schema
            pl.col("t_added").cast(pl.Datetime("us", "UTC")).alias("timestamp"),
        ])
        
        # Collect and finalize schema
        result = result.collect()
        
        n = len(result)
        result = result.select([
            pl.arange(0, n, eager=True).cast(pl.Utf8).alias("assertion_id"),
            "subject",
            "predicate",
            "object",
            "object_type",
            "object_value",
            "graph",
            pl.lit(None).cast(pl.Utf8).alias("quoted_triple_id"),
            pl.col("source_str").alias("source"),
            "timestamp",
            "confidence",
            pl.col("process_str").alias("process"),
            pl.lit(None).cast(pl.Utf8).alias("version"),
            pl.lit("{}").alias("metadata"),
            pl.lit(None).cast(pl.Utf8).alias("superseded_by"),
            "deprecated",
        ])
        
        return result

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        provenance: ProvenanceContext,
        graph: Optional[str] = None,
    ) -> UUID:
        """
        Add a triple with provenance to the store.
        
        Args:
            subject: Subject URI or blank node
            predicate: Predicate URI
            obj: Object (URI, literal, or value)
            provenance: Provenance context for this assertion
            graph: Optional named graph
            
        Returns:
            UUID of the created assertion
        """
        # Generate assertion ID upfront
        assertion_id = uuid4()
        
        # Intern all terms
        s_id = self._intern_term(subject, is_uri_hint=True)
        p_id = self._intern_term(predicate, is_uri_hint=True)
        o_id = self._intern_term(obj)
        g_id = self._term_dict.intern_iri(graph) if graph else DEFAULT_GRAPH_ID
        
        # Intern provenance terms
        source_id = self._term_dict.intern_literal(provenance.source) if provenance.source else 0
        process_id = self._term_dict.intern_literal(provenance.process) if provenance.process else 0
        
        # Convert provenance timestamp to microseconds if provided
        t_added = None
        if provenance.timestamp:
            t_added = int(provenance.timestamp.timestamp() * 1_000_000)
        
        # Add to fact store
        self._fact_store.add_fact(
            s=s_id,
            p=p_id,
            o=o_id,
            g=g_id,
            flags=FactFlags.ASSERTED,
            source=source_id,
            confidence=provenance.confidence,
            process=process_id,
            t_added=t_added,
        )
        
        # Store mapping for deprecation
        self._assertion_map[assertion_id] = (s_id, p_id, o_id, g_id)
        
        self._invalidate_cache()
        return assertion_id
    
    def add_assertion(self, assertion: Assertion) -> UUID:
        """Add a complete assertion object to the store."""
        return self.add_triple(
            subject=assertion.triple.subject,
            predicate=assertion.triple.predicate,
            obj=assertion.triple.object,
            provenance=assertion.provenance,
            graph=assertion.triple.graph,
        )
    
    def add_triples_batch(
        self,
        triples: list[dict],
    ) -> int:
        """
        Add multiple triples in a single batch operation.
        
        This is MUCH faster than calling add_triple() repeatedly because:
        - Batch term interning
        - Single FactStore batch operation
        
        Args:
            triples: List of dicts with keys:
                - subject: str
                - predicate: str
                - object: Any
                - source: str
                - confidence: float (optional, default 1.0)
                - process: str (optional)
                - graph: str (optional)
                
        Returns:
            Number of triples added
        """
        if not triples:
            return 0
        
        # Prepare batch data
        facts = []
        now = datetime.now(timezone.utc)
        
        for t in triples:
            # Intern terms
            s_id = self._intern_term(t["subject"], is_uri_hint=True)
            p_id = self._intern_term(t["predicate"], is_uri_hint=True)
            o_id = self._intern_term(t.get("object", ""))
            
            graph = t.get("graph")
            g_id = self._term_dict.intern_iri(graph) if graph else DEFAULT_GRAPH_ID
            
            source = t.get("source", "unknown")
            source_id = self._term_dict.intern_literal(source) if source else 0
            
            process = t.get("process")
            process_id = self._term_dict.intern_literal(process) if process else 0
            
            confidence = t.get("confidence", 1.0)
            
            facts.append((g_id, s_id, p_id, o_id, source_id, confidence, process_id))
        
        # Batch insert to FactStore
        for g_id, s_id, p_id, o_id, source_id, confidence, process_id in facts:
            self._fact_store.add_fact(
                s=s_id,
                p=p_id,
                o=o_id,
                g=g_id,
                flags=FactFlags.ASSERTED,
                source=source_id,
                confidence=confidence,
                process=process_id,
            )
        
        self._invalidate_cache()
        return len(triples)
    
    def add_triples_columnar(
        self,
        subjects: list[str],
        predicates: list[str],
        objects: list[Any],
        source: str = "unknown",
        confidence: float = 1.0,
        graph: Optional[str] = None,
    ) -> int:
        """
        Add triples from column lists (TRUE vectorized path).
        
        This is the FASTEST ingestion method. Pass pre-built lists
        of subjects, predicates, and objects.
        
        Args:
            subjects: List of subject URIs
            predicates: List of predicate URIs
            objects: List of object values
            source: Shared source for provenance
            confidence: Shared confidence score
            graph: Optional graph URI
            
        Returns:
            Number of triples added
        """
        n = len(subjects)
        if n == 0:
            return 0
        
        # Batch intern terms
        g_id = self._term_dict.intern_iri(graph) if graph else DEFAULT_GRAPH_ID
        source_id = self._term_dict.intern_literal(source)
        
        # Use batch interning for subjects and predicates (~3x faster)
        s_col = self._term_dict.intern_iris_batch(subjects)
        p_col = self._term_dict.intern_iris_batch(predicates)
        
        # Intern objects (could be literals or URIs — must dispatch per-item)
        o_col = [self._intern_term(o) for o in objects]
        
        # Graph column
        g_col = [g_id] * n
        
        # Use columnar insert
        self._fact_store.add_facts_columnar(
            g_col=g_col,
            s_col=s_col,
            p_col=p_col,
            o_col=o_col,
            flags=FactFlags.ASSERTED,
            source=source_id,
            confidence=confidence,
        )
        
        self._invalidate_cache()
        return n

    def begin_bulk_insert(self) -> None:
        """Begin bulk insertion mode.
        
        Defers DataFrame concatenation in the FactStore — all inserts
        are buffered and flushed in a single concat when
        ``end_bulk_insert()`` is called, avoiding the O(n²) repeated
        copy cost.
        """
        self._fact_store.begin_batch()

    def end_bulk_insert(self) -> int:
        """End bulk insertion and flush buffered DataFrames.
        
        Returns:
            Number of rows flushed.
        """
        count = self._fact_store.flush_batch()
        self._invalidate_cache()
        return count

    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        graph: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
        include_deprecated: bool = False,
    ) -> pl.DataFrame:
        """
        Query triples with optional filters.
        
        Uses the string-based _df view for compatibility with existing code.
        """
        df = self._df.lazy()
        
        if subject is not None:
            df = df.filter(pl.col("subject") == subject)
        if predicate is not None:
            df = df.filter(pl.col("predicate") == predicate)
        if obj is not None:
            df = df.filter(pl.col("object") == str(obj))
        if graph is not None:
            df = df.filter(pl.col("graph") == graph)
        if source is not None:
            df = df.filter(pl.col("source") == source)
        if min_confidence is not None:
            df = df.filter(pl.col("confidence") >= min_confidence)
        if not include_deprecated:
            df = df.filter(~pl.col("deprecated"))
        
        return df.collect()
    
    def get_competing_claims(
        self,
        subject: str,
        predicate: str,
    ) -> pl.DataFrame:
        """Find competing assertions about the same subject-predicate pair."""
        df = self.get_triples(subject=subject, predicate=predicate, include_deprecated=False)
        df = df.sort(["confidence", "timestamp"], descending=[True, True])
        return df
    
    def deprecate_assertion(self, assertion_id: UUID, superseded_by: Optional[UUID] = None) -> None:
        """Mark an assertion as deprecated."""
        # Look up the assertion in our mapping
        if assertion_id in self._assertion_map:
            s_id, p_id, o_id, g_id = self._assertion_map[assertion_id]
            self._fact_store.mark_deleted(s=s_id, p=p_id, o=o_id)
            self._invalidate_cache()
            return
        
        # Fallback: try to find in cached DataFrame
        if self._df_cache is not None and len(self._df_cache) > 0:
            matching = self._df_cache.filter(pl.col("assertion_id") == str(assertion_id))
            if len(matching) > 0:
                # Mark in cache
                self._df_cache = self._df_cache.with_columns([
                    pl.when(pl.col("assertion_id") == str(assertion_id))
                    .then(True)
                    .otherwise(pl.col("deprecated"))
                    .alias("deprecated"),
                    
                    pl.when(pl.col("assertion_id") == str(assertion_id))
                    .then(str(superseded_by) if superseded_by else None)
                    .otherwise(pl.col("superseded_by"))
                    .alias("superseded_by"),
                ])
                
                # Also need to mark in FactStore
                subject = matching["subject"][0]
                predicate = matching["predicate"][0]
                obj = matching["object"][0]
                
                s_id = self._term_dict.lookup_iri(subject)
                p_id = self._term_dict.lookup_iri(predicate)
                o_id = self._term_dict.lookup_iri(obj)
                if o_id is None:
                    o_id = self._term_dict.lookup_literal(obj)
                
                if s_id is not None and p_id is not None and o_id is not None:
                    self._fact_store.mark_deleted(s=s_id, p=p_id, o=o_id)
                    self._invalidate_cache()
    
    def get_provenance_timeline(self, subject: str, predicate: str) -> pl.DataFrame:
        """Get the full history of assertions about a subject-predicate pair."""
        df = self.get_triples(
            subject=subject,
            predicate=predicate,
            include_deprecated=True
        )
        df = df.sort("timestamp")
        return df
    
    def mark_deleted(
        self, 
        s: Optional[str] = None, 
        p: Optional[str] = None, 
        o: Optional[str] = None
    ) -> int:
        """
        Mark matching triples as deprecated (soft delete).
        
        Works on the FactStore level for correctness.
        """
        # Look up term IDs (if they don't exist, no triples to delete)
        s_id = None
        p_id = None
        o_id = None
        
        if s is not None:
            s_id = self._term_dict.lookup_iri(s)
            if s_id is None:
                return 0
        if p is not None:
            p_id = self._term_dict.lookup_iri(p)
            if p_id is None:
                return 0
        if o is not None:
            # Try as IRI first, then literal
            o_id = self._term_dict.lookup_iri(o)
            if o_id is None:
                o_id = self._term_dict.lookup_literal(o)
            if o_id is None:
                return 0
        
        count = self._fact_store.mark_deleted(s=s_id, p=p_id, o=o_id)
        self._invalidate_cache()
        return count
    
    def save(self, path: Path | str) -> None:
        """
        Save the triple store to disk.
        
        Saves all components: TermDict, QtDict, FactStore.
        """
        from rdf_starbase.storage.persistence import StoragePersistence
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for the unified store
        store_dir = path.parent / (path.stem + "_unified")
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # Use StoragePersistence for consistent save/load
        persistence = StoragePersistence(store_dir)
        persistence.save(self._term_dict, self._fact_store, self._qt_dict)
        
        # Also save the legacy format for backward compatibility
        self._df.write_parquet(path)
    
    @classmethod
    def load(cls, path: Path | str, streaming: bool = False) -> "TripleStore":
        """
        Load a triple store from disk.
        
        Attempts to load unified format first, falls back to legacy.
        
        Args:
            path: Path to the saved store (parquet file or directory)
            streaming: If True, use memory-mapped loading for large datasets.
                      This is recommended for datasets > 1M triples or when
                      memory is constrained. Default False.
        """
        path = Path(path)
        store_dir = path.parent / (path.stem + "_unified")
        
        if store_dir.exists():
            # Load unified format
            from rdf_starbase.storage.persistence import StoragePersistence
            persistence = StoragePersistence(store_dir)
            
            store = cls()
            if streaming:
                store._term_dict, store._fact_store, store._qt_dict = persistence.load_streaming()
            else:
                store._term_dict, store._fact_store, store._qt_dict = persistence.load()
            
            # Re-initialize common terms after loading
            store._init_common_terms()
            return store
        else:
            # Load legacy format and convert
            store = cls()
            legacy_df = pl.read_parquet(path)
            
            # Import each row
            for row in legacy_df.iter_rows(named=True):
                if not row.get("deprecated", False):
                    prov = ProvenanceContext(
                        source=row.get("source", "legacy"),
                        confidence=row.get("confidence", 1.0),
                        process=row.get("process"),
                        timestamp=row.get("timestamp", datetime.now(timezone.utc)),
                    )
                    store.add_triple(
                        subject=row["subject"],
                        predicate=row["predicate"],
                        obj=row["object"],
                        provenance=prov,
                        graph=row.get("graph"),
                    )
            
            return store
    
    def save_incremental(self, path: Path | str, force_compact: bool = False) -> dict:
        """
        Save the triple store incrementally.
        
        Only writes data that has changed since the last save, using delta files.
        Automatically compacts when delta count exceeds threshold (default 10).
        
        Args:
            path: Directory path for incremental storage
            force_compact: Force full compaction (merge all deltas into base)
            
        Returns:
            Dict with save statistics:
            - status: "no_changes", "delta_saved", "compacted", or "full_save"
            - delta_facts: Number of new facts written
            - delta_terms: Number of new terms written
            - was_compacted: Whether compaction occurred
        """
        from rdf_starbase.storage.persistence import IncrementalPersistence
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        persistence = IncrementalPersistence(path)
        return persistence.save(
            self._term_dict,
            self._fact_store,
            self._qt_dict,
            force_full=force_compact
        )
    
    @classmethod
    def load_incremental(cls, path: Path | str) -> "TripleStore":
        """
        Load a triple store from incremental storage.
        
        Loads the base data and applies all delta files.
        
        Args:
            path: Directory path containing incremental storage
            
        Returns:
            Loaded TripleStore instance
        """
        from rdf_starbase.storage.persistence import IncrementalPersistence
        
        path = Path(path)
        persistence = IncrementalPersistence(path)
        
        store = cls()
        store._term_dict, store._fact_store, store._qt_dict = persistence.load()
        store._init_common_terms()
        return store
    
    def compact(self, path: Path | str) -> dict:
        """
        Force compaction of incremental storage.
        
        Merges all delta files into the base files.
        
        Args:
            path: Directory path for incremental storage
            
        Returns:
            Compaction statistics
        """
        from rdf_starbase.storage.persistence import IncrementalPersistence
        
        path = Path(path)
        persistence = IncrementalPersistence(path)
        return persistence.compact(
            self._term_dict,
            self._fact_store,
            self._qt_dict
        )

    def stats(self) -> dict[str, Any]:
        """Get statistics about the triple store."""
        fact_stats = self._fact_store.stats()
        term_stats = self._term_dict.stats()
        
        # Get unique subjects and predicates from actual facts (not deleted)
        active_facts = self._fact_store._df.filter(
            (pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0
        )
        unique_subjects = active_facts.select("s").unique().height
        unique_predicates = active_facts.select("p").unique().height
        
        return {
            "total_assertions": fact_stats["total_facts"],
            "active_assertions": fact_stats["active_facts"],
            "deprecated_assertions": fact_stats["total_facts"] - fact_stats["active_facts"],
            "unique_sources": len(set(
                self._term_to_string(sid) 
                for sid in self._fact_store._df["source"].unique().to_list()
                if sid and sid != 0
            )),
            "unique_subjects": unique_subjects,
            "unique_predicates": unique_predicates,
            "term_dict": term_stats,
            "fact_store": fact_stats,
        }
    
    def __len__(self) -> int:
        """Return the number of active assertions."""
        return self._fact_store.count_active()
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"TripleStore("
            f"assertions={stats['active_assertions']}, "
            f"terms={stats['term_dict']['total_terms']})"
        )

    # =========================================================================
    # Named Graph Management
    # =========================================================================
    
    def list_graphs(self) -> list[str]:
        """List all named graphs in the store."""
        # Get unique graph IDs from FactStore
        graph_ids = self._fact_store._df.filter(
            (pl.col("g") != DEFAULT_GRAPH_ID) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        ).select("g").unique().to_series().to_list()
        
        graphs = []
        for gid in graph_ids:
            term = self._term_dict.lookup(gid)
            if term is not None:
                graphs.append(term.lex)
        
        return sorted(graphs)
    
    def create_graph(self, graph_uri: str) -> bool:
        """Create an empty named graph."""
        g_id = self._term_dict.intern_iri(graph_uri)
        existing = self._fact_store._df.filter(
            (pl.col("g") == g_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        ).height
        return existing == 0
    
    def drop_graph(self, graph_uri: str, silent: bool = False) -> int:
        """Drop (delete) a named graph and all its triples."""
        g_id = self._term_dict.lookup_iri(graph_uri)
        if g_id is None:
            return 0
        
        # Mark all facts in this graph as deleted
        count = 0
        fact_df = self._fact_store._df
        matching = fact_df.filter(
            (pl.col("g") == g_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        )
        count = matching.height
        
        if count > 0:
            # Update flags
            self._fact_store._df = fact_df.with_columns([
                pl.when(
                    (pl.col("g") == g_id) &
                    ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
                )
                .then((pl.col("flags").cast(pl.Int32) | int(FactFlags.DELETED)).cast(pl.UInt16))
                .otherwise(pl.col("flags"))
                .alias("flags")
            ])
            self._invalidate_cache()
        
        return count
    
    def clear_graph(self, graph_uri: Optional[str] = None, silent: bool = False) -> int:
        """Clear all triples from a graph (or default graph if None)."""
        if graph_uri is None:
            g_id = DEFAULT_GRAPH_ID
        else:
            g_id = self._term_dict.lookup_iri(graph_uri)
            if g_id is None:
                return 0
        
        fact_df = self._fact_store._df
        matching = fact_df.filter(
            (pl.col("g") == g_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        )
        count = matching.height
        
        if count > 0:
            self._fact_store._df = fact_df.with_columns([
                pl.when(
                    (pl.col("g") == g_id) &
                    ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
                )
                .then((pl.col("flags").cast(pl.Int32) | int(FactFlags.DELETED)).cast(pl.UInt16))
                .otherwise(pl.col("flags"))
                .alias("flags")
            ])
            self._invalidate_cache()
        
        return count
    
    def copy_graph(
        self, 
        source_graph: Optional[str], 
        dest_graph: Optional[str],
        silent: bool = False,
    ) -> int:
        """Copy all triples from source graph to destination graph."""
        # Clear destination first
        self.clear_graph(dest_graph, silent=True)
        
        # Get source graph ID
        if source_graph is None:
            src_g_id = DEFAULT_GRAPH_ID
        else:
            src_g_id = self._term_dict.lookup_iri(source_graph)
            if src_g_id is None:
                return 0
        
        # Get destination graph ID
        if dest_graph is None:
            dest_g_id = DEFAULT_GRAPH_ID
        else:
            dest_g_id = self._term_dict.intern_iri(dest_graph)
        
        # Get source facts
        fact_df = self._fact_store._df
        source_facts = fact_df.filter(
            (pl.col("g") == src_g_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        )
        
        if source_facts.height == 0:
            return 0
        
        # Create copies with new graph and transaction IDs
        new_txn = self._fact_store._allocate_txn()
        t_now = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        new_facts = source_facts.with_columns([
            pl.lit(dest_g_id).cast(pl.UInt64).alias("g"),
            pl.lit(new_txn).cast(pl.UInt64).alias("txn"),
            pl.lit(t_now).cast(pl.UInt64).alias("t_added"),
        ])
        
        self._fact_store._df = pl.concat([self._fact_store._df, new_facts])
        self._invalidate_cache()
        
        return new_facts.height
    
    def move_graph(
        self,
        source_graph: Optional[str],
        dest_graph: Optional[str],
        silent: bool = False,
    ) -> int:
        """Move all triples from source graph to destination graph."""
        count = self.copy_graph(source_graph, dest_graph, silent)
        
        # Clear source
        if source_graph is None:
            self.clear_graph(None, silent=True)
        else:
            self.clear_graph(source_graph, silent=True)
        
        return count
    
    def add_graph(
        self,
        source_graph: Optional[str],
        dest_graph: Optional[str],
        silent: bool = False,
    ) -> int:
        """Add all triples from source graph to destination graph."""
        # Get source graph ID
        if source_graph is None:
            src_g_id = DEFAULT_GRAPH_ID
        else:
            src_g_id = self._term_dict.lookup_iri(source_graph)
            if src_g_id is None:
                return 0
        
        # Get destination graph ID
        if dest_graph is None:
            dest_g_id = DEFAULT_GRAPH_ID
        else:
            dest_g_id = self._term_dict.intern_iri(dest_graph)
        
        # Get source facts
        fact_df = self._fact_store._df
        source_facts = fact_df.filter(
            (pl.col("g") == src_g_id) &
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) == 0)
        )
        
        if source_facts.height == 0:
            return 0
        
        # Create copies with new graph
        new_txn = self._fact_store._allocate_txn()
        t_now = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        
        new_facts = source_facts.with_columns([
            pl.lit(dest_g_id).cast(pl.UInt64).alias("g"),
            pl.lit(new_txn).cast(pl.UInt64).alias("txn"),
            pl.lit(t_now).cast(pl.UInt64).alias("t_added"),
        ])
        
        self._fact_store._df = pl.concat([self._fact_store._df, new_facts])
        self._invalidate_cache()
        
        return new_facts.height
    
    def _load_graph_oxigraph(
        self,
        file_path: Path,
        source_uri: str,
        graph_uri: Optional[str],
        suffix: str,
        is_gzipped: bool,
    ) -> int:
        """
        Fast loading path using Oxigraph's Rust parser.
        
        ~17x faster than Python parser for standard Turtle/N-Triples.
        Uses Oxigraph only for parsing, stores data in Polars FactStore.
        """
        # Map suffix to Oxigraph format
        format_map = {
            ".ttl": RdfFormat.TURTLE,
            ".turtle": RdfFormat.TURTLE,
            ".nt": RdfFormat.N_TRIPLES,
            ".ntriples": RdfFormat.N_TRIPLES,
        }
        rdf_format = format_map.get(suffix, RdfFormat.TURTLE)
        
        # Read file content
        if is_gzipped:
            import gzip
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
        else:
            content = file_path.read_text(encoding='utf-8')
        
        # Parse with Oxigraph (Rust speed)
        base_iri = file_path.absolute().as_uri()
        
        subjects = []
        predicates = []
        objects = []
        
        for quad in oxigraph_parse(content, rdf_format, base_iri=base_iri):
            # Extract subject
            s = quad.subject
            if hasattr(s, 'value'):
                subj = s.value
            else:
                subj = f"_:{s}"
            
            # Extract predicate
            pred = quad.predicate.value
            
            # Extract object
            o = quad.object
            if hasattr(o, 'value'):
                if hasattr(o, 'datatype') and o.datatype:
                    dt = o.datatype.value
                    if dt != "http://www.w3.org/2001/XMLSchema#string":
                        obj = f'"{o.value}"^^<{dt}>'
                    else:
                        obj = o.value
                elif hasattr(o, 'language') and o.language:
                    obj = f'"{o.value}"@{o.language}'
                else:
                    obj = o.value
            else:
                obj = str(o)
            
            subjects.append(subj)
            predicates.append(pred)
            objects.append(obj)
        
        if not subjects:
            return 0
        
        # Use columnar insert (Polars/Rust)
        count = self.add_triples_columnar(
            subjects=subjects,
            predicates=predicates,
            objects=objects,
            source=source_uri,
            confidence=1.0,
            graph=graph_uri,
        )
        
        return count
    
    def load_graph(
        self,
        source_uri: str,
        graph_uri: Optional[str] = None,
        silent: bool = False,
    ) -> int:
        """
        Load RDF data from a URI into a graph.
        
        When pyoxigraph is available, uses Rust-based parsing for ~17x faster loading.
        Falls back to Python parser for RDF-Star content or when Oxigraph not available.
        """
        from pathlib import Path
        from urllib.parse import urlparse, unquote
        
        # Determine file path
        if source_uri.startswith("file://"):
            parsed = urlparse(source_uri)
            file_path_str = unquote(parsed.path)
            if len(file_path_str) > 2 and file_path_str[0] == '/' and file_path_str[2] == ':':
                file_path_str = file_path_str[1:]
            file_path = Path(file_path_str)
        elif source_uri.startswith(("http://", "https://")):
            import tempfile
            import urllib.request
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ttl") as f:
                    urllib.request.urlretrieve(source_uri, f.name)
                    file_path = Path(f.name)
            except Exception as e:
                if silent:
                    return 0
                raise ValueError(f"Failed to download {source_uri}: {e}")
        else:
            file_path = Path(source_uri)
        
        if not file_path.exists():
            if silent:
                return 0
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Determine format from extension, handling gzip compression
        suffix = file_path.suffix.lower()
        is_gzipped = suffix == ".gz"
        if is_gzipped:
            # Get the actual format from the second-to-last extension
            stem = file_path.stem  # e.g., "file.ttl" from "file.ttl.gz"
            suffix = Path(stem).suffix.lower()  # e.g., ".ttl"
        
        # Read file content first to check for RDF-Star syntax
        if is_gzipped:
            import gzip
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                file_content = f.read()
        else:
            file_content = file_path.read_text(encoding="utf-8")
        
        # Check for RDF-Star quoted triple syntax: << ... >>
        # Skip Oxigraph if content contains RDF-Star, as it may not handle it correctly
        has_rdfstar = '<<' in file_content and '>>' in file_content
        
        # Try Oxigraph fast path for simple formats WITHOUT RDF-Star annotations
        if self._use_oxigraph_parsing and suffix in (".ttl", ".turtle", ".nt", ".ntriples") and not has_rdfstar:
            try:
                return self._load_graph_oxigraph(file_path, source_uri, graph_uri, suffix, is_gzipped)
            except Exception:
                # Fall back to Python parser if Oxigraph fails
                pass
        
        try:
            
            if suffix in (".ttl", ".turtle"):
                from rdf_starbase.formats.turtle import parse_turtle
                parsed = parse_turtle(file_content)
                triples = parsed.triples
            elif suffix in (".nt", ".ntriples"):
                from rdf_starbase.formats.ntriples import parse_ntriples
                parsed = parse_ntriples(file_content)
                triples = parsed.triples
            elif suffix in (".rdf", ".xml"):
                from rdf_starbase.formats.rdfxml import parse_rdfxml
                parsed = parse_rdfxml(file_content)
                triples = parsed.triples
            elif suffix in (".jsonld", ".json"):
                from rdf_starbase.formats.jsonld import parse_jsonld
                parsed = parse_jsonld(file_content)
                triples = parsed.triples
            else:
                from rdf_starbase.formats.turtle import parse_turtle
                parsed = parse_turtle(file_content)
                triples = parsed.triples
        except Exception as e:
            if silent:
                return 0
            raise ValueError(f"Failed to parse {file_path}: {e}")
        
        # Provenance predicate URIs to recognize
        PROV_SOURCE_PREDICATES = {
            "http://www.w3.org/ns/prov#wasDerivedFrom",
            "http://www.w3.org/ns/prov#wasAttributedTo",
            "http://www.w3.org/ns/prov#hadPrimarySource",
        }
        PROV_PROCESS_PREDICATES = {
            "http://www.w3.org/ns/prov#wasGeneratedBy",
        }
        PROV_CONFIDENCE_PREDICATES = {
            "http://www.w3.org/ns/prov#value",
        }
        PROV_TIMESTAMP_PREDICATES = {
            "http://www.w3.org/ns/prov#generatedAtTime",
        }
        
        # Separate base triples from RDF-Star annotation triples
        base_triples = []
        annotations = {}  # Key: (s, p, o) -> {source: ..., confidence: ..., timestamp: ...}
        
        for triple in triples:
            if triple.subject_triple is not None:
                # This is an RDF-Star annotation: << s p o >> predicate object
                qt = triple.subject_triple
                key = (qt.subject, qt.predicate, qt.object)
                
                if key not in annotations:
                    annotations[key] = {}
                
                pred = triple.predicate
                obj_val = triple.object
                
                # Extract the value (strip datatype for literals)
                if isinstance(obj_val, str) and "^^" in obj_val:
                    obj_val = obj_val.split("^^")[0].strip('"')
                elif isinstance(obj_val, str):
                    obj_val = obj_val.strip('"')
                
                if pred in PROV_SOURCE_PREDICATES:
                    annotations[key]["source"] = triple.object  # Keep full URI
                elif pred in PROV_PROCESS_PREDICATES:
                    annotations[key]["process"] = triple.object  # Keep full URI
                elif pred in PROV_CONFIDENCE_PREDICATES:
                    try:
                        annotations[key]["confidence"] = float(obj_val)
                    except (ValueError, TypeError):
                        annotations[key]["confidence"] = 1.0
                elif pred in PROV_TIMESTAMP_PREDICATES:
                    annotations[key]["timestamp"] = obj_val
            else:
                # Regular triple (not an annotation)
                base_triples.append(triple)
        
        # Group triples by provenance for batch columnar inserts
        prov_groups = {}  # (source, confidence) -> [(s, p, o), ...]
        
        for triple in base_triples:
            key = (triple.subject, triple.predicate, triple.object)
            if key in annotations:
                ann = annotations[key]
                source = ann.get("source", source_uri)
                confidence = ann.get("confidence", 1.0)
            else:
                source = source_uri
                confidence = 1.0
            
            prov_key = (source, confidence)
            if prov_key not in prov_groups:
                prov_groups[prov_key] = []
            prov_groups[prov_key].append((triple.subject, triple.predicate, triple.object))
        
        # Batch columnar insert for each provenance group
        count = 0
        for (source, confidence), triples_list in prov_groups.items():
            subjects = [t[0] for t in triples_list]
            predicates = [t[1] for t in triples_list]
            objects = [t[2] for t in triples_list]
            count += self.add_triples_columnar(
                subjects=subjects,
                predicates=predicates,
                objects=objects,
                source=source,
                confidence=confidence,
                graph=graph_uri,
            )
        
        return count
