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
"""

from datetime import datetime, timezone
from typing import Optional, Any, Literal
from uuid import UUID, uuid4
from pathlib import Path

import polars as pl

from rdf_starbase.models import Triple, QuotedTriple, Assertion, ProvenanceContext
from rdf_starbase.storage.terms import TermDict, TermKind, Term, TermId, get_term_kind
from rdf_starbase.storage.quoted_triples import QtDict
from rdf_starbase.storage.facts import FactStore, FactFlags, DEFAULT_GRAPH_ID


class TripleStore:
    """
    A high-performance RDF-Star triple store backed by dictionary-encoded Polars DataFrames.
    
    Unified architecture:
    - All terms are dictionary-encoded to integer IDs (TermDict)
    - Facts are stored as integer tuples for maximum join performance (FactStore)
    - String-based views are materialized on demand for SPARQL compatibility
    - Reasoner works directly on integer storage for efficient inference
    """
    
    def __init__(self):
        """Initialize an empty triple store with unified storage."""
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
        
        # Pre-intern common predicates and well-known IRIs
        self._init_common_terms()
    
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
        """
        if self._df_cache_valid and self._df_cache is not None:
            return self._df_cache
        
        # Get raw facts - include ALL facts (deleted too, for include_deprecated support)
        fact_df = self._fact_store._df
        
        if len(fact_df) == 0:
            self._df_cache = self._create_empty_dataframe()
            self._df_cache_valid = True
            return self._df_cache
        
        # Build term lookup DataFrame for fast Polars join
        # Include datatype_id for typed value conversion
        term_rows = [
            {"term_id": tid, "lex": term.lex, "kind": int(term.kind), 
             "datatype_id": term.datatype_id if term.datatype_id else 0}
            for tid, term in self._term_dict._id_to_term.items()
        ]
        
        if term_rows:
            term_df = pl.DataFrame(term_rows).cast({
                "term_id": pl.UInt64,
                "lex": pl.Utf8,
                "kind": pl.UInt8,
                "datatype_id": pl.UInt64,
            })
        else:
            term_df = pl.DataFrame({
                "term_id": pl.Series([], dtype=pl.UInt64),
                "lex": pl.Series([], dtype=pl.Utf8),
                "kind": pl.Series([], dtype=pl.UInt8),
                "datatype_id": pl.Series([], dtype=pl.UInt64),
            })
        
        # Rename fact columns to avoid conflicts during joins
        result = fact_df.rename({
            "source": "source_id",
            "process": "process_id"
        })
        
        # Subject - join drops right key automatically in Polars 1.x
        result = result.join(
            term_df.select([pl.col("term_id"), pl.col("lex").alias("subject")]),
            left_on="s", right_on="term_id", how="left"
        )
        
        # Predicate
        result = result.join(
            term_df.select([pl.col("term_id"), pl.col("lex").alias("predicate")]),
            left_on="p", right_on="term_id", how="left"
        )
        
        # Object + object type + typed object value
        # Get XSD numeric datatype IDs for typed value conversion
        xsd_integer_id = self._term_dict.xsd_integer_id
        xsd_decimal_id = self._term_dict.xsd_decimal_id
        xsd_double_id = self._term_dict.xsd_double_id
        xsd_boolean_id = self._term_dict.xsd_boolean_id
        
        obj_df = term_df.select([
            pl.col("term_id"),
            pl.col("lex").alias("object"),
            pl.col("datatype_id").alias("obj_datatype_id"),
            pl.when(pl.col("kind") == int(TermKind.IRI)).then(pl.lit("uri"))
              .when(pl.col("kind") == int(TermKind.BNODE)).then(pl.lit("bnode"))
              .otherwise(pl.lit("literal"))
              .alias("object_type")
        ])
        result = result.join(obj_df, left_on="o", right_on="term_id", how="left")
        
        # Create typed object_value column for numeric comparisons
        # Cast lex to float for numeric datatypes, null otherwise
        result = result.with_columns([
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
            .alias("object_value")
        ])
        
        # Graph (handle 0 = default graph)
        result = result.join(
            term_df.select([pl.col("term_id"), pl.col("lex").alias("graph")]),
            left_on="g", right_on="term_id", how="left"
        )
        
        # Source
        result = result.join(
            term_df.select([pl.col("term_id"), pl.col("lex").alias("source")]),
            left_on="source_id", right_on="term_id", how="left"
        )
        
        # Process
        result = result.join(
            term_df.select([pl.col("term_id"), pl.col("lex").alias("process")]),
            left_on="process_id", right_on="term_id", how="left"
        )
        
        # Build final schema
        result = result.select([
            pl.lit(None).cast(pl.Utf8).alias("assertion_id"),  # Deferred - generate on iteration
            "subject",
            "predicate",
            "object",
            "object_type",
            "object_value",  # Typed numeric value for FILTER comparisons
            "graph",
            pl.lit(None).cast(pl.Utf8).alias("quoted_triple_id"),
            "source",
            # Convert t_added microseconds to datetime
            (pl.col("t_added") * 1000).cast(pl.Datetime("ns", "UTC")).alias("timestamp"),
            "confidence",
            "process",
            pl.lit(None).cast(pl.Utf8).alias("version"),
            pl.lit("{}").alias("metadata"),
            pl.lit(None).cast(pl.Utf8).alias("superseded_by"),
            ((pl.col("flags").cast(pl.Int32) & int(FactFlags.DELETED)) != 0).alias("deprecated"),
        ])
        
        # Generate UUIDs using row_nr (fast)
        n = len(result)
        if n > 0:
            # Use sequential IDs for now (much faster than UUID generation)
            result = result.with_columns([
                pl.arange(0, n, eager=True).cast(pl.Utf8).alias("assertion_id")
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
        
        # Intern subjects (all URIs)
        s_col = [self._term_dict.intern_iri(s) for s in subjects]
        
        # Intern predicates (all URIs)
        p_col = [self._term_dict.intern_iri(p) for p in predicates]
        
        # Intern objects (could be literals or URIs)
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for the unified store
        store_dir = path.parent / (path.stem + "_unified")
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components
        self._term_dict.save(store_dir / "terms")
        self._qt_dict.save(store_dir / "quoted_triples")
        self._fact_store.save(store_dir / "facts")
        
        # Also save the legacy format for backward compatibility
        self._df.write_parquet(path)
    
    @classmethod
    def load(cls, path: Path | str) -> "TripleStore":
        """
        Load a triple store from disk.
        
        Attempts to load unified format first, falls back to legacy.
        """
        path = Path(path)
        store_dir = path.parent / (path.stem + "_unified")
        
        if store_dir.exists():
            # Load unified format
            store = cls()
            store._term_dict = TermDict.load(store_dir / "terms")
            store._qt_dict = QtDict.load(store_dir / "quoted_triples", store._term_dict)
            store._fact_store = FactStore.load(
                store_dir / "facts", 
                store._term_dict, 
                store._qt_dict
            )
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
    
    def load_graph(
        self,
        source_uri: str,
        graph_uri: Optional[str] = None,
        silent: bool = False,
    ) -> int:
        """Load RDF data from a URI into a graph."""
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
        
        # Determine format from extension
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in (".ttl", ".turtle"):
                from rdf_starbase.formats.turtle import parse_turtle
                parsed = parse_turtle(file_path.read_text())
                triples = parsed.triples
            elif suffix in (".nt", ".ntriples"):
                from rdf_starbase.formats.ntriples import parse_ntriples
                parsed = parse_ntriples(file_path.read_text())
                triples = parsed.triples
            elif suffix in (".rdf", ".xml"):
                from rdf_starbase.formats.rdfxml import parse_rdfxml
                parsed = parse_rdfxml(file_path.read_text())
                triples = parsed.triples
            elif suffix in (".jsonld", ".json"):
                from rdf_starbase.formats.jsonld import parse_jsonld
                parsed = parse_jsonld(file_path.read_text())
                triples = parsed.triples
            else:
                from rdf_starbase.formats.turtle import parse_turtle
                parsed = parse_turtle(file_path.read_text())
                triples = parsed.triples
        except Exception as e:
            if silent:
                return 0
            raise ValueError(f"Failed to parse {file_path}: {e}")
        
        # Add triples to the graph
        prov = ProvenanceContext(
            source=source_uri,
            confidence=1.0,
            process="LOAD",
        )
        
        count = 0
        for triple in triples:
            self.add_triple(
                subject=triple.subject,
                predicate=triple.predicate,
                obj=triple.object,
                provenance=prov,
                graph=graph_uri,
            )
            count += 1
        
        return count
