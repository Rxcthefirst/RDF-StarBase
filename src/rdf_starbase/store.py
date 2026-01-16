"""
Core TripleStore implementation using Polars.

The TripleStore is the heart of RDF-StarBase, leveraging Polars DataFrames
for blazingly fast RDF-Star operations.
"""

from datetime import datetime
from typing import Optional, Any, Literal
from uuid import UUID, uuid4
from pathlib import Path

import polars as pl

from rdf_starbase.models import Triple, QuotedTriple, Assertion, ProvenanceContext


class TripleStore:
    """
    A high-performance RDF-Star triple store backed by Polars DataFrames.
    
    Key design decisions:
    - Each assertion is a row in a Polars DataFrame
    - Quoted triples are stored with unique IDs for reference
    - Provenance columns are first-class (not metadata)
    - Uses Polars lazy evaluation for query optimization
    """
    
    def __init__(self):
        """Initialize an empty triple store."""
        self._df = self._create_empty_dataframe()
        self._quoted_triples: dict[UUID, QuotedTriple] = {}
    
    @staticmethod
    def _create_empty_dataframe() -> pl.DataFrame:
        """Create the schema for the assertion DataFrame."""
        return pl.DataFrame({
            "assertion_id": pl.Series([], dtype=pl.Utf8),
            "subject": pl.Series([], dtype=pl.Utf8),
            "predicate": pl.Series([], dtype=pl.Utf8),
            "object": pl.Series([], dtype=pl.Utf8),
            "object_type": pl.Series([], dtype=pl.Utf8),  # uri, literal, int, float, bool
            "graph": pl.Series([], dtype=pl.Utf8),
            "quoted_triple_id": pl.Series([], dtype=pl.Utf8),  # If subject/object is a quoted triple
            # Provenance columns
            "source": pl.Series([], dtype=pl.Utf8),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "confidence": pl.Series([], dtype=pl.Float64),
            "process": pl.Series([], dtype=pl.Utf8),
            "version": pl.Series([], dtype=pl.Utf8),
            "metadata": pl.Series([], dtype=pl.Utf8),  # JSON string
            # Status
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
        assertion_id = uuid4()
        
        # Determine object type
        if isinstance(obj, str) and obj.startswith("http"):
            obj_type = "uri"
        elif isinstance(obj, str):
            obj_type = "literal"
        elif isinstance(obj, bool):
            obj_type = "bool"
        elif isinstance(obj, int):
            obj_type = "int"
        elif isinstance(obj, float):
            obj_type = "float"
        else:
            obj_type = "literal"
            obj = str(obj)
        
        # Create new row
        new_row = pl.DataFrame({
            "assertion_id": [str(assertion_id)],
            "subject": [subject],
            "predicate": [predicate],
            "object": [str(obj)],
            "object_type": [obj_type],
            "graph": [graph],
            "quoted_triple_id": [None],
            "source": [provenance.source],
            "timestamp": [provenance.timestamp],
            "confidence": [provenance.confidence],
            "process": [provenance.process],
            "version": [provenance.version],
            "metadata": [str(provenance.metadata)],
            "superseded_by": [None],
            "deprecated": [False],
        })
        
        # Append to main dataframe
        self._df = pl.concat([self._df, new_row], how="vertical")
        
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
        
        This is a basic pattern matching query - the foundation of SPARQL.
        Uses Polars' lazy evaluation for optimization.
        
        Args:
            subject: Filter by subject (None = wildcard)
            predicate: Filter by predicate (None = wildcard)
            obj: Filter by object (None = wildcard)
            graph: Filter by graph (None = wildcard)
            source: Filter by provenance source
            min_confidence: Minimum confidence threshold
            include_deprecated: Whether to include deprecated assertions
            
        Returns:
            Filtered DataFrame of matching assertions
        """
        df = self._df.lazy()
        
        # Apply filters
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
        """
        Find competing assertions about the same subject-predicate pair.
        
        This implements the "Competing Claims View" primitive from the manifesto.
        
        Returns assertions sorted by confidence (desc) and recency (desc).
        """
        df = self.get_triples(subject=subject, predicate=predicate, include_deprecated=False)
        
        # Sort by confidence (descending) then timestamp (descending)
        df = df.sort(["confidence", "timestamp"], descending=[True, True])
        
        return df
    
    def deprecate_assertion(self, assertion_id: UUID, superseded_by: Optional[UUID] = None) -> None:
        """
        Mark an assertion as deprecated, optionally linking to superseding assertion.
        
        Args:
            assertion_id: ID of assertion to deprecate
            superseded_by: Optional ID of the assertion that supersedes this one
        """
        self._df = self._df.with_columns([
            pl.when(pl.col("assertion_id") == str(assertion_id))
            .then(True)
            .otherwise(pl.col("deprecated"))
            .alias("deprecated"),
            
            pl.when(pl.col("assertion_id") == str(assertion_id))
            .then(str(superseded_by) if superseded_by else None)
            .otherwise(pl.col("superseded_by"))
            .alias("superseded_by"),
        ])
    
    def get_provenance_timeline(self, subject: str, predicate: str) -> pl.DataFrame:
        """
        Get the full history of assertions about a subject-predicate pair.
        
        This implements the "Provenance Timeline" primitive from the manifesto.
        Shows the evolution of knowledge over time, including deprecated assertions.
        """
        df = self.get_triples(
            subject=subject,
            predicate=predicate,
            include_deprecated=True
        )
        
        # Sort by timestamp
        df = df.sort("timestamp")
        
        return df
    
    def save(self, path: Path | str) -> None:
        """
        Save the triple store to disk using Parquet format.
        
        Parquet is Polars' native format and provides excellent compression
        and query performance.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(path)
    
    @classmethod
    def load(cls, path: Path | str) -> "TripleStore":
        """
        Load a triple store from disk.
        
        Args:
            path: Path to the Parquet file
            
        Returns:
            Loaded TripleStore instance
        """
        store = cls()
        store._df = pl.read_parquet(path)
        return store
    
    def stats(self) -> dict[str, Any]:
        """Get statistics about the triple store."""
        total = len(self._df)
        active = len(self._df.filter(~pl.col("deprecated")))
        deprecated = total - active
        
        sources = self._df.select("source").unique().height
        
        return {
            "total_assertions": total,
            "active_assertions": active,
            "deprecated_assertions": deprecated,
            "unique_sources": sources,
            "unique_subjects": self._df.select("subject").unique().height,
            "unique_predicates": self._df.select("predicate").unique().height,
        }
    
    def __len__(self) -> int:
        """Return the number of active assertions."""
        return len(self._df.filter(~pl.col("deprecated")))
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"TripleStore("
            f"assertions={stats['active_assertions']}, "
            f"sources={stats['unique_sources']}, "
            f"subjects={stats['unique_subjects']})"
        )
