"""
Core TripleStore implementation using Polars.

The TripleStore is the heart of RDF-StarBase, leveraging Polars DataFrames
for blazingly fast RDF-Star operations.
"""

from datetime import datetime, timezone
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
    
    def add_triples_batch(
        self,
        triples: list[dict],
    ) -> int:
        """
        Add multiple triples in a single batch operation.
        
        This is MUCH faster than calling add_triple() repeatedly because:
        - Single DataFrame concatenation instead of N concatenations
        - Batch UUID generation
        - No intermediate DataFrame creation
        
        Args:
            triples: List of dicts with keys:
                - subject: str
                - predicate: str
                - object: Any
                - source: str
                - confidence: float (optional, default 1.0)
                - process: str (optional)
                - timestamp: datetime (optional)
                - graph: str (optional)
                
        Returns:
            Number of triples added
        """
        if not triples:
            return 0
        
        # Prepare batch data
        rows = {
            "assertion_id": [],
            "subject": [],
            "predicate": [],
            "object": [],
            "object_type": [],
            "graph": [],
            "quoted_triple_id": [],
            "source": [],
            "timestamp": [],
            "confidence": [],
            "process": [],
            "version": [],
            "metadata": [],
            "superseded_by": [],
            "deprecated": [],
        }
        
        now = datetime.now()
        
        for t in triples:
            obj = t.get("object", "")
            
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
            
            rows["assertion_id"].append(str(uuid4()))
            rows["subject"].append(t["subject"])
            rows["predicate"].append(t["predicate"])
            rows["object"].append(str(obj))
            rows["object_type"].append(obj_type)
            rows["graph"].append(t.get("graph"))
            rows["quoted_triple_id"].append(None)
            rows["source"].append(t.get("source", "unknown"))
            rows["timestamp"].append(t.get("timestamp", now))
            rows["confidence"].append(t.get("confidence", 1.0))
            rows["process"].append(t.get("process"))
            rows["version"].append(t.get("version"))
            rows["metadata"].append(str(t.get("metadata", {})))
            rows["superseded_by"].append(None)
            rows["deprecated"].append(False)
        
        # Create batch DataFrame
        batch_df = pl.DataFrame(rows)
        
        # Single concatenation
        self._df = pl.concat([self._df, batch_df], how="vertical")
        
        return len(triples)
    
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
    
    def mark_deleted(
        self, 
        s: Optional[str] = None, 
        p: Optional[str] = None, 
        o: Optional[str] = None
    ) -> int:
        """
        Mark matching triples as deprecated (soft delete).
        
        Args:
            s: Subject filter (optional)
            p: Predicate filter (optional)
            o: Object filter (optional)
            
        Returns:
            Number of triples marked as deleted
        """
        # Build filter condition
        condition = pl.lit(True)
        if s is not None:
            condition = condition & (pl.col("subject") == s)
        if p is not None:
            condition = condition & (pl.col("predicate") == p)
        if o is not None:
            condition = condition & (pl.col("object") == o)
        
        # Count matching rows
        count = self._df.filter(condition & ~pl.col("deprecated")).height
        
        # Mark matching rows as deprecated
        self._df = self._df.with_columns([
            pl.when(condition)
            .then(True)
            .otherwise(pl.col("deprecated"))
            .alias("deprecated"),
        ])
        
        return count
    
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

    # =========================================================================
    # Named Graph Management
    # =========================================================================
    
    def list_graphs(self) -> list[str]:
        """
        List all named graphs in the store.
        
        Returns:
            List of graph URIs (excluding None/default graph)
        """
        graphs = (
            self._df
            .filter(pl.col("graph").is_not_null() & ~pl.col("deprecated"))
            .select("graph")
            .unique()
            .to_series()
            .to_list()
        )
        return sorted(graphs)
    
    def create_graph(self, graph_uri: str) -> bool:
        """
        Create an empty named graph.
        
        In RDF-StarBase, graphs are created implicitly when triples are added.
        This method is provided for SPARQL compatibility and returns True
        if the graph didn't exist (was created) or False if it already exists.
        
        Args:
            graph_uri: The IRI of the graph to create
            
        Returns:
            True if graph was created, False if it already existed
        """
        existing = self._df.filter(
            (pl.col("graph") == graph_uri) & ~pl.col("deprecated")
        ).height
        return existing == 0
    
    def drop_graph(self, graph_uri: str, silent: bool = False) -> int:
        """
        Drop (delete) a named graph and all its triples.
        
        Args:
            graph_uri: The IRI of the graph to drop
            silent: If True, don't raise error if graph doesn't exist
            
        Returns:
            Number of triples removed
        """
        condition = (pl.col("graph") == graph_uri) & ~pl.col("deprecated")
        count = self._df.filter(condition).height
        
        if count == 0 and not silent:
            # Graph doesn't exist - in SPARQL, DROP on non-existent graph is fine
            return 0
        
        # Mark all triples in the graph as deprecated
        self._df = self._df.with_columns([
            pl.when(condition)
            .then(True)
            .otherwise(pl.col("deprecated"))
            .alias("deprecated"),
        ])
        
        return count
    
    def clear_graph(self, graph_uri: Optional[str] = None, silent: bool = False) -> int:
        """
        Clear all triples from a graph (or default graph if None).
        
        Unlike DROP, CLEAR keeps the graph existing but empty.
        For the default graph (None), removes all triples not in named graphs.
        
        Args:
            graph_uri: The IRI of the graph to clear, or None for default graph
            silent: If True, don't raise error if graph doesn't exist
            
        Returns:
            Number of triples removed
        """
        if graph_uri is None:
            # Clear default graph (where graph column is null)
            condition = pl.col("graph").is_null() & ~pl.col("deprecated")
        else:
            condition = (pl.col("graph") == graph_uri) & ~pl.col("deprecated")
        
        count = self._df.filter(condition).height
        
        # Mark matching triples as deprecated
        self._df = self._df.with_columns([
            pl.when(condition)
            .then(True)
            .otherwise(pl.col("deprecated"))
            .alias("deprecated"),
        ])
        
        return count
    
    def copy_graph(
        self, 
        source_graph: Optional[str], 
        dest_graph: str,
        silent: bool = False,
    ) -> int:
        """
        Copy all triples from source graph to destination graph.
        
        The destination graph is cleared first, then populated with
        copies of all triples from the source graph.
        
        Args:
            source_graph: Source graph IRI (None for default graph)
            dest_graph: Destination graph IRI
            silent: If True, don't fail if source doesn't exist
            
        Returns:
            Number of triples copied
        """
        # Clear destination first
        self.clear_graph(dest_graph, silent=True)
        
        # Get source triples
        if source_graph is None:
            source_df = self._df.filter(
                pl.col("graph").is_null() & ~pl.col("deprecated")
            )
        else:
            source_df = self._df.filter(
                (pl.col("graph") == source_graph) & ~pl.col("deprecated")
            )
        
        if source_df.height == 0:
            return 0
        
        # Create copies with new assertion IDs and target graph
        from uuid import uuid4
        
        new_rows = source_df.with_columns([
            pl.lit(str(uuid4())).alias("assertion_id"),
            pl.lit(dest_graph).alias("graph"),
            pl.lit(datetime.now(timezone.utc)).alias("timestamp"),
        ])
        
        self._df = pl.concat([self._df, new_rows])
        return new_rows.height
    
    def move_graph(
        self,
        source_graph: Optional[str],
        dest_graph: str,
        silent: bool = False,
    ) -> int:
        """
        Move all triples from source graph to destination graph.
        
        Like COPY but also removes triples from source graph.
        
        Args:
            source_graph: Source graph IRI (None for default graph)
            dest_graph: Destination graph IRI
            silent: If True, don't fail if source doesn't exist
            
        Returns:
            Number of triples moved
        """
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
        dest_graph: str,
        silent: bool = False,
    ) -> int:
        """
        Add all triples from source graph to destination graph.
        
        Unlike COPY, doesn't clear destination first - adds to existing triples.
        
        Args:
            source_graph: Source graph IRI (None for default graph)
            dest_graph: Destination graph IRI
            silent: If True, don't fail if source doesn't exist
            
        Returns:
            Number of triples added
        """
        # Get source triples
        if source_graph is None:
            source_df = self._df.filter(
                pl.col("graph").is_null() & ~pl.col("deprecated")
            )
        else:
            source_df = self._df.filter(
                (pl.col("graph") == source_graph) & ~pl.col("deprecated")
            )
        
        if source_df.height == 0:
            return 0
        
        # Create copies with new assertion IDs and target graph
        from uuid import uuid4
        
        new_rows = source_df.with_columns([
            pl.lit(str(uuid4())).alias("assertion_id"),
            pl.lit(dest_graph).alias("graph"),
            pl.lit(datetime.now(timezone.utc)).alias("timestamp"),
        ])
        
        self._df = pl.concat([self._df, new_rows])
        return new_rows.height
    
    def load_graph(
        self,
        source_uri: str,
        graph_uri: Optional[str] = None,
        silent: bool = False,
    ) -> int:
        """
        Load RDF data from a URI into a graph.
        
        Supports loading from:
        - Local files (file:// or plain paths)
        - HTTP/HTTPS URLs
        - Formats: Turtle, N-Triples, RDF/XML, JSON-LD (auto-detected)
        
        Args:
            source_uri: URI to load data from
            graph_uri: Target graph (None for default graph)
            silent: If True, don't fail on errors
            
        Returns:
            Number of triples loaded
        """
        from pathlib import Path
        from urllib.parse import urlparse, unquote
        from rdf_starbase.models import ProvenanceContext
        
        # Determine file path
        if source_uri.startswith("file://"):
            # Properly parse file:// URI
            parsed = urlparse(source_uri)
            # unquote handles percent-encoded characters
            file_path_str = unquote(parsed.path)
            # On Windows, file:///C:/path becomes /C:/path, remove leading /
            if len(file_path_str) > 2 and file_path_str[0] == '/' and file_path_str[2] == ':':
                file_path_str = file_path_str[1:]
            file_path = Path(file_path_str)
        elif source_uri.startswith(("http://", "https://")):
            # Download to temp file
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
                # Default to Turtle
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
