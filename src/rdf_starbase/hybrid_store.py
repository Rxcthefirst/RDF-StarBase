"""
Hybrid TripleStore using Oxigraph for core storage + Polars for metadata.

This prototype shows how to get Oxigraph's Rust performance while keeping
RDF-StarBase's unique metadata features (confidence, source, timestamps).
"""
import time
import hashlib
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import polars as pl

try:
    from pyoxigraph import Store, Quad, Triple, NamedNode, Literal, BlankNode, RdfFormat
    OXIGRAPH_AVAILABLE = True
except ImportError:
    OXIGRAPH_AVAILABLE = False
    print("Warning: pyoxigraph not installed. Install with: pip install pyoxigraph")


@dataclass
class AssertionMetadata:
    """Metadata about a triple assertion."""
    source: Optional[str] = None
    confidence: float = 1.0
    timestamp: Optional[str] = None
    process: Optional[str] = None


def hash_triple(s: str, p: str, o: str) -> int:
    """Hash a triple to a 64-bit integer for fast lookup."""
    key = f"{s}\x00{p}\x00{o}"
    h = hashlib.md5(key.encode()).digest()
    return int.from_bytes(h[:8], 'little')


class HybridTripleStore:
    """
    Hybrid store combining Oxigraph (Rust) with Polars metadata.
    
    Architecture:
    - Oxigraph: Stores raw triples, handles parsing, basic SPARQL
    - Polars: Stores assertion metadata (confidence, source, timestamps)
    
    This gives us:
    - Oxigraph's ~500K triples/sec parsing and loading
    - Oxigraph's optimized SPARQL execution for basic queries
    - Our custom metadata for AI grounding and provenance
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Create a hybrid store.
        
        Args:
            path: Optional path for persistent storage
        """
        if not OXIGRAPH_AVAILABLE:
            raise ImportError("pyoxigraph required for HybridTripleStore")
        
        # Core triple storage (Rust)
        self._store = Store(path) if path else Store()
        
        # Metadata storage (Polars/Rust)
        self._metadata = pl.DataFrame({
            "triple_hash": pl.Series([], dtype=pl.UInt64),
            "source": pl.Series([], dtype=pl.Utf8),
            "confidence": pl.Series([], dtype=pl.Float64),
            "timestamp": pl.Series([], dtype=pl.Utf8),
            "process": pl.Series([], dtype=pl.Utf8),
        })
        
        # Cache for hash -> triple mapping (for metadata queries)
        self._hash_to_triple: Dict[int, tuple] = {}
    
    def __len__(self) -> int:
        """Return number of triples."""
        return len(list(self._store))
    
    def _to_oxigraph_term(self, term: str):
        """Convert string term to Oxigraph type."""
        if term.startswith("_:"):
            return BlankNode(term[2:])
        elif term.startswith('"'):
            # Parse literal
            if "^^" in term:
                val, dtype = term.rsplit("^^", 1)
                val = val.strip('"')
                dtype = dtype.strip("<>")
                return Literal(val, datatype=NamedNode(dtype))
            elif "@" in term and term.count('"') == 2:
                val, lang = term.rsplit("@", 1)
                val = val.strip('"')
                return Literal(val, language=lang)
            else:
                return Literal(term.strip('"'))
        else:
            # IRI
            return NamedNode(term.strip("<>"))
    
    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
        confidence: float = 1.0,
        graph: Optional[str] = None,
    ) -> None:
        """
        Add a triple with optional metadata.
        
        Args:
            subject: Subject IRI or blank node
            predicate: Predicate IRI
            obj: Object (IRI, blank node, or literal)
            source: Optional data source
            confidence: Confidence score (0.0-1.0)
            graph: Optional named graph
        """
        # Add to Oxigraph (Rust-fast)
        s = self._to_oxigraph_term(subject)
        p = self._to_oxigraph_term(predicate)
        o = self._to_oxigraph_term(obj)
        
        if graph:
            g = self._to_oxigraph_term(graph)
            self._store.add(Quad(s, p, o, g))
        else:
            self._store.add(Quad(s, p, o))
        
        # Store metadata if provided
        if source or confidence != 1.0:
            h = hash_triple(subject, predicate, obj)
            self._hash_to_triple[h] = (subject, predicate, obj)
            
            new_row = pl.DataFrame({
                "triple_hash": [h],
                "source": [source],
                "confidence": [confidence],
                "timestamp": [None],
                "process": [None],
            }).cast({
                "triple_hash": pl.UInt64,
                "confidence": pl.Float64,
            })
            self._metadata = pl.concat([self._metadata, new_row])
    
    def add_batch(
        self,
        subjects: List[str],
        predicates: List[str],
        objects: List[str],
        source: Optional[str] = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Add multiple triples at once (vectorized).
        
        Returns number of triples added.
        """
        count = 0
        for s, p, o in zip(subjects, predicates, objects):
            self.add(s, p, o, source=source, confidence=confidence)
            count += 1
        return count
    
    def load_file(
        self,
        path: str,
        format: str = "turtle",
        source: Optional[str] = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Load triples from a file using Oxigraph's Rust parser.
        
        This is the FAST PATH - uses Oxigraph's native Rust parsing.
        
        Args:
            path: Path to RDF file
            format: RDF format (turtle, ntriples, rdfxml, etc.)
            source: Optional source metadata for all loaded triples
            confidence: Confidence for all loaded triples
            
        Returns:
            Number of triples loaded
        """
        format_map = {
            "turtle": RdfFormat.TURTLE,
            "ttl": RdfFormat.TURTLE,
            "nt": RdfFormat.N_TRIPLES,
            "ntriples": RdfFormat.N_TRIPLES,
            "nq": RdfFormat.N_QUADS,
            "nquads": RdfFormat.N_QUADS,
            "trig": RdfFormat.TRIG,
            "xml": RdfFormat.RDF_XML,
            "rdfxml": RdfFormat.RDF_XML,
        }
        
        rdf_format = format_map.get(format.lower(), RdfFormat.TURTLE)
        
        before = len(self)
        
        with open(path, 'rb') as f:
            self._store.load(f, rdf_format, base_iri=f"file://{path}")
        
        after = len(self)
        loaded = after - before
        
        # If metadata requested, we'd need to iterate and store hashes
        # For now, skip metadata for bulk loads (can be added later)
        
        return loaded
    
    def query(self, sparql: str) -> Union[pl.DataFrame, List[Dict], bool]:
        """
        Execute a SPARQL query.
        
        For basic queries, uses Oxigraph's Rust SPARQL engine (FAST).
        For metadata queries, falls back to custom handling.
        
        Args:
            sparql: SPARQL query string
            
        Returns:
            Query results (DataFrame for SELECT, bool for ASK)
        """
        # Detect if query needs metadata
        needs_metadata = any(kw in sparql.lower() for kw in [
            "confidence", "source", "timestamp", "process",
            "prov:", "dqv:"
        ])
        
        if needs_metadata:
            return self._query_with_metadata(sparql)
        
        # Use Oxigraph's Rust SPARQL engine (FAST PATH)
        results = self._store.query(sparql)
        
        # Convert to DataFrame
        if hasattr(results, '__iter__'):
            rows = list(results)
            if not rows:
                return pl.DataFrame()
            
            # Extract variable names and values
            if hasattr(rows[0], 'values'):
                # SELECT query
                cols = {}
                for var in rows[0].variables:
                    cols[str(var)] = [
                        str(row[var]) if row[var] else None 
                        for row in rows
                    ]
                return pl.DataFrame(cols)
            else:
                # Other result type
                return rows
        
        return results
    
    def _query_with_metadata(self, sparql: str) -> pl.DataFrame:
        """Handle queries that need assertion metadata."""
        # For now, delegate to Oxigraph and join with metadata
        # A full implementation would parse the query and rewrite it
        
        # Get base results from Oxigraph
        base_results = self._store.query(sparql.replace("?confidence", "").replace("?source", ""))
        
        # TODO: Join with metadata DataFrame
        # This would require query parsing to know which triples to join
        
        raise NotImplementedError(
            "Metadata queries require query rewriting. "
            "Use get_metadata() for specific triples."
        )
    
    def get_metadata(self, subject: str, predicate: str, obj: str) -> Optional[AssertionMetadata]:
        """Get metadata for a specific triple."""
        h = hash_triple(subject, predicate, obj)
        
        result = self._metadata.filter(pl.col("triple_hash") == h)
        if len(result) == 0:
            return None
        
        row = result.row(0, named=True)
        return AssertionMetadata(
            source=row["source"],
            confidence=row["confidence"],
            timestamp=row["timestamp"],
            process=row["process"],
        )


def benchmark_hybrid():
    """Benchmark the hybrid store."""
    print("=" * 60)
    print("HYBRID STORE BENCHMARK")
    print("=" * 60)
    
    store = HybridTripleStore()
    
    # Test loading
    test_file = "data/sample/data.ttl"
    
    print(f"\nLoading: {test_file}")
    t0 = time.time()
    count = store.load_file(test_file, source="benchmark", confidence=0.95)
    t1 = time.time()
    
    print(f"Loaded {count:,} triples in {t1-t0:.3f}s ({count/(t1-t0):,.0f}/sec)")
    
    # Test basic query
    print("\nQuery: COUNT(*)")
    t0 = time.time()
    for _ in range(10):
        result = store.query("SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Time: {(t1-t0)/10*1000:.2f}ms avg")
    
    # Test pattern query
    print("\nQuery: SELECT ?s ?p ?o LIMIT 1000")
    t0 = time.time()
    for _ in range(10):
        result = store.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1000")
    t1 = time.time()
    print(f"Rows: {len(result)}")
    print(f"Time: {(t1-t0)/10*1000:.2f}ms avg")
    
    # Test filter query  
    print("\nQuery: FILTER(isIRI(?o)) LIMIT 1000")
    t0 = time.time()
    for _ in range(10):
        result = store.query("""
            SELECT ?s ?p ?o WHERE { 
                ?s ?p ?o 
                FILTER(isIRI(?o))
            } LIMIT 1000
        """)
    t1 = time.time()
    print(f"Rows: {len(result)}")
    print(f"Time: {(t1-t0)/10*1000:.2f}ms avg")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if OXIGRAPH_AVAILABLE:
        benchmark_hybrid()
    else:
        print("Install pyoxigraph: pip install pyoxigraph")
