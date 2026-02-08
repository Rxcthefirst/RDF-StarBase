"""
Bulk loader for large RDF files with streaming support.

Processes files in chunks to handle files larger than available memory.
Supports both native Python parser and Oxigraph (Rust) parser for maximum performance.
"""
import gzip
import time
from pathlib import Path
from typing import Optional, Iterator, List, Tuple, Any
from dataclasses import dataclass
from src.rdf_starbase import TripleStore
from src.rdf_starbase.formats.turtle import TurtleParser
from src.rdf_starbase.models import Triple

# Try to import Oxigraph for fast parsing (~17x faster)
try:
    from pyoxigraph import parse as oxigraph_parse, RdfFormat, NamedNode, Literal, BlankNode
    OXIGRAPH_AVAILABLE = True
except ImportError:
    OXIGRAPH_AVAILABLE = False
    NamedNode = Literal = BlankNode = None


@dataclass
class ParsedChunk:
    """A chunk of parsed triples in columnar format."""
    subjects: List[str]
    predicates: List[str]
    objects: List[Any]
    count: int


def _oxigraph_quad_to_strings(quad) -> Tuple[str, str, str]:
    """Convert Oxigraph Quad to (subject, predicate, object) strings."""
    # Subject
    s = quad.subject
    if hasattr(s, 'value'):
        subj = s.value
    else:
        subj = str(s)
    
    # Predicate
    pred = quad.predicate.value
    
    # Object
    o = quad.object
    if hasattr(o, 'value'):
        if hasattr(o, 'datatype'):
            # Typed literal
            dt = o.datatype.value if o.datatype else None
            if dt and dt != "http://www.w3.org/2001/XMLSchema#string":
                obj = f'"{o.value}"^^<{dt}>'
            else:
                obj = o.value
        elif hasattr(o, 'language') and o.language:
            obj = f'"{o.value}"@{o.language}'
        else:
            obj = o.value
    else:
        obj = str(o)
    
    return subj, pred, obj


def intern_quads_direct(store: "TripleStore", quads, graph_id: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """
    Intern Oxigraph quads directly to TermIds - FASTEST PATH.
    
    Avoids string conversion overhead by accessing pyoxigraph attributes directly
    and using the TermDict's fast-path methods.
    Uses isinstance() instead of hasattr() for faster type checking.
    
    Args:
        store: TripleStore with term_dict
        quads: Iterator of pyoxigraph Quad objects
        graph_id: Pre-interned graph TermId
        
    Returns:
        Tuple of (subject_ids, predicate_ids, object_ids) lists
    """
    term_dict = store._term_dict
    
    # Cache method references for faster lookup
    intern_iri = term_dict.intern_iri
    intern_literal = term_dict.intern_literal
    intern_bnode = term_dict.intern_bnode
    
    s_ids = []
    p_ids = []
    o_ids = []
    
    # Pre-compute constant
    XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
    
    for quad in quads:
        # Subject: NamedNode or BlankNode
        s = quad.subject
        if isinstance(s, NamedNode):
            s_ids.append(intern_iri(s.value))
        else:  # BlankNode
            s_ids.append(intern_bnode(str(s)[2:]))  # Strip "_:" prefix
        
        # Predicate: always NamedNode
        p_ids.append(intern_iri(quad.predicate.value))
        
        # Object: Literal, NamedNode, or BlankNode
        o = quad.object
        if isinstance(o, Literal):
            val = o.value
            lang = o.language
            if lang:
                o_ids.append(intern_literal(val, lang=lang))
            else:
                dt = o.datatype
                if dt is None or dt.value == XSD_STRING:
                    o_ids.append(intern_literal(val))
                else:
                    o_ids.append(intern_literal(val, datatype=dt.value))
        elif isinstance(o, NamedNode):
            o_ids.append(intern_iri(o.value))
        else:  # BlankNode
            o_ids.append(intern_bnode(str(o)[2:]))
    
    return s_ids, p_ids, o_ids


def intern_quads_batch(store: "TripleStore", quads: list) -> Tuple[List[int], List[int], List[int]]:
    """
    Batch-optimized quad interning - 2-3x faster than intern_quads_direct.
    
    Collects all strings first, deduplicates, batch-interns, then maps back.
    Best for large batches (>10K quads).
    
    Args:
        store: TripleStore with term_dict
        quads: List of pyoxigraph Quad objects
        
    Returns:
        Tuple of (subject_ids, predicate_ids, object_ids) lists
    """
    if not quads:
        return [], [], []
    
    n = len(quads)
    term_dict = store._term_dict
    
    # Pre-compute constant
    XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
    
    # Phase 1: Extract all strings into parallel arrays
    # Subjects
    subject_iris = []  # (index, iri)
    subject_bnodes = []  # (index, label)
    
    # Predicates (always IRIs)
    predicate_iris = []
    
    # Objects  
    object_iris = []  # (index, iri)
    object_bnodes = []  # (index, label)
    object_literals = []  # (index, value, datatype, lang)
    
    for i, quad in enumerate(quads):
        # Subject
        s = quad.subject
        if isinstance(s, NamedNode):
            subject_iris.append((i, s.value))
        else:
            subject_bnodes.append((i, str(s)[2:]))
        
        # Predicate
        predicate_iris.append(quad.predicate.value)
        
        # Object
        o = quad.object
        if isinstance(o, Literal):
            val = o.value
            lang = o.language
            dt = o.datatype
            if lang:
                object_literals.append((i, val, None, lang))
            elif dt is None or dt.value == XSD_STRING:
                object_literals.append((i, val, None, None))
            else:
                object_literals.append((i, val, dt.value, None))
        elif isinstance(o, NamedNode):
            object_iris.append((i, o.value))
        else:
            object_bnodes.append((i, str(o)[2:]))
    
    # Phase 2: Batch intern by category
    s_ids = [0] * n
    p_ids = [0] * n
    o_ids = [0] * n
    
    # Subject IRIs
    if subject_iris:
        indices, iris = zip(*subject_iris)
        ids = term_dict.intern_iris_batch(list(iris))
        for idx, tid in zip(indices, ids):
            s_ids[idx] = tid
    
    # Subject blank nodes
    intern_bnode = term_dict.intern_bnode
    for idx, label in subject_bnodes:
        s_ids[idx] = intern_bnode(label)
    
    # Predicate IRIs (all predicates are IRIs)
    pred_term_ids = term_dict.intern_iris_batch(predicate_iris)
    p_ids = pred_term_ids
    
    # Object IRIs
    if object_iris:
        indices, iris = zip(*object_iris)
        ids = term_dict.intern_iris_batch(list(iris))
        for idx, tid in zip(indices, ids):
            o_ids[idx] = tid
    
    # Object blank nodes
    for idx, label in object_bnodes:
        o_ids[idx] = intern_bnode(label)
    
    # Object literals (batch for plain strings, individual for typed)
    if object_literals:
        plain_indices = []
        plain_values = []
        typed_items = []
        
        for idx, val, dt, lang in object_literals:
            if lang is None and dt is None:
                plain_indices.append(idx)
                plain_values.append(val)
            else:
                typed_items.append((idx, val, dt, lang))
        
        # Batch plain string literals
        if plain_values:
            lit_ids = term_dict.intern_literals_batch(plain_values)
            for idx, tid in zip(plain_indices, lit_ids):
                o_ids[idx] = tid
        
        # Individual typed/lang-tagged literals
        intern_literal = term_dict.intern_literal
        for idx, val, dt, lang in typed_items:
            o_ids[idx] = intern_literal(val, datatype=dt, lang=lang)
    
    return s_ids, p_ids, o_ids


def parse_and_intern_oxigraph(
    store: "TripleStore",
    chunk_text: str,
    graph_id: int = 0,
    base_iri: str = "http://example.org/"
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Parse AND intern in one pass using Oxigraph - FASTEST COMBINED PATH.
    
    Combines parsing and interning, avoiding intermediate string lists.
    
    Returns:
        Tuple of (s_ids, p_ids, o_ids, count)
    """
    quads = oxigraph_parse(chunk_text, RdfFormat.TURTLE, base_iri=base_iri)
    s_ids, p_ids, o_ids = intern_quads_direct(store, quads, graph_id)
    return s_ids, p_ids, o_ids, len(s_ids)


def parse_chunk_oxigraph(chunk_text: str, base_iri: str = "http://example.org/") -> ParsedChunk:
    """
    Parse a chunk using Oxigraph (Rust) - FAST PATH.
    
    ~17x faster than Python parser for spec-compliant Turtle.
    """
    subjects = []
    predicates = []
    objects = []
    
    for quad in oxigraph_parse(chunk_text, RdfFormat.TURTLE, base_iri=base_iri):
        s, p, o = _oxigraph_quad_to_strings(quad)
        subjects.append(s)
        predicates.append(p)
        objects.append(o)
    
    return ParsedChunk(subjects, predicates, objects, len(subjects))


def parse_chunk_native(chunk_text: str) -> ParsedChunk:
    """
    Parse a chunk using native Python parser.
    
    Handles RDF-Star and edge cases that Oxigraph may reject.
    """
    parser = TurtleParser()
    result = parser.parse(chunk_text)
    
    subjects = [t.subject for t in result.triples]
    predicates = [t.predicate for t in result.triples]
    objects = [t.object for t in result.triples]
    
    return ParsedChunk(subjects, predicates, objects, len(subjects))


def stream_turtle_chunks(
    file_path: str,
    chunk_lines: int = 500_000,
    progress_interval: int = 1_000_000,
    use_oxigraph: bool = True,
) -> Iterator[ParsedChunk]:
    """
    Stream parse a Turtle file in chunks.
    
    Yields ParsedChunk objects with columnar data ready for fast insertion.
    Automatically falls back to native parser if Oxigraph fails.
    
    Args:
        file_path: Path to .ttl or .ttl.gz file
        chunk_lines: Lines per chunk (default 500K)
        progress_interval: Print progress every N lines
        use_oxigraph: Use Oxigraph (Rust) parser if available (default True)
    """
    path = Path(file_path)
    is_gzipped = path.suffix.lower() == ".gz"
    
    # Determine parser to use
    use_fast_parser = use_oxigraph and OXIGRAPH_AVAILABLE
    parser_name = "Oxigraph (Rust)" if use_fast_parser else "Native Python"
    print(f"  Parser: {parser_name}")
    
    # Open file
    if is_gzipped:
        f = gzip.open(path, 'rt', encoding='utf-8')
    else:
        f = open(path, 'r', encoding='utf-8')
    
    try:
        prefix_lines = []
        data_lines = []
        total_lines = 0
        oxigraph_failures = 0
        
        for line in f:
            total_lines += 1
            
            # Accumulate prefix declarations
            stripped = line.strip()
            if stripped.startswith('@prefix') or stripped.startswith('@base') or \
               stripped.startswith('PREFIX') or stripped.startswith('BASE'):
                prefix_lines.append(line)
            else:
                data_lines.append(line)
            
            # Yield chunk when we have enough data lines AND we're at a statement boundary
            # Statement boundaries: empty line or line ending with '.'
            is_boundary = (stripped == '' or stripped.endswith('.'))
            
            if len(data_lines) >= chunk_lines and is_boundary:
                # Build chunk with all prefixes
                chunk_text = ''.join(prefix_lines) + ''.join(data_lines)
                
                # Try fast parser first, fall back to native
                if use_fast_parser:
                    try:
                        yield parse_chunk_oxigraph(chunk_text)
                    except Exception as e:
                        oxigraph_failures += 1
                        if oxigraph_failures == 1:
                            print(f"  Oxigraph parse error, falling back to native: {e}")
                        try:
                            yield parse_chunk_native(chunk_text)
                        except Exception as e2:
                            print(f"Warning: Both parsers failed at ~line {total_lines}: {e2}")
                else:
                    try:
                        yield parse_chunk_native(chunk_text)
                    except Exception as e:
                        print(f"Warning: Parse error at ~line {total_lines}: {e}")
                
                data_lines = []
            
            if total_lines % progress_interval == 0:
                print(f"  Read {total_lines:,} lines...")
        
        # Final chunk
        if data_lines:
            chunk_text = ''.join(prefix_lines) + ''.join(data_lines)
            if use_fast_parser:
                try:
                    yield parse_chunk_oxigraph(chunk_text)
                except Exception:
                    try:
                        yield parse_chunk_native(chunk_text)
                    except Exception as e:
                        print(f"Warning: Final parse error: {e}")
            else:
                try:
                    yield parse_chunk_native(chunk_text)
                except Exception as e:
                    print(f"Warning: Final parse error: {e}")
        
        print(f"  Total: {total_lines:,} lines")
        
    finally:
        f.close()


def bulk_load_turtle(
    store: TripleStore,
    file_path: str,
    chunk_lines: int = 500_000,
    graph_uri: Optional[str] = None,
    use_oxigraph: bool = True,
) -> int:
    """
    Bulk load a large Turtle file using streaming chunks and columnar insertion.
    
    This is the FASTEST loading path, combining:
    - Oxigraph (Rust) parser: ~200-500K triples/sec parsing
    - Columnar term interning: ~500K-4M terms/sec
    - Direct FactStore insertion: ~10M triples/sec
    
    Args:
        store: The TripleStore to load into
        file_path: Path to .ttl or .ttl.gz file
        chunk_lines: Lines per chunk (default 500K)
        graph_uri: Optional named graph URI
        use_oxigraph: Use Oxigraph parser if available (default True)
    
    Returns:
        Total number of triples loaded
    """
    parser_info = ""
    if use_oxigraph:
        if OXIGRAPH_AVAILABLE:
            parser_info = " [Oxigraph enabled - 17x faster]"
        else:
            parser_info = " [Oxigraph not installed - pip install pyoxigraph for 17x speedup]"
    else:
        parser_info = " [native parser]"
    
    print(f"Bulk loading: {file_path}{parser_info}")
    t0 = time.time()
    
    total_triples = 0
    chunk_count = 0
    
    for chunk in stream_turtle_chunks(file_path, chunk_lines, use_oxigraph=use_oxigraph):
        chunk_count += 1
        
        if chunk.count == 0:
            continue
        
        # Use add_triples_columnar - the TRUE vectorized path
        added = store.add_triples_columnar(
            subjects=chunk.subjects,
            predicates=chunk.predicates,
            objects=chunk.objects,
            source="bulk_load",
            confidence=1.0,
            graph=graph_uri,
        )
        
        total_triples += added
        elapsed = time.time() - t0
        rate = total_triples / elapsed if elapsed > 0 else 0
        print(f"  Chunk {chunk_count}: +{added:,} triples, total: {total_triples:,}, rate: {rate:,.0f}/sec")
    
    elapsed = time.time() - t0
    print(f"\nCompleted: {total_triples:,} triples in {elapsed:.1f}s ({total_triples/elapsed:,.0f} triples/sec)")
    
    return total_triples


def bulk_load_turtle_oneshot(
    store: TripleStore,
    file_path: str,
    graph_uri: Optional[str] = None,
    deduplicate: bool = True,
) -> int:
    """
    FASTEST bulk load - single-shot loading for small/medium files.
    
    Reads entire file at once, parses with Oxigraph, and inserts.
    Best for files that fit in memory (<100MB).
    
    Args:
        store: The TripleStore to load into
        file_path: Path to .ttl or .ttl.gz file
        graph_uri: Optional named graph URI
        deduplicate: If True (default), remove duplicate triples per RDF set semantics
    
    Returns:
        Total number of unique triples loaded
    """
    if not OXIGRAPH_AVAILABLE:
        return bulk_load_turtle(store, file_path, graph_uri=graph_uri, use_oxigraph=False)
    
    from rdf_starbase.storage.facts import FactFlags, DEFAULT_GRAPH_ID
    
    path = Path(file_path)
    is_gzipped = path.suffix.lower() == ".gz"
    
    # Read entire file
    if is_gzipped:
        with gzip.open(path, 'rb') as f:
            content = f.read()
    else:
        with open(path, 'rb') as f:
            content = f.read()
    
    # Pre-intern graph
    graph_id = store._term_dict.intern_iri(graph_uri) if graph_uri else DEFAULT_GRAPH_ID
    source_id = store._term_dict.intern_literal("bulk_load")
    
    # Parse all at once with Oxigraph
    quads = list(oxigraph_parse(content, RdfFormat.TURTLE))
    
    # Intern directly (batch interning has bugs, using original method)
    s_ids, p_ids, o_ids = intern_quads_direct(store, quads, graph_id)
    
    # Deduplicate if requested (RDF defines graphs as sets, not multisets)
    if deduplicate and len(s_ids) > 0:
        seen = set()
        unique_s, unique_p, unique_o = [], [], []
        for s, p, o in zip(s_ids, p_ids, o_ids):
            triple = (s, p, o)
            if triple not in seen:
                seen.add(triple)
                unique_s.append(s)
                unique_p.append(p)
                unique_o.append(o)
        s_ids, p_ids, o_ids = unique_s, unique_p, unique_o
    
    count = len(s_ids)
    
    if count > 0:
        store._fact_store.add_facts_columnar(
            g_col=[graph_id] * count,
            s_col=s_ids,
            p_col=p_ids,
            o_col=o_ids,
            flags=FactFlags.ASSERTED,
            source=source_id,
            confidence=1.0,
        )
    
    store._invalidate_cache()
    return count


def bulk_load_turtle_fast(
    store: TripleStore,
    file_path: str,
    chunk_lines: int = 500_000,
    graph_uri: Optional[str] = None,
) -> int:
    """
    FASTEST bulk load path - direct Oxigraph parsing + interning.
    
    Eliminates string conversion overhead by interning directly from
    pyoxigraph objects. ~2x faster than bulk_load_turtle.
    
    Requires pyoxigraph. Falls back to bulk_load_turtle if not available.
    
    Args:
        store: The TripleStore to load into
        file_path: Path to .ttl or .ttl.gz file
        chunk_lines: Lines per chunk (default 500K)
        graph_uri: Optional named graph URI
    
    Returns:
        Total number of triples loaded
    """
    if not OXIGRAPH_AVAILABLE:
        print("  pyoxigraph not available, falling back to standard loader")
        return bulk_load_turtle(store, file_path, chunk_lines, graph_uri, use_oxigraph=False)
    
    from rdf_starbase.storage.facts import FactFlags, DEFAULT_GRAPH_ID
    
    print(f"Bulk loading (FAST): {file_path}")
    t0 = time.time()
    
    # Pre-intern graph and source
    graph_id = store._term_dict.intern_iri(graph_uri) if graph_uri else DEFAULT_GRAPH_ID
    source_id = store._term_dict.intern_literal("bulk_load")
    
    path = Path(file_path)
    is_gzipped = path.suffix.lower() == ".gz"
    
    if is_gzipped:
        f = gzip.open(path, 'rt', encoding='utf-8')
    else:
        f = open(path, 'r', encoding='utf-8')
    
    total_triples = 0
    chunk_count = 0
    prefix_lines = []
    data_lines = []
    total_lines = 0
    
    try:
        for line in f:
            total_lines += 1
            stripped = line.strip()
            
            # Accumulate prefix declarations
            if stripped.startswith('@prefix') or stripped.startswith('@base') or \
               stripped.startswith('PREFIX') or stripped.startswith('BASE'):
                prefix_lines.append(line)
            else:
                data_lines.append(line)
            
            # Yield chunk at statement boundary
            is_boundary = (stripped == '' or stripped.endswith('.'))
            
            if len(data_lines) >= chunk_lines and is_boundary:
                chunk_text = ''.join(prefix_lines) + ''.join(data_lines)
                chunk_count += 1
                
                try:
                    # Direct parse + intern (no string conversion)
                    s_ids, p_ids, o_ids, count = parse_and_intern_oxigraph(
                        store, chunk_text, graph_id
                    )
                    
                    if count > 0:
                        # Direct insert to FactStore
                        store._fact_store.add_facts_columnar(
                            g_col=[graph_id] * count,
                            s_col=s_ids,
                            p_col=p_ids,
                            o_col=o_ids,
                            flags=FactFlags.ASSERTED,
                            source=source_id,
                            confidence=1.0,
                        )
                        total_triples += count
                    
                    elapsed = time.time() - t0
                    rate = total_triples / elapsed if elapsed > 0 else 0
                    print(f"  Chunk {chunk_count}: +{count:,} triples, total: {total_triples:,}, rate: {rate:,.0f}/sec")
                    
                except Exception as e:
                    print(f"  Warning: Parse error in chunk {chunk_count}: {e}")
                
                data_lines = []
        
        # Final chunk
        if data_lines:
            chunk_text = ''.join(prefix_lines) + ''.join(data_lines)
            chunk_count += 1
            
            try:
                s_ids, p_ids, o_ids, count = parse_and_intern_oxigraph(
                    store, chunk_text, graph_id
                )
                
                if count > 0:
                    store._fact_store.add_facts_columnar(
                        g_col=[graph_id] * count,
                        s_col=s_ids,
                        p_col=p_ids,
                        o_col=o_ids,
                        flags=FactFlags.ASSERTED,
                        source=source_id,
                        confidence=1.0,
                    )
                    total_triples += count
                
            except Exception as e:
                print(f"  Warning: Final parse error: {e}")
        
        print(f"  Total: {total_lines:,} lines")
        
    finally:
        f.close()
    
    store._invalidate_cache()
    elapsed = time.time() - t0
    print(f"\nCompleted: {total_triples:,} triples in {elapsed:.1f}s ({total_triples/elapsed:,.0f} triples/sec)")
    
    return total_triples


def benchmark_parsers(file_path: str, limit_lines: int = 100_000) -> dict:
    """
    Benchmark different parsing approaches.
    
    Returns dict with timing results for comparison.
    """
    import gzip
    from pathlib import Path
    
    path = Path(file_path)
    is_gzipped = path.suffix.lower() == ".gz"
    
    # Read test data
    lines = []
    if is_gzipped:
        f = gzip.open(path, 'rt', encoding='utf-8')
    else:
        f = open(path, 'r', encoding='utf-8')
    
    for i, line in enumerate(f):
        lines.append(line)
        if i >= limit_lines:
            break
    f.close()
    chunk = ''.join(lines)
    
    results = {
        "file": file_path,
        "lines": len(lines),
        "chars": len(chunk),
    }
    
    # Native parser
    t0 = time.time()
    try:
        parsed = parse_chunk_native(chunk)
        t1 = time.time()
        results["native_triples"] = parsed.count
        results["native_time"] = t1 - t0
        results["native_rate"] = parsed.count / (t1 - t0) if t1 > t0 else 0
    except Exception as e:
        results["native_error"] = str(e)
    
    # Oxigraph parser
    if OXIGRAPH_AVAILABLE:
        t0 = time.time()
        try:
            parsed = parse_chunk_oxigraph(chunk)
            t1 = time.time()
            results["oxigraph_triples"] = parsed.count
            results["oxigraph_time"] = t1 - t0
            results["oxigraph_rate"] = parsed.count / (t1 - t0) if t1 > t0 else 0
        except Exception as e:
            results["oxigraph_error"] = str(e)
    
    # Print summary
    print(f"\n=== Parser Benchmark: {file_path} ===")
    print(f"Lines: {results['lines']:,}, Characters: {results['chars']:,}")
    
    if "native_rate" in results:
        print(f"Native parser:   {results['native_triples']:,} triples in {results['native_time']:.2f}s ({results['native_rate']:,.0f}/sec)")
    if "oxigraph_rate" in results:
        print(f"Oxigraph parser: {results['oxigraph_triples']:,} triples in {results['oxigraph_time']:.2f}s ({results['oxigraph_rate']:,.0f}/sec)")
        if "native_rate" in results:
            speedup = results["oxigraph_rate"] / results["native_rate"]
            print(f"Speedup: {speedup:.1f}x faster with Oxigraph")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bulk_loader.py <file.ttl|file.ttl.gz> [limit_lines] [--benchmark] [--no-oxigraph]")
        print(f"\nOxigraph available: {OXIGRAPH_AVAILABLE}")
        if not OXIGRAPH_AVAILABLE:
            print("Install with: pip install pyoxigraph")
        sys.exit(1)
    
    file_path = sys.argv[1]
    limit = None
    benchmark = False
    use_oxigraph = True
    
    for arg in sys.argv[2:]:
        if arg == "--benchmark":
            benchmark = True
        elif arg == "--no-oxigraph":
            use_oxigraph = False
        elif arg.isdigit():
            limit = int(arg)
    
    if benchmark:
        benchmark_parsers(file_path, limit or 100_000)
        sys.exit(0)
    
    store = TripleStore()
    
    if limit:
        # Limited test
        print(f"Testing with first {limit:,} lines...")
        import gzip
        path = Path(file_path)
        
        if path.suffix.lower() == ".gz":
            f = gzip.open(path, 'rt', encoding='utf-8')
        else:
            f = open(path, 'r', encoding='utf-8')
        
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if i >= limit:
                break
        f.close()
        
        # Write to temp file
        temp_path = "temp_test.ttl"
        with open(temp_path, 'w', encoding='utf-8') as tf:
            tf.writelines(lines)
        
        count = bulk_load_turtle(store, temp_path, use_oxigraph=use_oxigraph)
        Path(temp_path).unlink()
    else:
        count = bulk_load_turtle(store, file_path, use_oxigraph=use_oxigraph)
    
    print(f"\nStore statistics:")
    print(f"  Total triples: {len(store):,}")
    
    # Quick query test
    from src.rdf_starbase import execute_sparql
    result = execute_sparql(store, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    print(f"  SPARQL count: {result}")
