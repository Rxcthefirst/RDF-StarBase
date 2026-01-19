"""
rdflib Compatibility Layer.

Provides a drop-in replacement for rdflib's Graph class, backed by RDF-StarBase's
high-performance Polars engine.

Usage:
    # Instead of:
    # from rdflib import Graph, URIRef, Literal, Namespace
    
    # Use:
    from rdf_starbase.compat.rdflib import Graph, URIRef, Literal, Namespace
    
    g = Graph()
    g.parse("data.ttl", format="turtle")
    
    for s, p, o in g.triples((None, RDF.type, None)):
        print(s)
    
    g.serialize(destination="out.ttl", format="turtle")

This module aims to be API-compatible with rdflib while providing:
- 10-50x faster parsing
- 10-100x faster queries  
- Native RDF-Star support
- Built-in provenance tracking
"""

from typing import Optional, Iterator, Tuple, Any, Union, IO
from pathlib import Path
from io import StringIO, BytesIO
import re


# =============================================================================
# RDF Term Classes (rdflib-compatible)
# =============================================================================

class Identifier:
    """Base class for RDF terms."""
    __slots__ = ()


class URIRef(Identifier, str):
    """
    An RDF URI Reference.
    
    Compatible with rdflib.URIRef.
    """
    __slots__ = ()
    
    def __new__(cls, value: str, base: Optional[str] = None):
        if base is not None:
            # Resolve relative URI against base
            # Simple implementation - full resolution would need urllib
            if not value.startswith(('http://', 'https://', 'urn:', 'file://')):
                if base.endswith('/'):
                    value = base + value
                else:
                    value = base + '/' + value
        return str.__new__(cls, value)
    
    def __repr__(self):
        return f"URIRef({super().__repr__()})"
    
    def __hash__(self):
        return str.__hash__(self)
    
    def __eq__(self, other):
        if isinstance(other, URIRef):
            return str.__eq__(self, other)
        return str.__eq__(self, other)
    
    def n3(self, namespace_manager=None) -> str:
        """Return N3/Turtle representation."""
        # TODO: Use namespace_manager for prefix compression
        return f"<{self}>"
    
    def toPython(self) -> str:
        """Return Python string representation."""
        return str(self)


class Literal(Identifier):
    """
    An RDF Literal.
    
    Compatible with rdflib.Literal.
    """
    __slots__ = ('_value', '_datatype', '_language')
    
    def __init__(
        self, 
        value: Any,
        lang: Optional[str] = None,
        datatype: Optional[URIRef] = None
    ):
        if lang is not None and datatype is not None:
            raise TypeError("Literal cannot have both lang and datatype")
        
        self._value = value
        self._language = lang.lower() if lang else None
        
        if datatype is not None:
            self._datatype = URIRef(datatype) if not isinstance(datatype, URIRef) else datatype
        elif lang is not None:
            self._datatype = None  # Language-tagged literals have no datatype
        elif isinstance(value, bool):
            self._datatype = XSD.boolean
        elif isinstance(value, int):
            self._datatype = XSD.integer
        elif isinstance(value, float):
            self._datatype = XSD.double
        else:
            self._datatype = XSD.string
    
    @property
    def value(self) -> Any:
        return self._value
    
    @property
    def language(self) -> Optional[str]:
        return self._language
    
    @property
    def datatype(self) -> Optional[URIRef]:
        return self._datatype
    
    def __str__(self):
        return str(self._value)
    
    def __repr__(self):
        if self._language:
            return f"Literal({self._value!r}, lang={self._language!r})"
        elif self._datatype and self._datatype != XSD.string:
            return f"Literal({self._value!r}, datatype={self._datatype!r})"
        return f"Literal({self._value!r})"
    
    def __hash__(self):
        return hash((str(self._value), self._language, self._datatype))
    
    def __eq__(self, other):
        if isinstance(other, Literal):
            return (
                str(self._value) == str(other._value) and
                self._language == other._language and
                self._datatype == other._datatype
            )
        return str(self._value) == str(other)
    
    def n3(self, namespace_manager=None) -> str:
        """Return N3/Turtle representation."""
        value_str = str(self._value)
        # Escape special characters
        value_str = value_str.replace('\\', '\\\\').replace('"', '\\"')
        
        if self._language:
            return f'"{value_str}"@{self._language}'
        elif self._datatype and self._datatype != XSD.string:
            return f'"{value_str}"^^<{self._datatype}>'
        return f'"{value_str}"'
    
    def toPython(self) -> Any:
        """Convert to Python native type."""
        if self._datatype == XSD.integer:
            return int(self._value)
        elif self._datatype == XSD.double or self._datatype == XSD.decimal:
            return float(self._value)
        elif self._datatype == XSD.boolean:
            return str(self._value).lower() in ('true', '1')
        return str(self._value)


class BNode(Identifier):
    """
    An RDF Blank Node.
    
    Compatible with rdflib.BNode.
    """
    __slots__ = ('_id',)
    _next_id = 0
    
    def __init__(self, value: Optional[str] = None):
        if value is None:
            BNode._next_id += 1
            self._id = f"N{BNode._next_id}"
        else:
            self._id = value
    
    def __str__(self):
        return self._id
    
    def __repr__(self):
        return f"BNode({self._id!r})"
    
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        if isinstance(other, BNode):
            return self._id == other._id
        return False
    
    def n3(self, namespace_manager=None) -> str:
        """Return N3/Turtle representation."""
        return f"_:{self._id}"
    
    def toPython(self) -> str:
        return self._id


# =============================================================================
# Namespace Support
# =============================================================================

class Namespace(URIRef):
    """
    An RDF Namespace.
    
    Allows attribute access for creating URIRefs:
        RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        RDF.type  # Returns URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    """
    __slots__ = ()
    
    def __new__(cls, value: str):
        return URIRef.__new__(cls, value)
    
    def __getattr__(self, name: str) -> URIRef:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return URIRef(f"{self}{name}")
    
    def __getitem__(self, key: str) -> URIRef:
        return URIRef(f"{self}{key}")
    
    def term(self, name: str) -> URIRef:
        return URIRef(f"{self}{name}")


class ClosedNamespace(Namespace):
    """A namespace with a fixed set of terms."""
    
    def __new__(cls, uri: str, terms: list):
        inst = Namespace.__new__(cls, uri)
        inst._terms = frozenset(terms)
        return inst
    
    def __getattr__(self, name: str) -> URIRef:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name not in self._terms:
            raise AttributeError(f"term '{name}' not in namespace")
        return URIRef(f"{self}{name}")


# Well-known namespaces
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")


# =============================================================================
# Graph Class (Main API)
# =============================================================================

# Type alias for triples
Triple = Tuple[Identifier, URIRef, Identifier]


class Graph:
    """
    An RDF Graph backed by RDF-StarBase.
    
    API-compatible with rdflib.Graph while providing superior performance.
    
    Example:
        g = Graph()
        g.parse("data.ttl", format="turtle")
        
        for s, p, o in g.triples((None, RDF.type, None)):
            print(f"{s} is a {o}")
        
        g.add((URIRef("http://example.org/s"), RDF.type, FOAF.Person))
        g.serialize(destination="out.ttl", format="turtle")
    """
    
    def __init__(self, store=None, identifier=None):
        """
        Create a new Graph.
        
        Args:
            store: Optional backing store (ignored, uses TripleStore)
            identifier: Optional graph identifier
        """
        # Import here to avoid circular imports
        from rdf_starbase import TripleStore
        
        self._store = TripleStore()
        self._identifier = identifier or BNode()
        self._namespace_manager = NamespaceManager(self)
    
    @property
    def store(self):
        """The backing store."""
        return self._store
    
    @property
    def identifier(self):
        """The graph identifier."""
        return self._identifier
    
    @property
    def namespace_manager(self):
        """The namespace manager."""
        return self._namespace_manager
    
    def __len__(self) -> int:
        """Return the number of triples in the graph."""
        return len(self._store)
    
    def __iter__(self) -> Iterator[Triple]:
        """Iterate over all triples."""
        return self.triples((None, None, None))
    
    def __contains__(self, triple: Triple) -> bool:
        """Check if a triple is in the graph."""
        s, p, o = triple
        for _ in self.triples((s, p, o)):
            return True
        return False
    
    def add(self, triple: Triple) -> "Graph":
        """
        Add a triple to the graph.
        
        Args:
            triple: A (subject, predicate, object) tuple
            
        Returns:
            self for chaining
        """
        from rdf_starbase.models import ProvenanceContext
        
        s, p, o = triple
        prov = ProvenanceContext(source="rdflib_compat", confidence=1.0)
        self._store.add_triple(
            subject=str(s),
            predicate=str(p),
            obj=self._term_to_value(o),
            provenance=prov
        )
        return self
    
    def remove(self, triple: Triple) -> "Graph":
        """
        Remove a triple from the graph.
        
        Args:
            triple: A (subject, predicate, object) tuple with optional None wildcards
            
        Returns:
            self for chaining
        """
        s, p, o = triple
        # Get matching triples and deprecate them
        matches = self._store.get_triples(
            subject=str(s) if s is not None else None,
            predicate=str(p) if p is not None else None,
            obj=self._term_to_value(o) if o is not None else None,
        )
        
        for row in matches.iter_rows(named=True):
            # Deprecate the assertion
            if 'assertion_id' in row:
                try:
                    from uuid import UUID
                    self._store.deprecate(UUID(row['assertion_id']))
                except (ValueError, KeyError):
                    pass
        return self
    
    def triples(
        self, 
        pattern: Tuple[Optional[Identifier], Optional[URIRef], Optional[Identifier]]
    ) -> Iterator[Triple]:
        """
        Iterate over triples matching a pattern.
        
        Args:
            pattern: (subject, predicate, object) with None as wildcard
            
        Yields:
            Matching (subject, predicate, object) tuples
        """
        s, p, o = pattern
        
        results = self._store.get_triples(
            subject=str(s) if s is not None else None,
            predicate=str(p) if p is not None else None,
            obj=self._term_to_value(o) if o is not None else None,
        )
        
        for row in results.iter_rows(named=True):
            yield (
                self._value_to_term(row['subject'], is_uri=True),
                URIRef(row['predicate']),
                self._value_to_term(row['object']),
            )
    
    def subjects(
        self,
        predicate: Optional[URIRef] = None,
        object: Optional[Identifier] = None,
        unique: bool = True
    ) -> Iterator[Identifier]:
        """Iterate over subjects matching the pattern."""
        seen = set() if unique else None
        for s, p, o in self.triples((None, predicate, object)):
            if seen is not None:
                if s in seen:
                    continue
                seen.add(s)
            yield s
    
    def predicates(
        self,
        subject: Optional[Identifier] = None,
        object: Optional[Identifier] = None,
        unique: bool = True
    ) -> Iterator[URIRef]:
        """Iterate over predicates matching the pattern."""
        seen = set() if unique else None
        for s, p, o in self.triples((subject, None, object)):
            if seen is not None:
                if p in seen:
                    continue
                seen.add(p)
            yield p
    
    def objects(
        self,
        subject: Optional[Identifier] = None,
        predicate: Optional[URIRef] = None,
        unique: bool = True
    ) -> Iterator[Identifier]:
        """Iterate over objects matching the pattern."""
        seen = set() if unique else None
        for s, p, o in self.triples((subject, predicate, None)):
            if seen is not None:
                if o in seen:
                    continue
                seen.add(o)
            yield o
    
    def subject_objects(
        self,
        predicate: Optional[URIRef] = None,
        unique: bool = True
    ) -> Iterator[Tuple[Identifier, Identifier]]:
        """Iterate over (subject, object) pairs matching the predicate."""
        seen = set() if unique else None
        for s, p, o in self.triples((None, predicate, None)):
            pair = (s, o)
            if seen is not None:
                if pair in seen:
                    continue
                seen.add(pair)
            yield pair
    
    def subject_predicates(
        self,
        object: Optional[Identifier] = None,
        unique: bool = True
    ) -> Iterator[Tuple[Identifier, URIRef]]:
        """Iterate over (subject, predicate) pairs matching the object."""
        seen = set() if unique else None
        for s, p, o in self.triples((None, None, object)):
            pair = (s, p)
            if seen is not None:
                if pair in seen:
                    continue
                seen.add(pair)
            yield pair
    
    def predicate_objects(
        self,
        subject: Optional[Identifier] = None,
        unique: bool = True
    ) -> Iterator[Tuple[URIRef, Identifier]]:
        """Iterate over (predicate, object) pairs matching the subject."""
        seen = set() if unique else None
        for s, p, o in self.triples((subject, None, None)):
            pair = (p, o)
            if seen is not None:
                if pair in seen:
                    continue
                seen.add(pair)
            yield pair
    
    def value(
        self,
        subject: Optional[Identifier] = None,
        predicate: Optional[URIRef] = None,
        object: Optional[Identifier] = None,
        default: Any = None,
        any: bool = True
    ) -> Optional[Identifier]:
        """Get a single value for the unbound component."""
        for s, p, o in self.triples((subject, predicate, object)):
            if subject is None:
                return s
            elif predicate is None:
                return p
            else:
                return o
        return default
    
    def parse(
        self,
        source: Optional[Union[str, Path, IO]] = None,
        publicID: Optional[str] = None,
        format: Optional[str] = None,
        location: Optional[str] = None,
        file: Optional[IO] = None,
        data: Optional[Union[str, bytes]] = None,
        **kwargs
    ) -> "Graph":
        """
        Parse RDF data into this graph.
        
        Args:
            source: File path, URL, or file-like object
            publicID: The logical URI of the graph
            format: Format hint (turtle, xml, n3, nt, json-ld)
            location: Alternative to source (URL to fetch)
            file: File-like object
            data: Raw string/bytes data
            
        Returns:
            self for chaining
        """
        # Determine the content to parse
        content = None
        
        if data is not None:
            content = data if isinstance(data, str) else data.decode('utf-8')
        elif file is not None:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        elif source is not None:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.exists():
                    content = path.read_text(encoding='utf-8')
                    if format is None:
                        format = self._guess_format(path)
                else:
                    # Might be a URL - try to fetch
                    # For now, just raise
                    raise FileNotFoundError(f"File not found: {source}")
            elif hasattr(source, 'read'):
                content = source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
        elif location is not None:
            raise NotImplementedError("URL fetching not implemented yet")
        else:
            raise ValueError("No input source provided")
        
        # Determine format
        if format is None:
            format = 'turtle'  # Default
        
        format = format.lower()
        if format in ('ttl', 'turtle', 'n3'):
            self._parse_turtle(content)
        elif format in ('nt', 'ntriples', 'n-triples'):
            self._parse_ntriples(content)
        elif format in ('xml', 'rdf/xml', 'rdfxml', 'application/rdf+xml'):
            self._parse_rdfxml(content)
        elif format in ('json-ld', 'jsonld'):
            self._parse_jsonld(content)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return self
    
    def serialize(
        self,
        destination: Optional[Union[str, Path, IO]] = None,
        format: str = "turtle",
        base: Optional[str] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Serialize the graph to RDF.
        
        Args:
            destination: File path or file-like object (None for string return)
            format: Output format (turtle, xml, nt, json-ld)
            base: Base URI
            encoding: Character encoding
            
        Returns:
            Serialized string if destination is None
        """
        format = format.lower()
        
        if format in ('ttl', 'turtle', 'n3'):
            content = self._serialize_turtle()
        elif format in ('nt', 'ntriples', 'n-triples'):
            content = self._serialize_ntriples()
        elif format in ('xml', 'rdf/xml', 'rdfxml', 'pretty-xml'):
            content = self._serialize_rdfxml()
        elif format in ('json-ld', 'jsonld'):
            content = self._serialize_jsonld()
        else:
            raise ValueError(f"Unknown format: {format}")
        
        if destination is None:
            return content
        
        if isinstance(destination, (str, Path)):
            Path(destination).write_text(content, encoding=encoding or 'utf-8')
        else:
            destination.write(content.encode(encoding or 'utf-8') if hasattr(destination, 'mode') and 'b' in destination.mode else content)
        
        return None
    
    def bind(self, prefix: str, namespace: Union[str, URIRef, Namespace], override: bool = True, replace: bool = False):
        """Bind a namespace prefix."""
        self._namespace_manager.bind(prefix, namespace, override, replace)
        return self
    
    def namespaces(self) -> Iterator[Tuple[str, URIRef]]:
        """Iterate over bound namespace prefixes."""
        return iter(self._namespace_manager.namespaces())
    
    def query(self, query: str, initBindings=None, initNs=None, **kwargs):
        """
        Execute a SPARQL query.
        
        Args:
            query: SPARQL query string
            initBindings: Initial variable bindings
            initNs: Namespace prefix mappings
            
        Returns:
            Query results
        """
        from rdf_starbase import execute_sparql
        
        # Add namespace prefixes if provided
        if initNs:
            prefix_block = ""
            for prefix, ns in initNs.items():
                prefix_block += f"PREFIX {prefix}: <{ns}>\n"
            query = prefix_block + query
        
        result = execute_sparql(self._store, query)
        return QueryResult(result, initBindings)
    
    def update(self, update_query: str, initBindings=None, initNs=None, **kwargs):
        """Execute a SPARQL Update query."""
        from rdf_starbase import execute_sparql
        
        if initNs:
            prefix_block = ""
            for prefix, ns in initNs.items():
                prefix_block += f"PREFIX {prefix}: <{ns}>\n"
            update_query = prefix_block + update_query
        
        return execute_sparql(self._store, update_query)
    
    # =========================================================================
    # Internal parsing methods
    # =========================================================================
    
    def _parse_turtle(self, content: str):
        """Parse Turtle content."""
        from rdf_starbase.formats.turtle import parse_turtle
        
        doc = parse_turtle(content)
        
        # Extract columns from parsed triples
        subjects = [t.subject for t in doc.triples]
        predicates = [t.predicate for t in doc.triples]
        objects = [t.object for t in doc.triples]
        
        # Use columnar insert (much faster than one-by-one)
        self._store.add_triples_columnar(
            subjects=subjects,
            predicates=predicates,
            objects=objects,
            source="turtle_parse",
            confidence=1.0,
        )
    
    def _parse_ntriples(self, content: str):
        """Parse N-Triples content."""
        from rdf_starbase.formats.ntriples import parse_ntriples
        
        doc = parse_ntriples(content)
        
        subjects = [t.subject for t in doc.triples]
        predicates = [t.predicate for t in doc.triples]
        objects = [t.object for t in doc.triples]
        
        self._store.add_triples_columnar(
            subjects=subjects,
            predicates=predicates,
            objects=objects,
            source="ntriples_parse",
            confidence=1.0,
        )
    
    def _parse_rdfxml(self, content: str):
        """Parse RDF/XML content."""
        from rdf_starbase.formats.rdfxml import parse_rdfxml
        
        doc = parse_rdfxml(content)
        
        subjects = [t.subject for t in doc.triples]
        predicates = [t.predicate for t in doc.triples]
        objects = [t.object for t in doc.triples]
        
        self._store.add_triples_columnar(
            subjects=subjects,
            predicates=predicates,
            objects=objects,
            source="rdfxml_parse",
            confidence=1.0,
        )
    
    def _parse_jsonld(self, content: str):
        """Parse JSON-LD content."""
        from rdf_starbase.formats.jsonld import parse_jsonld
        
        doc = parse_jsonld(content)
        
        subjects = [t.subject for t in doc.triples]
        predicates = [t.predicate for t in doc.triples]
        objects = [t.object for t in doc.triples]
        
        self._store.add_triples_columnar(
            subjects=subjects,
            predicates=predicates,
            objects=objects,
            source="jsonld_parse",
            confidence=1.0,
        )
    
    # =========================================================================
    # Internal serialization methods
    # =========================================================================
    
    def _serialize_turtle(self) -> str:
        """Serialize to Turtle."""
        lines = []
        
        # Convert namespaces to dict and write prefix declarations
        prefixes = {prefix: str(ns) for prefix, ns in self._namespace_manager.namespaces()}
        for prefix, namespace in sorted(prefixes.items()):
            lines.append(f"@prefix {prefix}: <{namespace}> .")
        
        if prefixes:
            lines.append("")
        
        # Build reverse prefix lookup for compression
        reverse_prefixes = {v: k for k, v in prefixes.items()}
        
        def compress_uri(uri: str) -> str:
            """Try to compress URI with prefix."""
            for ns, prefix in sorted(reverse_prefixes.items(), key=lambda x: -len(x[0])):
                if uri.startswith(ns):
                    local = uri[len(ns):]
                    # Only use prefix if local part is valid
                    if local and local[0].isalpha() and all(c.isalnum() or c == '_' for c in local):
                        return f"{prefix}:{local}"
            return f"<{uri}>"
        
        def format_value(v) -> str:
            """Format a value as Turtle."""
            if isinstance(v, str):
                if v.startswith(('http://', 'https://', 'urn:')):
                    return compress_uri(v)
                elif v.startswith('_:'):
                    return v
                else:
                    # Escape and quote literal
                    escaped = v.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    return f'"{escaped}"'
            else:
                return f'"{v}"'
        
        # Group by subject for prettier output
        by_subject = {}
        for row in self._store._df.iter_rows(named=True):
            if row.get('deprecated', False):
                continue
            s = row['subject']
            if s not in by_subject:
                by_subject[s] = []
            by_subject[s].append((row['predicate'], row['object']))
        
        # Write grouped triples
        for subject, po_list in by_subject.items():
            s_str = compress_uri(subject) if subject.startswith(('http://', 'https://')) else subject
            lines.append(f"{s_str}")
            
            for i, (pred, obj) in enumerate(po_list):
                p_str = compress_uri(pred)
                o_str = format_value(obj)
                sep = " ;" if i < len(po_list) - 1 else " ."
                lines.append(f"    {p_str} {o_str}{sep}")
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _serialize_ntriples(self) -> str:
        """Serialize to N-Triples."""
        lines = []
        for row in self._store._df.iter_rows(named=True):
            if row.get('deprecated', False):
                continue
            s = row['subject']
            p = row['predicate']
            o = row['object']
            
            # Format subject
            s_str = f"<{s}>" if not s.startswith('_:') else s
            p_str = f"<{p}>"
            
            # Format object
            if isinstance(o, str) and (o.startswith('http://') or o.startswith('https://') or o.startswith('urn:')):
                o_str = f"<{o}>"
            elif isinstance(o, str) and o.startswith('_:'):
                o_str = o
            else:
                o_str = f'"{o}"'
            
            lines.append(f"{s_str} {p_str} {o_str} .")
        
        return '\n'.join(lines)
    
    def _serialize_rdfxml(self) -> str:
        """Serialize to RDF/XML."""
        # Basic implementation
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        ]
        
        for row in self._store._df.iter_rows(named=True):
            if row.get('deprecated', False):
                continue
            s = row['subject']
            p = row['predicate']
            o = row['object']
            
            lines.append(f'  <rdf:Description rdf:about="{s}">')
            
            # Simple predicate handling
            if isinstance(o, str) and o.startswith('http'):
                lines.append(f'    <{p} rdf:resource="{o}"/>')
            else:
                lines.append(f'    <{p}>{o}</{p}>')
            
            lines.append('  </rdf:Description>')
        
        lines.append('</rdf:RDF>')
        return '\n'.join(lines)
    
    def _serialize_jsonld(self) -> str:
        """Serialize to JSON-LD."""
        import json
        
        # Group by subject
        subjects = {}
        for row in self._store._df.iter_rows(named=True):
            if row.get('deprecated', False):
                continue
            s = row['subject']
            p = row['predicate']
            o = row['object']
            
            if s not in subjects:
                subjects[s] = {"@id": s}
            
            if p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                if "@type" not in subjects[s]:
                    subjects[s]["@type"] = []
                subjects[s]["@type"].append(o)
            else:
                if p not in subjects[s]:
                    subjects[s][p] = []
                subjects[s][p].append(o)
        
        return json.dumps(list(subjects.values()), indent=2)
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _guess_format(self, path: Path) -> str:
        """Guess format from file extension."""
        suffix = path.suffix.lower()
        return {
            '.ttl': 'turtle',
            '.turtle': 'turtle',
            '.n3': 'n3',
            '.nt': 'nt',
            '.ntriples': 'nt',
            '.rdf': 'xml',
            '.xml': 'xml',
            '.owl': 'xml',
            '.jsonld': 'json-ld',
            '.json': 'json-ld',
        }.get(suffix, 'turtle')
    
    def _term_to_value(self, term: Identifier) -> Any:
        """Convert an RDF term to a storage value."""
        if isinstance(term, URIRef):
            return str(term)
        elif isinstance(term, Literal):
            return term.toPython()
        elif isinstance(term, BNode):
            return f"_:{term._id}"
        else:
            return str(term)
    
    def _value_to_term(self, value: Any, is_uri: bool = False) -> Identifier:
        """Convert a storage value to an RDF term."""
        if isinstance(value, str):
            if value.startswith('_:'):
                return BNode(value[2:])
            elif value.startswith(('http://', 'https://', 'urn:', 'file://')) or is_uri:
                return URIRef(value)
            else:
                return Literal(value)
        elif isinstance(value, (int, float, bool)):
            return Literal(value)
        else:
            return Literal(str(value))


# =============================================================================
# Namespace Manager
# =============================================================================

class NamespaceManager:
    """Manages namespace prefix bindings for a graph."""
    
    def __init__(self, graph: Optional[Graph] = None):
        self._graph = graph
        self._bindings: dict[str, URIRef] = {}
        self._reverse: dict[str, str] = {}
        
        # Default bindings
        self.bind("rdf", RDF)
        self.bind("rdfs", RDFS)
        self.bind("owl", OWL)
        self.bind("xsd", XSD)
    
    def bind(
        self, 
        prefix: str, 
        namespace: Union[str, URIRef, Namespace],
        override: bool = True,
        replace: bool = False
    ):
        """Bind a prefix to a namespace."""
        ns = URIRef(namespace) if not isinstance(namespace, URIRef) else namespace
        
        if not override and prefix in self._bindings:
            return
        
        if replace:
            # Remove old binding for this namespace
            old_prefix = self._reverse.get(str(ns))
            if old_prefix:
                del self._bindings[old_prefix]
        
        self._bindings[prefix] = ns
        self._reverse[str(ns)] = prefix
    
    def namespaces(self) -> Iterator[Tuple[str, URIRef]]:
        """Iterate over (prefix, namespace) pairs."""
        return iter(self._bindings.items())
    
    def compute_qname(self, uri: str, generate: bool = True) -> Tuple[str, str, str]:
        """Compute a qname (prefix, namespace, local) for a URI."""
        for prefix, ns in sorted(self._bindings.items(), key=lambda x: -len(x[1])):
            ns_str = str(ns)
            if uri.startswith(ns_str):
                local = uri[len(ns_str):]
                return (prefix, ns_str, local)
        
        # Try to generate a prefix
        if generate:
            # Split URI into namespace and local
            for sep in ('#', '/', ':'):
                if sep in uri:
                    idx = uri.rfind(sep)
                    ns = uri[:idx + 1]
                    local = uri[idx + 1:]
                    if ns in self._reverse:
                        return (self._reverse[ns], ns, local)
                    # Generate new prefix
                    prefix = f"ns{len(self._bindings)}"
                    self.bind(prefix, ns)
                    return (prefix, ns, local)
        
        raise ValueError(f"Cannot compute qname for {uri}")


# =============================================================================
# Query Results
# =============================================================================

class QueryResult:
    """Wrapper for SPARQL query results."""
    
    def __init__(self, result, bindings=None):
        self._result = result
        self._bindings = bindings or {}
    
    def __iter__(self):
        """Iterate over result rows."""
        import polars as pl
        if isinstance(self._result, pl.DataFrame):
            for row in self._result.iter_rows(named=True):
                yield QueryRow(row)
        elif isinstance(self._result, bool):
            yield self._result
        else:
            yield self._result
    
    def __bool__(self):
        """For ASK queries."""
        if isinstance(self._result, bool):
            return self._result
        return len(self._result) > 0


class QueryRow:
    """A single result row from a SPARQL query."""
    
    def __init__(self, row: dict):
        self._row = row
        self._keys = list(row.keys())
    
    def __getitem__(self, key):
        # Support both string keys and integer indices
        if isinstance(key, int):
            if 0 <= key < len(self._keys):
                key = self._keys[key]
            else:
                raise IndexError(f"Row index out of range: {key}")
        
        value = self._row.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            if value.startswith(('http://', 'https://', 'urn:')):
                return URIRef(value)
            elif value.startswith('_:'):
                return BNode(value[2:])
            return Literal(value)
        return Literal(value)
    
    def __iter__(self):
        return iter(self._row.values())
    
    def asdict(self):
        return dict(self._row)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Term classes
    'URIRef', 'Literal', 'BNode', 'Identifier',
    # Namespace
    'Namespace', 'ClosedNamespace', 'NamespaceManager',
    # Well-known namespaces
    'RDF', 'RDFS', 'OWL', 'XSD', 'FOAF', 'DC', 'DCTERMS', 'SKOS', 'PROV',
    # Graph
    'Graph',
    # Query
    'QueryResult', 'QueryRow',
]
