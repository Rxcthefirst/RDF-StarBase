"""
Turtle and Turtle-Star Parser.

Implements parsing of Turtle (Terse RDF Triple Language) format
with RDF-Star extensions for quoted triples.

Grammar based on W3C Turtle specification:
https://www.w3.org/TR/turtle/

With Turtle-Star extensions:
https://w3c.github.io/rdf-star/cg-spec/editors_draft.html
"""

from typing import Iterator, Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
from io import StringIO


@dataclass
class Triple:
    """A parsed RDF triple."""
    subject: str
    predicate: str
    object: str
    # For RDF-Star: if subject or object is a quoted triple
    subject_triple: Optional["Triple"] = None
    object_triple: Optional["Triple"] = None
    
    def __str__(self) -> str:
        s = f"<<{self.subject_triple}>>" if self.subject_triple else self.subject
        o = f"<<{self.object_triple}>>" if self.object_triple else self.object
        return f"{s} {self.predicate} {o}"


@dataclass
class ParsedDocument:
    """Result of parsing a Turtle document."""
    prefixes: Dict[str, str] = field(default_factory=dict)
    base: Optional[str] = None
    triples: List[Triple] = field(default_factory=list)
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast ingestion.
        
        Handles RDF-Star quoted triples by serializing them as << s p o >> strings.
        """
        subjects = []
        predicates = []
        objects = []
        
        for t in self.triples:
            # Handle subject - may be a quoted triple
            if t.subject_triple:
                qt = t.subject_triple
                s = f"<< {qt.subject} {qt.predicate} {qt.object} >>"
            else:
                s = t.subject
            
            # Handle object - may be a quoted triple
            if t.object_triple:
                qt = t.object_triple
                o = f"<< {qt.subject} {qt.predicate} {qt.object} >>"
            else:
                o = t.object
            
            subjects.append(s)
            predicates.append(t.predicate)
            objects.append(o)
        
        return (subjects, predicates, objects)


class TurtleParser:
    """
    Parser for Turtle and Turtle-Star format.
    
    Supports:
    - @prefix and @base directives
    - PREFIX and BASE (SPARQL-style)
    - Prefixed names (foaf:name)
    - Full IRIs (<http://...>)
    - Literals with language tags ("hello"@en)
    - Literals with datatypes ("42"^^xsd:integer)
    - Blank nodes (_:b1, [ ])
    - Collections (a b c)
    - RDF-Star quoted triples (<< s p o >>)
    - Predicate-object lists (; separation)
    - Object lists (, separation)
    """
    
    # Standard prefixes that are commonly used
    STANDARD_PREFIXES = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "prov": "http://www.w3.org/ns/prov#",
    }
    
    # Token patterns
    IRI_PATTERN = re.compile(r'<([^>]*)>')
    PREFIXED_NAME_PATTERN = re.compile(r'([a-zA-Z_][\w-]*)?:([a-zA-Z_][\w.-]*)?')
    BLANK_NODE_PATTERN = re.compile(r'_:([a-zA-Z_][\w.-]*)')
    STRING_PATTERN = re.compile(r'"""([^"]*(?:""?(?!"))?)*"""|\'\'\'([^\']*(?:\'\'?(?!\'))?)*\'\'\'|"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'')
    INTEGER_PATTERN = re.compile(r'[+-]?\d+')
    DECIMAL_PATTERN = re.compile(r'[+-]?\d*\.\d+')
    DOUBLE_PATTERN = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+')
    BOOLEAN_PATTERN = re.compile(r'true|false', re.IGNORECASE)
    COMMENT_PATTERN = re.compile(r'#[^\n]*')
    
    def __init__(self):
        self.prefixes: Dict[str, str] = {}
        self.base: Optional[str] = None
        self.blank_node_counter = 0
        self.text = ""
        self.pos = 0
        self.triples: List[Triple] = []
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse a Turtle document.
        
        Args:
            source: Turtle content as string, file path, or StringIO
            
        Returns:
            ParsedDocument with prefixes, base, and triples
        """
        if isinstance(source, Path):
            self.text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            self.text = source.read()
        else:
            self.text = source
        
        self.pos = 0
        self.prefixes = {}
        self.base = None
        self.triples = []
        self.blank_node_counter = 0
        
        self._parse_document()
        
        return ParsedDocument(
            prefixes=self.prefixes.copy(),
            base=self.base,
            triples=self.triples.copy()
        )
    
    def parse_file(self, path: Union[str, Path]) -> ParsedDocument:
        """Parse a Turtle file."""
        return self.parse(Path(path))
    
    def _parse_document(self):
        """Parse the entire document."""
        while self.pos < len(self.text):
            self._skip_ws_and_comments()
            if self.pos >= len(self.text):
                break
            
            # Check for directives
            if self._peek_text("@prefix"):
                self._parse_prefix_directive()
            elif self._peek_text("@base"):
                self._parse_base_directive()
            elif self._peek_text("PREFIX", case_insensitive=True):
                self._parse_sparql_prefix()
            elif self._peek_text("BASE", case_insensitive=True):
                self._parse_sparql_base()
            else:
                # Parse statement (triples)
                self._parse_statement()
    
    def _skip_ws_and_comments(self):
        """Skip whitespace and comments."""
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c in ' \t\n\r':
                self.pos += 1
            elif c == '#':
                # Skip to end of line
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self.pos += 1
            else:
                break
    
    def _peek_text(self, text: str, case_insensitive: bool = False) -> bool:
        """Check if text appears at current position."""
        end = self.pos + len(text)
        if end > len(self.text):
            return False
        actual = self.text[self.pos:end]
        if case_insensitive:
            return actual.lower() == text.lower()
        return actual == text
    
    def _consume(self, text: str, case_insensitive: bool = False):
        """Consume expected text or raise error."""
        if not self._peek_text(text, case_insensitive):
            context = self.text[max(0, self.pos-20):self.pos+20]
            raise ValueError(f"Expected '{text}' at position {self.pos}, context: ...{context}...")
        self.pos += len(text)
    
    def _parse_prefix_directive(self):
        """Parse @prefix directive."""
        self._consume("@prefix")
        self._skip_ws_and_comments()
        
        # Parse prefix name
        prefix = self._parse_prefix_name()
        self._skip_ws_and_comments()
        
        # Parse IRI
        iri = self._parse_iri_ref()
        self._skip_ws_and_comments()
        
        # Consume period
        self._consume(".")
        
        self.prefixes[prefix] = iri
    
    def _parse_sparql_prefix(self):
        """Parse PREFIX directive (SPARQL-style)."""
        self._consume("PREFIX", case_insensitive=True)
        self._skip_ws_and_comments()
        
        prefix = self._parse_prefix_name()
        self._skip_ws_and_comments()
        
        iri = self._parse_iri_ref()
        
        self.prefixes[prefix] = iri
    
    def _parse_base_directive(self):
        """Parse @base directive."""
        self._consume("@base")
        self._skip_ws_and_comments()
        
        self.base = self._parse_iri_ref()
        self._skip_ws_and_comments()
        
        self._consume(".")
    
    def _parse_sparql_base(self):
        """Parse BASE directive (SPARQL-style)."""
        self._consume("BASE", case_insensitive=True)
        self._skip_ws_and_comments()
        
        self.base = self._parse_iri_ref()
    
    def _parse_prefix_name(self) -> str:
        """Parse a prefix name (e.g., 'foaf:')."""
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ': \t\n\r':
            self.pos += 1
        prefix = self.text[start:self.pos]
        self._consume(":")
        return prefix
    
    def _parse_iri_ref(self) -> str:
        """Parse an IRI reference (<...>)."""
        self._consume("<")
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != '>':
            self.pos += 1
        iri = self.text[start:self.pos]
        self._consume(">")
        
        # Resolve relative IRI against base
        if self.base and not iri.startswith(('http://', 'https://', 'urn:', 'file:')):
            iri = self.base + iri
        
        return iri
    
    def _parse_statement(self):
        """Parse a triple statement."""
        subject = self._parse_subject()
        if subject is None:
            return
        
        self._skip_ws_and_comments()
        
        self._parse_predicate_object_list(subject)
        
        self._skip_ws_and_comments()
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            self.pos += 1
    
    def _parse_subject(self) -> Optional[Union[str, Triple]]:
        """Parse a subject (IRI, blank node, or quoted triple)."""
        self._skip_ws_and_comments()
        if self.pos >= len(self.text):
            return None
        
        # Check for quoted triple (RDF-Star)
        if self._peek_text("<<"):
            return self._parse_quoted_triple()
        
        # Check for blank node
        if self._peek_text("["):
            return self._parse_blank_node_property_list()
        
        if self._peek_text("_:"):
            return self._parse_blank_node_label()
        
        # Check for collection
        if self._peek_text("("):
            return self._parse_collection()
        
        # Otherwise, parse IRI or prefixed name
        return self._parse_iri_or_prefixed()
    
    def _parse_predicate_object_list(self, subject: Union[str, Triple]):
        """Parse predicate-object list (supports ; and ,)."""
        while True:
            self._skip_ws_and_comments()
            if self.pos >= len(self.text):
                break
            
            # Parse predicate
            predicate = self._parse_predicate()
            if predicate is None:
                break
            
            self._skip_ws_and_comments()
            
            # Parse object list (comma-separated)
            self._parse_object_list(subject, predicate)
            
            self._skip_ws_and_comments()
            
            # Check for more predicates (;)
            if self.pos < len(self.text) and self.text[self.pos] == ';':
                self.pos += 1
                self._skip_ws_and_comments()
                # Check for trailing semicolon before period
                if self.pos < len(self.text) and self.text[self.pos] in '.]:':
                    break
                continue
            else:
                break
    
    def _parse_predicate(self) -> Optional[str]:
        """Parse a predicate."""
        self._skip_ws_and_comments()
        if self.pos >= len(self.text):
            return None
        
        # Check for 'a' (shorthand for rdf:type)
        if self.text[self.pos] == 'a' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] in ' \t\n\r':
            self.pos += 1
            return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        
        # Statement terminator or separator - no predicate here
        if self.text[self.pos] in '.;,]':
            return None
        
        return self._parse_iri_or_prefixed()
    
    def _parse_object_list(self, subject: Union[str, Triple], predicate: str):
        """Parse object list (comma-separated)."""
        while True:
            self._skip_ws_and_comments()
            
            obj = self._parse_object()
            if obj is None:
                break
            
            # Create triple
            if isinstance(subject, Triple):
                triple = Triple(
                    subject="",
                    predicate=predicate,
                    object=obj if isinstance(obj, str) else "",
                    subject_triple=subject,
                    object_triple=obj if isinstance(obj, Triple) else None
                )
            elif isinstance(obj, Triple):
                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object="",
                    object_triple=obj
                )
            else:
                triple = Triple(subject=subject, predicate=predicate, object=obj)
            
            self.triples.append(triple)
            
            self._skip_ws_and_comments()
            
            # Check for more objects (,)
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
                continue
            else:
                break
    
    def _parse_object(self) -> Optional[Union[str, Triple]]:
        """Parse an object (IRI, blank node, literal, or quoted triple)."""
        self._skip_ws_and_comments()
        if self.pos >= len(self.text):
            return None
        
        c = self.text[self.pos]
        
        # Quoted triple (RDF-Star)
        if self._peek_text("<<"):
            return self._parse_quoted_triple()
        
        # Blank node property list
        if c == '[':
            return self._parse_blank_node_property_list()
        
        # Blank node label
        if self._peek_text("_:"):
            return self._parse_blank_node_label()
        
        # Collection
        if c == '(':
            return self._parse_collection()
        
        # Literal
        if c in '"\'':
            return self._parse_literal()
        
        # Numeric literals
        if c in '+-' or c.isdigit():
            return self._parse_numeric()
        
        # Boolean
        if self._peek_text("true") or self._peek_text("false"):
            return self._parse_boolean()
        
        # Otherwise IRI or prefixed name
        if c not in '.;,]':
            return self._parse_iri_or_prefixed()
        
        return None
    
    def _parse_iri_or_prefixed(self) -> str:
        """Parse an IRI (<...>) or prefixed name (prefix:local)."""
        if self.text[self.pos] == '<':
            return self._parse_iri_ref()
        else:
            return self._parse_prefixed_name()
    
    def _parse_prefixed_name(self) -> str:
        """Parse a prefixed name (e.g., foaf:name)."""
        start = self.pos
        
        # Parse prefix part
        prefix = ""
        while self.pos < len(self.text) and self.text[self.pos] not in ': \t\n\r.;,[]()':
            self.pos += 1
        prefix = self.text[start:self.pos]
        
        if self.pos >= len(self.text) or self.text[self.pos] != ':':
            # Handle 'a' as special case for rdf:type
            if prefix == 'a':
                return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            raise ValueError(f"Expected ':' in prefixed name at position {self.pos}")
        
        self.pos += 1  # Skip ':'
        
        # Parse local part
        local_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n\r.;,[]()<>"\'':
            self.pos += 1
        local = self.text[local_start:self.pos]
        
        # Expand prefix
        if prefix in self.prefixes:
            return self.prefixes[prefix] + local
        elif prefix in self.STANDARD_PREFIXES:
            return self.STANDARD_PREFIXES[prefix] + local
        elif prefix == "":
            # Empty prefix, use base or default
            if self.base:
                return self.base + local
            return local
        else:
            # Unknown prefix - keep as-is for now
            return f"{prefix}:{local}"
    
    def _parse_blank_node_label(self) -> str:
        """Parse a blank node label (_:name)."""
        self._consume("_:")
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n\r.;,[]()':
            self.pos += 1
        label = self.text[start:self.pos]
        return f"_:{label}"
    
    def _parse_blank_node_property_list(self) -> str:
        """Parse a blank node property list [ ... ]."""
        self._consume("[")
        
        # Generate unique blank node ID
        self.blank_node_counter += 1
        bnode = f"_:b{self.blank_node_counter}"
        
        self._skip_ws_and_comments()
        
        # Check for empty blank node
        if self.pos < len(self.text) and self.text[self.pos] == ']':
            self.pos += 1
            return bnode
        
        # Parse property list
        self._parse_predicate_object_list(bnode)
        
        self._skip_ws_and_comments()
        self._consume("]")
        
        return bnode
    
    def _parse_collection(self) -> str:
        """Parse a collection ( ... )."""
        self._consume("(")
        self._skip_ws_and_comments()
        
        items = []
        while self.pos < len(self.text) and self.text[self.pos] != ')':
            item = self._parse_object()
            if item is not None:
                items.append(item)
            self._skip_ws_and_comments()
        
        self._consume(")")
        
        if not items:
            return "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        # Build collection as linked list
        RDF_FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        RDF_REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        RDF_NIL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        
        head = None
        prev = None
        
        for item in items:
            self.blank_node_counter += 1
            node = f"_:list{self.blank_node_counter}"
            
            if head is None:
                head = node
            
            if prev is not None:
                self.triples.append(Triple(prev, RDF_REST, node))
            
            if isinstance(item, Triple):
                self.triples.append(Triple(node, RDF_FIRST, "", object_triple=item))
            else:
                self.triples.append(Triple(node, RDF_FIRST, item))
            
            prev = node
        
        if prev is not None:
            self.triples.append(Triple(prev, RDF_REST, RDF_NIL))
        
        return head or RDF_NIL
    
    def _parse_literal(self) -> str:
        """Parse a literal string."""
        # Check for long string (triple quotes)
        if self._peek_text('"""'):
            return self._parse_long_string('"""')
        elif self._peek_text("'''"):
            return self._parse_long_string("'''")
        
        # Regular string
        quote = self.text[self.pos]
        self.pos += 1
        
        value = []
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c == quote:
                self.pos += 1
                break
            elif c == '\\':
                self.pos += 1
                if self.pos < len(self.text):
                    escaped = self.text[self.pos]
                    if escaped == 'n':
                        value.append('\n')
                    elif escaped == 't':
                        value.append('\t')
                    elif escaped == 'r':
                        value.append('\r')
                    elif escaped == '\\':
                        value.append('\\')
                    elif escaped == quote:
                        value.append(quote)
                    elif escaped == 'u':
                        # Unicode escape \uXXXX
                        self.pos += 1
                        hex_chars = self.text[self.pos:self.pos+4]
                        value.append(chr(int(hex_chars, 16)))
                        self.pos += 3
                    else:
                        value.append(escaped)
                    self.pos += 1
            else:
                value.append(c)
                self.pos += 1
        
        string_value = ''.join(value)
        
        # Check for language tag or datatype
        if self.pos < len(self.text) and self.text[self.pos] == '@':
            self.pos += 1
            lang_start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n\r.;,[]':
                self.pos += 1
            lang = self.text[lang_start:self.pos]
            return f'"{string_value}"@{lang}'
        elif self._peek_text("^^"):
            self.pos += 2
            datatype = self._parse_iri_or_prefixed()
            return f'"{string_value}"^^<{datatype}>'
        
        return f'"{string_value}"'
    
    def _parse_long_string(self, delimiter: str) -> str:
        """Parse a long string (triple-quoted)."""
        self._consume(delimiter)
        
        value = []
        while self.pos < len(self.text):
            if self._peek_text(delimiter):
                self._consume(delimiter)
                break
            value.append(self.text[self.pos])
            self.pos += 1
        
        string_value = ''.join(value)
        
        # Check for language tag or datatype
        if self.pos < len(self.text) and self.text[self.pos] == '@':
            self.pos += 1
            lang_start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] not in ' \t\n\r.;,[]':
                self.pos += 1
            lang = self.text[lang_start:self.pos]
            return f'"{string_value}"@{lang}'
        elif self._peek_text("^^"):
            self.pos += 2
            datatype = self._parse_iri_or_prefixed()
            return f'"{string_value}"^^<{datatype}>'
        
        return f'"{string_value}"'
    
    def _parse_numeric(self) -> str:
        """Parse a numeric literal."""
        start = self.pos
        
        # Handle sign
        if self.text[self.pos] in '+-':
            self.pos += 1
        
        # Parse digits
        has_decimal = False
        has_exponent = False
        
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isdigit():
                self.pos += 1
            elif c == '.' and not has_decimal:
                has_decimal = True
                self.pos += 1
            elif c in 'eE' and not has_exponent:
                has_exponent = True
                self.pos += 1
                if self.pos < len(self.text) and self.text[self.pos] in '+-':
                    self.pos += 1
            else:
                break
        
        value = self.text[start:self.pos]
        
        # Determine datatype
        if has_exponent:
            return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#double>'
        elif has_decimal:
            return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#decimal>'
        else:
            return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#integer>'
    
    def _parse_boolean(self) -> str:
        """Parse a boolean literal."""
        if self._peek_text("true"):
            self._consume("true")
            return '"true"^^<http://www.w3.org/2001/XMLSchema#boolean>'
        else:
            self._consume("false")
            return '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>'
    
    def _parse_quoted_triple(self) -> Triple:
        """Parse an RDF-Star quoted triple << s p o >>."""
        self._consume("<<")
        self._skip_ws_and_comments()
        
        subject = self._parse_subject()
        self._skip_ws_and_comments()
        
        predicate = self._parse_predicate()
        self._skip_ws_and_comments()
        
        obj = self._parse_object()
        self._skip_ws_and_comments()
        
        self._consume(">>")
        
        if isinstance(subject, Triple):
            return Triple("", predicate, obj if isinstance(obj, str) else "", 
                         subject_triple=subject,
                         object_triple=obj if isinstance(obj, Triple) else None)
        elif isinstance(obj, Triple):
            return Triple(subject, predicate, "", object_triple=obj)
        else:
            return Triple(subject, predicate, obj)


class TurtleSerializer:
    """
    Serializer for Turtle format.
    
    Converts triples to Turtle format with:
    - Prefix declarations
    - Predicate-object grouping
    - Proper escaping
    - RDF-Star quoted triples
    """
    
    def __init__(self, prefixes: Optional[Dict[str, str]] = None):
        """
        Initialize serializer with optional prefixes.
        
        Args:
            prefixes: Dict mapping prefix to namespace IRI
        """
        self.prefixes = prefixes or {}
        self._reverse_prefixes: Dict[str, str] = {}
        self._update_reverse_prefixes()
    
    def _update_reverse_prefixes(self):
        """Build reverse prefix lookup."""
        self._reverse_prefixes = {v: k for k, v in self.prefixes.items()}
    
    def add_prefix(self, prefix: str, namespace: str):
        """Add a prefix mapping."""
        self.prefixes[prefix] = namespace
        self._reverse_prefixes[namespace] = prefix
    
    def serialize(self, triples: List[Triple], base: Optional[str] = None) -> str:
        """
        Serialize triples to Turtle format.
        
        Args:
            triples: List of Triple objects
            base: Optional base IRI
            
        Returns:
            Turtle formatted string
        """
        lines = []
        
        # Write base if provided
        if base:
            lines.append(f"@base <{base}> .")
            lines.append("")
        
        # Write prefixes
        for prefix, namespace in sorted(self.prefixes.items()):
            lines.append(f"@prefix {prefix}: <{namespace}> .")
        
        if self.prefixes:
            lines.append("")
        
        # Group triples by subject
        by_subject: Dict[str, List[Triple]] = {}
        for triple in triples:
            key = self._subject_key(triple)
            if key not in by_subject:
                by_subject[key] = []
            by_subject[key].append(triple)
        
        # Write triples
        for subject_key, subject_triples in by_subject.items():
            # Write subject
            subject = subject_triples[0]
            lines.append(f"{self._format_subject(subject)}")
            
            # Group by predicate
            by_predicate: Dict[str, List[Triple]] = {}
            for triple in subject_triples:
                pred = triple.predicate
                if pred not in by_predicate:
                    by_predicate[pred] = []
                by_predicate[pred].append(triple)
            
            pred_items = list(by_predicate.items())
            for i, (pred, pred_triples) in enumerate(pred_items):
                pred_str = self._compress_iri(pred)
                if pred == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    pred_str = "a"
                
                objects = [self._format_object(t) for t in pred_triples]
                objects_str = " , ".join(objects)
                
                if i < len(pred_items) - 1:
                    lines.append(f"    {pred_str} {objects_str} ;")
                else:
                    lines.append(f"    {pred_str} {objects_str} .")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def serialize_to_file(self, triples: List[Triple], path: Union[str, Path], 
                          base: Optional[str] = None):
        """Serialize triples to a Turtle file."""
        content = self.serialize(triples, base)
        Path(path).write_text(content, encoding="utf-8")
    
    def _subject_key(self, triple: Triple) -> str:
        """Get a key for grouping by subject."""
        if triple.subject_triple:
            return f"<<{triple.subject_triple}>>"
        return triple.subject
    
    def _format_subject(self, triple: Triple) -> str:
        """Format the subject of a triple."""
        if triple.subject_triple:
            return self._format_quoted_triple(triple.subject_triple)
        return self._compress_iri(triple.subject)
    
    def _format_object(self, triple: Triple) -> str:
        """Format the object of a triple."""
        if triple.object_triple:
            return self._format_quoted_triple(triple.object_triple)
        return self._format_term(triple.object)
    
    def _format_quoted_triple(self, triple: Triple) -> str:
        """Format a quoted triple."""
        s = self._format_subject(triple) if not triple.subject_triple else self._format_quoted_triple(triple.subject_triple)
        if triple.subject and not triple.subject_triple:
            s = self._compress_iri(triple.subject)
        p = self._compress_iri(triple.predicate)
        o = self._format_object(triple) if not triple.object_triple else self._format_quoted_triple(triple.object_triple)
        if triple.object and not triple.object_triple:
            o = self._format_term(triple.object)
        return f"<< {s} {p} {o} >>"
    
    def _format_term(self, term: str) -> str:
        """Format a term (IRI, blank node, or literal)."""
        if term.startswith('_:'):
            return term
        elif term.startswith('"'):
            return term  # Already formatted literal
        else:
            return self._compress_iri(term)
    
    def _compress_iri(self, iri: str) -> str:
        """Compress IRI using prefixes if possible."""
        if iri.startswith('_:'):
            return iri
        
        for namespace, prefix in self._reverse_prefixes.items():
            if iri.startswith(namespace):
                local = iri[len(namespace):]
                # Check if local part is valid for prefixed name
                if self._is_valid_local(local):
                    return f"{prefix}:{local}"
        
        return f"<{iri}>"
    
    def _is_valid_local(self, local: str) -> bool:
        """Check if a local name is valid for a prefixed name."""
        if not local:
            return True
        if local[0].isdigit():
            return False
        for c in local:
            if not (c.isalnum() or c in '_-'):
                return False
        return True


# Convenience functions
def parse_turtle(source: Union[str, Path]) -> ParsedDocument:
    """Parse Turtle content or file."""
    parser = TurtleParser()
    if isinstance(source, Path) or (isinstance(source, str) and len(source) < 500 and Path(source).exists()):
        return parser.parse_file(source)
    return parser.parse(source)


def serialize_turtle(triples: List[Triple], prefixes: Optional[Dict[str, str]] = None) -> str:
    """Serialize triples to Turtle format."""
    serializer = TurtleSerializer(prefixes)
    return serializer.serialize(triples)
