"""
N-Triples and N-Quads Parser and Serializer.

N-Triples is the simplest RDF format: one triple per line.
N-Quads extends N-Triples with an optional graph name.

Grammar:
  ntriplesDoc  ::= triple? (EOL triple)* EOL?
  triple       ::= subject predicate object '.'
  subject      ::= IRIREF | BLANK_NODE_LABEL | quotedTriple
  predicate    ::= IRIREF
  object       ::= IRIREF | BLANK_NODE_LABEL | literal | quotedTriple
  quotedTriple ::= '<<' subject predicate object '>>'

Reference: https://www.w3.org/TR/n-triples/
"""

from typing import Iterator, Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import re
from io import StringIO

from rdf_starbase.formats.turtle import Triple, ParsedDocument


class NTriplesParser:
    """
    Parser for N-Triples and N-Triples-Star format.
    
    N-Triples is line-oriented: each line is one triple.
    This makes it efficient for streaming large files.
    
    Format:
        <subject> <predicate> <object> .
        <subject> <predicate> "literal" .
        <subject> <predicate> "literal"@lang .
        <subject> <predicate> "literal"^^<datatype> .
    
    N-Triples-Star adds quoted triples:
        << <s> <p> <o> >> <p2> <o2> .
    """
    
    # Regex patterns for N-Triples tokens
    IRI_PATTERN = re.compile(r'<([^>]*)>')
    BLANK_NODE_PATTERN = re.compile(r'_:([A-Za-z_][A-Za-z0-9_.-]*)')
    LITERAL_PATTERN = re.compile(
        r'"((?:[^"\\]|\\.)*)"|'  # Double-quoted string
        r"'((?:[^'\\]|\\.)*)'"   # Single-quoted string
    )
    LANG_TAG_PATTERN = re.compile(r'@([a-zA-Z]+(?:-[a-zA-Z0-9]+)*)')
    DATATYPE_PATTERN = re.compile(r'\^\^<([^>]*)>')
    QUOTED_TRIPLE_START = re.compile(r'<<')
    QUOTED_TRIPLE_END = re.compile(r'>>')
    
    def __init__(self):
        self.line_number = 0
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse N-Triples content.
        
        Args:
            source: N-Triples content as string, file path, or StringIO
            
        Returns:
            ParsedDocument with triples (no prefixes in N-Triples)
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        triples = list(self.parse_lines(text.splitlines()))
        return ParsedDocument(triples=triples)
    
    def parse_file(self, path: Union[str, Path]) -> ParsedDocument:
        """Parse an N-Triples file."""
        return self.parse(Path(path))
    
    def parse_lines(self, lines: List[str]) -> Iterator[Triple]:
        """
        Parse lines of N-Triples.
        
        Args:
            lines: List of N-Triples lines
            
        Yields:
            Triple objects
        """
        for i, line in enumerate(lines):
            self.line_number = i + 1
            
            # Strip whitespace and skip empty lines/comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                triple = self._parse_line(line)
                if triple:
                    yield triple
            except Exception as e:
                raise ValueError(f"Error parsing line {self.line_number}: {e}\nLine: {line}")
    
    def parse_stream(self, stream) -> Iterator[Triple]:
        """
        Parse N-Triples from a stream (file-like object).
        
        Useful for processing large files without loading into memory.
        
        Args:
            stream: File-like object with readline()
            
        Yields:
            Triple objects
        """
        self.line_number = 0
        for line in stream:
            self.line_number += 1
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            triple = self._parse_line(line)
            if triple:
                yield triple
    
    def _parse_line(self, line: str) -> Optional[Triple]:
        """Parse a single N-Triples line."""
        pos = 0
        
        # Parse subject
        subject, pos = self._parse_subject(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Parse predicate
        predicate, pos = self._parse_iri(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Parse object
        obj, pos = self._parse_object(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Expect period
        if pos < len(line) and line[pos] == '.':
            pos += 1
        
        # Handle RDF-Star: subject or object might be Triple
        if isinstance(subject, Triple):
            return Triple(
                subject="",
                predicate=predicate,
                object=obj if isinstance(obj, str) else "",
                subject_triple=subject,
                object_triple=obj if isinstance(obj, Triple) else None
            )
        elif isinstance(obj, Triple):
            return Triple(
                subject=subject,
                predicate=predicate,
                object="",
                object_triple=obj
            )
        else:
            return Triple(subject=subject, predicate=predicate, object=obj)
    
    def _skip_ws(self, line: str, pos: int) -> int:
        """Skip whitespace."""
        while pos < len(line) and line[pos] in ' \t':
            pos += 1
        return pos
    
    def _parse_subject(self, line: str, pos: int) -> Tuple[Union[str, Triple], int]:
        """Parse a subject (IRI, blank node, or quoted triple)."""
        pos = self._skip_ws(line, pos)
        
        # Check for quoted triple
        if line[pos:pos+2] == '<<':
            return self._parse_quoted_triple(line, pos)
        
        # Check for blank node
        if line[pos:pos+2] == '_:':
            return self._parse_blank_node(line, pos)
        
        # Otherwise IRI
        return self._parse_iri(line, pos)
    
    def _parse_object(self, line: str, pos: int) -> Tuple[Union[str, Triple], int]:
        """Parse an object (IRI, blank node, literal, or quoted triple)."""
        pos = self._skip_ws(line, pos)
        
        # Check for quoted triple
        if line[pos:pos+2] == '<<':
            return self._parse_quoted_triple(line, pos)
        
        # Check for literal
        if line[pos] == '"':
            return self._parse_literal(line, pos)
        
        # Check for blank node
        if line[pos:pos+2] == '_:':
            return self._parse_blank_node(line, pos)
        
        # Otherwise IRI
        return self._parse_iri(line, pos)
    
    def _parse_iri(self, line: str, pos: int) -> Tuple[str, int]:
        """Parse an IRI <...>."""
        if pos >= len(line) or line[pos] != '<':
            raise ValueError(f"Expected '<' at position {pos}")
        
        end = line.find('>', pos + 1)
        if end == -1:
            raise ValueError(f"Unclosed IRI at position {pos}")
        
        iri = line[pos+1:end]
        return iri, end + 1
    
    def _parse_blank_node(self, line: str, pos: int) -> Tuple[str, int]:
        """Parse a blank node _:label."""
        match = self.BLANK_NODE_PATTERN.match(line, pos)
        if not match:
            raise ValueError(f"Invalid blank node at position {pos}")
        
        return f"_:{match.group(1)}", match.end()
    
    def _parse_literal(self, line: str, pos: int) -> Tuple[str, int]:
        """Parse a literal "..."."""
        if line[pos] != '"':
            raise ValueError(f"Expected '\"' at position {pos}")
        
        pos += 1
        value = []
        
        while pos < len(line):
            c = line[pos]
            if c == '"':
                pos += 1
                break
            elif c == '\\':
                pos += 1
                if pos < len(line):
                    escaped = line[pos]
                    if escaped == 'n':
                        value.append('\n')
                    elif escaped == 't':
                        value.append('\t')
                    elif escaped == 'r':
                        value.append('\r')
                    elif escaped == '\\':
                        value.append('\\')
                    elif escaped == '"':
                        value.append('"')
                    elif escaped == 'u':
                        # Unicode escape \uXXXX
                        hex_chars = line[pos+1:pos+5]
                        value.append(chr(int(hex_chars, 16)))
                        pos += 4
                    elif escaped == 'U':
                        # Long unicode escape \UXXXXXXXX
                        hex_chars = line[pos+1:pos+9]
                        value.append(chr(int(hex_chars, 16)))
                        pos += 8
                    else:
                        value.append(escaped)
                    pos += 1
            else:
                value.append(c)
                pos += 1
        
        string_value = ''.join(value)
        
        # Check for language tag
        if pos < len(line) and line[pos] == '@':
            lang_match = self.LANG_TAG_PATTERN.match(line, pos)
            if lang_match:
                lang = lang_match.group(1)
                return f'"{string_value}"@{lang}', lang_match.end()
        
        # Check for datatype
        if pos < len(line) and line[pos:pos+2] == '^^':
            pos += 2
            datatype, pos = self._parse_iri(line, pos)
            return f'"{string_value}"^^<{datatype}>', pos
        
        return f'"{string_value}"', pos
    
    def _parse_quoted_triple(self, line: str, pos: int) -> Tuple[Triple, int]:
        """Parse an RDF-Star quoted triple << s p o >>."""
        if line[pos:pos+2] != '<<':
            raise ValueError(f"Expected '<<' at position {pos}")
        pos += 2
        pos = self._skip_ws(line, pos)
        
        # Parse subject
        subject, pos = self._parse_subject(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Parse predicate
        predicate, pos = self._parse_iri(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Parse object
        obj, pos = self._parse_object(line, pos)
        pos = self._skip_ws(line, pos)
        
        # Expect >>
        if line[pos:pos+2] != '>>':
            raise ValueError(f"Expected '>>' at position {pos}")
        pos += 2
        
        if isinstance(subject, Triple):
            triple = Triple("", predicate, obj if isinstance(obj, str) else "",
                          subject_triple=subject,
                          object_triple=obj if isinstance(obj, Triple) else None)
        elif isinstance(obj, Triple):
            triple = Triple(subject, predicate, "", object_triple=obj)
        else:
            triple = Triple(subject, predicate, obj)
        
        return triple, pos


class NTriplesSerializer:
    """
    Serializer for N-Triples format.
    
    Output is one triple per line, fully expanded (no prefixes).
    """
    
    def serialize(self, triples: List[Triple]) -> str:
        """
        Serialize triples to N-Triples format.
        
        Args:
            triples: List of Triple objects
            
        Returns:
            N-Triples formatted string
        """
        lines = []
        for triple in triples:
            lines.append(self._format_triple(triple))
        return '\n'.join(lines) + '\n' if lines else ''
    
    def serialize_to_file(self, triples: List[Triple], path: Union[str, Path]):
        """Serialize triples to an N-Triples file."""
        content = self.serialize(triples)
        Path(path).write_text(content, encoding="utf-8")
    
    def _format_triple(self, triple: Triple) -> str:
        """Format a single triple as N-Triples line."""
        s = self._format_subject(triple)
        p = self._format_iri(triple.predicate)
        o = self._format_object(triple)
        return f"{s} {p} {o} ."
    
    def _format_subject(self, triple: Triple) -> str:
        """Format the subject."""
        if triple.subject_triple:
            return self._format_quoted_triple(triple.subject_triple)
        return self._format_term(triple.subject)
    
    def _format_object(self, triple: Triple) -> str:
        """Format the object."""
        if triple.object_triple:
            return self._format_quoted_triple(triple.object_triple)
        return self._format_term(triple.object)
    
    def _format_quoted_triple(self, triple: Triple) -> str:
        """Format a quoted triple."""
        s = self._format_subject(triple)
        p = self._format_iri(triple.predicate)
        o = self._format_object(triple)
        return f"<< {s} {p} {o} >>"
    
    def _format_term(self, term: str) -> str:
        """Format a term (IRI, blank node, or literal)."""
        if term.startswith('_:'):
            return term
        elif term.startswith('"'):
            return self._format_literal(term)
        else:
            return self._format_iri(term)
    
    def _format_iri(self, iri: str) -> str:
        """Format an IRI."""
        if iri.startswith('<') and iri.endswith('>'):
            return iri
        return f"<{iri}>"
    
    def _format_literal(self, literal: str) -> str:
        """Format a literal, escaping special characters."""
        # Already formatted literal
        if literal.startswith('"'):
            return literal
        
        # Need to escape
        escaped = literal.replace('\\', '\\\\').replace('"', '\\"')
        escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'"{escaped}"'


# Convenience functions
def parse_ntriples(source: Union[str, Path]) -> ParsedDocument:
    """Parse N-Triples content or file."""
    parser = NTriplesParser()
    if isinstance(source, Path) or (isinstance(source, str) and len(source) < 500 and Path(source).exists()):
        return parser.parse_file(source)
    return parser.parse(source)


def serialize_ntriples(triples: List[Triple]) -> str:
    """Serialize triples to N-Triples format."""
    serializer = NTriplesSerializer()
    return serializer.serialize(triples)
