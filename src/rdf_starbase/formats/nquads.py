"""
N-Quads Parser and Serializer.

N-Quads extends N-Triples with a fourth element: the graph name.
Each line contains: subject predicate object [graph] .

Grammar:
  nquadsDoc ::= quad? (EOL quad)* EOL?
  quad      ::= subject predicate object graphLabel? '.'
  graphLabel ::= IRIREF | BLANK_NODE_LABEL

Reference: https://www.w3.org/TR/n-quads/
"""

from typing import Iterator, Optional, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from io import StringIO

from rdf_starbase.formats.ntriples import NTriplesParser, NTriplesSerializer
from rdf_starbase.formats.turtle import Triple, ParsedDocument


@dataclass
class Quad:
    """A quad (triple + graph) representation."""
    subject: str
    predicate: str
    object: str
    graph: Optional[str] = None
    subject_triple: Optional[Triple] = None
    object_triple: Optional[Triple] = None


@dataclass
class ParsedQuads:
    """Result of parsing N-Quads."""
    quads: List[Quad]
    
    def to_triples(self) -> List[Triple]:
        """Convert quads to triples (discarding graph info)."""
        return [
            Triple(
                subject=q.subject,
                predicate=q.predicate,
                object=q.object,
                subject_triple=q.subject_triple,
                object_triple=q.object_triple
            )
            for q in self.quads
        ]
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast insertion."""
        quads = self.quads
        return (
            [q.subject if not q.subject_triple else f"<<{q.subject_triple.subject} {q.subject_triple.predicate} {q.subject_triple.object}>>" for q in quads],
            [q.predicate for q in quads],
            [q.object if not q.object_triple else f"<<{q.object_triple.subject} {q.object_triple.predicate} {q.object_triple.object}>>" for q in quads],
        )


class NQuadsParser(NTriplesParser):
    """
    Parser for N-Quads format.
    
    Extends NTriplesParser to handle the optional fourth element (graph name).
    
    Format:
        <subject> <predicate> <object> .
        <subject> <predicate> <object> <graph> .
    """
    
    def __init__(self):
        super().__init__()
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedQuads:
        """
        Parse N-Quads content.
        
        Args:
            source: N-Quads content as string, file path, or StringIO
            
        Returns:
            ParsedQuads with quads list
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        quads = list(self.parse_quads(text.splitlines()))
        return ParsedQuads(quads=quads)
    
    def parse_as_triples(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse N-Quads and return as triples (discarding graph info).
        
        Args:
            source: N-Quads content
            
        Returns:
            ParsedDocument with triples
        """
        result = self.parse(source)
        return ParsedDocument(triples=result.to_triples())
    
    def parse_quads(self, lines: List[str]) -> Iterator[Quad]:
        """
        Parse lines of N-Quads.
        
        Args:
            lines: List of N-Quads lines
            
        Yields:
            Quad objects
        """
        for i, line in enumerate(lines):
            self.line_number = i + 1
            
            # Strip whitespace and skip empty lines/comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                quad = self._parse_quad_line(line)
                if quad:
                    yield quad
            except Exception as e:
                raise ValueError(f"Error parsing line {self.line_number}: {e}\nLine: {line}")
    
    def _parse_quad_line(self, line: str) -> Optional[Quad]:
        """Parse a single N-Quads line."""
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
        
        # Check for optional graph
        graph = None
        if pos < len(line) and line[pos] != '.':
            graph, pos = self._parse_graph(line, pos)
            pos = self._skip_ws(line, pos)
        
        # Expect period
        if pos < len(line) and line[pos] == '.':
            pos += 1
        
        # Handle RDF-Star triples
        subject_triple = subject if isinstance(subject, Triple) else None
        object_triple = obj if isinstance(obj, Triple) else None
        
        return Quad(
            subject=subject if isinstance(subject, str) else "",
            predicate=predicate,
            object=obj if isinstance(obj, str) else "",
            graph=graph,
            subject_triple=subject_triple,
            object_triple=object_triple
        )
    
    def _parse_graph(self, line: str, pos: int) -> Tuple[str, int]:
        """Parse the optional graph label (IRI or blank node)."""
        pos = self._skip_ws(line, pos)
        
        if pos >= len(line):
            return None, pos
        
        # Check for IRI
        if line[pos] == '<':
            return self._parse_iri(line, pos)
        
        # Check for blank node
        if line[pos:pos+2] == '_:':
            return self._parse_blank_node(line, pos)
        
        return None, pos


class NQuadsSerializer(NTriplesSerializer):
    """
    Serializer for N-Quads format.
    
    Extends NTriplesSerializer to output graph names.
    """
    
    def serialize_quads(self, quads: List[Quad]) -> str:
        """
        Serialize quads to N-Quads format.
        
        Args:
            quads: List of Quad objects
            
        Returns:
            N-Quads formatted string
        """
        lines = []
        for quad in quads:
            subject = self._format_term(quad.subject) if quad.subject else self._format_quoted_triple(quad.subject_triple)
            obj = self._format_term(quad.object) if quad.object else self._format_quoted_triple(quad.object_triple)
            
            if quad.graph:
                line = f"{subject} <{quad.predicate}> {obj} <{quad.graph}> ."
            else:
                line = f"{subject} <{quad.predicate}> {obj} ."
            
            lines.append(line)
        
        return '\n'.join(lines)


def parse_nquads(source: Union[str, Path, StringIO]) -> ParsedQuads:
    """
    Parse N-Quads content.
    
    Args:
        source: N-Quads content as string, file path, or StringIO
        
    Returns:
        ParsedQuads with quads list
    """
    parser = NQuadsParser()
    return parser.parse(source)


def parse_nquads_as_triples(source: Union[str, Path, StringIO]) -> ParsedDocument:
    """
    Parse N-Quads and return as triples (discarding graph info).
    
    For import into the triple store (which handles graphs separately).
    
    Args:
        source: N-Quads content
        
    Returns:
        ParsedDocument with triples
    """
    parser = NQuadsParser()
    return parser.parse_as_triples(source)
