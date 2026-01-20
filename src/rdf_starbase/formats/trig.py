"""
TriG Parser and Serializer.

TriG extends Turtle to support named graphs with GRAPH blocks.

Grammar:
  trigDoc ::= (directive | block)*
  block   ::= triplesOrGraph | wrappedGraph | triples2 | "GRAPH" labelOrSubject wrappedGraph
  wrappedGraph ::= '{' triplesBlock? '}'

TriG-Star adds RDF-Star quoted triple support within graph blocks.

Reference: https://www.w3.org/TR/trig/
"""

from typing import List, Union, Optional, Iterator, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import re

from rdf_starbase.formats.turtle import TurtleParser, TurtleSerializer, Triple, ParsedDocument


@dataclass
class GraphBlock:
    """A named graph block containing triples."""
    name: Optional[str]  # None for default graph
    triples: List[Triple] = field(default_factory=list)


@dataclass
class ParsedTriG:
    """Result of parsing TriG document."""
    graphs: Dict[str, List[Triple]] = field(default_factory=dict)
    prefixes: Dict[str, str] = field(default_factory=dict)
    base: Optional[str] = None
    
    @property
    def default_graph(self) -> List[Triple]:
        """Get triples in the default graph."""
        return self.graphs.get("", [])
    
    def all_triples(self) -> List[Triple]:
        """Get all triples from all graphs."""
        result = []
        for triples in self.graphs.values():
            result.extend(triples)
        return result
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast insertion."""
        all_triples = self.all_triples()
        return (
            [t.subject for t in all_triples],
            [t.predicate for t in all_triples],
            [t.object for t in all_triples],
        )


class TriGParser(TurtleParser):
    """
    Parser for TriG (and TriG-Star) format.
    
    TriG extends Turtle with named graph blocks:
    
        @prefix ex: <http://example.org/> .
        
        # Default graph
        ex:s ex:p ex:o .
        
        # Named graph
        GRAPH ex:graph1 {
            ex:s1 ex:p1 ex:o1 .
            ex:s2 ex:p2 ex:o2 .
        }
        
        # Alternative syntax
        ex:graph2 {
            ex:s3 ex:p3 ex:o3 .
        }
    """
    
    GRAPH_PATTERN = re.compile(r'(?:GRAPH\s+)?(<[^>]+>|_:\w+|\w+:\w*)\s*\{', re.IGNORECASE)
    
    def __init__(self):
        super().__init__()
        self.graphs = {}
        self.current_graph = ""  # Empty string = default graph
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedTriG:
        """
        Parse TriG content.
        
        Args:
            source: TriG content as string, file path, or StringIO
            
        Returns:
            ParsedTriG with graphs and prefixes
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        self.graphs = {"": []}  # Initialize default graph
        self.prefixes = {}
        self.base = None
        
        # Split into graph blocks and parse each
        self._parse_trig(text)
        
        return ParsedTriG(
            graphs=self.graphs.copy(),
            prefixes=self.prefixes.copy(),
            base=self.base
        )
    
    def parse_as_document(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse TriG and return as flat ParsedDocument (all graphs merged).
        
        Args:
            source: TriG content
            
        Returns:
            ParsedDocument with all triples
        """
        result = self.parse(source)
        return ParsedDocument(
            triples=result.all_triples(),
            prefixes=result.prefixes,
            base=result.base
        )
    
    def _parse_trig(self, text: str):
        """Parse TriG document structure."""
        pos = 0
        current_content = []
        self.current_graph = ""
        brace_depth = 0
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip comments
            if line.startswith('#'):
                i += 1
                continue
            
            # Handle prefix and base
            if line.startswith('@prefix') or line.startswith('@base') or \
               line.upper().startswith('PREFIX') or line.upper().startswith('BASE'):
                self._parse_directive(line)
                i += 1
                continue
            
            # Check for GRAPH block start
            graph_match = self.GRAPH_PATTERN.match(line)
            if graph_match and brace_depth == 0:
                # Flush current content to current graph
                if current_content:
                    self._parse_graph_content('\n'.join(current_content))
                    current_content = []
                
                # Extract graph name
                graph_name = graph_match.group(1)
                self.current_graph = self._expand_iri(graph_name)
                if self.current_graph not in self.graphs:
                    self.graphs[self.current_graph] = []
                
                # Get content after {
                after_brace = line[graph_match.end():]
                brace_depth = 1
                
                if after_brace.strip():
                    current_content.append(after_brace)
                
                i += 1
                continue
            
            # Track brace depth
            if '{' in line and brace_depth == 0:
                # Anonymous graph block
                brace_idx = line.index('{')
                before = line[:brace_idx].strip()
                after = line[brace_idx + 1:].strip()
                
                if before:
                    # Graph name before {
                    if current_content:
                        self._parse_graph_content('\n'.join(current_content))
                        current_content = []
                    
                    self.current_graph = self._expand_iri(before)
                    if self.current_graph not in self.graphs:
                        self.graphs[self.current_graph] = []
                
                brace_depth = 1
                if after:
                    current_content.append(after)
                i += 1
                continue
            
            # Count braces in line
            opens = line.count('{')
            closes = line.count('}')
            
            if brace_depth > 0:
                # Inside a graph block
                brace_depth += opens - closes
                
                if closes > 0 and brace_depth == 0:
                    # End of graph block
                    line_without_close = line.rsplit('}', 1)[0]
                    if line_without_close.strip():
                        current_content.append(line_without_close)
                    
                    self._parse_graph_content('\n'.join(current_content))
                    current_content = []
                    self.current_graph = ""
                else:
                    current_content.append(line)
            else:
                # Default graph content
                if line:
                    current_content.append(line)
            
            i += 1
        
        # Flush remaining content
        if current_content:
            self._parse_graph_content('\n'.join(current_content))
    
    def _parse_directive(self, line: str):
        """Parse @prefix or @base directive."""
        line_lower = line.lower()
        
        if '@prefix' in line_lower or line_lower.startswith('prefix'):
            # Extract prefix: value
            match = re.search(r'(?:@prefix|PREFIX)\s+(\w*):?\s*<([^>]+)>', line, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                uri = match.group(2)
                self.prefixes[prefix] = uri
        
        elif '@base' in line_lower or line_lower.startswith('base'):
            match = re.search(r'(?:@base|BASE)\s+<([^>]+)>', line, re.IGNORECASE)
            if match:
                self.base = match.group(1)
    
    def _parse_graph_content(self, content: str):
        """Parse Turtle content within a graph block."""
        if not content.strip():
            return
        
        # Add prefixes to content for proper parsing
        prefix_block = '\n'.join(f"@prefix {p}: <{u}> ." for p, u in self.prefixes.items())
        if self.base:
            prefix_block = f"@base <{self.base}> .\n{prefix_block}"
        
        full_content = f"{prefix_block}\n{content}"
        
        try:
            result = super().parse(full_content)
            if self.current_graph not in self.graphs:
                self.graphs[self.current_graph] = []
            self.graphs[self.current_graph].extend(result.triples)
        except Exception as e:
            # If parsing fails, try line by line
            pass
    
    def _expand_iri(self, term: str) -> str:
        """Expand a prefixed name or IRI."""
        if term.startswith('<') and term.endswith('>'):
            return term[1:-1]
        
        if ':' in term:
            prefix, local = term.split(':', 1)
            if prefix in self.prefixes:
                return self.prefixes[prefix] + local
        
        return term


class TriGSerializer:
    """
    Serializer for TriG format.
    
    Outputs triples organized by named graphs.
    """
    
    def __init__(self):
        self.turtle_serializer = TurtleSerializer()
    
    def serialize(self, graphs: Dict[str, List[Triple]], prefixes: dict = None) -> str:
        """
        Serialize graphs to TriG format.
        
        Args:
            graphs: Dict mapping graph names to triple lists
            prefixes: Optional namespace prefixes
            
        Returns:
            TriG formatted string
        """
        lines = []
        
        # Output prefixes
        if prefixes:
            for prefix, uri in prefixes.items():
                lines.append(f"@prefix {prefix}: <{uri}> .")
            lines.append("")
        
        # Output each graph
        for graph_name, triples in graphs.items():
            if not triples:
                continue
            
            if graph_name:  # Named graph
                lines.append(f"<{graph_name}> {{")
                
                # Serialize triples with indentation
                turtle_content = self.turtle_serializer.serialize(triples, {})
                for line in turtle_content.split('\n'):
                    if line.strip():
                        lines.append(f"    {line}")
                
                lines.append("}")
                lines.append("")
            else:  # Default graph
                lines.append("# Default graph")
                lines.append(self.turtle_serializer.serialize(triples, {}))
                lines.append("")
        
        return '\n'.join(lines)


def parse_trig(source: Union[str, Path, StringIO]) -> ParsedTriG:
    """
    Parse TriG content.
    
    Args:
        source: TriG content as string, file path, or StringIO
        
    Returns:
        ParsedTriG with graphs and prefixes
    """
    parser = TriGParser()
    return parser.parse(source)


def parse_trig_as_document(source: Union[str, Path, StringIO]) -> ParsedDocument:
    """
    Parse TriG and return as flat ParsedDocument.
    
    For import into the triple store (which handles graphs separately).
    
    Args:
        source: TriG content
        
    Returns:
        ParsedDocument with all triples
    """
    parser = TriGParser()
    return parser.parse_as_document(source)
