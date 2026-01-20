"""
TriX Parser and Serializer.

TriX is an XML-based format for serializing named graphs.
Each graph is enclosed in a <graph> element, with optional <uri> for the name.

Structure:
    <TriX xmlns="http://www.w3.org/2004/03/trix/trix-1/">
        <graph>
            <uri>http://example.org/graph1</uri>
            <triple>
                <uri>http://example.org/subject</uri>
                <uri>http://example.org/predicate</uri>
                <uri>http://example.org/object</uri>
            </triple>
        </graph>
    </TriX>

Reference: https://www.w3.org/2004/03/trix/
"""

from typing import List, Union, Optional, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import re
import xml.etree.ElementTree as ET

from rdf_starbase.formats.turtle import Triple, ParsedDocument


TRIX_NS = "http://www.w3.org/2004/03/trix/trix-1/"


@dataclass
class ParsedTriX:
    """Result of parsing TriX document."""
    graphs: Dict[str, List[Triple]] = field(default_factory=dict)
    
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


class TriXParser:
    """
    Parser for TriX format.
    
    TriX uses XML to represent named graphs:
    
        <TriX xmlns="http://www.w3.org/2004/03/trix/trix-1/">
            <graph>
                <uri>http://example.org/graph</uri>
                <triple>
                    <uri>http://subject</uri>
                    <uri>http://predicate</uri>
                    <plainLiteral>object</plainLiteral>
                </triple>
            </graph>
        </TriX>
    """
    
    def __init__(self):
        self.graphs = {}
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedTriX:
        """
        Parse TriX content.
        
        Args:
            source: TriX XML content as string, file path, or StringIO
            
        Returns:
            ParsedTriX with graphs
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        self.graphs = {"": []}  # Default graph
        
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise ValueError(f"Invalid TriX XML: {e}")
        
        # Handle namespace
        ns = {"trix": TRIX_NS}
        
        # Find all graph elements
        for graph_elem in root.findall(".//trix:graph", ns):
            graph_name = ""
            
            # Check for graph URI
            uri_elem = graph_elem.find("trix:uri", ns)
            if uri_elem is not None and uri_elem.text:
                graph_name = uri_elem.text.strip()
            
            if graph_name not in self.graphs:
                self.graphs[graph_name] = []
            
            # Parse triples in this graph
            for triple_elem in graph_elem.findall("trix:triple", ns):
                triple = self._parse_triple(triple_elem, ns)
                if triple:
                    self.graphs[graph_name].append(triple)
        
        # Also check for triples at root level (without namespace prefix sometimes)
        for graph_elem in root.findall(".//graph"):
            graph_name = ""
            uri_elem = graph_elem.find("uri")
            if uri_elem is not None and uri_elem.text:
                graph_name = uri_elem.text.strip()
            
            if graph_name not in self.graphs:
                self.graphs[graph_name] = []
            
            for triple_elem in graph_elem.findall("triple"):
                triple = self._parse_triple_no_ns(triple_elem)
                if triple:
                    self.graphs[graph_name].append(triple)
        
        return ParsedTriX(graphs=self.graphs.copy())
    
    def parse_as_document(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse TriX and return as flat ParsedDocument.
        
        Args:
            source: TriX content
            
        Returns:
            ParsedDocument with all triples
        """
        result = self.parse(source)
        return ParsedDocument(triples=result.all_triples())
    
    def _parse_triple(self, triple_elem: ET.Element, ns: dict) -> Optional[Triple]:
        """Parse a triple element with namespace."""
        children = list(triple_elem)
        if len(children) < 3:
            return None
        
        subject = self._parse_term(children[0], ns)
        predicate = self._parse_term(children[1], ns)
        obj = self._parse_term(children[2], ns)
        
        if subject and predicate and obj:
            return Triple(subject=subject, predicate=predicate, object=obj)
        return None
    
    def _parse_triple_no_ns(self, triple_elem: ET.Element) -> Optional[Triple]:
        """Parse a triple element without namespace."""
        children = list(triple_elem)
        if len(children) < 3:
            return None
        
        subject = self._parse_term_no_ns(children[0])
        predicate = self._parse_term_no_ns(children[1])
        obj = self._parse_term_no_ns(children[2])
        
        if subject and predicate and obj:
            return Triple(subject=subject, predicate=predicate, object=obj)
        return None
    
    def _parse_term(self, elem: ET.Element, ns: dict) -> Optional[str]:
        """Parse a term element (uri, id, plainLiteral, typedLiteral)."""
        tag = elem.tag.replace(f"{{{TRIX_NS}}}", "")
        return self._parse_term_by_tag(tag, elem)
    
    def _parse_term_no_ns(self, elem: ET.Element) -> Optional[str]:
        """Parse a term element without namespace."""
        tag = elem.tag
        return self._parse_term_by_tag(tag, elem)
    
    def _parse_term_by_tag(self, tag: str, elem: ET.Element) -> Optional[str]:
        """Parse term based on tag name."""
        text = elem.text.strip() if elem.text else ""
        
        if tag == "uri":
            return text
        elif tag == "id":
            return f"_:{text}"
        elif tag == "plainLiteral":
            lang = elem.get("{http://www.w3.org/XML/1998/namespace}lang") or elem.get("lang")
            if lang:
                return f'"{text}"@{lang}'
            return f'"{text}"'
        elif tag == "typedLiteral":
            datatype = elem.get("datatype", "")
            return f'"{text}"^^<{datatype}>'
        
        return text


class TriXSerializer:
    """
    Serializer for TriX format.
    
    Outputs triples as XML.
    """
    
    def serialize(self, graphs: Dict[str, List[Triple]]) -> str:
        """
        Serialize graphs to TriX format.
        
        Args:
            graphs: Dict mapping graph names to triple lists
            
        Returns:
            TriX XML string
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<TriX xmlns="{TRIX_NS}">')
        
        for graph_name, triples in graphs.items():
            if not triples:
                continue
            
            lines.append("  <graph>")
            
            if graph_name:
                lines.append(f"    <uri>{self._escape_xml(graph_name)}</uri>")
            
            for triple in triples:
                lines.append("    <triple>")
                lines.append(f"      {self._serialize_term(triple.subject)}")
                lines.append(f"      {self._serialize_term(triple.predicate)}")
                lines.append(f"      {self._serialize_term(triple.object)}")
                lines.append("    </triple>")
            
            lines.append("  </graph>")
        
        lines.append("</TriX>")
        return '\n'.join(lines)
    
    def _serialize_term(self, term: str) -> str:
        """Serialize a term to TriX XML."""
        if term.startswith("_:"):
            return f"<id>{self._escape_xml(term[2:])}</id>"
        elif term.startswith('"'):
            # Parse literal
            if '"^^<' in term:
                # Typed literal
                match = re.match(r'"(.*)"\^\^<(.*)>$', term)
                if match:
                    value, datatype = match.groups()
                    return f'<typedLiteral datatype="{self._escape_xml(datatype)}">{self._escape_xml(value)}</typedLiteral>'
            elif '"@' in term:
                # Language-tagged literal
                match = re.match(r'"(.*)"@(.*)$', term)
                if match:
                    value, lang = match.groups()
                    return f'<plainLiteral xml:lang="{lang}">{self._escape_xml(value)}</plainLiteral>'
            else:
                # Plain literal
                value = term[1:-1] if term.endswith('"') else term[1:]
                return f"<plainLiteral>{self._escape_xml(value)}</plainLiteral>"
        else:
            # URI
            return f"<uri>{self._escape_xml(term)}</uri>"
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def parse_trix(source: Union[str, Path, StringIO]) -> ParsedTriX:
    """
    Parse TriX content.
    
    Args:
        source: TriX XML content
        
    Returns:
        ParsedTriX with graphs
    """
    parser = TriXParser()
    return parser.parse(source)


def parse_trix_as_document(source: Union[str, Path, StringIO]) -> ParsedDocument:
    """
    Parse TriX and return as flat ParsedDocument.
    
    Args:
        source: TriX content
        
    Returns:
        ParsedDocument with all triples
    """
    parser = TriXParser()
    return parser.parse_as_document(source)
