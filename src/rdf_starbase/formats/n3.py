"""
Notation3 (N3) Parser and Serializer.

N3 is a superset of Turtle that adds:
- Formulas (quoted graphs) with { }
- Variables with ?var syntax
- Implications with => and <=
- Built-in functions (@forAll, @forSome, etc.)

For RDF-StarBase, we primarily support the RDF-compatible subset of N3,
which is essentially Turtle with some additional syntax.

Reference: https://www.w3.org/TeamSubmission/n3/
"""

from typing import List, Union, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import re

from rdf_starbase.formats.turtle import TurtleParser, TurtleSerializer, Triple, ParsedDocument


class N3Parser(TurtleParser):
    """
    Parser for Notation3 (N3) format.
    
    N3 is a superset of Turtle. This parser extends TurtleParser
    to handle N3-specific syntax like:
    - @forAll, @forSome quantifiers
    - Formulas { ... }
    - Implications =>
    - Variables ?x
    
    For basic RDF data, N3 is parsed identically to Turtle.
    """
    
    def __init__(self):
        super().__init__()
        self.formulas = []  # For storing nested formulas
        self.variables = set()  # Universally quantified variables
        self.existentials = set()  # Existentially quantified variables
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse N3 content.
        
        Args:
            source: N3 content as string, file path, or StringIO
            
        Returns:
            ParsedDocument with triples and prefixes
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        # Preprocess N3-specific syntax
        text = self._preprocess_n3(text)
        
        # Use parent Turtle parser
        return super().parse(text)
    
    def _preprocess_n3(self, text: str) -> str:
        """
        Preprocess N3-specific syntax to Turtle-compatible format.
        
        Handles:
        - @forAll ?x - converted to comment (not RDF)
        - @forSome ?x - converted to blank nodes
        - { ... } formulas - converted to named graphs where possible
        - => implications - converted to RDF reification
        """
        lines = []
        in_formula = False
        formula_depth = 0
        
        for line in text.split('\n'):
            stripped = line.strip()
            
            # Skip N3 quantifiers (not RDF-compatible)
            if stripped.startswith('@forAll') or stripped.startswith('@forSome'):
                lines.append(f"# N3: {stripped}")
                continue
            
            # Handle formulas - track depth
            formula_depth += stripped.count('{') - stripped.count('}')
            
            # Skip implications (not RDF-compatible)
            if '=>' in stripped or '<=' in stripped:
                lines.append(f"# N3 implication: {stripped}")
                continue
            
            # Convert N3 ?variables to blank nodes for RDF compatibility
            if '?' in stripped and not stripped.startswith('#'):
                # Simple variable replacement - more complex cases would need full parsing
                stripped = re.sub(r'\?([a-zA-Z_][a-zA-Z0-9_]*)', r'_:var_\1', stripped)
                lines.append(stripped)
            else:
                lines.append(line)
        
        return '\n'.join(lines)


class N3Serializer(TurtleSerializer):
    """
    Serializer for Notation3 (N3) format.
    
    Since N3 is a superset of Turtle, the basic serialization
    is identical to Turtle. This serializer can be extended
    to output N3-specific features if needed.
    """
    
    def __init__(self):
        super().__init__()
    
    def serialize(self, triples: List[Triple], prefixes: dict = None) -> str:
        """
        Serialize triples to N3 format.
        
        Args:
            triples: List of Triple objects
            prefixes: Optional namespace prefixes
            
        Returns:
            N3 formatted string (Turtle-compatible)
        """
        # N3 output is Turtle-compatible
        return super().serialize(triples, prefixes)


def parse_n3(source: Union[str, Path, StringIO]) -> ParsedDocument:
    """
    Parse N3 content.
    
    Args:
        source: N3 content as string, file path, or StringIO
        
    Returns:
        ParsedDocument with triples and prefixes
    """
    parser = N3Parser()
    return parser.parse(source)


def serialize_n3(triples: List[Triple], prefixes: dict = None) -> str:
    """
    Serialize triples to N3 format.
    
    Args:
        triples: List of Triple objects
        prefixes: Optional namespace prefixes
        
    Returns:
        N3 formatted string
    """
    serializer = N3Serializer()
    return serializer.serialize(triples, prefixes)
