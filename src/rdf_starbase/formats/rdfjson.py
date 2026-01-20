"""
RDF/JSON Parser and Serializer.

RDF/JSON (not to be confused with JSON-LD) is a direct JSON encoding
of RDF triples. It's structured as nested objects:

{
  "subject1": {
    "predicate1": [
      { "type": "uri", "value": "object1" },
      { "type": "literal", "value": "object2", "lang": "en" }
    ]
  }
}

Reference: https://www.w3.org/TR/rdf-json/
"""

from typing import List, Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import json

from rdf_starbase.formats.turtle import Triple, ParsedDocument


@dataclass
class ParsedRDFJSON:
    """Result of parsing RDF/JSON."""
    triples: List[Triple] = field(default_factory=list)
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast insertion."""
        return (
            [t.subject for t in self.triples],
            [t.predicate for t in self.triples],
            [t.object for t in self.triples],
        )


class RDFJSONParser:
    """
    Parser for RDF/JSON format.
    
    RDF/JSON structure:
    {
        "http://example.org/subject": {
            "http://example.org/predicate": [
                { "type": "uri", "value": "http://example.org/object" },
                { "type": "literal", "value": "text", "lang": "en" },
                { "type": "literal", "value": "42", "datatype": "http://www.w3.org/2001/XMLSchema#integer" },
                { "type": "bnode", "value": "b1" }
            ]
        }
    }
    """
    
    def __init__(self):
        self.triples = []
    
    def parse(self, source: Union[str, Path, StringIO, dict]) -> ParsedRDFJSON:
        """
        Parse RDF/JSON content.
        
        Args:
            source: RDF/JSON content as string, file path, StringIO, or dict
            
        Returns:
            ParsedRDFJSON with triples
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
            data = json.loads(text)
        elif isinstance(source, StringIO):
            data = json.loads(source.read())
        elif isinstance(source, str):
            data = json.loads(source)
        else:
            data = source
        
        self.triples = []
        
        for subject, predicates in data.items():
            subject_str = self._format_subject(subject)
            
            for predicate, objects in predicates.items():
                for obj in objects:
                    obj_str = self._format_object(obj)
                    self.triples.append(Triple(
                        subject=subject_str,
                        predicate=predicate,
                        object=obj_str
                    ))
        
        return ParsedRDFJSON(triples=self.triples.copy())
    
    def parse_as_document(self, source: Union[str, Path, StringIO, dict]) -> ParsedDocument:
        """
        Parse RDF/JSON and return as ParsedDocument.
        
        Args:
            source: RDF/JSON content
            
        Returns:
            ParsedDocument with triples
        """
        result = self.parse(source)
        return ParsedDocument(triples=result.triples)
    
    def _format_subject(self, subject: str) -> str:
        """Format subject (URI or blank node)."""
        if subject.startswith("_:"):
            return subject
        return subject  # Already a URI
    
    def _format_object(self, obj: dict) -> str:
        """Format object based on type."""
        obj_type = obj.get("type", "literal")
        value = obj.get("value", "")
        
        if obj_type == "uri":
            return value
        elif obj_type == "bnode":
            return f"_:{value}"
        elif obj_type == "literal":
            lang = obj.get("lang")
            datatype = obj.get("datatype")
            
            # Escape quotes in value
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            
            if lang:
                return f'"{escaped}"@{lang}'
            elif datatype:
                return f'"{escaped}"^^<{datatype}>'
            else:
                return f'"{escaped}"'
        
        return f'"{value}"'


class RDFJSONSerializer:
    """
    Serializer for RDF/JSON format.
    """
    
    def serialize(self, triples: List[Triple]) -> str:
        """
        Serialize triples to RDF/JSON format.
        
        Args:
            triples: List of Triple objects
            
        Returns:
            RDF/JSON string
        """
        data: Dict[str, Dict[str, List[dict]]] = {}
        
        for triple in triples:
            subject = triple.subject
            predicate = triple.predicate
            obj = triple.object
            
            if subject not in data:
                data[subject] = {}
            
            if predicate not in data[subject]:
                data[subject][predicate] = []
            
            data[subject][predicate].append(self._format_object(obj))
        
        return json.dumps(data, indent=2)
    
    def _format_object(self, obj: str) -> dict:
        """Convert object string to RDF/JSON object."""
        if obj.startswith("_:"):
            return {"type": "bnode", "value": obj[2:]}
        elif obj.startswith('"'):
            # Parse literal
            if '"^^<' in obj:
                # Typed literal
                match_idx = obj.rfind('"^^<')
                value = obj[1:match_idx]
                datatype = obj[match_idx + 4:-1]
                return {"type": "literal", "value": value, "datatype": datatype}
            elif '"@' in obj:
                # Language-tagged
                match_idx = obj.rfind('"@')
                value = obj[1:match_idx]
                lang = obj[match_idx + 2:]
                return {"type": "literal", "value": value, "lang": lang}
            else:
                # Plain literal
                value = obj[1:-1] if obj.endswith('"') else obj[1:]
                return {"type": "literal", "value": value}
        else:
            # URI
            return {"type": "uri", "value": obj}


def parse_rdfjson(source: Union[str, Path, StringIO, dict]) -> ParsedRDFJSON:
    """
    Parse RDF/JSON content.
    
    Args:
        source: RDF/JSON content
        
    Returns:
        ParsedRDFJSON with triples
    """
    parser = RDFJSONParser()
    return parser.parse(source)


def parse_rdfjson_as_document(source: Union[str, Path, StringIO, dict]) -> ParsedDocument:
    """
    Parse RDF/JSON and return as ParsedDocument.
    
    Args:
        source: RDF/JSON content
        
    Returns:
        ParsedDocument with triples
    """
    parser = RDFJSONParser()
    return parser.parse_as_document(source)
