"""
NDJSON-LD Parser and Serializer.

NDJSON-LD (Newline-Delimited JSON-LD) is a streaming format where
each line is a complete JSON-LD document. This is useful for:
- Streaming large datasets
- Log-style append-only data
- Line-by-line processing

Each line is parsed as an independent JSON-LD document.

Reference: https://json-ld.org/ (streaming extension)
"""

from typing import List, Union, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
import json

from rdf_starbase.formats.jsonld import JSONLDParser, JSONLDSerializer, JSONLDDocument
from rdf_starbase.formats.turtle import Triple, ParsedDocument


@dataclass
class ParsedNDJSONLD:
    """Result of parsing NDJSON-LD."""
    documents: List[JSONLDDocument] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast insertion."""
        return (
            [t.subject for t in self.triples],
            [t.predicate for t in self.triples],
            [t.object for t in self.triples],
        )


class NDJSONLDParser:
    """
    Parser for NDJSON-LD (Newline-Delimited JSON-LD) format.
    
    Each line is a complete JSON-LD document:
    
        {"@context": {...}, "@id": "...", "name": "..."}
        {"@context": {...}, "@id": "...", "name": "..."}
        {"@context": {...}, "@id": "...", "name": "..."}
    
    Useful for streaming large datasets.
    """
    
    def __init__(self):
        self.jsonld_parser = JSONLDParser()
    
    def parse(self, source: Union[str, Path, StringIO]) -> ParsedNDJSONLD:
        """
        Parse NDJSON-LD content.
        
        Args:
            source: NDJSON-LD content (one JSON-LD document per line)
            
        Returns:
            ParsedNDJSONLD with all documents and merged triples
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, StringIO):
            text = source.read()
        else:
            text = source
        
        documents = []
        all_triples = []
        
        for line_num, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = self.jsonld_parser.parse(line)
                documents.append(doc)
                all_triples.extend(doc.triples)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num}: {e}")
        
        return ParsedNDJSONLD(documents=documents, triples=all_triples)
    
    def parse_as_document(self, source: Union[str, Path, StringIO]) -> ParsedDocument:
        """
        Parse NDJSON-LD and return as flat ParsedDocument.
        
        Args:
            source: NDJSON-LD content
            
        Returns:
            ParsedDocument with all triples
        """
        result = self.parse(source)
        return ParsedDocument(triples=result.triples)
    
    def parse_stream(self, stream) -> Iterator[JSONLDDocument]:
        """
        Parse NDJSON-LD from a stream (file-like object).
        
        Useful for processing large files without loading into memory.
        
        Args:
            stream: File-like object with readline()
            
        Yields:
            JSONLDDocument for each line
        """
        line_num = 0
        for line in stream:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                yield self.jsonld_parser.parse(line)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num}: {e}")


class NDJSONLDSerializer:
    """
    Serializer for NDJSON-LD format.
    
    Groups triples by subject and outputs one JSON-LD object per line.
    """
    
    def __init__(self):
        self.jsonld_serializer = JSONLDSerializer()
    
    def serialize(self, triples: List[Triple], context: dict = None) -> str:
        """
        Serialize triples to NDJSON-LD format.
        
        Groups triples by subject and outputs one line per subject.
        
        Args:
            triples: List of Triple objects
            context: Optional shared @context for all documents
            
        Returns:
            NDJSON-LD string (one JSON object per line)
        """
        # Group triples by subject
        by_subject: dict = {}
        for triple in triples:
            if triple.subject not in by_subject:
                by_subject[triple.subject] = []
            by_subject[triple.subject].append(triple)
        
        lines = []
        for subject, subject_triples in by_subject.items():
            doc = self._triples_to_jsonld(subject, subject_triples, context)
            lines.append(json.dumps(doc, separators=(',', ':')))
        
        return '\n'.join(lines)
    
    def _triples_to_jsonld(self, subject: str, triples: List[Triple], context: dict = None) -> dict:
        """Convert triples with same subject to JSON-LD object."""
        doc = {}
        
        if context:
            doc["@context"] = context
        
        doc["@id"] = subject
        
        for triple in triples:
            pred = triple.predicate
            obj = triple.object
            
            # Convert predicate to key (could use context to shorten)
            key = pred
            
            # Convert object to value
            if obj.startswith('"'):
                # Literal
                if '"^^<' in obj:
                    match_idx = obj.rfind('"^^<')
                    value = obj[1:match_idx]
                    datatype = obj[match_idx + 4:-1]
                    obj_value = {"@value": value, "@type": datatype}
                elif '"@' in obj:
                    match_idx = obj.rfind('"@')
                    value = obj[1:match_idx]
                    lang = obj[match_idx + 2:]
                    obj_value = {"@value": value, "@language": lang}
                else:
                    obj_value = obj[1:-1] if obj.endswith('"') else obj[1:]
            else:
                # URI or blank node
                obj_value = {"@id": obj}
            
            # Handle multiple values for same predicate
            if key in doc:
                if isinstance(doc[key], list):
                    doc[key].append(obj_value)
                else:
                    doc[key] = [doc[key], obj_value]
            else:
                doc[key] = obj_value
        
        return doc


def parse_ndjsonld(source: Union[str, Path, StringIO]) -> ParsedNDJSONLD:
    """
    Parse NDJSON-LD content.
    
    Args:
        source: NDJSON-LD content (one JSON-LD document per line)
        
    Returns:
        ParsedNDJSONLD with all documents and merged triples
    """
    parser = NDJSONLDParser()
    return parser.parse(source)


def parse_ndjsonld_as_document(source: Union[str, Path, StringIO]) -> ParsedDocument:
    """
    Parse NDJSON-LD and return as flat ParsedDocument.
    
    Args:
        source: NDJSON-LD content
        
    Returns:
        ParsedDocument with all triples
    """
    parser = NDJSONLDParser()
    return parser.parse_as_document(source)
