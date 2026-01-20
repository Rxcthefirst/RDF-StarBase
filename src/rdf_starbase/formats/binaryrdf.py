"""
Binary RDF Parser and Serializer.

This implements a simple binary format for RDF data, inspired by HDT
(Header-Dictionary-Triples) but simplified for practical use.

Structure:
- Header: Magic bytes, version, counts
- Dictionary: Unique terms with varint IDs
- Triples: Subject, predicate, object as varint IDs

This format is:
- Compact: Terms stored once, referenced by ID
- Fast to load: No parsing, just memory mapping
- Streamable: Can append new triples

Format:
    Header (16 bytes):
        Magic: "RDFB" (4 bytes)
        Version: uint16 (2 bytes)
        Flags: uint16 (2 bytes)
        Term count: uint32 (4 bytes)
        Triple count: uint32 (4 bytes)
    
    Dictionary:
        For each term:
            Length: varint
            UTF-8 bytes
    
    Triples:
        For each triple:
            Subject ID: varint
            Predicate ID: varint
            Object ID: varint
"""

from typing import List, Union, Optional, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
import struct

from rdf_starbase.formats.turtle import Triple, ParsedDocument


MAGIC = b"RDFB"
VERSION = 1


@dataclass
class ParsedBinaryRDF:
    """Result of parsing Binary RDF."""
    triples: List[Triple] = field(default_factory=list)
    dictionary: List[str] = field(default_factory=list)
    
    def to_columnar(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract columnar data for fast insertion."""
        return (
            [t.subject for t in self.triples],
            [t.predicate for t in self.triples],
            [t.object for t in self.triples],
        )


class BinaryRDFParser:
    """
    Parser for Binary RDF format.
    
    Reads the binary format and reconstructs triples.
    """
    
    def __init__(self):
        self.dictionary = []
        self.triples = []
    
    def parse(self, source: Union[bytes, Path, BytesIO]) -> ParsedBinaryRDF:
        """
        Parse Binary RDF content.
        
        Args:
            source: Binary data as bytes, file path, or BytesIO
            
        Returns:
            ParsedBinaryRDF with triples and dictionary
        """
        if isinstance(source, Path):
            data = source.read_bytes()
        elif isinstance(source, BytesIO):
            data = source.read()
        else:
            data = source
        
        stream = BytesIO(data)
        
        # Read header
        magic = stream.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid Binary RDF magic: {magic}")
        
        version, flags = struct.unpack("<HH", stream.read(4))
        if version > VERSION:
            raise ValueError(f"Unsupported version: {version}")
        
        term_count, triple_count = struct.unpack("<II", stream.read(8))
        
        # Read dictionary
        self.dictionary = []
        for _ in range(term_count):
            length = self._read_varint(stream)
            term = stream.read(length).decode("utf-8")
            self.dictionary.append(term)
        
        # Read triples
        self.triples = []
        for _ in range(triple_count):
            s_id = self._read_varint(stream)
            p_id = self._read_varint(stream)
            o_id = self._read_varint(stream)
            
            if s_id < len(self.dictionary) and p_id < len(self.dictionary) and o_id < len(self.dictionary):
                self.triples.append(Triple(
                    subject=self.dictionary[s_id],
                    predicate=self.dictionary[p_id],
                    object=self.dictionary[o_id]
                ))
        
        return ParsedBinaryRDF(
            triples=self.triples.copy(),
            dictionary=self.dictionary.copy()
        )
    
    def parse_as_document(self, source: Union[bytes, Path, BytesIO]) -> ParsedDocument:
        """
        Parse Binary RDF and return as ParsedDocument.
        
        Args:
            source: Binary RDF content
            
        Returns:
            ParsedDocument with triples
        """
        result = self.parse(source)
        return ParsedDocument(triples=result.triples)
    
    def _read_varint(self, stream: BytesIO) -> int:
        """Read a variable-length integer."""
        result = 0
        shift = 0
        while True:
            byte = stream.read(1)
            if not byte:
                raise ValueError("Unexpected end of stream")
            b = byte[0]
            result |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
        return result


class BinaryRDFSerializer:
    """
    Serializer for Binary RDF format.
    
    Encodes triples as compact binary data.
    """
    
    def serialize(self, triples: List[Triple]) -> bytes:
        """
        Serialize triples to Binary RDF format.
        
        Args:
            triples: List of Triple objects
            
        Returns:
            Binary data
        """
        # Build dictionary
        term_to_id: Dict[str, int] = {}
        dictionary: List[str] = []
        
        for triple in triples:
            for term in [triple.subject, triple.predicate, triple.object]:
                if term not in term_to_id:
                    term_to_id[term] = len(dictionary)
                    dictionary.append(term)
        
        stream = BytesIO()
        
        # Write header
        stream.write(MAGIC)
        stream.write(struct.pack("<HH", VERSION, 0))  # version, flags
        stream.write(struct.pack("<II", len(dictionary), len(triples)))
        
        # Write dictionary
        for term in dictionary:
            term_bytes = term.encode("utf-8")
            self._write_varint(stream, len(term_bytes))
            stream.write(term_bytes)
        
        # Write triples
        for triple in triples:
            self._write_varint(stream, term_to_id[triple.subject])
            self._write_varint(stream, term_to_id[triple.predicate])
            self._write_varint(stream, term_to_id[triple.object])
        
        return stream.getvalue()
    
    def _write_varint(self, stream: BytesIO, value: int):
        """Write a variable-length integer."""
        while value >= 0x80:
            stream.write(bytes([value & 0x7F | 0x80]))
            value >>= 7
        stream.write(bytes([value]))


def parse_binaryrdf(source: Union[bytes, Path, BytesIO]) -> ParsedBinaryRDF:
    """
    Parse Binary RDF content.
    
    Args:
        source: Binary RDF data
        
    Returns:
        ParsedBinaryRDF with triples and dictionary
    """
    parser = BinaryRDFParser()
    return parser.parse(source)


def parse_binaryrdf_as_document(source: Union[bytes, Path, BytesIO]) -> ParsedDocument:
    """
    Parse Binary RDF and return as ParsedDocument.
    
    Args:
        source: Binary RDF data
        
    Returns:
        ParsedDocument with triples
    """
    parser = BinaryRDFParser()
    return parser.parse_as_document(source)


def serialize_binaryrdf(triples: List[Triple]) -> bytes:
    """
    Serialize triples to Binary RDF format.
    
    Args:
        triples: List of Triple objects
        
    Returns:
        Binary data
    """
    serializer = BinaryRDFSerializer()
    return serializer.serialize(triples)
